// Package jobs runs pipeline jobs: validated request -> argv -> a queued
// process, with live state (steps, progress, logs) published to
// subscribers and persisted to <data>/jobs/<id>/job.json so the job list
// survives a restart. Outputs are only ever served from the job's own
// manifest, after checking each path is a regular file inside the job's
// output folder.
package jobs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tomott12345/process-audio/web/internal/caps"
	"github.com/tomott12345/process-audio/web/internal/runner"
	"github.com/tomott12345/process-audio/web/internal/upload"
)

type State string

const (
	Queued      State = "queued"
	Running     State = "running"
	Succeeded   State = "succeeded"
	Failed      State = "failed"
	Canceled    State = "canceled"
	Interrupted State = "interrupted"
)

func (s State) Finished() bool { return s != Queued && s != Running }

type Step struct {
	Step   string `json:"step"`
	Label  string `json:"label"`
	Status string `json:"status"` // pending | running | done | failed
	Done   int    `json:"done,omitempty"`
	Total  int    `json:"total,omitempty"`
	Detail string `json:"detail,omitempty"`
}

type File struct {
	Index  int    `json:"index"`
	Kind   string `json:"kind"`
	Format string `json:"format"`
	Name   string `json:"name"`
	Size   int64  `json:"size"`
	Path   string `json:"-"`
}

type Job struct {
	ID         string       `json:"id"`
	UploadID   string       `json:"upload_id"`
	UploadName string       `json:"upload_name"`
	Request    caps.Request `json:"request"`
	Argv       []string     `json:"argv"`
	State      State        `json:"state"`
	Error      string       `json:"error,omitempty"`
	ExitCode   *int         `json:"exit_code,omitempty"`
	CreatedAt  time.Time    `json:"created_at"`
	StartedAt  *time.Time   `json:"started_at,omitempty"`
	FinishedAt *time.Time   `json:"finished_at,omitempty"`
	Steps      []Step       `json:"steps"`
	Files      []File       `json:"files"`
	// Path is stored in job.json (FilePaths) but never sent to clients.
	FilePaths []string `json:"file_paths,omitempty"`
}

// Public is the client view: no server paths, no full argv paths.
func (j *Job) Public() map[string]any {
	b, _ := json.Marshal(j)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	delete(m, "file_paths")
	m["command"] = displayArgv(j.Argv)
	delete(m, "argv")
	return m
}

// displayArgv shows the command with paths shortened to their base names.
func displayArgv(argv []string) string {
	parts := make([]string, 0, len(argv))
	for i, a := range argv {
		switch {
		case i == 0:
			parts = append(parts, "python3")
		case strings.Contains(a, "=") && strings.HasPrefix(a, "--out-dir="):
			parts = append(parts, "--out-dir=<job>")
		case filepath.IsAbs(a):
			parts = append(parts, filepath.Base(a))
		default:
			if strings.ContainsAny(a, " '\"") {
				a = "'" + strings.ReplaceAll(a, "'", `'\''`) + "'"
			}
			parts = append(parts, a)
		}
	}
	return strings.Join(parts, " ")
}

// Message is what subscribers receive: a job snapshot or a log line.
type Message struct {
	Type string         `json:"type"` // "job" | "log"
	Job  map[string]any `json:"job,omitempty"`
	Line string         `json:"line,omitempty"`
}

type entry struct {
	mu     sync.Mutex
	job    *Job
	cancel context.CancelFunc
	logs   []string // tail
	subs   map[chan Message]struct{}
}

type Manager struct {
	dir     string
	caps    *caps.Store
	uploads *upload.Store
	queue   chan string
	mu      sync.Mutex
	jobs    map[string]*entry
	wg      sync.WaitGroup
	baseCtx context.Context
	stop    context.CancelFunc
}

const logTail = 400

func NewManager(dataDir string, cs *caps.Store, us *upload.Store, workers int) (*Manager, error) {
	dir := filepath.Join(dataDir, "jobs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	ctx, stop := context.WithCancel(context.Background())
	m := &Manager{dir: dir, caps: cs, uploads: us, queue: make(chan string, 1000),
		jobs: map[string]*entry{}, baseCtx: ctx, stop: stop}
	if err := m.load(); err != nil {
		return nil, err
	}
	if workers < 1 {
		workers = 1
	}
	for i := 0; i < workers; i++ {
		m.wg.Add(1)
		go m.worker()
	}
	return m, nil
}

// Shutdown cancels running jobs (they are recorded as interrupted) and
// waits for the workers.
func (m *Manager) Shutdown() {
	m.stop()
	close(m.queue)
	m.wg.Wait()
}

func (m *Manager) jobDir(id string) string { return filepath.Join(m.dir, id) }
func (m *Manager) OutDir(id string) string { return filepath.Join(m.dir, id, "out") }

func (m *Manager) load() error {
	ents, err := os.ReadDir(m.dir)
	if err != nil {
		return err
	}
	for _, e := range ents {
		if !e.IsDir() || !upload.ValidID(e.Name()) {
			continue
		}
		b, err := os.ReadFile(filepath.Join(m.dir, e.Name(), "job.json"))
		if err != nil {
			continue
		}
		var j Job
		if json.Unmarshal(b, &j) != nil {
			continue
		}
		for i := range j.Files {
			if i < len(j.FilePaths) {
				j.Files[i].Path = j.FilePaths[i]
			}
		}
		if !j.State.Finished() {
			j.State = Interrupted
			j.Error = "the server stopped while this job was " + map[bool]string{true: "running", false: "queued"}[j.StartedAt != nil]
			now := time.Now().UTC()
			j.FinishedAt = &now
			for i := range j.Steps {
				if j.Steps[i].Status == "running" {
					j.Steps[i].Status = "failed"
				}
			}
		}
		en := &entry{job: &j, subs: map[chan Message]struct{}{}, logs: tailLines(filepath.Join(m.dir, j.ID, "log.txt"), logTail)}
		m.jobs[j.ID] = en
		m.persist(en)
	}
	return nil
}

func (m *Manager) persist(en *entry) {
	j := en.job
	j.FilePaths = j.FilePaths[:0]
	for _, f := range j.Files {
		j.FilePaths = append(j.FilePaths, f.Path)
	}
	b, _ := json.MarshalIndent(j, "", "  ")
	tmp := filepath.Join(m.jobDir(j.ID), "job.json.tmp")
	if err := os.WriteFile(tmp, b, 0o644); err == nil {
		_ = os.Rename(tmp, filepath.Join(m.jobDir(j.ID), "job.json"))
	}
}

// Create validates the request, records the job, and queues it.
func (m *Manager) Create(ctx context.Context, uploadID string, req caps.Request) (*Job, error) {
	meta, err := m.uploads.Get(uploadID)
	if err != nil {
		return nil, err
	}
	c, _, err := m.caps.Get(ctx)
	if err != nil {
		return nil, err
	}
	id := upload.NewID()
	argv, err := c.BuildArgv(req, m.uploads.InputPath(uploadID), m.OutDir(id), false, true)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(m.OutDir(id), 0o755); err != nil {
		return nil, err
	}
	j := &Job{ID: id, UploadID: uploadID, UploadName: meta.OriginalName, Request: req, Argv: argv,
		State: Queued, CreatedAt: time.Now().UTC(), Steps: []Step{}, Files: []File{}}
	en := &entry{job: j, subs: map[chan Message]struct{}{}}
	m.mu.Lock()
	m.jobs[id] = en
	m.mu.Unlock()
	en.mu.Lock()
	m.persist(en)
	en.mu.Unlock()
	select {
	case m.queue <- id:
	default:
		return nil, errors.New("the job queue is full -- try again later")
	}
	return j, nil
}

func (m *Manager) get(id string) (*entry, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	en, ok := m.jobs[id]
	return en, ok
}

// Get returns a copy of the job's client view.
func (m *Manager) Get(id string) (map[string]any, bool) {
	en, ok := m.get(id)
	if !ok {
		return nil, false
	}
	en.mu.Lock()
	defer en.mu.Unlock()
	return en.job.Public(), true
}

// File returns a manifest file of a finished job by index.
func (m *Manager) File(id string, index int) (*File, bool) {
	en, ok := m.get(id)
	if !ok {
		return nil, false
	}
	en.mu.Lock()
	defer en.mu.Unlock()
	if index < 0 || index >= len(en.job.Files) {
		return nil, false
	}
	f := en.job.Files[index]
	return &f, true
}

// Files returns all manifest files of a job.
func (m *Manager) Files(id string) ([]File, string, bool) {
	en, ok := m.get(id)
	if !ok {
		return nil, "", false
	}
	en.mu.Lock()
	defer en.mu.Unlock()
	return append([]File(nil), en.job.Files...), en.job.UploadName, true
}

// List returns all jobs, newest first.
func (m *Manager) List() []map[string]any {
	m.mu.Lock()
	ens := make([]*entry, 0, len(m.jobs))
	for _, en := range m.jobs {
		ens = append(ens, en)
	}
	m.mu.Unlock()
	out := make([]map[string]any, 0, len(ens))
	for _, en := range ens {
		en.mu.Lock()
		out = append(out, en.job.Public())
		en.mu.Unlock()
	}
	sort.Slice(out, func(a, b int) bool { return out[a]["created_at"].(string) > out[b]["created_at"].(string) })
	return out
}

var ErrNotFound = errors.New("no such job")
var ErrBusy = errors.New("the job is still running -- cancel it first")

func (m *Manager) Cancel(id string) error {
	en, ok := m.get(id)
	if !ok {
		return ErrNotFound
	}
	en.mu.Lock()
	defer en.mu.Unlock()
	switch en.job.State {
	case Queued:
		m.finish(en, Canceled, "canceled before it started", nil)
	case Running:
		if en.cancel != nil {
			en.cancel()
		}
	}
	return nil
}

func (m *Manager) Delete(id string) error {
	en, ok := m.get(id)
	if !ok {
		return ErrNotFound
	}
	en.mu.Lock()
	if !en.job.State.Finished() {
		en.mu.Unlock()
		return ErrBusy
	}
	en.mu.Unlock()
	m.mu.Lock()
	delete(m.jobs, id)
	m.mu.Unlock()
	return os.RemoveAll(m.jobDir(id))
}

// Subscribe returns a channel of updates (starting with the current
// snapshot and recent log lines) and a function to stop.
func (m *Manager) Subscribe(id string) (<-chan Message, func(), bool) {
	en, ok := m.get(id)
	if !ok {
		return nil, nil, false
	}
	ch := make(chan Message, 512)
	en.mu.Lock()
	for _, l := range en.logs {
		ch <- Message{Type: "log", Line: l}
	}
	ch <- Message{Type: "job", Job: en.job.Public()}
	en.subs[ch] = struct{}{}
	en.mu.Unlock()
	return ch, func() {
		en.mu.Lock()
		delete(en.subs, ch)
		en.mu.Unlock()
	}, true
}

// publish must be called with en.mu held.
func (m *Manager) publish(en *entry, msg Message) {
	for ch := range en.subs {
		select {
		case ch <- msg:
		default: // a slow client misses a line; the next snapshot catches it up
		}
	}
}

func (m *Manager) snapshot(en *entry) {
	m.publish(en, Message{Type: "job", Job: en.job.Public()})
}

// finish must be called with en.mu held.
func (m *Manager) finish(en *entry, st State, msg string, code *int) {
	now := time.Now().UTC()
	en.job.State, en.job.Error, en.job.FinishedAt, en.job.ExitCode = st, msg, &now, code
	for i := range en.job.Steps {
		if en.job.Steps[i].Status == "running" {
			if st == Succeeded {
				en.job.Steps[i].Status = "done"
			} else {
				en.job.Steps[i].Status = "failed"
			}
		}
	}
	m.persist(en)
	m.snapshot(en)
}

func (m *Manager) worker() {
	defer m.wg.Done()
	for id := range m.queue {
		en, ok := m.get(id)
		if !ok {
			continue
		}
		m.run(en)
	}
}

func (m *Manager) run(en *entry) {
	en.mu.Lock()
	if en.job.State != Queued {
		en.mu.Unlock()
		return
	}
	if m.baseCtx.Err() != nil {
		m.finish(en, Interrupted, "the server stopped before this job started", nil)
		en.mu.Unlock()
		return
	}
	ctx, cancel := context.WithCancel(m.baseCtx)
	defer cancel()
	en.cancel = cancel
	now := time.Now().UTC()
	en.job.State, en.job.StartedAt = Running, &now
	argv := append([]string(nil), en.job.Argv...)
	outDir := m.OutDir(en.job.ID)
	m.persist(en)
	m.snapshot(en)
	en.mu.Unlock()

	c, _, err := m.caps.Get(ctx)
	if err != nil {
		en.mu.Lock()
		m.finish(en, Failed, err.Error(), nil)
		en.mu.Unlock()
		return
	}
	logf, _ := os.Create(filepath.Join(m.jobDir(en.job.ID), "log.txt"))
	if logf != nil {
		defer logf.Close()
	}
	var lastErrLine string
	code, runErr := runner.Stream(ctx, argv, outDir, c.Run.ProgressPrefix+" ",
		func(ev runner.Event) {
			en.mu.Lock()
			defer en.mu.Unlock()
			if apply(en.job, ev, outDir) {
				m.persist(en)
			}
			m.snapshot(en)
		},
		func(stream, line string) {
			if logf != nil {
				fmt.Fprintln(logf, line)
			}
			if strings.HasPrefix(line, "error:") {
				lastErrLine = strings.TrimSpace(strings.TrimPrefix(line, "error:"))
			}
			en.mu.Lock()
			en.logs = append(en.logs, line)
			if len(en.logs) > logTail {
				en.logs = en.logs[len(en.logs)-logTail:]
			}
			m.publish(en, Message{Type: "log", Line: line})
			en.mu.Unlock()
		})

	en.mu.Lock()
	defer en.mu.Unlock()
	switch {
	case m.baseCtx.Err() != nil:
		m.finish(en, Interrupted, "the server stopped while this job was running", &code)
	case ctx.Err() != nil:
		m.finish(en, Canceled, "canceled", &code)
	case runErr != nil || code != 0:
		msg := lastErrLine
		if msg == "" {
			msg = fmt.Sprintf("the pipeline exited with code %d", code)
		}
		m.finish(en, Failed, msg, &code)
	case len(en.job.Files) == 0:
		m.finish(en, Failed, "the pipeline finished but reported no output files", &code)
	default:
		m.finish(en, Succeeded, "", &code)
	}
	log.Printf("job %s %s (exit %d)", en.job.ID, en.job.State, code)
}

// apply folds one progress event into the job. Returns true if it
// changed something worth persisting (not every frame tick).
func apply(j *Job, ev runner.Event, outDir string) bool {
	name := ev.Str("step")
	parent := ev.Str("parent")
	find := func(id string) *Step {
		for i := range j.Steps {
			if j.Steps[i].Step == id {
				return &j.Steps[i]
			}
		}
		return nil
	}
	if parent != "" {
		// a child process working on behalf of one of our steps
		st := find(parent)
		if st == nil {
			return false
		}
		switch ev.Str("event") {
		case "step_start":
			st.Detail = ev.Str("label")
			st.Done, st.Total = 0, 0
		case "progress":
			if d, ok := ev.Int("done"); ok {
				st.Done = d
			}
			if t, ok := ev.Int("total"); ok {
				st.Total = t
			}
			if det := ev.Str("detail"); det != "" && det != "frames" {
				st.Detail = det
			}
		}
		return false
	}
	switch ev.Str("event") {
	case "plan":
		steps, _ := ev["steps"].([]any)
		j.Steps = j.Steps[:0]
		for _, s := range steps {
			if sm, ok := s.(map[string]any); ok {
				id, _ := sm["step"].(string)
				label, _ := sm["label"].(string)
				j.Steps = append(j.Steps, Step{Step: id, Label: label, Status: "pending"})
			}
		}
		return true
	case "step_start":
		if st := find(name); st != nil {
			st.Status = "running"
		} else {
			j.Steps = append(j.Steps, Step{Step: name, Label: ev.Str("label"), Status: "running"})
		}
		return true
	case "step_end":
		if st := find(name); st != nil {
			st.Status, st.Detail = "done", ""
			if st.Total > 0 {
				st.Done = st.Total
			}
		}
		return true
	case "step_error":
		if st := find(name); st != nil {
			st.Status, st.Detail = "failed", ev.Str("message")
		}
		return true
	case "progress":
		if st := find(name); st != nil {
			if d, ok := ev.Int("done"); ok {
				st.Done = d
			}
			if t, ok := ev.Int("total"); ok {
				st.Total = t
			}
			if det := ev.Str("detail"); det != "" && det != "frames" {
				st.Detail = det
			}
		}
		return false
	case "manifest":
		files, _ := ev["files"].([]any)
		j.Files = j.Files[:0]
		for _, f := range files {
			fm, ok := f.(map[string]any)
			if !ok {
				continue
			}
			p, _ := fm["path"].(string)
			safe, size, ok := inside(outDir, p)
			if !ok {
				log.Printf("job %s: ignoring manifest path outside the job folder: %q", j.ID, p)
				continue
			}
			kind, _ := fm["kind"].(string)
			format, _ := fm["format"].(string)
			j.Files = append(j.Files, File{Index: len(j.Files), Kind: kind, Format: format,
				Name: filepath.Base(safe), Size: size, Path: safe})
		}
		return true
	}
	return false
}

// tailLines returns the last n lines of a file (the job log after a restart).
func tailLines(path string, n int) []string {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	lines := strings.Split(strings.TrimRight(string(b), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	if len(lines) == 1 && lines[0] == "" {
		return nil
	}
	return lines
}

// inside checks that p is a regular file (not a symlink) within dir.
func inside(dir, p string) (string, int64, bool) {
	if p == "" || !filepath.IsAbs(p) {
		return "", 0, false
	}
	realDir, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return "", 0, false
	}
	clean := filepath.Clean(p)
	realP, err := filepath.EvalSymlinks(filepath.Dir(clean))
	if err != nil {
		return "", 0, false
	}
	full := filepath.Join(realP, filepath.Base(clean))
	rel, err := filepath.Rel(realDir, full)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", 0, false
	}
	st, err := os.Lstat(full)
	if err != nil || !st.Mode().IsRegular() {
		return "", 0, false
	}
	return full, st.Size(), true
}
