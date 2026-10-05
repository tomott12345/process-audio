package httpapi

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"testing/fstest"
	"time"

	"github.com/tomott12345/process-audio/web/internal/caps"
	"github.com/tomott12345/process-audio/web/internal/jobs"
	"github.com/tomott12345/process-audio/web/internal/upload"
)

// Integration tests: a real server, the real Python scripts, synthetic
// audio from tests/make_test_tracks.py. Skipped with -short or when
// python3/ffmpeg aren't available.

type env struct {
	t     *testing.T
	srv   *httptest.Server
	jobs  *jobs.Manager
	audio map[string]string
}

func repoRoot() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
}

func setup(t *testing.T) *env {
	t.Helper()
	if testing.Short() {
		t.Skip("integration test")
	}
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not on PATH")
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	repo := repoRoot()
	tracks := t.TempDir()
	out, err := exec.Command(py, filepath.Join(repo, "tests", "make_test_tracks.py"), tracks).CombinedOutput()
	if err != nil {
		t.Fatalf("make_test_tracks: %v\n%s", err, out)
	}
	audio := map[string]string{}
	for _, n := range []string{"techno", "speech", "rain", "trap"} {
		audio[n] = filepath.Join(tracks, n+"_test.wav")
	}
	data := t.TempDir()
	cs := caps.NewStore(py, repo)
	us := &upload.Store{Dir: filepath.Join(data, "uploads"), MaxSize: 200 << 20}
	jm, err := jobs.NewManager(data, cs, us, 1)
	if err != nil {
		t.Fatal(err)
	}
	ui := fstest.MapFS{"index.html": {Data: []byte("<!doctype html><title>t</title>")}}
	srv := httptest.NewServer(New(cs, us, jm, ui).Handler())
	t.Cleanup(func() { srv.Close(); jm.Shutdown() })
	return &env{t: t, srv: srv, jobs: jm, audio: audio}
}

func (e *env) do(method, path string, body any, ctype string) (*http.Response, []byte) {
	e.t.Helper()
	var rd io.Reader
	switch b := body.(type) {
	case nil:
	case []byte:
		rd = bytes.NewReader(b)
	default:
		j, _ := json.Marshal(b)
		rd = bytes.NewReader(j)
		ctype = "application/json"
	}
	req, _ := http.NewRequest(method, e.srv.URL+path, rd)
	if ctype != "" {
		req.Header.Set("Content-Type", ctype)
	}
	if method != "GET" {
		req.Header.Set(CSRFHeader, CSRFValue)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	return resp, data
}

func (e *env) upload(path, name string) map[string]any {
	e.t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	fw, _ := mw.CreateFormFile("file", name)
	f, err := os.Open(path)
	if err != nil {
		e.t.Fatal(err)
	}
	io.Copy(fw, f)
	f.Close()
	mw.Close()
	resp, body := e.do("POST", "/api/uploads", buf.Bytes(), mw.FormDataContentType())
	if resp.StatusCode != 201 {
		e.t.Fatalf("upload: %d %s", resp.StatusCode, body)
	}
	var m map[string]any
	json.Unmarshal(body, &m)
	return m
}

// waitJob follows the SSE stream until the job finishes.
func (e *env) waitJob(id string, timeout time.Duration) (map[string]any, []string) {
	e.t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", e.srv.URL+"/api/jobs/"+id+"/events", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		e.t.Fatalf("events content type %q", ct)
	}
	var last map[string]any
	var logs []string
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 8<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var msg jobs.Message
		json.Unmarshal([]byte(line[6:]), &msg)
		if msg.Type == "log" {
			logs = append(logs, msg.Line)
		} else {
			last = msg.Job
		}
	}
	if last == nil {
		e.t.Fatal("no job snapshot on the event stream")
	}
	return last, logs
}

func TestUploadValidation(t *testing.T) {
	e := setup(t)
	// not a WAV
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	fw, _ := mw.CreateFormFile("file", "song.wav")
	fw.Write([]byte("ID3\x03 this is an mp3, honest"))
	mw.Close()
	resp, body := e.do("POST", "/api/uploads", buf.Bytes(), mw.FormDataContentType())
	if resp.StatusCode != http.StatusUnsupportedMediaType || !strings.Contains(string(body), "isn't a WAV") {
		t.Errorf("non-WAV: %d %s", resp.StatusCode, body)
	}
	// missing CSRF header
	req, _ := http.NewRequest("POST", e.srv.URL+"/api/uploads", bytes.NewReader(buf.Bytes()))
	req.Header.Set("Content-Type", mw.FormDataContentType())
	r2, _ := http.DefaultClient.Do(req)
	if r2.StatusCode != http.StatusForbidden {
		t.Errorf("no CSRF header: %d", r2.StatusCode)
	}
	// cross-origin
	req, _ = http.NewRequest("POST", e.srv.URL+"/api/jobs", strings.NewReader("{}"))
	req.Header.Set(CSRFHeader, CSRFValue)
	req.Header.Set("Origin", "https://evil.example")
	r3, _ := http.DefaultClient.Do(req)
	if r3.StatusCode != http.StatusForbidden {
		t.Errorf("cross-origin: %d", r3.StatusCode)
	}
	// a path-ish upload name is kept for display only
	m := e.upload(e.audio["speech"], "../../etc/passwd.wav")
	if m["original_name"] != "passwd.wav" {
		t.Errorf("display name %v", m["original_name"])
	}
	p := m["probe"].(map[string]any)
	if p["channels"].(float64) != 2 || p["sample_rate"].(float64) != 48000 || p["bits"].(float64) != 24 {
		t.Errorf("probe %v", p)
	}
	// bad ids
	for _, path := range []string{"/api/uploads/..%2f..%2fetc", "/api/uploads/zzz", "/api/jobs/nope", "/api/jobs/nope/files/0"} {
		if resp, _ := e.do("GET", path, nil, ""); resp.StatusCode != 404 {
			t.Errorf("%s: %d", path, resp.StatusCode)
		}
	}
}

func TestCapabilitiesHideServerPaths(t *testing.T) {
	e := setup(t)
	resp, body := e.do("GET", "/api/capabilities", nil, "")
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, body)
	}
	var m map[string]any
	json.Unmarshal(body, &m)
	if _, ok := m["python"]; ok {
		t.Error("python path leaked")
	}
	if len(m["pipelines"].([]any)) != 3 {
		t.Error("expected 3 pipelines")
	}
	if csp := resp.Header.Get("Content-Security-Policy"); !strings.Contains(csp, "default-src 'self'") {
		t.Errorf("CSP %q", csp)
	}
}

func TestSpeechJobEndToEnd(t *testing.T) {
	e := setup(t)
	up := e.upload(e.audio["speech"], "interview.wav")
	id := up["id"].(string)

	resp, body := e.do("GET", "/api/uploads/"+id+"/peaks?points=200", nil, "")
	var pk map[string]any
	json.Unmarshal(body, &pk)
	if resp.StatusCode != 200 || pk["points"].(float64) != 200 {
		t.Fatalf("peaks: %d %s", resp.StatusCode, body[:min(len(body), 300)])
	}

	req := map[string]any{"upload_id": id, "pipeline": "speech", "type": "",
		"values":  map[string]any{"title": "-Ep 1-", "trim_silence": true, "preset": "apple"},
		"outputs": []string{"master"}, "formats": []string{"wav24", "flac", "mp3"}}
	resp, body = e.do("POST", "/api/plan", req, "")
	if resp.StatusCode != 200 || !strings.Contains(string(body), `"pipeline": "speech"`) {
		t.Fatalf("plan: %d %s", resp.StatusCode, body)
	}
	bad := map[string]any{"upload_id": id, "pipeline": "speech", "values": map[string]any{"preset": "radio"},
		"outputs": []string{"master"}, "formats": []string{"wav24"}}
	if resp, body := e.do("POST", "/api/jobs", bad, ""); resp.StatusCode != 400 || !strings.Contains(string(body), "not one of the choices") {
		t.Errorf("bad preset: %d %s", resp.StatusCode, body)
	}

	resp, body = e.do("POST", "/api/jobs", req, "")
	if resp.StatusCode != 201 {
		t.Fatalf("create job: %d %s", resp.StatusCode, body)
	}
	var j map[string]any
	json.Unmarshal(body, &j)
	jid := j["id"].(string)
	if strings.Contains(j["command"].(string), e.srv.URL) || strings.Contains(string(body), "/uploads/") {
		t.Errorf("server paths leaked into the job: %s", body)
	}
	final, logs := e.waitJob(jid, 3*time.Minute)
	if final["state"] != "succeeded" {
		t.Fatalf("job %v: %v\nlog tail: %v", final["state"], final["error"], logs[max(0, len(logs)-15):])
	}
	steps := final["steps"].([]any)
	for _, s := range steps {
		if s.(map[string]any)["status"] != "done" {
			t.Errorf("step not done: %v", s)
		}
	}
	files := final["files"].([]any)
	got := map[string]bool{}
	for _, f := range files {
		fm := f.(map[string]any)
		got[fm["kind"].(string)+"/"+fm["format"].(string)] = true
	}
	for _, want := range []string{"master/wav24", "master/flac", "master/mp3", "report/txt"} {
		if !got[want] {
			t.Errorf("missing output %s in %v", want, got)
		}
	}
	// download one with a range request (players seek)
	hreq, _ := http.NewRequest("GET", e.srv.URL+"/api/jobs/"+jid+"/files/1?inline=1", nil)
	hreq.Header.Set("Range", "bytes=0-3")
	hr, err := http.DefaultClient.Do(hreq)
	if err != nil {
		t.Fatal(err)
	}
	b4, _ := io.ReadAll(hr.Body)
	hr.Body.Close()
	if hr.StatusCode != 206 || string(b4) != "fLaC" || !strings.HasPrefix(hr.Header.Get("Content-Disposition"), "inline") {
		t.Errorf("range download: %d %q %s", hr.StatusCode, b4, hr.Header.Get("Content-Disposition"))
	}
	if resp, _ := e.do("GET", "/api/jobs/"+jid+"/files/99", nil, ""); resp.StatusCode != 404 {
		t.Errorf("out-of-range file index: %d", resp.StatusCode)
	}
	// zip of everything
	resp, body = e.do("GET", "/api/jobs/"+jid+"/zip", nil, "")
	zr, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil || len(zr.File) != len(files) {
		t.Fatalf("zip: %v, %d entries for %d files", err, len(zr.File), len(files))
	}
	// the job list survives a restart (reload from job.json)
	resp, body = e.do("GET", "/api/jobs", nil, "")
	if !strings.Contains(string(body), jid) {
		t.Error("job missing from the list")
	}
	if resp, _ := e.do("DELETE", "/api/jobs/"+jid, nil, ""); resp.StatusCode != 204 {
		t.Errorf("delete: %d", resp.StatusCode)
	}
}

func TestAnalyzeMusicAndNature(t *testing.T) {
	e := setup(t)
	id := e.upload(e.audio["techno"], "techno.wav")["id"].(string)
	resp, body := e.do("POST", "/api/uploads/"+id+"/analyze", map[string]any{"pipeline": "music", "type": "techno"}, "")
	if resp.StatusCode != 200 {
		t.Fatalf("music analyze: %d %s", resp.StatusCode, body)
	}
	var a struct {
		Analysis struct {
			Tempo struct {
				HasGrid bool    `json:"has_grid"`
				BPM     float64 `json:"bpm"`
			} `json:"tempo"`
		} `json:"analysis"`
	}
	json.Unmarshal(body, &a)
	if !a.Analysis.Tempo.HasGrid || a.Analysis.Tempo.BPM < 127.9 || a.Analysis.Tempo.BPM > 128.1 {
		t.Errorf("tempo %+v", a.Analysis.Tempo)
	}
	if resp, _ := e.do("POST", "/api/uploads/"+id+"/analyze", map[string]any{"pipeline": "music", "type": "polka"}, ""); resp.StatusCode != 400 {
		t.Errorf("unknown genre: %d", resp.StatusCode)
	}
	rid := e.upload(e.audio["rain"], "rain.wav")["id"].(string)
	resp, body = e.do("POST", "/api/uploads/"+rid+"/analyze", map[string]any{"pipeline": "nature", "type": "rain"}, "")
	if resp.StatusCode != 200 || !strings.Contains(string(body), `"bed"`) {
		t.Errorf("nature analyze: %d %s", resp.StatusCode, body)
	}
}

func TestCancelKillsTheWholeProcessGroup(t *testing.T) {
	e := setup(t)
	id := e.upload(e.audio["trap"], "trap.wav")["id"].(string)
	req := map[string]any{"upload_id": id, "pipeline": "music", "type": "trap",
		"values": map[string]any{}, "outputs": []string{"video_16x9"}, "formats": []string{"wav24"}}
	resp, body := e.do("POST", "/api/jobs", req, "")
	if resp.StatusCode != 201 {
		t.Fatalf("%d %s", resp.StatusCode, body)
	}
	var j map[string]any
	json.Unmarshal(body, &j)
	jid := j["id"].(string)
	// wait until the video render (a grandchild process) is running
	deadline := time.Now().Add(4 * time.Minute)
	for {
		_, b := e.do("GET", "/api/jobs/"+jid, nil, "")
		var cur struct {
			State string      `json:"state"`
			Steps []jobs.Step `json:"steps"`
		}
		json.Unmarshal(b, &cur)
		rendering := false
		for _, st := range cur.Steps {
			rendering = rendering || (st.Step == "video_16x9" && st.Status == "running" && st.Done > 0)
		}
		if rendering {
			break
		}
		if cur.State != "queued" && cur.State != "running" {
			t.Fatalf("job finished before it could be canceled: %s", b)
		}
		if time.Now().After(deadline) {
			t.Fatalf("render never started: %s", b)
		}
		time.Sleep(500 * time.Millisecond)
	}
	time.Sleep(2 * time.Second)
	if resp, b := e.do("POST", "/api/jobs/"+jid+"/cancel", nil, ""); resp.StatusCode != 200 {
		t.Fatalf("cancel: %d %s", resp.StatusCode, b)
	}
	final, _ := e.waitJob(jid, time.Minute)
	if final["state"] != "canceled" {
		t.Fatalf("state %v (%v)", final["state"], final["error"])
	}
	time.Sleep(1 * time.Second)
	out, _ := exec.Command("pgrep", "-f", jid).Output()
	if s := strings.TrimSpace(string(out)); s != "" {
		t.Errorf("processes for the canceled job are still running: %s", s)
	}
}
