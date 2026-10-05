// Package httpapi is the JSON + server-sent-events API the browser UI
// talks to. Standard library only.
//
//	GET    /api/health
//	GET    /api/capabilities
//	POST   /api/uploads                     multipart "file" -> upload meta + probe
//	GET    /api/uploads/{id}
//	DELETE /api/uploads/{id}
//	GET    /api/uploads/{id}/peaks?points=N waveform overview
//	POST   /api/uploads/{id}/analyze        {pipeline, type} -> analysis JSON
//	POST   /api/plan                        {upload_id, pipeline, type, values, outputs, formats} -> plan JSON
//	POST   /api/jobs                        same body -> job
//	GET    /api/jobs
//	GET    /api/jobs/{id}
//	DELETE /api/jobs/{id}
//	POST   /api/jobs/{id}/cancel
//	GET    /api/jobs/{id}/events            text/event-stream: "job" snapshots + "log" lines
//	GET    /api/jobs/{id}/files/{index}     one output (?inline=1 to play in the browser)
//	GET    /api/jobs/{id}/zip               every output as a zip
//
// Every state-changing request must carry "X-Requested-With:
// process-audio" (a custom header a cross-site page can't send without a
// CORS preflight this server never approves) and, if the browser sends an
// Origin, it must be this server's own.
package httpapi

import (
	"archive/zip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tomott12345/process-audio/web/internal/caps"
	"github.com/tomott12345/process-audio/web/internal/jobs"
	"github.com/tomott12345/process-audio/web/internal/runner"
	"github.com/tomott12345/process-audio/web/internal/upload"
)

const CSRFHeader = "X-Requested-With"
const CSRFValue = "process-audio"

type Server struct {
	Caps    *caps.Store
	Uploads *upload.Store
	Jobs    *jobs.Manager
	UI      fs.FS
	tools   chan struct{} // bounds concurrent analyze/plan/peaks runs
}

func New(c *caps.Store, u *upload.Store, j *jobs.Manager, ui fs.FS) *Server {
	return &Server{Caps: c, Uploads: u, Jobs: j, UI: ui, tools: make(chan struct{}, 2)}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
	})
	mux.HandleFunc("GET /api/capabilities", s.capabilities)
	mux.HandleFunc("POST /api/uploads", s.createUpload)
	mux.HandleFunc("GET /api/uploads/{id}", s.getUpload)
	mux.HandleFunc("DELETE /api/uploads/{id}", s.deleteUpload)
	mux.HandleFunc("GET /api/uploads/{id}/peaks", s.peaks)
	mux.HandleFunc("POST /api/uploads/{id}/analyze", s.analyze)
	mux.HandleFunc("POST /api/plan", s.plan)
	mux.HandleFunc("POST /api/jobs", s.createJob)
	mux.HandleFunc("GET /api/jobs", s.listJobs)
	mux.HandleFunc("GET /api/jobs/{id}", s.getJob)
	mux.HandleFunc("DELETE /api/jobs/{id}", s.deleteJob)
	mux.HandleFunc("POST /api/jobs/{id}/cancel", s.cancelJob)
	mux.HandleFunc("GET /api/jobs/{id}/events", s.events)
	mux.HandleFunc("GET /api/jobs/{id}/files/{index}", s.file)
	mux.HandleFunc("GET /api/jobs/{id}/zip", s.zip)
	mux.Handle("GET /", http.FileServerFS(s.UI))
	return s.middleware(mux)
}

func (s *Server) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "no-referrer")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Content-Security-Policy",
			"default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			if r.Header.Get(CSRFHeader) != CSRFValue {
				writeErr(w, http.StatusForbidden, "missing "+CSRFHeader+" header")
				return
			}
			if o := r.Header.Get("Origin"); o != "" {
				u, err := url.Parse(o)
				if err != nil || u.Host != r.Host {
					writeErr(w, http.StatusForbidden, "cross-origin request refused")
					return
				}
			}
		}
		if !strings.HasPrefix(r.URL.Path, "/api/") {
			h.Set("Cache-Control", "no-cache") // the UI is embedded; always revalidate
		}
		start := time.Now()
		next.ServeHTTP(w, r)
		// log changes and slow reads, not the UI's routine polling
		if d := time.Since(start); strings.HasPrefix(r.URL.Path, "/api/") && !strings.HasSuffix(r.URL.Path, "/events") &&
			(r.Method != http.MethodGet || d > time.Second) {
			log.Printf("%s %s (%s)", r.Method, r.URL.Path, d.Round(time.Millisecond))
		}
	})
}

// ------------------------------------------------------------- helpers

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": msg})
}

// fail maps known error types to HTTP statuses; anything else is a 500
// whose details go to the server log, not the client.
func fail(w http.ResponseWriter, err error) {
	var ve *caps.ValidationError
	var ue *upload.Error
	switch {
	case errors.As(err, &ve):
		writeErr(w, http.StatusBadRequest, ve.Msg)
	case errors.As(err, &ue):
		writeErr(w, ue.Status, ue.Msg)
	case errors.Is(err, jobs.ErrNotFound):
		writeErr(w, http.StatusNotFound, err.Error())
	case errors.Is(err, jobs.ErrBusy):
		writeErr(w, http.StatusConflict, err.Error())
	default:
		log.Printf("internal error: %v", err)
		writeErr(w, http.StatusInternalServerError, "internal error -- see the server log")
	}
}

func decode(r *http.Request, v any) error {
	r.Body = http.MaxBytesReader(nil, r.Body, 1<<20)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return &caps.ValidationError{Msg: "invalid JSON body: " + err.Error()}
	}
	return nil
}

func (s *Server) acquire(ctx context.Context) bool {
	select {
	case s.tools <- struct{}{}:
		return true
	case <-ctx.Done():
		return false
	}
}

func (s *Server) release() { <-s.tools }

// ------------------------------------------------------------- handlers

func (s *Server) capabilities(w http.ResponseWriter, r *http.Request) {
	_, raw, err := s.Caps.Get(r.Context())
	if err != nil {
		fail(w, err)
		return
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		fail(w, err)
		return
	}
	delete(m, "python") // server paths stay on the server
	delete(m, "repo")
	writeJSON(w, http.StatusOK, m)
}

func (s *Server) createUpload(w http.ResponseWriter, r *http.Request) {
	m, err := s.Uploads.Save(r.Context(), w, r)
	if err != nil {
		fail(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, m)
}

func (s *Server) getUpload(w http.ResponseWriter, r *http.Request) {
	m, err := s.Uploads.Get(r.PathValue("id"))
	if err != nil {
		fail(w, err)
		return
	}
	writeJSON(w, http.StatusOK, m)
}

func (s *Server) deleteUpload(w http.ResponseWriter, r *http.Request) {
	if _, err := s.Uploads.Get(r.PathValue("id")); err != nil {
		fail(w, err)
		return
	}
	if err := s.Uploads.Delete(r.PathValue("id")); err != nil {
		fail(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) peaks(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, err := s.Uploads.Get(id); err != nil {
		fail(w, err)
		return
	}
	points := 1000
	if p := r.URL.Query().Get("points"); p != "" {
		n, err := strconv.Atoi(p)
		if err != nil || n < 10 || n > 20000 {
			writeErr(w, http.StatusBadRequest, "points must be 10..20000")
			return
		}
		points = n
	}
	cache := s.Uploads.WorkDir(id, fmt.Sprintf("peaks-%d.json", points))
	if b, err := os.ReadFile(cache); err == nil {
		writeRawJSON(w, b)
		return
	}
	c, _, err := s.Caps.Get(r.Context())
	if err != nil {
		fail(w, err)
		return
	}
	argv := []string{c.Python, c.Script(c.Run.Peaks.Script)}
	argv = append(argv, substitute(c.Run.Peaks.Args, map[string]string{
		"input": s.Uploads.InputPath(id), "points": strconv.Itoa(points)})...)
	out, err := s.tool(r.Context(), argv, c.Repo, 2*time.Minute)
	if err != nil {
		fail(w, err)
		return
	}
	_ = os.MkdirAll(filepath.Dir(cache), 0o755)
	_ = os.WriteFile(cache, out, 0o644)
	writeRawJSON(w, out)
}

func writeRawJSON(w http.ResponseWriter, b []byte) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(b)
}

func (s *Server) tool(ctx context.Context, argv []string, dir string, timeout time.Duration) ([]byte, error) {
	if !s.acquire(ctx) {
		return nil, ctx.Err()
	}
	defer s.release()
	out, stderr, err := runner.Capture(ctx, argv, dir, timeout)
	if err != nil {
		msg := runner.LastLine(stderr)
		msg = strings.TrimPrefix(msg, "error: ")
		if msg == "" {
			msg = err.Error()
		}
		return nil, &caps.ValidationError{Msg: msg}
	}
	return out, nil
}

func substitute(args []string, vars map[string]string) []string {
	out := make([]string, len(args))
	for i, a := range args {
		for k, v := range vars {
			a = strings.ReplaceAll(a, "{"+k+"}", v)
		}
		out[i] = a
	}
	return out
}

type analyzeBody struct {
	Pipeline string `json:"pipeline"`
	Type     string `json:"type"`
}

func (s *Server) analyze(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, err := s.Uploads.Get(id); err != nil {
		fail(w, err)
		return
	}
	var b analyzeBody
	if err := decode(r, &b); err != nil {
		fail(w, err)
		return
	}
	c, _, err := s.Caps.Get(r.Context())
	if err != nil {
		fail(w, err)
		return
	}
	p, ok := c.Pipeline(b.Pipeline)
	if !ok || !p.Available {
		writeErr(w, http.StatusBadRequest, "unknown or unavailable pipeline")
		return
	}
	if p.Analyze == nil {
		writeJSON(w, http.StatusOK, map[string]any{"analysis": nil})
		return
	}
	if p.Type != nil && !p.HasType(b.Type) {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("%s: unknown type %q", p.Type.Label, b.Type))
		return
	}
	dir := s.Uploads.WorkDir(id, "analysis", p.ID+"-"+b.Type)
	cache := filepath.Join(dir, "result.json")
	if out, err := os.ReadFile(cache); err == nil {
		writeRawJSON(w, out)
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fail(w, err)
		return
	}
	argv := append([]string{c.Python, c.Script(p.Analyze.Script)},
		substitute(p.Analyze.Args, map[string]string{"input": s.Uploads.InputPath(id), "type": b.Type, "dir": dir})...)
	out, err := s.tool(r.Context(), argv, dir, 10*time.Minute)
	if err != nil {
		fail(w, err)
		return
	}
	if p.Analyze.ResultFile != nil {
		if out, err = os.ReadFile(filepath.Join(dir, *p.Analyze.ResultFile)); err != nil {
			fail(w, err)
			return
		}
	}
	if !json.Valid(out) {
		fail(w, errors.New("analysis did not produce valid JSON"))
		return
	}
	wrapped, _ := json.Marshal(map[string]any{"pipeline": p.ID, "type": b.Type, "analysis": json.RawMessage(out)})
	_ = os.WriteFile(cache, wrapped, 0o644)
	writeRawJSON(w, wrapped)
}

type jobBody struct {
	UploadID string `json:"upload_id"`
	caps.Request
}

func (s *Server) plan(w http.ResponseWriter, r *http.Request) {
	var b jobBody
	if err := decode(r, &b); err != nil {
		fail(w, err)
		return
	}
	if _, err := s.Uploads.Get(b.UploadID); err != nil {
		fail(w, err)
		return
	}
	c, _, err := s.Caps.Get(r.Context())
	if err != nil {
		fail(w, err)
		return
	}
	dir := s.Uploads.WorkDir(b.UploadID, "plan")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fail(w, err)
		return
	}
	argv, err := c.BuildArgv(b.Request, s.Uploads.InputPath(b.UploadID), dir, true, false)
	if err != nil {
		fail(w, err)
		return
	}
	out, err := s.tool(r.Context(), argv, dir, 10*time.Minute)
	if err != nil {
		fail(w, err)
		return
	}
	if !json.Valid(out) {
		fail(w, errors.New("plan did not produce valid JSON"))
		return
	}
	writeRawJSON(w, out)
}

func (s *Server) createJob(w http.ResponseWriter, r *http.Request) {
	var b jobBody
	if err := decode(r, &b); err != nil {
		fail(w, err)
		return
	}
	j, err := s.Jobs.Create(r.Context(), b.UploadID, b.Request)
	if err != nil {
		fail(w, err)
		return
	}
	pub, _ := s.Jobs.Get(j.ID)
	writeJSON(w, http.StatusCreated, pub)
}

func (s *Server) listJobs(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"jobs": s.Jobs.List()})
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	j, ok := s.Jobs.Get(r.PathValue("id"))
	if !ok {
		fail(w, jobs.ErrNotFound)
		return
	}
	writeJSON(w, http.StatusOK, j)
}

func (s *Server) deleteJob(w http.ResponseWriter, r *http.Request) {
	if err := s.Jobs.Delete(r.PathValue("id")); err != nil {
		fail(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) cancelJob(w http.ResponseWriter, r *http.Request) {
	if err := s.Jobs.Cancel(r.PathValue("id")); err != nil {
		fail(w, err)
		return
	}
	j, _ := s.Jobs.Get(r.PathValue("id"))
	writeJSON(w, http.StatusOK, j)
}

func (s *Server) events(w http.ResponseWriter, r *http.Request) {
	ch, stop, ok := s.Jobs.Subscribe(r.PathValue("id"))
	if !ok {
		fail(w, jobs.ErrNotFound)
		return
	}
	defer stop()
	fl, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-store")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	fl.Flush()
	ping := time.NewTicker(15 * time.Second)
	defer ping.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ping.C:
			fmt.Fprint(w, ": ping\n\n")
			fl.Flush()
		case msg := <-ch:
			b, _ := json.Marshal(msg)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", msg.Type, b)
			fl.Flush()
			if msg.Type == "job" {
				if st, _ := msg.Job["state"].(string); jobs.State(st).Finished() {
					return
				}
			}
		}
	}
}

func (s *Server) file(w http.ResponseWriter, r *http.Request) {
	idx, err := strconv.Atoi(r.PathValue("index"))
	if err != nil {
		fail(w, jobs.ErrNotFound)
		return
	}
	f, ok := s.Jobs.File(r.PathValue("id"), idx)
	if !ok {
		writeErr(w, http.StatusNotFound, "no such file")
		return
	}
	fh, err := os.Open(f.Path)
	if err != nil {
		writeErr(w, http.StatusGone, "that file is no longer on disk")
		return
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil {
		fail(w, err)
		return
	}
	disp := "attachment"
	if r.URL.Query().Get("inline") == "1" {
		disp = "inline"
	}
	w.Header().Set("Content-Disposition", mime.FormatMediaType(disp, map[string]string{"filename": f.Name}))
	if ct := contentType(f.Name); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	http.ServeContent(w, r, f.Name, st.ModTime(), fh) // Range support: players can seek
}

func contentType(name string) string {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".wav":
		return "audio/wav"
	case ".flac":
		return "audio/flac"
	case ".mp3":
		return "audio/mpeg"
	case ".mp4":
		return "video/mp4"
	case ".txt":
		return "text/plain; charset=utf-8"
	case ".json":
		return "application/json"
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	}
	return "application/octet-stream"
}

func (s *Server) zip(w http.ResponseWriter, r *http.Request) {
	files, uploadName, ok := s.Jobs.Files(r.PathValue("id"))
	if !ok {
		fail(w, jobs.ErrNotFound)
		return
	}
	if len(files) == 0 {
		writeErr(w, http.StatusNotFound, "this job has no output files")
		return
	}
	stem := strings.TrimSuffix(uploadName, filepath.Ext(uploadName))
	name := fmt.Sprintf("%s_%s.zip", safeStem(stem), r.PathValue("id")[:8])
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{"filename": name}))
	zw := zip.NewWriter(w)
	used := map[string]bool{}
	for _, f := range files {
		entry := f.Name
		if used[entry] {
			entry = f.Kind + "_" + entry
		}
		used[entry] = true
		method := zip.Deflate
		switch strings.ToLower(filepath.Ext(f.Name)) {
		case ".mp4", ".mp3", ".flac", ".png", ".jpg":
			method = zip.Store // already compressed
		}
		hw, err := zw.CreateHeader(&zip.FileHeader{Name: entry, Method: method, Modified: time.Now()})
		if err != nil {
			return
		}
		fh, err := os.Open(f.Path)
		if err != nil {
			continue
		}
		_, err = io.Copy(hw, fh)
		fh.Close()
		if err != nil {
			return // client went away
		}
	}
	_ = zw.Close()
}

func safeStem(s string) string {
	s = strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		}
		return '_'
	}, s)
	if s == "" {
		s = "release"
	}
	if len(s) > 60 {
		s = s[:60]
	}
	return s
}
