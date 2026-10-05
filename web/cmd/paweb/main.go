// Command paweb serves the process-audio web app: upload a WAV, pick a
// pipeline (music / nature / speech) and its options, run it, download the
// results. The audio work is done by the repo's Python scripts; this
// binary validates requests, runs jobs, streams progress, and serves files.
//
//	go run ./cmd/paweb                    # http://127.0.0.1:8765
//	go run ./cmd/paweb -addr 127.0.0.1:9000 -workers 2
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/tomott12345/process-audio/web/internal/caps"
	"github.com/tomott12345/process-audio/web/internal/httpapi"
	"github.com/tomott12345/process-audio/web/internal/jobs"
	"github.com/tomott12345/process-audio/web/internal/upload"
	"github.com/tomott12345/process-audio/web/ui"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8765", "listen address (loopback only for now)")
	repo := flag.String("repo", "", "process-audio repo folder (default: found by walking up from here)")
	python := flag.String("python", "python3", "Python interpreter with the repo's dependencies")
	data := flag.String("data", "", "folder for uploads and jobs (default: <repo>/web/data)")
	workers := flag.Int("workers", 1, "jobs to run at once (video renders use every core; 1 is usually right)")
	maxMB := flag.Int64("max-upload-mb", 2048, "largest accepted upload, in MB")
	flag.Parse()

	if err := checkLoopback(*addr); err != nil {
		log.Fatal(err)
	}
	root := *repo
	if root == "" {
		var err error
		if root, err = findRepo(); err != nil {
			log.Fatal(err)
		}
	}
	root, _ = filepath.Abs(root)
	py, err := exec.LookPath(*python)
	if err != nil {
		log.Fatalf("python interpreter %q not found: %v", *python, err)
	}
	for _, bin := range []string{"ffmpeg", "ffprobe"} {
		if _, err := exec.LookPath(bin); err != nil {
			log.Fatalf("%s not found on PATH -- install ffmpeg (e.g. brew install ffmpeg)", bin)
		}
	}
	dataDir := *data
	if dataDir == "" {
		dataDir = filepath.Join(root, "web", "data")
	}
	if err := os.MkdirAll(filepath.Join(dataDir, "uploads"), 0o755); err != nil {
		log.Fatal(err)
	}

	cs := caps.NewStore(py, root)
	c, _, err := cs.Get(context.Background())
	if err != nil {
		log.Fatalf("loading pipeline capabilities: %v", err)
	}
	for _, p := range c.Pipelines {
		status := "ok"
		if !p.Available {
			status = p.DisabledReason
		}
		log.Printf("pipeline %-7s %d types, %d options, %d outputs (%s)", p.ID, len(p.Types), len(p.Options), len(p.Outputs), status)
	}

	us := &upload.Store{Dir: filepath.Join(dataDir, "uploads"), MaxSize: *maxMB << 20}
	jm, err := jobs.NewManager(dataDir, cs, us, *workers)
	if err != nil {
		log.Fatal(err)
	}
	srv := &http.Server{
		Addr:              *addr,
		Handler:           httpapi.New(cs, us, jm, ui.FS()).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		log.Printf("process-audio web app on http://%s  (repo %s, data %s, python %s)", *addr, root, dataDir, c.Python)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}()
	<-ctx.Done()
	log.Print("shutting down: stopping running jobs")
	shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutCtx)
	jm.Shutdown()
}

// checkLoopback refuses to listen beyond this machine until the token
// login (WEB_APP_PLAN.md phase 5) exists: the app runs scripts on
// uploaded files.
func checkLoopback(addr string) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("bad -addr %q: %v", addr, err)
	}
	if host == "localhost" {
		return nil
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return fmt.Errorf("-addr %q isn't a loopback address; network access waits for the token login (phase 5) -- use 127.0.0.1", addr)
	}
	return nil
}

func findRepo() (string, error) {
	starts := []string{}
	if wd, err := os.Getwd(); err == nil {
		starts = append(starts, wd)
	}
	if exe, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	for _, dir := range starts {
		for d := dir; ; d = filepath.Dir(d) {
			if _, err := os.Stat(filepath.Join(d, "pipeline_capabilities.py")); err == nil {
				return d, nil
			}
			if filepath.Dir(d) == d {
				break
			}
		}
	}
	return "", errors.New("couldn't find the process-audio repo (pipeline_capabilities.py) -- pass -repo")
}
