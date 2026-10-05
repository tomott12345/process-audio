package jobs

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tomott12345/process-audio/web/internal/runner"
)

func ev(t *testing.T, s string) runner.Event {
	var e runner.Event
	if err := json.Unmarshal([]byte(s), &e); err != nil {
		t.Fatal(err)
	}
	return e
}

func TestApplyFoldsEvents(t *testing.T) {
	out := t.TempDir()
	master := filepath.Join(out, "audio", "master.flac")
	os.MkdirAll(filepath.Dir(master), 0o755)
	os.WriteFile(master, []byte("fLaC"), 0o644)
	j := &Job{ID: "t"}
	for _, s := range []string{
		`{"event":"plan","steps":[{"step":"master","label":"Mastering"},{"step":"video_9x16","label":"Short"}]}`,
		`{"event":"step_start","step":"master","label":"Mastering"}`,
		`{"event":"plan","steps":[{"step":"master","label":"inner"}],"parent":"master"}`,
		`{"event":"step_start","step":"deliverables","label":"Writing audio formats","parent":"master"}`,
		`{"event":"progress","step":"master","detail":"loudness pass","done":2,"total":5,"parent":"master"}`,
		`{"event":"manifest","files":[{"kind":"master","format":"wav24","path":"/elsewhere/x.wav"}],"parent":"master"}`,
		`{"event":"step_end","step":"master"}`,
		`{"event":"step_start","step":"video_9x16","label":"Short"}`,
		`{"event":"progress","step":"video_9x16","done":30,"total":90,"detail":"frames","parent":"video_9x16"}`,
	} {
		apply(j, ev(t, s), out)
	}
	if len(j.Steps) != 2 || j.Steps[0].Status != "done" || j.Steps[1].Status != "running" {
		t.Fatalf("steps: %+v", j.Steps)
	}
	if j.Steps[1].Done != 30 || j.Steps[1].Total != 90 {
		t.Errorf("child frame progress not attributed to the parent step: %+v", j.Steps[1])
	}
	if len(j.Files) != 0 {
		t.Errorf("a child's manifest must not become the job's: %+v", j.Files)
	}
	apply(j, ev(t, `{"event":"manifest","files":[
		{"kind":"master","format":"flac","path":"`+master+`"},
		{"kind":"evil","format":"txt","path":"/etc/passwd"},
		{"kind":"evil","format":"txt","path":"`+out+`/../../etc/passwd"},
		{"kind":"missing","format":"wav","path":"`+out+`/nope.wav"}]}`), out)
	if len(j.Files) != 1 || j.Files[0].Kind != "master" || j.Files[0].Size != 4 {
		t.Fatalf("manifest filtering: %+v", j.Files)
	}
}

func TestInsideRejectsSymlinkEscape(t *testing.T) {
	out := t.TempDir()
	outside := filepath.Join(t.TempDir(), "secret.txt")
	os.WriteFile(outside, []byte("x"), 0o644)
	link := filepath.Join(out, "link.txt")
	if err := os.Symlink(outside, link); err != nil {
		t.Skip(err)
	}
	if _, _, ok := inside(out, link); ok {
		t.Error("a symlink pointing outside the job folder was accepted")
	}
	dirLink := filepath.Join(out, "d")
	os.Symlink(filepath.Dir(outside), dirLink)
	if _, _, ok := inside(out, filepath.Join(dirLink, "secret.txt")); ok {
		t.Error("a file reached through a symlinked folder was accepted")
	}
}

func TestDisplayArgvHidesPaths(t *testing.T) {
	got := displayArgv([]string{"/opt/py/bin/python3", "/repo/release.py", "/data/uploads/x/input.wav",
		"--genre=techno", "--title=Night Drive", "--out-dir=/data/jobs/y/out", "--progress-json"})
	want := "python3 release.py input.wav --genre=techno '--title=Night Drive' --out-dir=<job> --progress-json"
	if got != want {
		t.Errorf("\n got %s\nwant %s", got, want)
	}
}
