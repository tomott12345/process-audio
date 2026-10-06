package caps

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func repoRoot(t *testing.T) string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
}

func loadCaps(t *testing.T) *Capabilities {
	t.Helper()
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not on PATH")
	}
	c, _, err := Load(context.Background(), py, repoRoot(t))
	if err != nil {
		t.Fatalf("load capabilities: %v", err)
	}
	return c
}

// sampleValue mirrors tests/test_pipeline_contract.py: a valid,
// non-default value for an option.
func sampleValue(o *Option, typeID string) any {
	d := o.OptionDefault(typeID)
	switch o.Kind {
	case "bool":
		b, _ := d.(bool)
		return !b
	case "number":
		for _, c := range []float64{*o.Max, *o.Min, (*o.Min + *o.Max) / 2} {
			if !equal(c, d) {
				return c
			}
		}
	case "choice":
		for _, c := range o.Choices {
			if !equal(c.Value, d) {
				return c.Value
			}
		}
	case "time":
		switch o.ID {
		case "end":
			return "60"
		case "drop_at":
			return "0:20"
		case "clip_start":
			return "0:10"
		case "loop_target":
			return "1:00:00"
		}
		return "1"
	case "time_list":
		return []any{"10", "end"}
	case "text":
		if s, _ := d.(string); s != "" && o.FlagEmpty != "" {
			return ""
		}
		if o.ID == "emoji" {
			return "⭐"
		}
		return "-Leading dash and spaces"
	}
	return nil
}

type tcase struct {
	Req  Request `json:"req"`
	Plan bool    `json:"plan"`
}

func cases(c *Capabilities) []tcase {
	var out []tcase
	for _, p := range c.Pipelines {
		types := []string{""}
		if p.Type != nil {
			types = nil
			for _, t := range p.Types {
				types = append(types, t.ID)
			}
		}
		var allOut, defOut []string
		for _, o := range p.Outputs {
			if o.Available {
				allOut = append(allOut, o.ID)
				if o.Default {
					defOut = append(defOut, o.ID)
				}
			}
		}
		for _, typeID := range types {
			// every option alone, against every type (inactive ones must vanish)
			for i := range p.Options {
				o := &p.Options[i]
				vals := map[string]any{o.ID: sampleValue(o, typeID)}
				out = append(out, tcase{Req: Request{p.ID, typeID, vals, allOut, p.DefaultFormats}})
				// and set to its own default (must be dropped)
				if d := o.OptionDefault(typeID); d != nil {
					out = append(out, tcase{Req: Request{p.ID, typeID, map[string]any{o.ID: d}, defOut, []string{"mp3"}}, Plan: true})
				}
			}
			// everything at once, default outputs, all formats
			all := map[string]any{}
			for i := range p.Options {
				all[p.Options[i].ID] = sampleValue(&p.Options[i], typeID)
			}
			out = append(out, tcase{Req: Request{p.ID, typeID, all, defOut, []string{"wav24", "wav16", "flac", "mp3"}}})
			// everything at once under each value of every option that others
			// depend on (e.g. each visual style: the ffmpeg knobs follow it)
			deps := map[string]bool{}
			for _, o := range p.Options {
				for d := range o.RequiresOption {
					deps[d] = true
				}
			}
			for i := range p.Options {
				o := &p.Options[i]
				if !deps[o.ID] || o.Kind != "choice" {
					continue
				}
				for _, c := range o.Choices {
					vals := map[string]any{}
					for k, v := range all {
						vals[k] = v
					}
					vals[o.ID] = c.Value
					vals["viz_glow"], vals["viz_trails"] = true, true
					if _, ok := p.Option("viz_glow"); !ok {
						delete(vals, "viz_glow")
						delete(vals, "viz_trails")
					}
					out = append(out, tcase{Req: Request{p.ID, typeID, vals, allOut, []string{"wav24"}}})
				}
			}
			// one output at a time
			for _, o := range allOut {
				out = append(out, tcase{Req: Request{p.ID, typeID, nil, []string{o}, []string{"flac"}}})
			}
		}
	}
	// rejections both sides must agree on
	bad := []Request{
		{"music", "house", nil, []string{"master"}, []string{"wav24"}},
		{"music", "techno", map[string]any{"width": 9.0}, []string{"master"}, []string{"wav24"}},
		{"music", "techno", map[string]any{"preset": "spotify"}, []string{"master"}, []string{"wav24"}},
		{"music", "techno", map[string]any{"nope": 1.0}, []string{"master"}, []string{"wav24"}},
		{"music", "techno", map[string]any{"glue": "yes"}, []string{"master"}, []string{"wav24"}},
		{"music", "techno", nil, []string{"master"}, []string{"ogg"}},
		{"music", "techno", nil, []string{}, []string{"wav24"}},
		{"music", "techno", nil, []string{"podcast"}, []string{"wav24"}},
		{"music", "techno", nil, []string{"master"}, []string{}},
		{"speech", "rain", nil, []string{"master"}, []string{"wav24"}},
		{"nature", "", nil, []string{"master_long"}, []string{"wav24"}},
		{"karaoke", "", nil, []string{"master"}, []string{"wav24"}},
	}
	for _, r := range bad {
		out = append(out, tcase{Req: r})
	}
	return out
}

const pyRef = `
import json, sys
sys.path.insert(0, sys.argv[1])
import pipeline_capabilities as pc
caps = pc.capabilities()
res = []
for case in json.load(sys.stdin):
    r = case["req"]
    try:
        res.append({"argv": pc.build_argv(caps, r["pipeline"], r["type"] or None, r["values"] or {},
                    r["outputs"] or [], r["formats"] or [], "IN.wav", "/OUT", plan=case["plan"])})
    except (ValueError, KeyError, TypeError) as e:
        res.append({"error": str(e)})
json.dump(res, sys.stdout)
`

func TestBuildArgvMatchesPythonReference(t *testing.T) {
	c := loadCaps(t)
	cs := cases(c)
	in, _ := json.Marshal(cs)
	cmd := exec.Command(c.Python, "-c", pyRef, c.Repo)
	cmd.Stdin = bytes.NewReader(in)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("python reference: %v\n%s", err, stderr.String())
	}
	var ref []struct {
		Argv  []string `json:"argv"`
		Error string   `json:"error"`
	}
	if err := json.Unmarshal(out, &ref); err != nil {
		t.Fatal(err)
	}
	if len(ref) != len(cs) {
		t.Fatalf("reference returned %d results for %d cases", len(ref), len(cs))
	}
	okCount, errCount := 0, 0
	for i, tc := range cs {
		got, gerr := c.BuildArgv(tc.Req, "IN.wav", "/OUT", tc.Plan, true)
		want := ref[i]
		switch {
		case want.Error != "" && gerr == nil:
			t.Errorf("case %d %+v: python rejected (%s), go accepted %v", i, tc.Req, want.Error, got)
		case want.Error == "" && gerr != nil:
			t.Errorf("case %d %+v: go rejected (%v), python built %v", i, tc.Req, gerr, want.Argv)
		case want.Error == "" && !reflect.DeepEqual(got, want.Argv):
			t.Errorf("case %d %+v:\n  go:     %q\n  python: %q", i, tc.Req, got, want.Argv)
		case want.Error != "":
			var ve *ValidationError
			if !errors.As(gerr, &ve) {
				t.Errorf("case %d: rejection should be a ValidationError, got %T", i, gerr)
			}
			errCount++
		default:
			okCount++
		}
	}
	t.Logf("%d command lines identical to the Python reference, %d rejections agreed", okCount, errCount)
	if okCount < 200 {
		t.Errorf("only %d positive cases -- the case generator is broken", okCount)
	}
}

func TestStricterThanReference(t *testing.T) {
	c := loadCaps(t)
	for _, v := range []map[string]any{
		{"start": "1;rm -rf /"},
		{"start": "abc"},
		{"title": "line1\nline2"},
		{"title": strings.Repeat("x", 201)},
	} {
		_, err := c.BuildArgv(Request{"music", "techno", v, []string{"master"}, []string{"wav24"}}, "i", "o", false, true)
		var ve *ValidationError
		if !errors.As(err, &ve) {
			t.Errorf("%v: want a validation error, got %v", v, err)
		}
	}
}

func TestFmtGMatchesPython(t *testing.T) {
	for in, want := range map[float64]string{100: "100", 0.012: "0.012", -14: "-14", 1.5: "1.5", 0.0005: "0.0005", 1e-5: "1e-05", 123456789: "1.23457e+08"} {
		if got := fmtG(in); got != want {
			t.Errorf("fmtG(%v) = %q, want %q", in, got, want)
		}
	}
}
