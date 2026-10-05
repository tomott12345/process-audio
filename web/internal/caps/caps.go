// Package caps loads pipeline_capabilities.py's JSON -- the description of
// every pipeline, option, output, and format -- and turns validated form
// values into a command line (see argv.go). The Python side is the single
// source of truth; nothing here hard-codes a genre, option, or flag.
package caps

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

type Capabilities struct {
	Version      int             `json:"version"`
	Python       string          `json:"python"`
	Repo         string          `json:"repo"`
	Dependencies map[string]bool `json:"dependencies"`
	Formats      []Format        `json:"formats"`
	Run          RunInfo         `json:"run"`
	Pipelines    []Pipeline      `json:"pipelines"`
}

type Format struct {
	ID             string `json:"id"`
	Label          string `json:"label"`
	Available      bool   `json:"available"`
	DisabledReason string `json:"disabled_reason,omitempty"`
}

type RunInfo struct {
	ProgressFlag   string   `json:"progress_flag"`
	ProgressPrefix string   `json:"progress_prefix"`
	PlanFlags      []string `json:"plan_flags"`
	OutDirFlag     string   `json:"out_dir_flag"`
	Peaks          Script   `json:"peaks"`
}

type Script struct {
	Script     string   `json:"script"`
	Args       []string `json:"args"`
	ResultFile *string  `json:"result_file,omitempty"`
	Result     string   `json:"result,omitempty"`
	Provides   []string `json:"provides,omitempty"`
}

type Pipeline struct {
	ID                 string     `json:"id"`
	Label              string     `json:"label"`
	Description        string     `json:"description"`
	Script             string     `json:"script"`
	RequiresDependency string     `json:"requires_dependency,omitempty"`
	Available          bool       `json:"available"`
	DisabledReason     string     `json:"disabled_reason,omitempty"`
	Type               *TypeParam `json:"type"`
	Types              []Type     `json:"types"`
	Groups             []Group    `json:"groups"`
	Options            []Option   `json:"options"`
	Outputs            []Output   `json:"outputs"`
	OutputsParam       *string    `json:"outputs_param"`
	FormatsParam       string     `json:"formats_param"`
	DefaultFormats     []string   `json:"default_formats"`
	Analyze            *Script    `json:"analyze"`
}

type TypeParam struct {
	Param    string `json:"param"`
	Label    string `json:"label"`
	Required bool   `json:"required"`
}

type Type struct {
	ID       string         `json:"id"`
	Label    string         `json:"label"`
	Untested bool           `json:"untested"`
	Summary  string         `json:"summary"`
	Facts    map[string]any `json:"facts"`
}

type Group struct {
	ID    string `json:"id"`
	Label string `json:"label"`
}

type Choice struct {
	Value string `json:"value"`
	Label string `json:"label"`
}

type Option struct {
	ID                 string            `json:"id"`
	Group              string            `json:"group"`
	Label              string            `json:"label"`
	Kind               string            `json:"kind"`
	Advanced           bool              `json:"advanced"`
	Help               string            `json:"help,omitempty"`
	Detail             string            `json:"detail,omitempty"`
	Flag               string            `json:"flag,omitempty"`
	FlagTrue           string            `json:"flag_true,omitempty"`
	FlagFalse          string            `json:"flag_false,omitempty"`
	FlagEmpty          string            `json:"flag_empty,omitempty"`
	Default            any               `json:"default,omitempty"`
	DefaultsByType     map[string]any    `json:"defaults_by_type,omitempty"`
	DetailByType       map[string]string `json:"detail_by_type,omitempty"`
	Min                *float64          `json:"min,omitempty"`
	Max                *float64          `json:"max,omitempty"`
	Step               *float64          `json:"step,omitempty"`
	Unit               string            `json:"unit,omitempty"`
	Choices            []Choice          `json:"choices,omitempty"`
	AppliesTo          []string          `json:"applies_to,omitempty"`
	RequiresOption     map[string]any    `json:"requires_option,omitempty"`
	RequiresOutput     []string          `json:"requires_output,omitempty"`
	RequiresAnalysis   string            `json:"requires_analysis,omitempty"`
	RequiresDependency string            `json:"requires_dependency,omitempty"`
	Available          bool              `json:"available"`
	DisabledReason     string            `json:"disabled_reason,omitempty"`
}

type Output struct {
	ID                 string `json:"id"`
	Label              string `json:"label"`
	Formats            bool   `json:"formats,omitempty"`
	Default            bool   `json:"default"`
	FlagWhenOff        string `json:"flag_when_off,omitempty"`
	RequiresDependency string `json:"requires_dependency,omitempty"`
	Slow               bool   `json:"slow,omitempty"`
	Available          bool   `json:"available"`
	DisabledReason     string `json:"disabled_reason,omitempty"`
}

func (c *Capabilities) Pipeline(id string) (*Pipeline, bool) {
	for i := range c.Pipelines {
		if c.Pipelines[i].ID == id {
			return &c.Pipelines[i], true
		}
	}
	return nil, false
}

func (c *Capabilities) Format(id string) (*Format, bool) {
	for i := range c.Formats {
		if c.Formats[i].ID == id {
			return &c.Formats[i], true
		}
	}
	return nil, false
}

func (p *Pipeline) Option(id string) (*Option, bool) {
	for i := range p.Options {
		if p.Options[i].ID == id {
			return &p.Options[i], true
		}
	}
	return nil, false
}

func (p *Pipeline) OutputByID(id string) (*Output, bool) {
	for i := range p.Outputs {
		if p.Outputs[i].ID == id {
			return &p.Outputs[i], true
		}
	}
	return nil, false
}

func (p *Pipeline) HasType(id string) bool {
	for _, t := range p.Types {
		if t.ID == id {
			return true
		}
	}
	return false
}

// Script resolves a repo-relative script name to an absolute path.
func (c *Capabilities) Script(name string) string {
	return filepath.Join(c.Repo, name)
}

// Load runs pipeline_capabilities.py and decodes its output.
func Load(ctx context.Context, python, repo string) (*Capabilities, []byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, filepath.Join(repo, "pipeline_capabilities.py"), "--compact")
	cmd.Dir = repo
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, nil, fmt.Errorf("pipeline_capabilities.py failed: %v: %s", err, lastBytes(stderr.Bytes(), 2000))
	}
	var c Capabilities
	if err := json.Unmarshal(out, &c); err != nil {
		return nil, nil, fmt.Errorf("pipeline_capabilities.py printed invalid JSON: %w", err)
	}
	if c.Version != 1 {
		return nil, nil, fmt.Errorf("unsupported capabilities version %d (this server speaks 1)", c.Version)
	}
	return &c, out, nil
}

func lastBytes(b []byte, n int) []byte {
	if len(b) > n {
		return b[len(b)-n:]
	}
	return b
}

// Store caches the capabilities and reloads them when a recipe file the
// Python side reads changes, so editing music_recipes.json shows up in the
// UI without a server restart.
type Store struct {
	python, repo string
	mu           sync.Mutex
	caps         *Capabilities
	raw          []byte
	stamp        string
}

func NewStore(python, repo string) *Store { return &Store{python: python, repo: repo} }

var watched = []string{"music_recipes.json", "recipes.json", "pipeline_capabilities.py", "process_speech_wav.py"}

func (s *Store) fingerprint() string {
	var b bytes.Buffer
	for _, f := range watched {
		if st, err := os.Stat(filepath.Join(s.repo, f)); err == nil {
			fmt.Fprintf(&b, "%s:%d:%d;", f, st.ModTime().UnixNano(), st.Size())
		}
	}
	return b.String()
}

// Get returns the current capabilities (and their raw JSON), reloading if
// a watched file changed since the last load.
func (s *Store) Get(ctx context.Context) (*Capabilities, []byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fp := s.fingerprint()
	if s.caps != nil && fp == s.stamp {
		return s.caps, s.raw, nil
	}
	c, raw, err := Load(ctx, s.python, s.repo)
	if err != nil {
		if s.caps != nil {
			// keep serving the last good copy; a half-edited recipe file
			// shouldn't take the app down
			return s.caps, s.raw, nil
		}
		return nil, nil, err
	}
	s.caps, s.raw, s.stamp = c, raw, fp
	return c, raw, nil
}
