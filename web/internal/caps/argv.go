package caps

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// Request is what the browser sends: a pipeline, its content type, the
// option values the user touched, and the outputs/formats they want.
type Request struct {
	Pipeline string         `json:"pipeline"`
	Type     string         `json:"type"`
	Values   map[string]any `json:"values"`
	Outputs  []string       `json:"outputs"`
	Formats  []string       `json:"formats"`
}

// ValidationError is a problem with the user's request (HTTP 400), as
// opposed to a server failure.
type ValidationError struct{ Msg string }

func (e *ValidationError) Error() string { return e.Msg }

func invalid(format string, a ...any) error { return &ValidationError{Msg: fmt.Sprintf(format, a...)} }

const maxTextLen = 200

var timeRe = regexp.MustCompile(`^(\d+(\.\d+)?|\d+:\d{1,2}(\.\d+)?|\d+:\d{1,2}:\d{1,2}(\.\d+)?)$`)

// OptionDefault mirrors pipeline_capabilities.option_default.
func (o *Option) OptionDefault(typeID string) any {
	if typeID != "" {
		if v, ok := o.DefaultsByType[typeID]; ok {
			return v
		}
	}
	return o.Default
}

// Active mirrors pipeline_capabilities.option_active: whether the option
// means anything for this type, these outputs, and the other values.
func (p *Pipeline) Active(o *Option, typeID string, values map[string]any, outputs []string) bool {
	if len(o.AppliesTo) > 0 && !contains(o.AppliesTo, typeID) {
		return false
	}
	if len(o.RequiresOutput) > 0 && !intersects(o.RequiresOutput, outputs) {
		return false
	}
	for depID, want := range o.RequiresOption {
		dep, ok := p.Option(depID)
		if !ok {
			return false
		}
		have, set := values[depID]
		if !set {
			have = dep.OptionDefault(typeID)
		}
		if !equal(have, want) {
			return false
		}
	}
	return o.Available
}

// BuildArgv is the Go port of pipeline_capabilities.build_argv -- the
// tests run both on the same cases and require identical output. It is
// also the validation boundary between web input and a process launch,
// so it is stricter than the Python reference: times must look like
// times, text is length-limited and has no control characters.
func (c *Capabilities) BuildArgv(r Request, inputPath, outDir string, plan, progress bool) ([]string, error) {
	p, ok := c.Pipeline(r.Pipeline)
	if !ok {
		return nil, invalid("unknown pipeline %q", r.Pipeline)
	}
	if !p.Available {
		return nil, invalid("the %s pipeline is unavailable: %s", p.Label, p.DisabledReason)
	}
	argv := []string{c.Python, c.Script(p.Script), inputPath}
	if p.Type != nil {
		if !p.HasType(r.Type) {
			return nil, invalid("%s: unknown type %q", p.Type.Label, r.Type)
		}
		argv = append(argv, p.Type.Param+"="+r.Type)
	} else if r.Type != "" {
		return nil, invalid("pipeline %q has no types", r.Pipeline)
	}

	if len(r.Outputs) == 0 {
		return nil, invalid("choose at least one output")
	}
	seen := map[string]bool{}
	for _, id := range r.Outputs {
		o, ok := p.OutputByID(id)
		if !ok {
			return nil, invalid("unknown output %q", id)
		}
		if !o.Available {
			return nil, invalid("output %q unavailable: %s", id, o.DisabledReason)
		}
		if seen[id] {
			return nil, invalid("output %q listed twice", id)
		}
		seen[id] = true
	}

	values := r.Values
	if values == nil {
		values = map[string]any{}
	}
	for id := range values {
		if _, ok := p.Option(id); !ok {
			return nil, invalid("unknown option %q", id)
		}
	}
	for i := range p.Options {
		o := &p.Options[i]
		v, set := values[o.ID]
		if !set || !p.Active(o, r.Type, values, r.Outputs) {
			continue
		}
		if o.Kind != "time_list" && equal(v, o.OptionDefault(r.Type)) {
			continue
		}
		args, err := optionArgs(o, v)
		if err != nil {
			return nil, err
		}
		argv = append(argv, args...)
	}

	if p.OutputsParam != nil && *p.OutputsParam != "" {
		argv = append(argv, *p.OutputsParam+"="+strings.Join(r.Outputs, ","))
	}
	for _, o := range p.Outputs {
		if o.FlagWhenOff != "" && !seen[o.ID] {
			argv = append(argv, o.FlagWhenOff)
		}
	}
	needFormats := false
	for _, id := range r.Outputs {
		if o, _ := p.OutputByID(id); o.Formats {
			needFormats = true
		}
	}
	if needFormats {
		if len(r.Formats) == 0 {
			return nil, invalid("choose at least one audio format")
		}
		for _, f := range r.Formats {
			fm, ok := c.Format(f)
			if !ok || !fm.Available {
				return nil, invalid("format %q unavailable", f)
			}
		}
		argv = append(argv, p.FormatsParam+"="+strings.Join(r.Formats, ","))
	}
	argv = append(argv, c.Run.OutDirFlag+"="+outDir)
	if plan {
		argv = append(argv, c.Run.PlanFlags...)
	} else if progress {
		argv = append(argv, c.Run.ProgressFlag)
	}
	return argv, nil
}

func optionArgs(o *Option, v any) ([]string, error) {
	switch o.Kind {
	case "bool":
		b, ok := v.(bool)
		if !ok {
			return nil, invalid("%s: expected true/false", o.Label)
		}
		if b && o.FlagTrue != "" {
			return []string{o.FlagTrue}, nil
		}
		if !b && o.FlagFalse != "" {
			return []string{o.FlagFalse}, nil
		}
		return nil, nil
	case "number":
		f, ok := v.(float64)
		if !ok || math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, invalid("%s: expected a number", o.Label)
		}
		if (o.Min != nil && f < *o.Min) || (o.Max != nil && f > *o.Max) {
			return nil, invalid("%s: %s is outside %s..%s", o.Label, fmtG(f), fmtG(deref(o.Min)), fmtG(deref(o.Max)))
		}
		return []string{o.Flag + "=" + fmtG(f)}, nil
	case "choice":
		s, ok := v.(string)
		if !ok {
			return nil, invalid("%s: expected one of the choices", o.Label)
		}
		for _, c := range o.Choices {
			if c.Value == s {
				return []string{o.Flag + "=" + s}, nil
			}
		}
		return nil, invalid("%s: %q is not one of the choices", o.Label, s)
	case "time":
		s, ok := v.(string)
		if !ok {
			return nil, invalid("%s: expected a time like 90, 1:30, or 1:02:30", o.Label)
		}
		s = strings.TrimSpace(s)
		if s == "" {
			return nil, nil
		}
		if !timeRe.MatchString(s) {
			return nil, invalid("%s: %q is not a time (use 90, 1:30, or 1:02:30)", o.Label, s)
		}
		return []string{o.Flag + "=" + s}, nil
	case "time_list":
		items, ok := v.([]any)
		if !ok {
			return nil, invalid("%s: expected a list of times", o.Label)
		}
		if len(items) > 50 {
			return nil, invalid("%s: at most 50 entries", o.Label)
		}
		var out []string
		for _, it := range items {
			s, ok := it.(string)
			s = strings.TrimSpace(s)
			if !ok || (s != "end" && !timeRe.MatchString(s)) {
				return nil, invalid("%s: %v is not a time (or 'end')", o.Label, it)
			}
			out = append(out, o.Flag+"="+s)
		}
		return out, nil
	case "text":
		s, ok := v.(string)
		if !ok {
			return nil, invalid("%s: expected text", o.Label)
		}
		if len([]rune(s)) > maxTextLen {
			return nil, invalid("%s: at most %d characters", o.Label, maxTextLen)
		}
		for _, r := range s {
			if unicode.IsControl(r) {
				return nil, invalid("%s: control characters aren't allowed", o.Label)
			}
		}
		if s == "" {
			if o.FlagEmpty != "" {
				return []string{o.FlagEmpty}, nil
			}
			return nil, nil
		}
		return []string{o.Flag + "=" + s}, nil
	}
	return nil, fmt.Errorf("option %s has unknown kind %q", o.ID, o.Kind)
}

// fmtG matches Python's f"{v:g}".
func fmtG(f float64) string { return strconv.FormatFloat(f, 'g', 6, 64) }

func deref(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}

// equal compares decoded JSON values the way Python's == does for them
// (numbers by value, no bool/number mixing).
func equal(a, b any) bool {
	switch x := a.(type) {
	case nil:
		return b == nil
	case bool:
		y, ok := b.(bool)
		return ok && x == y
	case float64:
		y, ok := b.(float64)
		return ok && x == y
	case string:
		y, ok := b.(string)
		return ok && x == y
	}
	return false
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func intersects(a, b []string) bool {
	for _, x := range a {
		if contains(b, x) {
			return true
		}
	}
	return false
}
