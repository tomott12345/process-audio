// Package upload stores uploaded WAV files: streamed straight to disk
// (never buffered in memory), checked by their RIFF/WAVE header rather
// than the file extension, and probed with ffprobe. Each upload lives in
// <data>/uploads/<id>/ as input.wav + meta.json; the original file name is
// kept for display only and never touches the filesystem.
package upload

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"
)

type Probe struct {
	Duration   float64 `json:"duration"`
	SampleRate int     `json:"sample_rate"`
	Channels   int     `json:"channels"`
	Bits       int     `json:"bits"`
	Codec      string  `json:"codec"`
}

type Meta struct {
	ID           string    `json:"id"`
	OriginalName string    `json:"original_name"`
	Size         int64     `json:"size"`
	CreatedAt    time.Time `json:"created_at"`
	Probe        Probe     `json:"probe"`
}

// Error is a problem with the uploaded file itself (HTTP 4xx).
type Error struct {
	Status int
	Msg    string
}

func (e *Error) Error() string { return e.Msg }

type Store struct {
	Dir     string // <data>/uploads
	MaxSize int64
}

var idRe = regexp.MustCompile(`^[0-9a-f]{20}$`)

func NewID() string {
	b := make([]byte, 10)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}

func ValidID(id string) bool { return idRe.MatchString(id) }

func (s *Store) dir(id string) string { return filepath.Join(s.Dir, id) }

// InputPath is the stored WAV for an upload.
func (s *Store) InputPath(id string) string { return filepath.Join(s.dir(id), "input.wav") }

// WorkDir is a scratch folder inside the upload for analysis results.
func (s *Store) WorkDir(id string, parts ...string) string {
	return filepath.Join(append([]string{s.dir(id), "work"}, parts...)...)
}

func (s *Store) Get(id string) (*Meta, error) {
	if !ValidID(id) {
		return nil, &Error{http.StatusNotFound, "no such upload"}
	}
	b, err := os.ReadFile(filepath.Join(s.dir(id), "meta.json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, &Error{http.StatusNotFound, "no such upload"}
	} else if err != nil {
		return nil, err
	}
	var m Meta
	return &m, json.Unmarshal(b, &m)
}

func (s *Store) Delete(id string) error {
	if !ValidID(id) {
		return &Error{http.StatusNotFound, "no such upload"}
	}
	return os.RemoveAll(s.dir(id))
}

// Save streams the "file" part of a multipart request to disk.
func (s *Store) Save(ctx context.Context, w http.ResponseWriter, r *http.Request) (*Meta, error) {
	r.Body = http.MaxBytesReader(w, r.Body, s.MaxSize+1<<20) // + room for multipart framing
	mr, err := r.MultipartReader()
	if err != nil {
		return nil, &Error{http.StatusBadRequest, "expected a multipart/form-data upload with a \"file\" field"}
	}
	var part *multipart.Part
	for {
		part, err = mr.NextPart()
		if err == io.EOF {
			return nil, &Error{http.StatusBadRequest, "no \"file\" field in the upload"}
		}
		if err != nil {
			return nil, uploadErr(err)
		}
		if part.FormName() == "file" {
			break
		}
		part.Close()
	}
	defer part.Close()

	id := NewID()
	dir := s.dir(id)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			os.RemoveAll(dir)
		}
	}()

	// header check before writing anything big
	head := make([]byte, 12)
	if _, err := io.ReadFull(part, head); err != nil {
		return nil, &Error{http.StatusBadRequest, "that file is too short to be a WAV"}
	}
	if !isWAV(head) {
		return nil, &Error{http.StatusUnsupportedMediaType, "that isn't a WAV file (no RIFF/WAVE header) -- export or record as .wav"}
	}
	f, err := os.Create(filepath.Join(dir, "input.wav"))
	if err != nil {
		return nil, err
	}
	n, err := io.Copy(f, io.MultiReader(bytes.NewReader(head), part))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return nil, uploadErr(err)
	}
	if n > s.MaxSize {
		return nil, &Error{http.StatusRequestEntityTooLarge, fmt.Sprintf("the file is larger than the %s limit", human(s.MaxSize))}
	}

	pr, err := ProbeFile(ctx, filepath.Join(dir, "input.wav"))
	if err != nil {
		return nil, &Error{http.StatusUnprocessableEntity, "ffprobe couldn't read that WAV: " + err.Error()}
	}
	if pr.Channels < 1 || pr.Channels > 2 {
		return nil, &Error{http.StatusUnprocessableEntity,
			fmt.Sprintf("that WAV has %d channels -- upload a mono or stereo file (for a Bluebox session, the stereo mix)", pr.Channels)}
	}
	if pr.Duration < 1 {
		return nil, &Error{http.StatusUnprocessableEntity, "that WAV is shorter than a second"}
	}
	m := &Meta{ID: id, OriginalName: displayName(part.FileName()), Size: n, CreatedAt: time.Now().UTC(), Probe: *pr}
	b, _ := json.MarshalIndent(m, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, "meta.json"), b, 0o644); err != nil {
		return nil, err
	}
	ok = true
	return m, nil
}

func isWAV(h []byte) bool {
	riff := string(h[0:4])
	return (riff == "RIFF" || riff == "RF64" || riff == "BW64") && string(h[8:12]) == "WAVE"
}

func uploadErr(err error) error {
	var mbe *http.MaxBytesError
	if errors.As(err, &mbe) {
		return &Error{http.StatusRequestEntityTooLarge, fmt.Sprintf("the file is larger than the %s limit", human(mbe.Limit))}
	}
	return &Error{http.StatusBadRequest, "upload interrupted: " + err.Error()}
}

// displayName keeps the user's file name for display: base name only,
// printable, bounded.
func displayName(name string) string {
	name = filepath.Base(strings.ReplaceAll(name, "\\", "/"))
	name = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) {
			return -1
		}
		return r
	}, name)
	if r := []rune(name); len(r) > 120 {
		name = string(r[:120])
	}
	if name == "" || name == "." || name == "/" {
		name = "upload.wav"
	}
	return name
}

func human(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MB", float64(n)/(1<<20))
	}
	return fmt.Sprintf("%d bytes", n)
}

// ProbeFile reads duration/rate/channels/bit depth with ffprobe.
func ProbeFile(ctx context.Context, path string) (*Probe, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ffprobe", "-v", "error", "-select_streams", "a:0",
		"-show_entries", "stream=codec_name,sample_rate,channels,bits_per_raw_sample,bits_per_sample:format=duration",
		"-of", "json", path).Output()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return nil, errors.New(strings.TrimSpace(string(ee.Stderr)))
		}
		return nil, err
	}
	var j struct {
		Streams []struct {
			Codec      string `json:"codec_name"`
			SampleRate string `json:"sample_rate"`
			Channels   int    `json:"channels"`
			BitsRaw    string `json:"bits_per_raw_sample"`
			Bits       int    `json:"bits_per_sample"`
		} `json:"streams"`
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
	}
	if err := json.Unmarshal(out, &j); err != nil {
		return nil, err
	}
	if len(j.Streams) == 0 {
		return nil, errors.New("no audio stream")
	}
	st := j.Streams[0]
	p := &Probe{Codec: st.Codec, Channels: st.Channels, Bits: st.Bits}
	p.SampleRate, _ = strconv.Atoi(st.SampleRate)
	p.Duration, _ = strconv.ParseFloat(j.Format.Duration, 64)
	if b, err := strconv.Atoi(st.BitsRaw); err == nil && b > 0 {
		p.Bits = b
	}
	return p, nil
}
