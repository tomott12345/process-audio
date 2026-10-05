# process-audio web app (Go)

A browser front end for the repo's three pipelines (music, nature, speech).
Go handles uploads, validation, the job queue, live progress, and
downloads; the Python scripts in the repo root do all the audio work. See
`../WEB_APP_PLAN.md` for the design.

Status: **phase 2 of 5**. The HTTP API is complete; the browser interface
(phase 3) is a placeholder page for now.

## Run

```bash
cd web
go build -o bin/paweb ./cmd/paweb
bin/paweb                      # http://127.0.0.1:8765
```

Needs Go 1.24+, `python3` with the repo's dependencies, and
`ffmpeg`/`ffprobe` on PATH.

| Flag | Default | Notes |
|---|---|---|
| `-addr` | `127.0.0.1:8765` | Loopback only until the token login exists (phase 5) |
| `-repo` | found by walking up | Folder holding `pipeline_capabilities.py` |
| `-python` | `python3` | Interpreter with numpy/scipy/librosa/... |
| `-data` | `<repo>/web/data` | Uploads and jobs (git-ignored) |
| `-workers` | `1` | Concurrent jobs; video renders already use every core |
| `-max-upload-mb` | `2048` | A 45-minute 24-bit/48k stereo jam is ~780 MB |

The first launch of a freshly built binary from another app (an IDE, the
Claude preview pane) can be held at a macOS privacy prompt for access to
Documents. If the server never starts listening, run it once from a
terminal.

## API

| Method & path | Purpose |
|---|---|
| `GET /api/capabilities` | Pipelines, types, options (with Basic/Advanced, per-type defaults, ranges, dependencies), outputs, formats |
| `POST /api/uploads` | Multipart `file` → upload id + ffprobe info. WAV only (checked by header), mono/stereo |
| `GET /api/uploads/{id}` · `DELETE` | Upload info / remove |
| `GET /api/uploads/{id}/peaks?points=N` | Waveform overview (cached) |
| `POST /api/uploads/{id}/analyze` | `{pipeline, type}` → music analysis or nature bed plan (cached) |
| `POST /api/plan` | Job body → the resolved plan (`--plan-only --json`), nothing rendered |
| `POST /api/jobs` | `{upload_id, pipeline, type, values, outputs, formats}` → queued job |
| `GET /api/jobs` · `GET /api/jobs/{id}` · `DELETE` | List / state / remove a finished job |
| `POST /api/jobs/{id}/cancel` | Stops the script and every process it started |
| `GET /api/jobs/{id}/events` | Server-sent events: `job` snapshots (steps, progress, files) and `log` lines |
| `GET /api/jobs/{id}/files/{n}` | Download output `n` (Range supported; `?inline=1` to play in the page) |
| `GET /api/jobs/{id}/zip` | All outputs as one zip |

Every POST/DELETE must send `X-Requested-With: process-audio` and, if
the browser sends an `Origin` header, it must be this server's own.
`values` holds only the options the user changed; anything that doesn't
apply to the chosen type/outputs is ignored, and anything out of range is
rejected with a 400 and a readable message.

## How it fits together

- `internal/caps`: loads `pipeline_capabilities.py` (reloaded when a
  recipe file changes) and ports its `build_argv()`. A test requires the
  Go and Python builders to produce identical command lines on hundreds
  of generated cases.
- `internal/upload`: streams uploads to disk, checks the header, runs ffprobe.
- `internal/jobs`: the queue, job state persisted to `job.json`, and
  progress events folded into steps. Only manifest files that are real
  files inside the job folder can be downloaded.
- `internal/runner`: process-group launch and cancel, `@@progress` parsing.
- `internal/httpapi`: the routes above, security headers, and the CSRF check.

## Tests

```bash
go test ./...            # includes integration tests against the real scripts (~1 min)
go test -short ./...     # unit tests only
```
