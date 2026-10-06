# process-audio web app (Go)

A browser front end for the repo's three pipelines (music, nature, speech).
Go handles uploads, validation, the job queue, live progress, and
downloads; the Python scripts in the repo root do all the audio work. See
`../WEB_APP_PLAN.md` for the design.

Status: **phase 3 of 5**. The HTTP API and the browser interface are
working. Waveform drag handles (trim, drop marker) are phase 4; the token
login for network access and data cleanup are phase 5.

## Using it

1. **Pick the kind of audio**: Music, Nature, or Speech.
2. **Drop in the WAV.** You'll see its length, format, and waveform.
3. **Pick the type** (music genre or nature recording). Music and nature
   files are analyzed right away. Music shows BPM, key, loudness, the
   sections, and the main drop on the waveform. Nature shows the clean bed
   it chose.
4. **Options**: the everyday ones show by default; everything else is
   under **Advanced**, which starts collapsed. Each option starts at its
   recipe default for the chosen type. Changed ones are highlighted with a
   *reset* link. Options that don't apply are hidden, and ones that can't
   be used right now are greyed out with the reason.
5. **Outputs and formats**: tick what you want back (WAV 24/16-bit, FLAC,
   MP3, videos, captions...).
6. **Show plan** previews exactly what will run; **Run** starts it.
   Progress streams live, including percentages on video renders, and you
   can cancel. Results come with players, per-file downloads, and a zip.
   **Edit & run again** loads a finished job back into the form.

The **Jobs** panel lists every run, survives restarts, and can delete old
jobs and their files.

![The web app: upload and waveform](../docs/images/web-1-upload.png)

More screenshots, one per step, are in the [main README](../README.md#web-app).

## Run

The simplest way is the Docker image. See "Run it with Docker" in the
[main README](../README.md#run-it-with-docker-easiest). Inside a container
the server runs with `-container`, which allows listening on the
container's interface. The port must then be published on the host's
loopback only (`-p 127.0.0.1:8765:8765`), because there is no login yet.

To run it directly:

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
| `-container` | off | Inside a container: allow listening on all interfaces (publish on 127.0.0.1 only) |

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
- `ui/static`: `index.html`, `app.css`, and `app.js`, embedded into the
  binary. Plain ES module, no framework or build step. Everything on screen
  is built from `/api/capabilities`. Rebuild the binary after editing them.

## Tests

```bash
go test ./...            # includes integration tests against the real scripts (~1 min)
go test -short ./...     # unit tests only
```
