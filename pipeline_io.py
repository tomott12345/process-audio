"""Output formats and machine-readable progress, shared by every pipeline
(process_field_wav.py, process_speech_wav.py, process_music_wav.py,
release.py, visualize_wav.py) so the web app -- or any other caller --
drives them all the same way.

Formats: every pipeline renders one 24-bit/48k WAV master first; the other
formats are encoded FROM that master, so they can never disagree with it.

  wav24  the master itself                     <prefix>.wav (name set by the pipeline)
  wav16  16-bit/44.1k, triangular-HP dither    <prefix>_16bit_44k1.wav
  flac   24-bit/48k lossless                   <prefix>.flac
  mp3    320 kbps CBR, 44.1k, ID3v2.3 tags     <prefix>_320k.mp3

Progress: when PROCESS_AUDIO_PROGRESS=1 is set in the environment (the
entry scripts' --progress-json flag sets it, and child processes inherit
it), events are printed to stdout as single lines:

  @@progress {"event": "step_start", "step": "master", "label": "Mastering"}
  @@progress {"event": "progress", "step": "video_9x16", "done": 120, "total": 900}
  @@progress {"event": "output", "kind": "master", "format": "flac", "path": "/.../master.flac"}
  @@progress {"event": "step_end", "step": "master", "seconds": 41.2}
  @@progress {"event": "manifest", "files": [{"kind": "master", "format": "flac", "path": "..."}]}

A run starts with a "plan" event (the ordered step list) and ends with a
"manifest" event (the deliverables). Children started by an orchestrator
carry "parent": <the orchestrator's step>.

Human-readable output is unchanged; a caller just picks out the
"@@progress " lines. Without the env var nothing extra is printed.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

PROGRESS_ENV = "PROCESS_AUDIO_PROGRESS"
PROGRESS_PREFIX = "@@progress "

AUDIO_FORMATS = {
    "wav24": {"label": "WAV 24-bit / 48 kHz (master)", "suffix": ".wav"},
    "wav16": {"label": "WAV 16-bit / 44.1 kHz (dithered)", "suffix": "_16bit_44k1.wav"},
    "flac": {"label": "FLAC 24-bit / 48 kHz", "suffix": ".flac"},
    "mp3": {"label": "MP3 320 kbps", "suffix": "_320k.mp3"},
}
DEFAULT_FORMATS = ("wav24",)


def enable_progress() -> None:
    os.environ[PROGRESS_ENV] = "1"


def progress_enabled() -> bool:
    return os.environ.get(PROGRESS_ENV) == "1"


# set by an orchestrator (release.py) in a child's environment so the
# child's events can be attributed to the orchestrator's step
PARENT_STEP_ENV = "PROCESS_AUDIO_STEP"


def emit(event: str, **fields) -> None:
    if not progress_enabled():
        return
    fields = {k: (str(v) if isinstance(v, Path) else v) for k, v in fields.items()}
    parent = os.environ.get(PARENT_STEP_ENV)
    if parent and "parent" not in fields:
        fields["parent"] = parent
    print(PROGRESS_PREFIX + json.dumps({"event": event, **fields}), flush=True)


def emit_plan(steps: list[tuple[str, str]]) -> None:
    """The steps this run will take, in order, before any of them start --
    so a UI can draw the whole checklist up front."""
    emit("plan", steps=[{"step": s, "label": label} for s, label in steps])


def emit_manifest(files: list[dict]) -> None:
    """The run's deliverables: [{kind, format, path}, ...]. The last
    manifest event a run prints is the authoritative list of outputs
    (an orchestrator's manifest supersedes its children's)."""
    emit("manifest", files=[{k: str(v) for k, v in f.items()} for f in files])


@contextmanager
def step(name: str, label: str):
    t0 = time.monotonic()
    emit("step_start", step=name, label=label)
    try:
        yield
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (0, None):
            emit("step_error", step=name, message=str(exc) or exc.__class__.__name__)
        raise
    emit("step_end", step=name, seconds=round(time.monotonic() - t0, 2))


def parse_formats(value: str) -> tuple[str, ...]:
    """argparse type for --formats: comma-separated, validated, de-duplicated."""
    import argparse

    out: list[str] = []
    for f in (x.strip().lower() for x in value.split(",")):
        if not f:
            continue
        if f not in AUDIO_FORMATS:
            raise argparse.ArgumentTypeError(
                f"unknown format {f!r}; choose from {', '.join(AUDIO_FORMATS)}"
            )
        if f not in out:
            out.append(f)
    if not out:
        raise argparse.ArgumentTypeError("--formats needs at least one format")
    return tuple(out)


def format_path(master24: Path, prefix: str, fmt: str) -> Path:
    if fmt == "wav24":
        return master24
    return master24.parent / f"{prefix}{AUDIO_FORMATS[fmt]['suffix']}"


def _encode_args(fmt: str) -> list[str]:
    if fmt == "wav16":
        return ["-af", "aresample=44100:filter_size=64:cutoff=0.97:osf=s16:dither_method=triangular_hp",
                "-c:a", "pcm_s16le"]
    if fmt == "flac":
        return ["-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", "-compression_level", "8"]
    if fmt == "mp3":
        return ["-af", "aresample=44100:filter_size=64:cutoff=0.97", "-c:a", "libmp3lame", "-b:a", "320k",
                "-id3v2_version", "3"]
    raise ValueError(fmt)


def export_formats(master24: Path, prefix: str, formats, meta: dict[str, str] | None = None,
                   kind: str = "master") -> dict[str, Path]:
    """Encode each requested format from the finished 24-bit master and
    announce every file (wav24 included) as an output event. Returns
    {format: path}."""
    meta_args: list[str] = []
    for k, v in (meta or {}).items():
        if v:
            meta_args += ["-metadata", f"{k}={v}"]
    out: dict[str, Path] = {}
    for fmt in formats:
        dst = format_path(master24, prefix, fmt)
        if fmt != "wav24":
            tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(master24),
                   *_encode_args(fmt), *meta_args, str(tmp)]
            print("+ " + shlex.join(cmd))
            try:
                subprocess.check_call(cmd)
                tmp.replace(dst)
            finally:
                tmp.unlink(missing_ok=True)
        out[fmt] = dst
        emit("output", kind=kind, format=fmt, path=dst)
    return out


def emit_output(kind: str, path: Path, fmt: str | None = None) -> None:
    emit("output", kind=kind, format=fmt or Path(path).suffix.lstrip("."), path=path)


def print_json(data: dict) -> None:
    """For --plan-only --json: exactly one JSON document on the real stdout
    (scripts point sys.stdout at stderr in --json mode so their human
    output can't corrupt it -- see json_mode())."""
    json.dump(data, sys.__stdout__, indent=2)
    sys.__stdout__.write("\n")
    sys.__stdout__.flush()


def json_mode() -> None:
    """Route ordinary print() output to stderr so stdout carries only the
    JSON document. Child processes still inherit fd 1, so callers in this
    mode must keep subprocess output off stdout (stdout=sys.stderr)."""
    sys.stdout = sys.stderr
