"""Shared helpers for the music scripts (music_analyze.py,
process_music_wav.py, pick_clip.py, release.py).

Kept separate from process_field_wav.py / process_speech_wav.py on purpose:
those two are stable, self-contained scripts and the music side shouldn't
be able to break them. Everything here is plain ffmpeg + stdlib; numpy /
librosa are imported only by the scripts that need them.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RECIPES_PATH = SCRIPT_DIR / "music_recipes.json"

REQUIRED_GENRE_KEYS = (
    "tempo_range", "expect_grid", "phrase_bars", "highpass_hz", "mono_bass_hz",
    "eq", "dynamic_eq", "glue", "softclip", "width", "loudness_offset_lu",
    "fade_in", "fade_out", "visual_style", "visual_args", "hashtags",
)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_timestamp(value: str) -> float:
    """Accept plain seconds ("5", "12.5") or a clock-style timestamp
    ("3:45" = 3m45s, "1:02:30" = 1h2m30s)."""
    value = value.strip()
    parts = value.split(":") if ":" in value else [value]
    if len(parts) not in (1, 2, 3):
        raise argparse.ArgumentTypeError(
            f"not a valid time: {value!r} (use seconds like 12.5, or mm:ss / hh:mm:ss)"
        )
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid time: {value!r}")
    seconds = 0.0
    for p in parts_f:
        seconds = seconds * 60 + p
    return seconds


def fmt_time(seconds: float) -> str:
    """mm:ss (or h:mm:ss) -- the form YouTube chapters and humans read."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def load_recipes() -> dict:
    try:
        recipes = json.loads(RECIPES_PATH.read_text())
    except FileNotFoundError:
        die(f"recipe file not found: {RECIPES_PATH}")
    except json.JSONDecodeError as e:
        die(f"{RECIPES_PATH.name} is not valid JSON: {e}")
    for name, g in recipes.get("genres", {}).items():
        missing = [k for k in REQUIRED_GENRE_KEYS if k not in g]
        if missing:
            die(f"{RECIPES_PATH.name}: genre {name!r} is missing required key(s): {', '.join(missing)}")
    return recipes


def genre_recipe(recipes: dict, genre: str | None) -> dict:
    """Refuse to guess: a missing --genre is a question for the user, not a
    default. Same rule as --label in the nature pipeline."""
    genres = recipes["genres"]
    if not genre:
        die(
            "which genre is this? pass --genre " + "|".join(sorted(genres))
            + " (it sets the tempo range, EQ/dynamics recipe, and visual style -- not guessed)",
            2,
        )
    if genre not in genres:
        die(f"unknown genre {genre!r}; known: {', '.join(sorted(genres))}", 2)
    return genres[genre]


def preflight(filters: tuple[str, ...] = ()) -> None:
    for exe in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([exe, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (OSError, subprocess.CalledProcessError):
            die(f"required binary not found on PATH: {exe}")
    if not filters:
        return
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], stdout=subprocess.PIPE, text=True, check=True).stdout
    have = {line.split()[1] for line in out.splitlines() if len(line.split()) > 2 and line.startswith(" ")}
    missing = [f for f in filters if f not in have]
    if missing:
        die(
            f"this ffmpeg build is missing required filter(s): {', '.join(missing)} -- "
            "install a normal full-featured ffmpeg build."
        )


def require_python(modules: dict[str, str]) -> None:
    """modules: import name -> pip install hint. Checked lazily by the
    scripts that need them, with an install hint instead of a traceback."""
    missing = []
    for mod, pip_name in modules.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        die(
            f"missing Python package(s): {', '.join(missing)} -- install with:\n"
            f"  pip3 install {' '.join(missing)} --break-system-packages"
        )


def ffmpeg_version() -> str:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-version"], stdout=subprocess.PIPE, text=True, check=True).stdout
    return out.splitlines()[0].strip() if out else "unknown"


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels,bits_per_raw_sample,bits_per_sample,codec_name:format=duration",
         "-of", "json", str(path)],
        stdout=subprocess.PIPE, text=True, check=True,
    ).stdout
    j = json.loads(out)
    st = (j.get("streams") or [{}])[0]
    bits = st.get("bits_per_raw_sample") or st.get("bits_per_sample") or 0
    return {
        "duration": float(j["format"]["duration"]),
        "sample_rate": int(st.get("sample_rate", 0)),
        "channels": int(st.get("channels", 0)),
        "bits": int(bits) if str(bits).isdigit() else 0,
        "codec": st.get("codec_name", ""),
    }


def ebur128(path: Path) -> dict:
    """Integrated loudness, LRA, and true peak via ffmpeg's ebur128 (the
    spec meter -- the same measurement platforms normalize by)."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    summary = proc.stderr[proc.stderr.rfind("Summary:"):]
    out = {"integrated_lufs": None, "lra_lu": None, "true_peak_dbtp": None}
    m = re.search(r"I:\s+(-?[\d.]+|-inf)\s+LUFS", summary)
    if m:
        out["integrated_lufs"] = float(m.group(1)) if m.group(1) != "-inf" else float("-inf")
    m = re.search(r"LRA:\s+(-?[\d.]+)\s+LU", summary)
    if m:
        out["lra_lu"] = float(m.group(1))
    m = re.search(r"True peak:\s+Peak:\s+(-?[\d.]+|-inf)\s+dBFS", summary)
    if m:
        out["true_peak_dbtp"] = float(m.group(1)) if m.group(1) != "-inf" else float("-inf")
    if out["integrated_lufs"] is None:
        die("ebur128 measurement produced no summary -- ffmpeg output:\n" + proc.stderr[-2000:])
    return out


def astats(path: Path) -> dict:
    """Overall DC offset, sample peak, and how many samples sit at that
    peak (a run of samples pinned at ~0 dBFS is clipping)."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "astats=measure_perchannel=none:measure_overall=DC_offset+Peak_level+Peak_count+RMS_level",
         "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    text = proc.stderr[proc.stderr.rfind("Overall"):]

    def grab(key: str) -> float | None:
        m = re.search(rf"{key}:\s+(-?[\d.]+|-inf)", text)
        if not m:
            return None
        return float(m.group(1)) if m.group(1) != "-inf" else float("-inf")

    return {
        "dc_offset": grab("DC offset"),
        "sample_peak_dbfs": grab("Peak level dB"),
        "peak_count": grab("Peak count"),
        "rms_dbfs": grab("RMS level dB"),
    }


def run(cmd: list[str], *, quiet: bool = False) -> None:
    print("+ " + shlex.join(cmd))
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL if quiet else None)


def ffmpeg_to(cmd: list[str], tmp: Path, dst: Path, *, verify: bool = True) -> None:
    """Run an ffmpeg cmd producing `tmp`, verify with ffprobe, then atomically
    replace `dst`. Always removes any leftover `tmp` -- success or failure."""
    if tmp.exists():
        tmp.unlink()
    try:
        run(cmd)
        if verify:
            subprocess.check_call(["ffprobe", "-v", "error", str(tmp)])
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
