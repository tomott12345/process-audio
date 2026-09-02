#!/usr/bin/env python3
"""Remove a specific, known steady noise (hum, hiss, distant traffic) from a
recording, using a short sample of "just that noise" as a spectral profile.

This is targeted noise-profile subtraction (via the `noisereduce` library),
not general denoising -- point it at a few seconds that are ONLY the
unwanted noise (no wanted ambience), and it estimates that noise's spectral
shape and subtracts it from the rest of the file. Works best on a genuinely
steady/repetitive noise (AC hum, a fridge, room tone, a steady low hiss); a
noise whose character drifts over time will subtract less cleanly -- try
--non-stationary in that case.

Meant to run on an already-trimmed bed (a few minutes at most), not a raw
multi-hour source -- the whole target file is held in memory.

Usage:
  # noise sample is a time range within the SAME file (e.g. handling hum
  # before the nature sound settles in)
  python3 denoise_profile.py INPUT.wav --noise-start 0 --noise-end 3 --out cleaned.wav

  # noise sample is a separate short recording of just the noise
  python3 denoise_profile.py INPUT.wav --noise-file hum_only.wav --out cleaned.wav
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


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


def decode_stereo_f32(path: str, sr: int, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Decode a WAV (optionally just [start, end)) to a stereo float32 array."""
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    if start is not None:
        cmd.extend(["-ss", f"{start:.3f}"])
    cmd.extend(["-i", str(path)])
    if start is not None and end is not None:
        cmd.extend(["-t", f"{max(0.01, end - start):.3f}"])
    raw = Path(tempfile.mkstemp(suffix=".f32")[1])
    try:
        cmd.extend(["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), str(raw)])
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        x = np.fromfile(raw, dtype=np.float32)
        if x.size == 0:
            raise SystemExit(f"empty decode: {path}")
        if x.size % 2:
            x = x[:-1]
        return x.reshape(-1, 2)
    finally:
        raw.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--noise-start", type=parse_timestamp, help="Start of a noise-only span within INPUT")
    p.add_argument("--noise-end", type=parse_timestamp, help="End of a noise-only span within INPUT")
    p.add_argument("--noise-file", type=Path, help="Alternative: a separate short WAV of just the noise")
    p.add_argument("--non-stationary", action="store_true", help="Use for a noise whose character drifts over time (default: stationary)")
    p.add_argument("--prop-decrease", type=float, default=1.0, help="How aggressively to subtract the noise profile, 0-1 (default 1.0 = full)")
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    have_range = args.noise_start is not None and args.noise_end is not None
    if not have_range and not args.noise_file:
        raise SystemExit("pass either --noise-start/--noise-end (a span within INPUT) or --noise-file")
    if have_range and args.noise_file:
        raise SystemExit("pass --noise-start/--noise-end OR --noise-file, not both")
    if have_range and args.noise_end <= args.noise_start:
        raise SystemExit("--noise-end must be after --noise-start")

    try:
        import noisereduce as nr
    except ImportError:
        raise SystemExit(
            "missing required python package: noisereduce. "
            "Install with `pip3 install noisereduce --break-system-packages`."
        )

    sr = args.sr
    audio = decode_stereo_f32(args.input, sr)
    if args.noise_file:
        noise = decode_stereo_f32(str(args.noise_file), sr)
    else:
        noise = decode_stereo_f32(args.input, sr, args.noise_start, args.noise_end)

    if noise.shape[0] < int(sr * 0.25):
        raise SystemExit(f"noise sample is only {noise.shape[0]/sr:.2f}s -- give it at least 0.5s, ideally 2-5s")

    out = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        out[:, ch] = nr.reduce_noise(
            y=audio[:, ch],
            sr=sr,
            y_noise=noise[:, ch],
            stationary=not args.non_stationary,
            prop_decrease=args.prop_decrease,
        ).astype(np.float32)

    raw = Path(tempfile.mkstemp(suffix=".f32")[1])
    try:
        out.astype(np.float32).tofile(raw)
        subprocess.check_call(
            [
                "ffmpeg", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2",
                "-i", str(raw), "-c:a", "pcm_s24le", args.out,
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    finally:
        raw.unlink(missing_ok=True)

    noise_desc = f"{args.noise_start:.2f}-{args.noise_end:.2f}s of {args.input}" if have_range else str(args.noise_file)
    print(
        f"wrote {args.out}  noise_profile={noise_desc}  "
        f"stationary={not args.non_stationary}  prop_decrease={args.prop_decrease}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
