#!/usr/bin/env python3
"""Strip a voice/speech-like foreground (talking, footsteps, a passing
conversation) out of a field recording, using Demucs' pretrained vocal-
separation model.

This is different from bed-selection or EQ: it's a source-separation model,
so it can remove unwanted foreground content that overlaps in TIME with the
ambience you want to keep -- something trimming/EQ alone can't do (those
only work when there's a voice-free span to select or a frequency band to
cut). Demucs was trained to split music into vocals vs. everything else, not
specifically for field recordings, so results vary: it tends to do best on a
clear, close voice/footsteps against a quieter ambient bed, and worst when
the "foreground" is soft or blends into the ambience.

This is a genuinely heavy, slow, OPTIONAL step -- it needs `torch` and
`demucs` installed (a large download, a GB or more) and pretrained model
weights (~80MB, fetched on first run), and can take a couple of minutes per
file on a CPU. Only reach for it on takes that actually need it; leave it
out of your normal workflow otherwise.

Usage:
  python3 remove_foreground.py INPUT.wav --out cleaned.wav

  # only process part of a long file (much faster) -- e.g. the span where
  # the unwanted talking actually happens
  python3 remove_foreground.py INPUT.wav --start 0:30 --end 3:00 --out cleaned.wav
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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


def check_deps() -> None:
    missing = []
    for mod in ("torch", "demucs"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            "missing required package(s) for voice removal: " + ", ".join(missing) + ".\n"
            "This is a heavy, optional dependency -- install with:\n"
            "  pip3 install torch --break-system-packages\n"
            "  pip3 install demucs --break-system-packages\n"
            "(a GB+ download; only needed if you use --remove-voice / remove_foreground.py)"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--start", type=parse_timestamp, help="Only process from here (seconds or mm:ss)")
    p.add_argument("--end", type=parse_timestamp, help="Only process until here (seconds or mm:ss)")
    p.add_argument("--model", default="htdemucs", help="Demucs model name (default: htdemucs)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if (args.start is None) != (args.end is None):
        raise SystemExit("pass both --start and --end, or neither")
    if args.start is not None and args.end <= args.start:
        raise SystemExit("--end must be after --start")

    check_deps()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"file not found: {src}")

    work = Path(tempfile.mkdtemp(prefix="remove_foreground_"))
    try:
        target = src
        if args.start is not None:
            target = work / "span.wav"
            subprocess.check_call(
                [
                    "ffmpeg", "-hide_banner", "-y",
                    "-ss", f"{args.start:.3f}", "-i", str(src),
                    "-t", f"{args.end - args.start:.3f}",
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(target),
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        print(f"+ running demucs ({args.model}, two-stems=vocals) on {target} -- this can take a while on CPU")
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "--two-stems", "vocals",
            "-n", args.model,
            "-o", str(work / "out"),
            str(target),
        ]
        print("+ " + " ".join(cmd))
        subprocess.check_call(cmd)

        stem = target.stem
        no_vocals = work / "out" / args.model / stem / "no_vocals.wav"
        if not no_vocals.exists():
            raise SystemExit(f"demucs did not produce the expected output: {no_vocals}")

        tmp_out = Path(args.out).with_suffix(".tmp.wav")
        subprocess.check_call(
            [
                "ffmpeg", "-hide_banner", "-y", "-i", str(no_vocals),
                "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(tmp_out),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        Path(tmp_out).replace(args.out)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"wrote {args.out} (voice/foreground-removed via demucs {args.model})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
