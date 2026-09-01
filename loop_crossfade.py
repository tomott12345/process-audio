#!/usr/bin/env python3
"""Build a seamless loop from a clean slice of a field recording.

Equal-power wrap at the cut. Writes 24-bit WAV via ffmpeg.

Only the requested slice (plus a small safety pad for seek accuracy on
non-PCM sources) is decoded to float32 -- not the whole source file -- so this
stays cheap even when INPUT is a multi-hour recording and the loop slice is
short.

Usage:
  python3 loop_crossfade.py INPUT.wav --start 136 --end 161 --target 180 --out loop.wav
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def decode_slice_f32(path: str, start: float, end: float, sr: int, pad: float = 2.0) -> np.ndarray:
    """Decode only [start, end) plus `pad` seconds of safety margin on each
    side (clamped at 0) to a stereo float32 array, then trim to the exact
    requested range in memory. Bounded by the slice length, not file length."""
    seek = max(0.0, start - pad)
    read_dur = (end - seek) + pad
    raw = Path(tempfile.mkstemp(suffix=".f32")[1])
    try:
        subprocess.check_call(
            [
                "ffmpeg", "-y", "-ss", f"{seek:.3f}", "-i", path,
                "-t", f"{read_dur:.3f}",
                "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr),
                str(raw),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        x = np.fromfile(raw, dtype=np.float32)
        if x.size == 0:
            raise SystemExit(f"empty decode: {path} [{start}-{end}]")
        if x.size % 2:
            x = x[:-1]
        x = x.reshape(-1, 2)
    finally:
        raw.unlink(missing_ok=True)

    offset = int(round((start - seek) * sr))
    length = int(round((end - start) * sr))
    if offset + length > x.shape[0]:
        # Ran off the end of the decoded window (e.g. `end` was past EOF);
        # clamp rather than reading garbage or raising on a bad slice.
        length = max(0, x.shape[0] - offset)
    if length <= 0:
        raise SystemExit(f"requested slice {start}-{end}s decoded to nothing for {path} (near end of file?)")
    return x[offset:offset + length].copy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--target", type=float, default=180.0)
    p.add_argument("--xfade", type=float, default=1.5)
    p.add_argument("--edge-fade", type=float, default=0.4)
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.end <= args.start:
        raise SystemExit("end must be after start")
    if args.xfade * 2 >= (args.end - args.start):
        raise SystemExit("xfade too long for this slice")

    sr = args.sr
    clip = decode_slice_f32(args.input, args.start, args.end, sr)
    xf = int(args.xfade * sr)
    if xf * 2 >= clip.shape[0]:
        raise SystemExit("xfade too long for the decoded slice")
    t = np.linspace(0, 1, xf, endpoint=True, dtype=np.float32)[:, None]
    seam = clip[-xf:] * np.cos(t * np.pi / 2) + clip[:xf] * np.sin(t * np.pi / 2)
    unit = np.concatenate([seam, clip[xf:-xf]], axis=0)
    n = int(np.ceil(args.target / (unit.shape[0] / sr))) + 1
    out = np.tile(unit, (n, 1))[: int(args.target * sr)].copy()

    edge = int(args.edge_fade * sr)
    if edge > 0:
        w = np.linspace(0, 1, edge, dtype=np.float32)
        out[:edge] *= w[:, None]
        out[-edge:] *= w[::-1, None]

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

    print(
        f"wrote {args.out}  slice={args.start}-{args.end}s  "
        f"unit={unit.shape[0]/sr:.3f}s  target={args.target:.3f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
