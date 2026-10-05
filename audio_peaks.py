#!/usr/bin/env python3
"""Print a compact waveform overview of any audio file as JSON: duration,
format, and N min/max peak pairs -- enough to draw a waveform in a UI
(trim handles, a drop marker) without shipping the audio itself.

Decodes through ffmpeg at a low sample rate in a streaming pass, so a
45-minute jam costs a few MB of memory, not a gigabyte.

Usage:
  python3 audio_peaks.py take.wav                 # 1000 points
  python3 audio_peaks.py take.wav --points 2000

Output:
  {"duration": 312.4, "sample_rate": 48000, "channels": 2, "bits": 24,
   "points": 1000, "seconds_per_point": 0.3124,
   "min": [-0.41, ...], "max": [0.43, ...], "peak_dbfs": -0.8}
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from music_common import die, preflight, probe

DECODE_RATE = 4000  # Hz -- plenty for a visual envelope


def peaks(path: Path, points: int) -> dict:
    import numpy as np

    info = probe(path)
    dur = info["duration"]
    total = max(1, int(dur * DECODE_RATE))
    per = max(1, math.ceil(total / points))
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(DECODE_RATE), "-f", "f32le", "-"],
        stdout=subprocess.PIPE,
    )
    mins: list[float] = []
    maxs: list[float] = []
    carry = np.zeros(0, dtype=np.float32)
    chunk_bytes = per * 4 * 256
    while True:
        buf = proc.stdout.read(chunk_bytes)
        if not buf:
            break
        x = np.concatenate([carry, np.frombuffer(buf, dtype=np.float32)])
        n = (len(x) // per) * per
        if n:
            blocks = x[:n].reshape(-1, per)
            mins += blocks.min(axis=1).tolist()
            maxs += blocks.max(axis=1).tolist()
        carry = x[n:]
    if len(carry):
        mins.append(float(carry.min()))
        maxs.append(float(carry.max()))
    if proc.wait() != 0:
        die(f"ffmpeg could not decode {path}")
    peak = max([abs(v) for v in mins + maxs] or [0.0])
    return {
        "duration": round(dur, 3),
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "bits": info["bits"],
        "points": len(maxs),
        "seconds_per_point": round(per / DECODE_RATE, 5),
        "min": [round(v, 4) for v in mins],
        "max": [round(v, 4) for v in maxs],
        "peak_dbfs": round(20 * math.log10(peak), 2) if peak > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--points", type=int, default=1000, help="Number of min/max pairs (default 1000)")
    args = ap.parse_args()
    if not 10 <= args.points <= 20000:
        die("--points must be between 10 and 20000", 2)
    preflight()
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        die(f"file not found: {src}")
    json.dump(peaks(src, args.points), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
