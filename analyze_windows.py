#!/usr/bin/env python3
"""Score a field recording in 1-second windows.

Streams the file in overlapping blocks (see --block-seconds/--pad-seconds) so
memory stays bounded regardless of recording length -- a several-hour take no
longer has to be decoded to float32 in RAM all at once (that was roughly
1.3 GB per hour of 48kHz stereo audio before this rewrite).

"Dirty" windows are flagged relative to the file's OWN measured baseline for
each band (see --*-factor), not fixed absolute levels. Fixed absolute levels
only worked for one gain-staging convention (a quiet porch take on a Zoom
recorder at modest gain); a continuously loud recording (a waterfall) or a
recording full of legitimate transients (bird calls) tripped the old fixed
thresholds on almost every window and never surfaced a usable clean run.
Genuine digital clipping is still flagged on an absolute basis (--peak-clip),
since that is a real defect regardless of the file's overall level.

Does not isolate stems. It only flags time ranges.

Usage:
  python3 analyze_windows.py INPUT.wav
  python3 analyze_windows.py INPUT.wav --json
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt


def ffprobe_duration(path: str) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1", path,
        ],
        text=True,
    )
    return float(out.strip())


def rms(sig: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sig**2) + 1e-20)) if sig.size else 0.0


def iter_blocks(path: str, sr: int, dur: float, block_seconds: float, pad_seconds: float):
    """Yield (block_start_t, stereo_interior) covering [0, dur) in
    non-overlapping, gap-free chunks of length `block_seconds`.

    Each chunk is decoded with `pad_seconds` of extra context on each side
    (clamped at the file's edges) so sosfiltfilt has room to settle before the
    region actually scored; that padding is then discarded. Only one chunk's
    worth of audio is ever in memory at a time.
    """
    t = 0.0
    while t < dur:
        block_end = min(dur, t + block_seconds)
        read_start = max(0.0, t - pad_seconds)
        read_end = min(dur, block_end + pad_seconds)
        read_dur = read_end - read_start
        raw = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-ss", f"{read_start:.3f}", "-i", path,
                "-t", f"{read_dur:.3f}", "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "2", "-ar", str(sr), "-",
            ],
            check=True, stdout=subprocess.PIPE,
        ).stdout
        block = np.frombuffer(raw, dtype=np.float32)
        if block.size % 2:
            block = block[:-1]
        block = block.reshape(-1, 2)
        lead = int(round((t - read_start) * sr))
        keep = int(round((block_end - t) * sr))
        interior = block[lead:lead + keep]
        yield t, interior
        t = block_end


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--json", action="store_true", help="Print a machine-readable summary after the human table")
    p.add_argument("--block-seconds", type=float, default=60.0, help="Streaming decode chunk size")
    p.add_argument("--pad-seconds", type=float, default=3.0, help="Filter-settling context read on each side of a chunk")
    p.add_argument("--baseline-pct", type=float, default=20.0, help="Percentile used as each band's typical/quiet baseline for this file")
    p.add_argument("--rumble-factor", type=float, default=3.0)
    p.add_argument("--thump-factor", type=float, default=3.0)
    p.add_argument("--mid-factor", type=float, default=3.0)
    p.add_argument("--rms-factor", type=float, default=2.5)
    p.add_argument("--peak-factor", type=float, default=6.0)
    p.add_argument("--peak-clip", type=float, default=0.97, help="Absolute sample amplitude treated as digital clipping regardless of file level")
    args = p.parse_args()

    sr = args.sr
    dur = ffprobe_duration(args.input)
    print(f"file={args.input}")
    print(f"dur={dur:.3f}s")

    sos_rumble = butter(4, 150, btype="low", fs=sr, output="sos")
    sos_thump = butter(4, 80, btype="low", fs=sr, output="sos")
    sos_activity = butter(4, [2500, 9000], btype="band", fs=sr, output="sos")
    sos_mid = butter(4, [200, 1500], btype="band", fs=sr, output="sos")

    rows: list[tuple[float, float, float, float, float, float, float, float]] = []
    peakL = 0.0
    peakR = 0.0
    for block_t, stereo in iter_blocks(args.input, sr, dur, args.block_seconds, args.pad_seconds):
        if stereo.shape[0] < int(sr * 0.5):
            continue
        if stereo.size:
            peakL = max(peakL, float(np.max(np.abs(stereo[:, 0]))))
            peakR = max(peakR, float(np.max(np.abs(stereo[:, 1]))))
        mono = stereo.mean(axis=1)
        rumble_full = sosfiltfilt(sos_rumble, mono)
        thump_full = sosfiltfilt(sos_thump, mono)
        mid_full = sosfiltfilt(sos_mid, mono)
        activity_full = sosfiltfilt(sos_activity, mono)
        n = mono.size
        t = block_t
        i = 0
        while i + int(sr * 0.5) <= n:
            j = min(n, i + sr)
            sl = mono[i:j]
            rumble = rms(rumble_full[i:j])
            thump = rms(thump_full[i:j])
            mid = rms(mid_full[i:j])
            activity = rms(activity_full[i:j])
            r_rms = rms(sl)
            peak = float(np.max(np.abs(sl))) if sl.size else 0.0
            ratio = activity / (rumble + 1e-12)
            rows.append((t, r_rms, peak, rumble, thump, mid, activity, ratio))
            t += 1.0
            i += sr

    print(f"peakL={peakL:.4f}  peakR={peakR:.4f}")
    print(f"windows={len(rows)}")

    if not rows:
        print("no usable windows (file shorter than 0.5s?)")
        if args.json:
            import json
            print("JSON_BEGIN")
            print(json.dumps({"file": args.input, "dur": dur, "clean_runs": [], "longest_clean": None}, indent=2))
            print("JSON_END")
        return 0

    arr = np.array(rows, dtype=float)  # columns: t, rms, peak, rumble, thump, mid, activity, ratio
    pct = args.baseline_pct
    rms_base = float(np.percentile(arr[:, 1], pct)) + 1e-9
    peak_base = float(np.percentile(arr[:, 2], pct)) + 1e-9
    rumble_base = float(np.percentile(arr[:, 3], pct)) + 1e-9
    thump_base = float(np.percentile(arr[:, 4], pct)) + 1e-9
    mid_base = float(np.percentile(arr[:, 5], pct)) + 1e-9

    print(
        f"baseline(p{pct:.0f}): rms={rms_base:.5f} peak={peak_base:.5f} "
        f"rumble={rumble_base:.5f} thump={thump_base:.5f} mid={mid_base:.5f}"
    )
    print(
        f"thresholds: rumble>{rumble_base*args.rumble_factor:.5f} thump>{thump_base*args.thump_factor:.5f} "
        f"mid>{mid_base*args.mid_factor:.5f} rms>{rms_base*args.rms_factor:.5f} "
        f"peak>{peak_base*args.peak_factor:.5f} or clip>{args.peak_clip}"
    )

    print("\nt0-t1  rms     peak    rumble  thump   mid     activity ratio  flags")
    dirty_flags = []
    for row in rows:
        t, r_rms, peak, rumble, thump, mid, activity, ratio = row
        flags = []
        if peak > args.peak_clip:
            flags.append("CLIP")
        if rumble > rumble_base * args.rumble_factor:
            flags.append("RUMBLE")
        if thump > thump_base * args.thump_factor:
            flags.append("THUMP")
        if peak > peak_base * args.peak_factor:
            flags.append("PEAK")
        if mid > mid_base * args.mid_factor:
            flags.append("MID")
        if r_rms > rms_base * args.rms_factor:
            flags.append("LOUD")
        dirty_flags.append(bool(flags))
        print(
            f"{t:5.0f}-{t+1:3.0f}  {r_rms:.5f} {peak:.4f} {rumble:.5f} {thump:.5f} "
            f"{mid:.5f} {activity:.5f} {ratio:6.2f} {' '.join(flags)}"
        )

    print("\nCLEAN RUNS")
    best = None
    runs = []
    i = 0
    while i < len(rows):
        if dirty_flags[i]:
            i += 1
            continue
        j = i
        while j < len(rows) and not dirty_flags[j]:
            j += 1
        length = j - i
        mean_ratio = float(np.mean([rows[k][7] for k in range(i, j)]))
        mean_activity = float(np.mean([rows[k][6] for k in range(i, j)]))
        t0 = rows[i][0]
        t1 = rows[j - 1][0] + 1
        print(
            f"  {t0:.0f}-{t1:.0f}s  len={length}s  mean_ratio={mean_ratio:.2f}  "
            f"mean_activity={mean_activity:.5f}"
        )
        run = {
            "start": float(t0), "end": float(t1), "len": int(length),
            "mean_ratio": mean_ratio, "mean_activity": mean_activity,
        }
        runs.append(run)
        cand = (length, mean_ratio, mean_activity, t0, t1)
        if best is None or cand[:3] > best[:3]:
            best = cand
        i = j

    if best:
        print(
            f"\nLONGEST_CLEAN start={best[3]:.0f} end={best[4]:.0f} "
            f"(prefer a shorter run if mean_activity is much higher)"
        )
    else:
        print(
            "\nNo clean run found at these factors -- either the file is uniformly "
            "busy (try relaxing --*-factor) or genuinely has a problem throughout."
        )

    if args.json:
        import json
        payload = {
            "file": args.input,
            "dur": dur,
            "peakL": peakL,
            "peakR": peakR,
            "clean_runs": runs,
            "longest_clean": (
                {
                    "start": float(best[3]), "end": float(best[4]), "len": int(best[0]),
                    "mean_ratio": float(best[1]), "mean_activity": float(best[2]),
                }
                if best else None
            ),
            "baselines": {
                "rms": rms_base, "peak": peak_base, "rumble": rumble_base,
                "thump": thump_base, "mid": mid_base, "percentile": pct,
            },
        }
        print("JSON_BEGIN")
        print(json.dumps(payload, indent=2))
        print("JSON_END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
