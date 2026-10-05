#!/usr/bin/env python3
"""Pick the best excerpt of a finished master for a YouTube Short or an
Instagram Reel, and cut it ready to post.

Default: start a few bars before the main drop (from music_analysis.json)
so the build lands inside the clip, cut on downbeats, keep the clip a
whole number of bars no longer than --length, then fade out over the last
bar. Lead-in is about a quarter of the clip, 1-4 bars.

--loop instead cuts an exact phrase (a multiple of 4 bars) starting AT the
drop, with only 2 ms edge fades, so when the platform auto-repeats the clip
the seam lands on a downbeat and the groove never breaks.

Ambient (no beat grid): the --length window with the most sustained
energy, with slow fades. --loop isn't available without a grid -- use
loop_crossfade.py for a seamless ambient loop.

Section labels are heuristic. If the chosen spot is wrong, --drop-at moves
the drop the clip is built around, or --start pins the exact start (both
snap to the nearest downbeat when there's a grid). --plan-only prints the
choice and the reasoning without cutting anything.

Platform length caps for Shorts/Reels change often -- check the current
limits rather than trusting a hard-coded number; this script doesn't
enforce one.

Writes clip_<N>s.wav (24-bit/48k) plus clip_grid.json: the analysis
shifted onto the clip's own timeline, for visualize_wav.py --grid.

Usage:
  python3 pick_clip.py master_24bit_48k.wav --genre techno --length 30
  python3 pick_clip.py master_24bit_48k.wav --genre trap --length 15 --loop
  python3 pick_clip.py master_24bit_48k.wav --genre psytrance --drop-at 2:10 --plan-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from music_common import (
    die, ebur128, ffmpeg_to, fmt_time, genre_recipe, load_recipes,
    parse_timestamp, preflight, probe, write_json,
)

# cut this far ahead of the downbeat so the fade-in finishes before the
# kick's attack instead of softening it
PREROLL = 0.010
LOOP_EDGE_FADE = 0.002


def nearest_index(times: list[float], t: float) -> int:
    return min(range(len(times)), key=lambda i: abs(times[i] - t))


def plan_grid(a: dict, recipe: dict, length: float, loop: bool, drop_at: float | None, start_at: float | None) -> dict:
    g = a["grid"]
    bar = g["bar_seconds"]
    downs = g["downbeats"]
    duration = a["duration"]
    why = []
    n_bars = int(length // bar)
    if loop:
        n_bars = (n_bars // 4) * 4
        if n_bars < 4:
            die(f"--loop needs at least 4 bars ({4 * bar:.1f}s at {a['tempo']['bpm']:g} BPM); --length {length:g} is too short")
    if n_bars < 1:
        die(f"--length {length:g}s is shorter than one bar ({bar:.2f}s)")

    if start_at is not None:
        si = nearest_index(downs, start_at)
        why.append(f"start pinned by --start {fmt_time(start_at)} -> nearest downbeat {downs[si]:.2f}s")
    else:
        if drop_at is not None:
            drop = downs[nearest_index(downs, drop_at)]
            why.append(f"drop set by --drop-at {fmt_time(drop_at)} -> nearest downbeat {drop:.2f}s")
        elif a.get("main_drop") is not None:
            drop = a["main_drop"]
            why.append(f"main drop from analysis at {fmt_time(drop)} ({drop:.2f}s)")
        else:
            best = max(a["sections"], key=lambda s: s["energy"])
            drop = best["start"]
            why.append(f"no drop detected -- using the highest-energy section ({best['label']}) at {fmt_time(drop)}")
        di = nearest_index(downs, drop)
        lead = 0 if loop else max(1, min(4, round(0.25 * length / bar)))
        si = max(0, di - lead)
        why.append(
            "loop mode: clip starts on the drop itself so the seam is drop->drop"
            if loop else f"lead-in {di - si} bar(s) before the drop so the build lands in the clip"
        )
    # keep the clip inside the track: slide earlier by whole bars if needed
    while si > 0 and downs[si] + n_bars * bar > duration + 1e-3:
        si -= 1
    if downs[si] + n_bars * bar > duration + 1e-3:
        n_bars = int((duration - downs[si]) // bar)
        if loop:
            n_bars = (n_bars // 4) * 4
        if n_bars < 1:
            die("track is too short for a clip of this length")
        why.append(f"track too short -- clip trimmed to {n_bars} bars")
    start = max(0.0, downs[si] - PREROLL)
    clip_len = n_bars * bar
    fi = PREROLL if not loop else LOOP_EDGE_FADE
    fo = LOOP_EDGE_FADE if loop else min(recipe["clip_fade_out"], bar)
    return {
        "start": start, "duration": clip_len, "bars": n_bars, "bar_seconds": bar,
        "fade_in": fi, "fade_out": fo, "loop": loop, "why": why,
    }


def plan_free(path: Path, a: dict, recipe: dict, length: float, start_at: float | None) -> dict:
    """No grid: the window with the most sustained energy."""
    import librosa
    import numpy as np

    duration = a["duration"]
    length = min(length, duration)
    why = []
    if start_at is not None:
        start = min(max(0.0, start_at), duration - length)
        why.append(f"start pinned by --start {fmt_time(start_at)}")
    else:
        y, sr = librosa.load(str(path), sr=11025, mono=True)
        hop = int(sr * 0.1)
        rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0] ** 2
        w = max(1, int(length / 0.1))
        if len(rms) <= w:
            start = 0.0
        else:
            sums = np.convolve(rms, np.ones(w), mode="valid")
            start = round(float(np.argmax(sums)) * 0.1, 1)
        start = min(start, max(0.0, duration - length))
        why.append(f"no beat grid -- the {length:g}s window with the most sustained energy starts at {fmt_time(start)}")
    return {
        "start": start, "duration": length, "bars": None, "bar_seconds": None,
        "fade_in": recipe["clip_fade_in"], "fade_out": recipe["clip_fade_out"], "loop": False, "why": why,
    }


def shift_analysis(a: dict, start: float, dur: float) -> dict:
    """The analysis on the clip's own timeline (0 = clip start)."""
    def inside(ts):
        return [round(t - start, 4) for t in ts if start - 1e-6 <= t < start + dur]

    out = {k: a[k] for k in ("version", "genre", "tempo", "key", "flags") if k in a}
    out["source"] = a.get("source")
    out["clip_of"] = {"start": round(start, 4), "duration": round(dur, 4)}
    out["duration"] = round(dur, 4)
    if a["tempo"].get("has_grid"):
        g = a["grid"]
        out["grid"] = {
            "beats_per_bar": g["beats_per_bar"], "bar_seconds": g["bar_seconds"],
            "beats": inside(g["beats"]), "downbeats": inside(g["downbeats"]),
            "beats_fullres": inside(g["beats_fullres"]),
        }
        out["grid"]["first_downbeat"] = out["grid"]["downbeats"][0] if out["grid"]["downbeats"] else None
    else:
        out["grid"] = {}
    secs = []
    for s in a["sections"]:
        s0, s1 = max(s["start"], start), min(s["end"], start + dur)
        if s1 > s0:
            secs.append({**s, "start": round(s0 - start, 3), "end": round(s1 - start, 3)})
    out["sections"] = secs
    out["drops"] = inside(a.get("drops", []))
    md = a.get("main_drop")
    out["main_drop"] = round(md - start, 4) if md is not None and start <= md < start + dur else None
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Finished master (e.g. master_24bit_48k.wav from process_music_wav.py)")
    p.add_argument("--genre", help="ambient | trap | techno | psytrance (required -- not guessed)")
    p.add_argument("--length", type=float, default=30.0, help="Target clip length in seconds (default 30; e.g. 15/30/60/90)")
    p.add_argument("--loop", action="store_true", help="Cut an exact 4/8/16-bar phrase from the drop so the clip loops seamlessly")
    p.add_argument("--drop-at", type=parse_timestamp, help="Override the drop the clip is built around (seconds or mm:ss)")
    p.add_argument("--start", type=parse_timestamp, help="Pin the exact clip start (seconds or mm:ss); snaps to a downbeat")
    p.add_argument("--analysis", type=Path, help="music_analysis.json for this master (default: the one beside it, else analyze)")
    p.add_argument("--out-dir", type=Path, help="Output folder (default: the master's folder)")
    p.add_argument("--plan-only", action="store_true", help="Print the chosen window and the reasoning, then exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    recipes = load_recipes()
    recipe = genre_recipe(recipes, args.genre)
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        die(f"file not found: {src}")
    preflight(("atrim", "afade", "ebur128"))

    apath = args.analysis or src.parent / "music_analysis.json"
    if apath.exists():
        a = json.loads(apath.read_text())
        if a.get("genre") != args.genre:
            die(f"{apath} was analyzed as {a.get('genre')!r}, not {args.genre!r} -- re-run music_analyze.py or pass --analysis")
        if abs(a["duration"] - probe(src)["duration"]) > 0.5:
            die(f"{apath} doesn't match this file's length -- it belongs to a different render; pass --analysis or delete it")
        print(f"analysis: {apath}")
    else:
        from music_analyze import analyze
        a = analyze(src, args.genre, recipe)

    if a["tempo"].get("has_grid"):
        plan = plan_grid(a, recipe, args.length, args.loop, args.drop_at, args.start)
    else:
        if args.loop:
            die("--loop needs a beat grid, and this track has none -- for a seamless ambient loop use loop_crossfade.py")
        plan = plan_free(src, a, recipe, args.length, args.start)

    s, d = plan["start"], plan["duration"]
    lines = [
        f"clip: {fmt_time(s)} -> {fmt_time(s + d)}  ({d:.2f}s"
        + (f", {plan['bars']} bars at {a['tempo']['bpm']:g} BPM" if plan["bars"] else "") + ")",
        f"mode: {'seamless loop' if plan['loop'] else 'excerpt'}   fade in {plan['fade_in'] * 1000:.0f} ms, out {plan['fade_out'] * 1000:.0f} ms",
    ] + [f"  - {w}" for w in plan["why"]]
    covered = [sec for sec in a["sections"] if min(sec["end"], s + d) - max(sec["start"], s) > 0.5]
    lines.append("  covers: " + " -> ".join(sec["label"] for sec in covered))
    print("\n".join(lines))
    if args.plan_only:
        return 0

    out_dir = (args.out_dir or src.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{int(round(args.length))}s" + ("_loop" if plan["loop"] else "")
    clip = out_dir / f"clip_{tag}.wav"
    tmp = clip.with_suffix(".tmp.wav")
    af = (
        f"atrim=start={s:.5f}:end={s + d:.5f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={plan['fade_in']},afade=t=out:st={max(0.0, d - plan['fade_out']):.5f}:d={plan['fade_out']}"
    )
    ffmpeg_to(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-af", af,
               "-ar", "48000", "-c:a", "pcm_s24le", str(tmp)], tmp, clip)
    grid_json = out_dir / f"clip_{tag}_grid.json"
    write_json(grid_json, shift_analysis(a, s, d))
    m = ebur128(clip)
    lines.append(f"clip loudness: {m['integrated_lufs']} LUFS, TP {m['true_peak_dbtp']} dBTP"
                 " (an excerpt of the loudest part measures hotter than the full track; platforms turn it down to their reference)")
    (out_dir / f"clip_{tag}_REPORT.txt").write_text("\n".join(lines) + "\n")
    print(lines[-1])
    print(f"\nWrote {clip}\nWrote {grid_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
