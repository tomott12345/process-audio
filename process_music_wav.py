#!/usr/bin/env python3
"""Master a finished track or Bluebox stereo mix for YouTube / Instagram.

A music sibling of process_speech_wav.py -- separate on purpose: the
nature pipeline's bed selection/looping and the speech chain's de-esser and
podcast targets are both wrong for EDM.

Chain (every step set by the genre recipe in music_recipes.json):
  1. subsonic high-pass (also removes DC offset -- common on analog gear)
  2. mono-bass: split at the genre's crossover (acrossover, phase-coherent
     Linkwitz-Riley), fold the low band to mono, recombine. Stereo synths,
     chorus on bass patches, and wide reverbs all smear the low end; mono
     lows are tighter on big systems and survive phone/mono playback.
     Optional stereo width (--width) is applied to the upper band only.
  3. tonal EQ, then optional dynamic EQ (resonant acid/filter peaks)
  4. glue compression (slow enough attack to let the kick through)
  5. loudness: measure, apply ONE static gain to hit the target, soft-clip
     the peaks that gain pushes over (asoftclip, 4x oversampled), then a
     true-peak-safe limiter (alimiter run at 4x the sample rate, since a
     plain sample-peak limiter misses inter-sample overs). Re-measure and
     correct once. This replaces loudnorm on purpose: loudnorm's dynamic
     mode pumps on music, and its linear mode silently falls back to
     dynamic whenever the gain it needs would exceed the true-peak ceiling
     -- which is every EDM master.
  6. fades, then deliverables:
       master_24bit_48k.wav  -- the master (48k = what video uses, so the
                                video mux doesn't resample)
       master_16bit_44k1.wav -- triangular high-pass dithered 16-bit
     Both tagged with title/artist/genre and BPM/key in the comment.

--genre is required (not guessed); --preset picks the loudness target
(youtube / instagram -- see music_recipes.json for the numbers and how
much to trust them). Ambient deliberately lands below the preset target
(loudness_offset_lu) and skips the soft-clipper to keep its dynamics.

Usage:
  python3 process_music_wav.py mix.wav --genre techno --title "Night Drive"
  python3 process_music_wav.py mix.wav --genre trap --preset instagram --start 0:12 --end 3:40
  python3 process_music_wav.py mix.wav --genre ambient --plan-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from music_common import (
    astats, die, ebur128, ffmpeg_to, ffmpeg_version, fmt_time, genre_recipe,
    load_recipes, parse_timestamp, preflight, probe, write_json,
)

REQUIRED_FILTERS = (
    "aformat", "highpass", "acrossover", "pan", "amix", "stereotools", "equalizer",
    "highshelf", "acompressor", "volume", "asoftclip", "alimiter",
    "aresample", "afade", "ebur128", "astats",
)
# the 4x-oversampled limiter's ceiling sits this far under the target so
# the downsample and any residual inter-sample overs still land under it
TP_MARGIN_DB = 0.3


def tone_graph(recipe: dict, width: float, no_mono_bass: bool) -> str:
    """ffmpeg -filter_complex graph from [0:a] to [tone]: everything up to
    (not including) the loudness stage."""
    # aformat first: a mono file becomes dual-mono so the stereo stages
    # (mono-bass pan, width) have two channels to work with
    hp = f"aformat=channel_layouts=stereo,highpass=f={recipe['highpass_hz']}:poles=2"
    xo = recipe["mono_bass_hz"]
    hi_tail = f",stereotools=slev={width}" if width != 1.0 else ""
    if no_mono_bass and not hi_tail:
        head = f"[0:a]{hp}[pre]"
    else:
        # width is only ever applied above the crossover -- widening the
        # low end is exactly what the mono-bass step exists to undo
        lo_chain = "anull" if no_mono_bass else "pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1"
        head = (
            f"[0:a]{hp},acrossover=split={xo}:order=4th[lo][hi];"
            f"[lo]{lo_chain}[lom];"
            f"[hi]anull{hi_tail}[hiw];"
            "[lom][hiw]amix=inputs=2:normalize=0[pre]"
        )
    chain = [recipe["eq"]]
    if recipe.get("dynamic_eq"):
        chain.append(recipe["dynamic_eq"])
    g = recipe["glue"]
    chain.append(
        f"acompressor=threshold={g['threshold_db']}dB:ratio={g['ratio']}:attack={g['attack_ms']}"
        f":release={g['release_ms']}:knee={10 ** (g['knee_db'] / 20):.3f}:makeup=1"
    )
    return head + ";[pre]" + ",".join(chain) + "[tone]"


def loud_chain(gain_db: float, recipe: dict, ceiling_db: float, use_softclip: bool) -> str:
    """Runs at 4x (192k) so the clipper and limiter see inter-sample peaks.
    The clipper's harmonics and the limiter's gain moves put energy above
    24 kHz; filtering that out on the way back down to 48k reshapes the
    peaks and overshoots the ceiling (measured: +1.5 dB on a hard-clipped
    808). So the band-limit happens BEFORE the limiter, and the caller
    still re-measures true peak and pulls the ceiling down if needed."""
    parts = ["aresample=192000", f"volume={gain_db:.2f}dB"]
    sc = recipe.get("softclip")
    if use_softclip and sc:
        thr = 10 ** (sc["threshold_db"] / 20)
        # tanh: unity gain well below the threshold, bending toward it above
        parts.append(f"asoftclip=type={sc['type']}:threshold={thr:.4f}:oversample=1")
    parts += ["lowpass=f=20000:poles=2", "lowpass=f=20000:poles=2"]
    ceiling = 10 ** (ceiling_db / 20)
    parts.append(f"alimiter=limit={ceiling:.4f}:level=false:attack=1:release=60:asc=1")
    parts.append("aresample=48000:filter_size=64:cutoff=0.97")
    return ",".join(parts)


def render(cmd_graph: str | None, cmd_af: str | None, src: Path, dst: Path, extra: list[str] | None = None) -> None:
    tmp = dst.with_suffix(".tmp.wav")
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error", "-y", "-i", str(src)]
    if cmd_graph:
        cmd += ["-filter_complex", cmd_graph, "-map", "[tone]"]
    if cmd_af:
        cmd += ["-af", cmd_af]
    cmd += ["-ar", "48000", "-c:a", "pcm_f32le"] + (extra or []) + [str(tmp)]
    ffmpeg_to(cmd, tmp, dst)


def tags_for(args, analysis: dict | None, genre: str) -> list[str]:
    comment = [f"genre={genre}"]
    if analysis:
        t, k = analysis["tempo"], analysis["key"]
        if t.get("has_grid"):
            comment.append(f"bpm={t['bpm']:g}")
        if k.get("name"):
            comment.append(f"key={k['name']} ({k['camelot']})")
    meta = ["-metadata", f"genre={genre}", "-metadata", "comment=" + "; ".join(comment)]
    if args.title:
        meta += ["-metadata", f"title={args.title}"]
    if args.artist:
        meta += ["-metadata", f"artist={args.artist}"]
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Finished track or Bluebox stereo mix")
    p.add_argument("--genre", help="ambient | trap | techno | psytrance (required -- not guessed)")
    p.add_argument("--preset", default="youtube", help="Loudness preset from music_recipes.json (default: youtube)")
    p.add_argument("--target-i", type=float, help="Override integrated loudness target (LUFS)")
    p.add_argument("--target-tp", type=float, help="Override true-peak ceiling (dBTP)")
    p.add_argument("--start", type=parse_timestamp, help="Seconds, or mm:ss/hh:mm:ss, to trim from the beginning")
    p.add_argument("--end", type=parse_timestamp, help="Seconds, or mm:ss/hh:mm:ss, to cut off at")
    p.add_argument("--title", help="Title tag (also used by release.py)")
    p.add_argument("--artist", help="Artist tag")
    p.add_argument("--width", type=float, help="Side level above the mono-bass crossover (1.0 = unchanged; overrides the recipe)")
    p.add_argument("--no-mono-bass", action="store_true", help="Skip folding the low end to mono")
    p.add_argument("--no-softclip", action="store_true", help="Skip the soft-clipper even if the genre recipe uses one")
    p.add_argument("--analysis", type=Path, help="music_analysis.json to take BPM/key tags from (default: analyze the input)")
    p.add_argument("--no-analysis", action="store_true", help="Skip analysis (no BPM/key tags, no post-master QC report)")
    p.add_argument("--out-dir", type=Path, help="Output folder (default: <stem>_music/)")
    p.add_argument("--plan-only", action="store_true", help="Print the resolved chain and targets, then exit without rendering")
    return p


def main() -> int:
    args = build_parser().parse_args()
    recipes = load_recipes()
    recipe = genre_recipe(recipes, args.genre)
    if args.preset not in recipes["presets"]:
        die(f"unknown preset {args.preset!r}; known: {', '.join(sorted(recipes['presets']))}", 2)
    preset = recipes["presets"][args.preset]
    target_i = (args.target_i if args.target_i is not None else preset["target_i"]) + (
        0.0 if args.target_i is not None else recipe["loudness_offset_lu"]
    )
    tp = args.target_tp if args.target_tp is not None else preset["target_tp"]
    width = args.width if args.width is not None else recipe["width"]
    use_softclip = bool(recipe.get("softclip")) and not args.no_softclip
    graph = tone_graph(recipe, width, args.no_mono_bass)

    plan = [
        f"genre: {args.genre}{'  (recipe untested -- A/B before trusting)' if recipe.get('untested') else ''}",
        f"preset: {args.preset}  target: I={target_i:g} LUFS  TP={tp:g} dBTP"
        + (f"  (genre offset {recipe['loudness_offset_lu']:+g} LU)" if recipe["loudness_offset_lu"] and args.target_i is None else ""),
        f"mono-bass: {'off' if args.no_mono_bass else str(recipe['mono_bass_hz']) + ' Hz'}  width: {width:g}",
        f"tone graph: {graph}",
        f"loudness: static gain -> {'softclip ' + json.dumps(recipe['softclip']) + ' -> ' if use_softclip else ''}"
        f"limiter at {tp - TP_MARGIN_DB:g} dBFS (4x oversampled)",
        f"fades: in {recipe['fade_in']}s  out {recipe['fade_out']}s",
    ]
    if args.plan_only:
        print("=== PLAN ONLY (no rendering) ===")
        print("\n".join(plan))
        return 0

    preflight(REQUIRED_FILTERS + ((recipe["dynamic_eq"].split("=")[0],) if recipe.get("dynamic_eq") else ()))
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        die(f"file not found: {src}")
    out_dir = (args.out_dir or src.parent / f"{src.stem}_music").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)

    info = probe(src)
    print(f"source: {src}  ({fmt_time(info['duration'])}, {info['sample_rate']} Hz, {info['channels']} ch)")
    print(f"out:    {out_dir}")
    print("\n".join(plan))
    if info["channels"] not in (1, 2):
        die(f"{info['channels']}-channel input -- pass the Bluebox stereo mix (or a stereo bounce), not a multichannel file")

    report = [
        f"run_at_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"ffmpeg={ffmpeg_version()}",
        f"source={src}",
    ] + plan

    # 0. trim (sample-accurate, before anything else)
    cur = src
    if args.start is not None or args.end is not None:
        start = args.start or 0.0
        end = args.end if args.end is not None else info["duration"]
        if end <= start:
            die(f"invalid span {start}-{end}")
        trimmed = work / "00_trim.wav"
        render(None, f"atrim=start={start:.4f}:end={end:.4f},asetpts=PTS-STARTPTS", cur, trimmed)
        cur = trimmed
        report.append(f"span={start:.3f}-{end:.3f}")

    # 1-4. tone
    toned = work / "01_tone.wav"
    render(graph, None, cur, toned)
    m0 = ebur128(toned)
    if m0["integrated_lufs"] == float("-inf"):
        die("the input is silent after processing -- nothing to master")
    gain = target_i - m0["integrated_lufs"]

    # 5. loudness: static gain -> softclip -> TP limiter. Re-measure and
    #    correct: the clipper/limiter shave a little loudness off the top
    #    (raise the gain), and any true-peak overshoot left after the
    #    downsample lowers the limiter ceiling by that much.
    loud = work / "02_loud.wav"
    ceiling_db = tp - TP_MARGIN_DB
    for attempt in range(5):
        render(None, loud_chain(gain, recipe, ceiling_db, use_softclip), toned, loud)
        m1 = ebur128(loud)
        err = target_i - m1["integrated_lufs"]
        over = m1["true_peak_dbtp"] - tp
        print(
            f"  pass {attempt + 1}: gain {gain:+.2f} dB, ceiling {ceiling_db:.2f} dBFS -> "
            f"{m1['integrated_lufs']} LUFS, TP {m1['true_peak_dbtp']} dBTP"
        )
        if abs(err) <= 0.3 and over <= 0.0:
            break
        if over > 0.0:
            ceiling_db -= over + 0.1
        if abs(err) > 0.3:
            gain += err
    report.append(f"limiter_ceiling={ceiling_db:.2f}dBFS")
    gr_db = gain - (m1["integrated_lufs"] - m0["integrated_lufs"])
    report.append(
        f"tone_stage_I={m0['integrated_lufs']} static_gain={gain:+.2f}dB "
        f"clip+limit_loss={gr_db:.2f}dB final_I={m1['integrated_lufs']} final_TP={m1['true_peak_dbtp']}"
    )
    if gr_db > 4.0:
        report.append(
            f"WARNING: the clipper/limiter took {gr_db:.1f} dB off the top to reach {target_i:g} LUFS -- "
            "that's heavy; transients will be audibly flattened. Consider a lower target or lighter glue."
        )
    if m1["true_peak_dbtp"] is not None and m1["true_peak_dbtp"] > tp + 0.05:
        report.append(f"WARNING: true peak {m1['true_peak_dbtp']} dBTP is above the {tp} ceiling")

    # 6. fades + deliverables (tags from the analysis)
    analysis = None
    if args.analysis:
        analysis = json.loads(args.analysis.read_text())
    elif not args.no_analysis:
        from music_analyze import analyze
        analysis = analyze(cur, args.genre, recipe)
        write_json(out_dir / "music_analysis_source.json", analysis)
    meta = tags_for(args, analysis, args.genre)

    dur = probe(loud)["duration"]
    fi, fo = recipe["fade_in"], min(recipe["fade_out"], dur / 4)
    fades = f"afade=t=in:st=0:d={fi},afade=t=out:st={max(0.0, dur - fo):.4f}:d={fo}"
    m24 = out_dir / "master_24bit_48k.wav"
    tmp = m24.with_suffix(".tmp.wav")
    ffmpeg_to(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(loud), "-af", fades,
               "-ar", "48000", "-c:a", "pcm_s24le", *meta, str(tmp)], tmp, m24)
    m16 = out_dir / "master_16bit_44k1.wav"
    tmp = m16.with_suffix(".tmp.wav")
    ffmpeg_to(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(loud), "-af",
               fades + ",aresample=44100:filter_size=64:cutoff=0.97:osf=s16:dither_method=triangular_hp",
               "-c:a", "pcm_s16le", *meta, str(tmp)], tmp, m16)

    # post-master QC: the numbers that matter for upload
    fin = ebur128(m24)
    st = astats(m24)
    report.append(
        f"master: I={fin['integrated_lufs']} LUFS  LRA={fin['lra_lu']} LU  TP={fin['true_peak_dbtp']} dBTP  "
        f"sample_peak={st['sample_peak_dbfs']} dBFS"
    )
    qc = {}
    if analysis is not None:
        from music_analyze import analyze
        qc = analyze(m24, args.genre, recipe)
        write_json(out_dir / "music_analysis.json", qc)
        s0, s1 = analysis["stereo"], qc["stereo"]
        report.append(
            f"low-band correlation: {s0['low_band_correlation']} -> {s1['low_band_correlation']}   "
            f"phone loss: {analysis['phone']['phone_speaker_loss_db']} -> {qc['phone']['phone_speaker_loss_db']} dB"
        )
        if qc["tempo"].get("has_grid"):
            report.append(f"tags: {qc['tempo']['bpm']:g} BPM, {qc['key']['name']} ({qc['key']['camelot']})")
        for f in qc["flags"]:
            report.append(f"flag: {f}")
    report += [f"product={m24}", f"product={m16}"]
    (out_dir / "REPORT.txt").write_text("\n".join(report) + "\n")

    print("\n=== REPORT ===")
    print("\n".join(report[3:]))
    print(f"\nWrote {m24}\nWrote {m16}\nWrote {out_dir / 'REPORT.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
