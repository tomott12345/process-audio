#!/usr/bin/env python3
"""Clean and level a spoken-word recording (interview, podcast, panel) for
publishing. This is a DIFFERENT pipeline from process_field_wav.py, on
purpose: there's no bed selection and no looping here, because every word
of a real conversation matters and it isn't repeat-tolerant the way an
ambient texture is. This script keeps the whole recording (or your
--start/--end span) and processes it once, start to finish.

Chain: highpass (rumble/plosives) -> gentle presence EQ (intelligibility,
less boxiness) -> de-esser (tame sibilance) -> gentle compressor (even out
level swings between mic distance/energy/speakers) -> two-pass loudnorm to
a podcast loudness target -> short fades -> true-peak limiter.

Loudness presets are commonly-cited platform targets as of early 2026 --
confirm against each platform's current published spec before assuming
these won't drift:
  apple    -16 LUFS  (Apple Podcasts' commonly cited mono/stereo target)
  spotify  -19 LUFS  (a conservative, widely-used creator target)
  youtube  -14 LUFS  (YouTube's loudness reference)
  general  -16 LUFS  (default -- same as apple, the most common podcast norm)
--target-i/--target-tp/--target-lra override individual preset values.

Usage:
  python3 process_speech_wav.py interview.wav --preset apple
  python3 process_speech_wav.py interview.wav --start 0:08 --end 42:10 --mono
  python3 process_speech_wav.py interview.wav --trim-silence --plan-only
  python3 process_speech_wav.py interview.wav --formats wav24,mp3 --title "Episode 12"

--formats picks the deliverables (wav24, wav16, flac, mp3; default wav24 --
the master.wav this script has always written). --plan-only --json prints
the plan as JSON; --progress-json emits machine-readable progress (see
pipeline_io.py).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_io import (
    AUDIO_FORMATS, emit, emit_manifest, emit_plan, enable_progress, export_formats,
    json_mode, parse_formats, print_json,
)

REQUIRED_FILTERS = (
    "highpass", "equalizer", "deesser", "acompressor",
    "loudnorm", "alimiter", "afade", "silenceremove", "pan", "areverse",
)

PRESETS = {
    "apple": (-16.0, -1.5, 9.0),
    "spotify": (-19.0, -1.5, 9.0),
    "youtube": (-14.0, -1.5, 9.0),
    "general": (-16.0, -1.5, 9.0),
}


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


def preflight() -> None:
    for exe in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([exe, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (OSError, subprocess.CalledProcessError):
            die(f"required binary not found on PATH: {exe}")
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], stdout=subprocess.PIPE, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        die("could not query ffmpeg -filters")
    missing = [f for f in REQUIRED_FILTERS if f" {f} " not in out and f" {f}\n" not in out]
    if missing:
        die(
            f"this ffmpeg build is missing required filter(s): {', '.join(missing)} -- "
            "install a normal full-featured ffmpeg build."
        )


def ffmpeg_version() -> str:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-version"], stdout=subprocess.PIPE, text=True, check=True).stdout
    return out.splitlines()[0].strip() if out else "unknown"


def duration_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        stdout=subprocess.PIPE, text=True, check=True,
    ).stdout.strip()
    return float(out)


def volumedetect(path: Path) -> dict:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    out = {"max_volume": None, "mean_volume": None}
    for line in proc.stderr.splitlines():
        if "max_volume" in line:
            out["max_volume"] = float(line.split(":")[1].strip().replace(" dB", ""))
        elif "mean_volume" in line:
            out["mean_volume"] = float(line.split(":")[1].strip().replace(" dB", ""))
    return out


def loudnorm_measure(path: Path, i: float, tp: float, lra: float) -> dict:
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(path),
        "-af", f"loudnorm=I={i}:TP={tp}:LRA={lra}:print_format=json",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    stderr = proc.stderr
    if "{" not in stderr:
        die("loudnorm measurement pass produced no JSON -- ffmpeg output:\n" + stderr[-2000:])
    blob = stderr[stderr.rindex("{"):stderr.rindex("}") + 1]
    return json.loads(blob)


def _ffmpeg_to(cmd: list[str], tmp: Path, dst: Path, *, verify: bool = True) -> None:
    """Run an ffmpeg cmd producing `tmp`, verify with ffprobe, then atomically
    replace `dst`. Always removes any leftover `tmp` -- success or failure --
    so a crashed run doesn't litter the output folder with partial files."""
    if tmp.exists():
        tmp.unlink()
    try:
        print("+ " + shlex.join(cmd))
        subprocess.check_call(cmd)
        if verify:
            subprocess.check_call(["ffprobe", "-v", "error", str(tmp)])
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def ffmpeg_wav(cmd_af: str, src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp.wav")
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(src), "-af", cmd_af, "-ar", "48000", "-c:a", "pcm_s24le", str(tmp)]
    _ffmpeg_to(cmd, tmp, dst)


def copy_trim(src: Path, dst: Path, start: float, end: float) -> None:
    tmp = dst.with_suffix(".tmp.wav")
    dur = max(0.01, end - start)
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
        "-ar", "48000", "-c:a", "pcm_s24le", str(tmp),
    ]
    _ffmpeg_to(cmd, tmp, dst)


def tone_chain(*, compress: bool, deess: bool) -> str:
    parts = [
        "highpass=f=90:poles=2",
        "equalizer=f=300:t=q:w=1.0:g=-2.0",
        "equalizer=f=3000:t=q:w=1.0:g=2.0",
        "equalizer=f=9000:t=q:w=1.2:g=1.0",
    ]
    if deess:
        parts.append("deesser=i=0.15:m=0.4:f=0.5:s=o")
    if compress:
        parts.append("acompressor=threshold=0.1:ratio=2.5:attack=15:release=200:makeup=1.3:knee=6")
    return ",".join(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="Source WAV (or any ffmpeg-readable audio)")
    p.add_argument("--start", type=parse_timestamp, help="Seconds, or mm:ss/hh:mm:ss, to trim from the beginning")
    p.add_argument("--end", type=parse_timestamp, help="Seconds, or mm:ss/hh:mm:ss, to cut off at")
    p.add_argument("--preset", choices=sorted(PRESETS), default="general", help="Platform loudness preset (default: general)")
    p.add_argument("--target-i", type=float, help="Override integrated loudness target (LUFS)")
    p.add_argument("--target-tp", type=float, help="Override true-peak ceiling (dBTP)")
    p.add_argument("--target-lra", type=float, help="Override loudness range target (LU)")
    p.add_argument("--no-compress", action="store_true", help="Skip the gentle leveling compressor")
    p.add_argument("--no-deess", action="store_true", help="Skip the de-esser")
    p.add_argument("--mono", action="store_true", help="Downmix to mono before loudness normalization (many podcast loudness specs are defined for mono)")
    p.add_argument("--trim-silence", action="store_true", help="Trim leading/trailing silence only (never mid-file)")
    p.add_argument("--silence-threshold", type=float, default=-45.0, help="dB threshold for --trim-silence (default -45)")
    p.add_argument("--out-dir", type=Path, help="Output folder (default: <stem>_podcast/)")
    p.add_argument("--plan-only", action="store_true", help="Print the resolved chain and targets, then exit without rendering")
    p.add_argument("--formats", type=parse_formats, default=("wav24",),
                   help=f"Comma-separated deliverables: {', '.join(AUDIO_FORMATS)} (default: wav24)")
    p.add_argument("--title", help="Title tag for the deliverables")
    p.add_argument("--artist", help="Artist / show tag for the deliverables")
    p.add_argument("--json", action="store_true", help="With --plan-only: print the plan as one JSON document")
    p.add_argument("--progress-json", action="store_true", help="Emit machine-readable progress lines (see pipeline_io.py)")
    return p


def write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    if args.json and not args.plan_only:
        die("--json only applies to --plan-only", 2)
    if args.json:
        json_mode()
    if args.progress_json:
        enable_progress()
    if not args.input:
        die("pass a WAV path as the first argument", 2)

    i, tp, lra = PRESETS[args.preset]
    if args.target_i is not None:
        i = args.target_i
    if args.target_tp is not None:
        tp = args.target_tp
    if args.target_lra is not None:
        lra = args.target_lra

    chain = tone_chain(compress=not args.no_compress, deess=not args.no_deess)

    plan_steps = []
    if args.start is not None or args.end is not None:
        plan_steps.append(("trim", "Trimming"))
    if args.trim_silence:
        plan_steps.append(("silence", "Trimming leading/trailing silence"))
    plan_steps += [("process", "Cleanup EQ / de-ess / compression"), ("loudness", "Loudness + limiter"),
                   ("deliverables", "Writing audio formats")]
    if args.plan_only and args.json:
        print_json({
            "pipeline": "speech", "script": "process_speech_wav.py", "preset": args.preset,
            "targets": {"i": i, "tp": tp, "lra": lra}, "chain": chain, "mono": args.mono,
            "trim_silence": args.trim_silence, "silence_threshold": args.silence_threshold,
            "trim": {"start": args.start, "end": args.end}, "formats": list(args.formats),
            "steps": [{"step": n, "label": lbl} for n, lbl in plan_steps],
        })
        return 0
    if args.plan_only:
        print("=== PLAN ONLY (no rendering) ===")
        print(f"preset: {args.preset}  targets: I={i} TP={tp} LRA={lra}")
        print(f"mono: {args.mono}  trim_silence: {args.trim_silence}")
        print(f"chain: {chain}")
        return 0

    preflight()

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        die(f"file not found: {src}")

    out_dir = (args.out_dir or src.parent / f"{src.stem}_podcast").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    emit_plan(plan_steps)
    print(f"source: {src}")
    print(f"out:    {out_dir}")
    print(f"preset: {args.preset}   targets: I={i} TP={tp} LRA={lra}")

    probe = subprocess.check_output(["ffprobe", "-hide_banner", "-i", str(src)], stderr=subprocess.STDOUT, text=True)
    (out_dir / "00_probe.txt").write_text(probe)

    vd0 = volumedetect(src)
    print(f"source max_volume={vd0['max_volume']} dB  mean={vd0['mean_volume']} dB")

    dur = duration_seconds(src)
    start = args.start if args.start is not None else 0.0
    end = args.end if args.end is not None else dur
    if end <= start:
        die(f"invalid span {start}-{end}")

    cur = src
    report = [
        f"run_at_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"ffmpeg={ffmpeg_version()}",
        f"source={src}",
        f"preset={args.preset}",
        f"targets=I={i} TP={tp} LRA={lra}",
        f"span={start:.3f}-{end:.3f}",
        f"chain={chain}",
        f"mono={args.mono}",
    ]

    if args.start is not None or args.end is not None:
        emit("step_start", step="trim", label="Trimming")
        trimmed = out_dir / "01_trim.wav"
        copy_trim(cur, trimmed, start, end)
        cur = trimmed
        emit("step_end", step="trim")

    if args.trim_silence:
        # Only "start_*" options -- never "stop_*". silenceremove's stop_periods
        # fires on the FIRST silence gap anywhere in the file that's long
        # enough, not specifically the trailing one -- so it will happily
        # mistake an ordinary mid-conversation pause for "the end" and
        # truncate a real interview right after its first breath. Trimming
        # leading silence, reversing, trimming leading silence again, then
        # reversing back only ever touches the true start and true end.
        emit("step_start", step="silence", label="Trimming leading/trailing silence")
        silence_trimmed = out_dir / "01b_silence_trim.wav"
        th = args.silence_threshold
        one_end = f"silenceremove=start_periods=1:start_duration=0.3:start_threshold={th}dB:start_silence=0.2"
        sr_chain = f"areverse,{one_end},areverse,{one_end}"
        ffmpeg_wav(sr_chain, cur, silence_trimmed)
        cur = silence_trimmed
        report.append(f"trim_silence=yes threshold={th}dB (leading+trailing only, never mid-file)")
        emit("step_end", step="silence")

    emit("step_start", step="process", label="Cleanup EQ / de-ess / compression")
    processed = out_dir / "02_processed.wav"
    full_chain = chain
    if args.mono:
        full_chain = full_chain + ",pan=mono|c0=0.5*c0+0.5*c1"
    ffmpeg_wav(full_chain, cur, processed)

    emit("step_end", step="process")
    emit("step_start", step="loudness", label="Loudness + limiter")
    meas = loudnorm_measure(processed, i, tp, lra)
    proc_dur = duration_seconds(processed)
    fade = 0.5 if proc_dur > 4.0 else max(0.05, proc_dur / 8)
    fade_out_at = max(0.0, proc_dur - fade)
    ln = (
        f"loudnorm=I={i}:TP={tp}:LRA={lra}"
        f":measured_I={meas['input_i']}:measured_LRA={meas['input_lra']}"
        f":measured_TP={meas['input_tp']}:measured_thresh={meas['input_thresh']}"
        f":offset={meas['target_offset']}:linear=true"
    )
    af = (
        f"{ln},"
        f"afade=t=in:st=0:d={fade},"
        f"afade=t=out:st={fade_out_at:.3f}:d={fade},"
        f"alimiter=limit={tp}dB:level=false:attack=5:release=50"
    )
    master = out_dir / "master.wav"
    ffmpeg_wav(af, processed, master)
    emit("step_end", step="loudness")
    emit("step_start", step="deliverables", label="Writing audio formats")
    meta = {"title": args.title or "", "artist": args.artist or "", "genre": "Speech",
            "comment": f"preset={args.preset}"}
    products = export_formats(master, "master", args.formats, meta)
    emit("step_end", step="deliverables")
    vd = volumedetect(master)
    report.append(
        f"measured_I={meas.get('input_i')} input_lra={meas.get('input_lra')} "
        f"final_max_volume={vd['max_volume']} final_dur={proc_dur:.3f}"
    )
    report += [f"product={pth}" for pth in products.values()]
    write_report(out_dir / "REPORT.txt", report)
    emit_manifest([{"kind": "master", "format": f, "path": pth} for f, pth in products.items()]
                  + [{"kind": "report", "format": "txt", "path": out_dir / "REPORT.txt"}])

    print("\n=== REPORT ===")
    print("\n".join(report))
    for pth in products.values():
        print(f"Wrote {pth}")
    print(f"Wrote {out_dir / 'REPORT.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
