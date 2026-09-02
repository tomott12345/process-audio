#!/usr/bin/env python3
"""Turn any field WAV into ASMR masters using the field-audio recipe.

Usage:
  process_field_wav.py INPUT.wav --label rain --place porch
  process_field_wav.py INPUT.wav --label insects --place woods --long clean
  process_field_wav.py INPUT.wav --label thunder --start 12 --end 240
  process_field_wav.py INPUT.wav --label thunder --start 5 --end 3:45
  process_field_wav.py INPUT.wav --label birds --place woods --plan-only

Does not delete the source. Stays pcm_s24le until optional AAC mux.
Does not invent stems. EQ is band work only.

If --label is omitted the script prints the four questions and exits.

Content-type recipes (EQ chain, loudness targets, bed-selection strategy,
correlated-water loop policy) live in recipes.json next to this script --
edit that file to retune a label or add a new one; the script fails loudly
at startup if a label is missing a required key rather than silently
reusing another label's recipe. --plan-only prints the chosen bed, EQ
chain, and loudness targets without rendering anything, useful before
committing to a long render.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _tool(name: str) -> Path:
    cand = SCRIPT_DIR / name
    if cand.exists():
        return cand
    die(f"missing helper script: {name} (expected next to process_field_wav.py)")


ANALYZE = _tool("analyze_windows.py")
LOOP = _tool("loop_crossfade.py")
REMOVE_FOREGROUND = _tool("remove_foreground.py")
DENOISE_PROFILE = _tool("denoise_profile.py")

# ---------------------------------------------------------------------------
# Recipes: EQ chains, loudness targets, bed-selection strategy, and the
# correlated-water loop policy all live in recipes.json, not in this file.
# ---------------------------------------------------------------------------

REQUIRED_LABEL_KEYS = ("eq", "bed_selection", "correlated_water", "targets")
VALID_BED_SELECTIONS = ("longest", "activity", "mixed")
REQUIRED_TOP_KEYS = ("labels", "places", "denoise_chain", "repair_chain")


def load_recipes(path: Path) -> dict:
    if not path.exists():
        die(f"recipe file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        die(f"recipe file {path} is not valid JSON: {exc}")
    for key in REQUIRED_TOP_KEYS:
        if key not in data:
            die(f"recipe file {path} is missing top-level {key!r}")
    labels = data["labels"]
    if not labels:
        die(f"recipe file {path} has no labels defined")
    for name, recipe in labels.items():
        missing = [k for k in REQUIRED_LABEL_KEYS if k not in recipe]
        if missing:
            die(f"recipe for label {name!r} in {path} is missing: {', '.join(missing)}")
        if recipe["bed_selection"] not in VALID_BED_SELECTIONS:
            die(
                f"recipe for label {name!r} has bed_selection {recipe['bed_selection']!r}, "
                f"must be one of {VALID_BED_SELECTIONS}"
            )
        for kind in ("long", "short"):
            vals = recipe["targets"].get(kind)
            if not vals or len(vals) != 3:
                die(f"recipe for label {name!r} needs targets.{kind} = [I, TP, LRA]")
    return data


def _peek_recipes_path(argv: list[str], default: Path) -> Path:
    for i, a in enumerate(argv):
        if a == "--recipes" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser().resolve()
        if a.startswith("--recipes="):
            return Path(a.split("=", 1)[1]).expanduser().resolve()
    return default


DEFAULT_RECIPES_PATH = SCRIPT_DIR / "recipes.json"
RECIPES_PATH = _peek_recipes_path(sys.argv[1:], DEFAULT_RECIPES_PATH)
RECIPES = load_recipes(RECIPES_PATH)
LABELS = tuple(RECIPES["labels"].keys())
PLACES = tuple(RECIPES["places"])
CORRELATED_WATER = tuple(name for name, r in RECIPES["labels"].items() if r.get("correlated_water"))

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

REQUIRED_BINARIES = ("ffmpeg", "ffprobe")
REQUIRED_FILTERS = (
    "afftdn", "adeclick", "adeclip", "acrossfade", "alimiter",
    "loudnorm", "highpass", "lowpass", "equalizer", "afade", "volumedetect",
)
REQUIRED_MODULES = ("numpy", "scipy")


def preflight() -> None:
    missing_bin = [b for b in REQUIRED_BINARIES if shutil.which(b) is None]
    if missing_bin:
        die(
            "missing required tool(s): " + ", ".join(missing_bin)
            + ". Install ffmpeg (e.g. `brew install ffmpeg`) and re-run."
        )
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        die(f"could not query ffmpeg's filter list: {exc}")
    missing_filters = [f for f in REQUIRED_FILTERS if f not in out]
    if missing_filters:
        die(
            "this ffmpeg build is missing required filter(s): " + ", ".join(missing_filters)
            + ". Install a full-featured ffmpeg build (e.g. `brew reinstall ffmpeg`)."
        )
    missing_mod = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
        except ImportError:
            missing_mod.append(mod)
    if missing_mod:
        die(
            "missing required python package(s) for the analysis/loop helper scripts: "
            + ", ".join(missing_mod)
            + f". Install with `pip3 install {' '.join(missing_mod)} --break-system-packages`."
        )


def preflight_voice_removal() -> None:
    """Only called when --remove-voice is passed -- this dependency (torch +
    demucs, a GB+ download) is not part of the normal preflight check."""
    missing = []
    for mod in ("torch", "demucs"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        die(
            "missing package(s) for --remove-voice: " + ", ".join(missing)
            + ". Install with `pip3 install torch --break-system-packages` then "
            "`pip3 install demucs --break-system-packages` (a GB+ download, only "
            "needed for this flag)."
        )


def preflight_denoise_profile() -> None:
    """Only called when --denoise-profile-* is passed."""
    try:
        __import__("noisereduce")
    except ImportError:
        die(
            "missing package for --denoise-profile-*: noisereduce. "
            "Install with `pip3 install noisereduce --break-system-packages`."
        )


def ffmpeg_version() -> str:
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], text=True, stderr=subprocess.STDOUT)
        return out.splitlines()[0].strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Small process/ffmpeg helpers
# ---------------------------------------------------------------------------

def shlex_join(cmd: list[str]) -> str:
    import shlex
    return " ".join(shlex.quote(c) for c in cmd)


def ffprobe_json(path: Path) -> dict:
    raw = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        text=True,
    )
    return json.loads(raw)


def duration_seconds(path: Path) -> float:
    info = ffprobe_json(path)
    return float(info["format"]["duration"])


def volumedetect(path: Path) -> dict:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    out = proc.stdout
    mean = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", out)
    peak = re.search(r"max_volume:\s*([-\d.]+)\s*dB", out)
    return {
        "mean_volume": float(mean.group(1)) if mean else None,
        "max_volume": float(peak.group(1)) if peak else None,
        "raw": out[-2000:],
    }


def loudnorm_measure(path: Path, i: float, tp: float, lra: float) -> dict:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", str(path), "-af",
            f"loudnorm=I={i}:TP={tp}:LRA={lra}:print_format=json",
            "-f", "null", "-",
        ],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    text = proc.stdout
    match = re.search(r"\{[^{}]*input_i[^{}]*\}", text, re.S)
    if not match:
        die("loudnorm did not print JSON measurements")
    return json.loads(match.group(0))


def analyze(path: Path, out_log: Path, recipe: dict | None = None) -> dict:
    crd = RECIPES.get("clean_run_detector", {})
    extra = []
    for key, flag in (
        ("baseline_percentile", "--baseline-pct"),
        ("rumble_factor", "--rumble-factor"),
        ("thump_factor", "--thump-factor"),
        ("mid_factor", "--mid-factor"),
        ("rms_factor", "--rms-factor"),
        ("peak_factor", "--peak-factor"),
        ("peak_clip_ceiling", "--peak-clip"),
    ):
        if key in crd:
            extra.extend([flag, str(crd[key])])
    activity_band = (recipe or {}).get("activity_band")
    if activity_band:
        if len(activity_band) != 2:
            die(f"recipe activity_band must be [lo, hi], got {activity_band!r}")
        extra.extend(["--activity-lo", str(activity_band[0]), "--activity-hi", str(activity_band[1])])
    proc = subprocess.run(
        ["python3", str(ANALYZE), str(path), "--json", *extra],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    out_log.write_text(proc.stdout)
    if "JSON_BEGIN" not in proc.stdout:
        die("analyze_windows.py did not emit JSON")
    blob = proc.stdout.split("JSON_BEGIN", 1)[1].split("JSON_END", 1)[0]
    return json.loads(blob)


def pick_bed(label: str, analysis: dict, long_mode: str, recipe: dict) -> tuple[float, float, str]:
    dur = float(analysis["dur"])
    runs = analysis.get("clean_runs") or []
    # Drop the typical Zoom start rumble if the whole file is used.
    full_start = 2.0 if dur > 6 else 0.0
    full_end = dur

    if not runs:
        return full_start, full_end, "no clean run at current detector settings; using file after 2 s start trim"

    longest = max(runs, key=lambda r: r["len"])
    buggiest = max(runs, key=lambda r: (r["mean_activity"], r["mean_ratio"], r["len"]))

    if long_mode == "full":
        return full_start, full_end, "user asked for full file after cleanup of start"

    strategy = recipe["bed_selection"]
    if strategy == "activity":
        return buggiest["start"], buggiest["end"], f"{label} bed by mean_activity={buggiest['mean_activity']:.5f}"
    if strategy == "longest":
        return longest["start"], longest["end"], f"weather/water bed by longest clean {longest['len']}s"
    if strategy == "mixed":
        if buggiest["mean_activity"] > longest["mean_activity"] * 1.8 and buggiest["len"] >= 12:
            return buggiest["start"], buggiest["end"], "mixed: activity-heavy run outranked longest weather run"
        return longest["start"], longest["end"], "mixed: longest clean run"
    die(f"recipe for label {label!r} has unknown bed_selection strategy {strategy!r}")


def eq_chain(recipe: dict, *, denoise: bool, repair: bool, pre_pad: bool) -> str:
    parts = []
    if repair:
        parts.append(RECIPES["repair_chain"])
    if pre_pad:
        parts.append("volume=-6dB")
    parts.append(recipe["eq"])
    if denoise:
        parts.append(RECIPES["denoise_chain"])
    return ",".join(parts)


def targets(recipe: dict, kind: str) -> tuple[float, float, float]:
    if kind not in ("long", "short"):
        die(f"unknown target kind {kind!r}")
    vals = recipe["targets"][kind]
    return float(vals[0]), float(vals[1]), float(vals[2])


def _ffmpeg_to(cmd: list[str], tmp: Path, dst: Path, *, verify: bool = True) -> None:
    """Run an ffmpeg cmd producing `tmp`, verify with ffprobe, then atomically
    replace `dst`. Always removes any leftover `tmp` -- success or failure --
    so a crashed run doesn't litter the output folder with partial files."""
    if tmp.exists():
        tmp.unlink()
    try:
        print("+ " + shlex_join(cmd))
        subprocess.check_call(cmd)
        if verify:
            subprocess.check_call(["ffprobe", "-v", "error", str(tmp)])
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def ffmpeg_wav(cmd_af: str, src: Path, dst: Path, extra_in: list[str] | None = None) -> None:
    tmp = dst.with_suffix(".tmp.wav")
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    if extra_in:
        cmd.extend(extra_in)
    cmd.extend(["-i", str(src), "-af", cmd_af, "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(tmp)])
    _ffmpeg_to(cmd, tmp, dst)


def copy_trim(src: Path, dst: Path, start: float, end: float) -> None:
    tmp = dst.with_suffix(".tmp.wav")
    dur = max(0.01, end - start)
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(tmp),
    ]
    _ffmpeg_to(cmd, tmp, dst)


def hard_splice_to_target(
    src: Path,
    dst: Path,
    *,
    target: float = 180.0,
    edge: float = 8.0,
    splice: float = 0.012,
    min_interior: float = 15.0,
    max_repeats: int = 10,
) -> tuple[float, int]:
    """Extend a bed to target seconds by tiling its interior with short
    (~12 ms) linear splices -- never a 1.5 s equal-power wrap, which dips on
    correlated (XY/MS) water.

    Repeats the SAME interior clip end-to-end with `splice`-second crossfades
    until the concatenation reaches `target`, then trims to exactly `target`.

    Refuses (raises via `die`) instead of silently returning something
    shorter than `target`, which the previous single-splice version did for
    any clean bed under roughly 100 seconds:
      - if the interior is below `min_interior` seconds, or
      - if reaching `target` would need more than `max_repeats` copies of the
        same clip (that many repeats of identical material would sound
        obviously looped, not just extended).

    Returns (interior_len, repeats_used).
    """
    dur = duration_seconds(src)
    if dur <= edge * 2 + 1:
        die(f"bed too short to interior-splice: {dur:.2f}s")
    interior_start = edge
    interior_end = dur - edge
    interior_len = interior_end - interior_start
    if interior_len < min_interior:
        die(
            f"clean interior is only {interior_len:.1f}s after dropping {edge:.0f}s from each edge -- "
            f"too short to hard-splice to {target:.0f}s. Pass --start/--end for a longer span, "
            f"or --loop never to keep the native length."
        )

    interior = dst.with_name(dst.stem + "_interior.tmp.wav")
    copy_trim(src, interior, interior_start, interior_end)
    try:
        if interior_len >= target:
            copy_trim(interior, dst, 0.0, target)
            return target, 1

        repeats = max(2, math.ceil((target - splice) / (interior_len - splice)))
        if repeats > max_repeats:
            die(
                f"reaching {target:.0f}s from a {interior_len:.1f}s clean interior would need "
                f"{repeats} repeats of the same material ({max_repeats} max) -- that would sound "
                f"obviously looped, not extended. Pick a longer clean span or lower the target."
            )

        tiled = dst.with_name(dst.stem + "_tiled.tmp.wav")
        raw_tmp = dst.with_name(dst.stem + "_tiled.raw.tmp.wav")
        cmd = ["ffmpeg", "-hide_banner", "-y"]
        for _ in range(repeats):
            cmd.extend(["-i", str(interior)])
        chain = []
        prev = "0:a"
        for idx in range(1, repeats):
            label = f"a{idx}"
            chain.append(f"[{prev}][{idx}:a]acrossfade=d={splice:.3f}:c1=tri:c2=tri[{label}]")
            prev = label
        cmd.extend(
            [
                "-filter_complex", ";".join(chain), "-map", f"[{prev}]",
                "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(raw_tmp),
            ]
        )
        _ffmpeg_to(cmd, raw_tmp, tiled)
        try:
            copy_trim(tiled, dst, 0.0, target)
        finally:
            tiled.unlink(missing_ok=True)
        return interior_len, repeats
    finally:
        interior.unlink(missing_ok=True)


def mux_still(still: Path, audio: Path, out_mp4: Path, *, short: bool) -> None:
    """Still + WAV. Cap the infinite image loop to the audio duration.

    `-loop 1` without `-t` on the still leaves dead air after the WAV.
    """
    tmp = out_mp4.with_suffix(".tmp.mp4")
    dur = duration_seconds(audio)
    dur_s = f"{dur:.3f}"
    aac_rate = "192k" if short else "256k"
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-loop", "1", "-framerate", "1", "-t", dur_s, "-i", str(still),
        "-i", str(audio),
        "-c:v", "libx264", "-tune", "stillimage", "-crf", "18", "-preset", "medium",
        "-threads", "4", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", aac_rate, "-ar", "48000",
        "-t", dur_s, "-shortest", "-movflags", "+faststart", str(tmp),
    ]
    out_dur = None
    try:
        print("+ " + shlex_join(cmd))
        subprocess.check_call(cmd)
        subprocess.check_call(["ffprobe", "-v", "error", str(tmp)])
        out_dur = duration_seconds(tmp)
        if abs(out_dur - dur) > 0.15:
            die(f"mux duration mismatch audio={dur:.3f}s mp4={out_dur:.3f}s")
        tmp.replace(out_mp4)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"mux {out_mp4.name}: audio={dur:.3f}s mp4={out_dur:.3f}s")


def crop_still(src: Path, dst: Path, width: int, height: int) -> None:
    tmp = dst.with_suffix(".tmp.jpg")
    if width > height:
        vf = f"crop=iw:'min(ih,iw*{height}/{width})',scale={width}:{height}:flags=lanczos,setsar=1"
    else:
        vf = f"crop='min(iw,ih*{width}/{height})':ih,scale={width}:{height}:flags=lanczos,setsar=1"
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(src), "-vf", vf, "-frames:v", "1", "-q:v", "2", str(tmp)]
    try:
        subprocess.check_call(cmd)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def questions() -> None:
    print(
        """What is this recording, before any processing?

  1. What is it of?   """ + " | ".join(LABELS) + """ | something else
  2. Where was it?    """ + " | ".join(PLACES) + """
  3. Long video:      full file after cleanup, or only the longest clean span?
  4. Still-frame:     any constraints? (no glass, two cups, no people)

Re-run with flags, for example:

  process_field_wav.py take.wav --label brook --place brook --long clean --loop never
"""
    )


def parse_timestamp(value: str) -> float:
    """Accept plain seconds ("5", "12.5") for --start, or a clock-style
    timestamp ("3:45" = 3m45s, "1:02:30" = 1h2m30s) for --end -- either flag
    takes either format. Used as the argparse `type=` for --start/--end."""
    value = value.strip()
    parts = value.split(":") if ":" in value else [value]
    if len(parts) not in (1, 2, 3):
        raise argparse.ArgumentTypeError(
            f"not a valid time: {value!r} (use seconds like 12.5, or mm:ss / hh:mm:ss)"
        )
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"not a valid time: {value!r} (use seconds like 12.5, or mm:ss / hh:mm:ss)"
        )
    seconds = 0.0
    for p in parts_f:
        seconds = seconds * 60 + p
    return seconds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean a field WAV into long + 3-minute Short masters."
    )
    p.add_argument("input", nargs="?", help="Source WAV (or any ffmpeg-readable audio)")
    p.add_argument("--label", choices=LABELS, help="Required. What the take is of.")
    p.add_argument("--place", choices=PLACES, default="unknown")
    p.add_argument(
        "--loop",
        choices=("auto", "never", "force"),
        default="auto",
        help="Short length. auto = never loop correlated water/brook/rain/thunder; "
        "insects/birds may still wrap to 180 s. force = short linear splices tiled to 180 s.",
    )
    p.add_argument(
        "--splice",
        type=float,
        default=0.012,
        help="Seconds of linear overlap when --loop force (default 0.012).",
    )
    p.add_argument(
        "--loop-target",
        type=parse_timestamp,
        help="Extend the LONG master to this length too -- seconds, or mm:ss/hh:mm:ss "
        "(e.g. 1:00:00 for an hour). Uses hard-splice tiling for correlated water "
        "(rain/thunder/water/brook, or whenever --loop force) and the equal-power wrap "
        "otherwise, same as the Short. If the clean bed is already this long or longer, "
        "it's just trimmed to exactly this length.",
    )
    p.add_argument(
        "--long",
        choices=("full", "clean"),
        default="clean",
        help="full = usable file after start trim; clean = best bed only",
    )
    p.add_argument(
        "--start",
        type=parse_timestamp,
        help="Seconds to trim from the beginning (e.g. 5), or an mm:ss / hh:mm:ss "
        "timestamp for the bed's start",
    )
    p.add_argument(
        "--end",
        type=parse_timestamp,
        help="Where to cut off the bed -- seconds, or an mm:ss / hh:mm:ss timestamp "
        "(e.g. 3:45 or 1:02:30)",
    )
    p.add_argument("--out-dir", type=Path, help="Output folder (default: <stem>_fieldaudio/)")
    p.add_argument("--denoise", action="store_true", help="Optional afftdn nr=8 only")
    p.add_argument("--repair", action="store_true", help="adeclick + adeclip before EQ")
    p.add_argument("--skip-short", action="store_true")
    p.add_argument("--skip-long", action="store_true")
    p.add_argument(
        "--still-16x9",
        type=Path,
        help="Existing landscape still; cropped to 1920x1080 and 1280x720",
    )
    p.add_argument(
        "--still-9x16",
        type=Path,
        help="Existing portrait still; cropped to 1080x1920",
    )
    p.add_argument(
        "--ask",
        action="store_true",
        help="Print the four questions and exit, even if flags are present",
    )
    p.add_argument(
        "--recipes",
        type=Path,
        default=DEFAULT_RECIPES_PATH,
        help="Path to the label recipe JSON (default: recipes.json next to this script)",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the chosen bed, EQ chain, and loudness targets, then exit without rendering",
    )
    p.add_argument(
        "--remove-voice",
        action="store_true",
        help="Run Demucs vocal separation on the trimmed bed to strip a speech-like "
        "foreground (talking, footsteps, a passing conversation) before EQ. Heavy "
        "optional dependency (torch + demucs, GB+ download); see remove_foreground.py.",
    )
    p.add_argument(
        "--voice-model",
        default="htdemucs",
        help="Demucs model name for --remove-voice (default: htdemucs)",
    )
    p.add_argument(
        "--denoise-profile-start",
        type=parse_timestamp,
        help="Start of a noise-only span within the trimmed bed (0 = start of bed), used "
        "to build a spectral profile of a known steady noise (hum, hiss, distant "
        "traffic) and subtract it. Pair with --denoise-profile-end, or use "
        "--denoise-profile-file instead. See denoise_profile.py.",
    )
    p.add_argument(
        "--denoise-profile-end",
        type=parse_timestamp,
        help="End of the noise-only span (see --denoise-profile-start).",
    )
    p.add_argument(
        "--denoise-profile-file",
        type=Path,
        help="Alternative to --denoise-profile-start/-end: a separate short WAV "
        "containing just the unwanted noise.",
    )
    p.add_argument(
        "--denoise-non-stationary",
        action="store_true",
        help="For --denoise-profile-*: the noise's character drifts over time "
        "(default assumes a steady/stationary noise).",
    )
    p.add_argument(
        "--denoise-prop-decrease",
        type=float,
        default=1.0,
        help="For --denoise-profile-*: how aggressively to subtract the noise profile, "
        "0-1 (default 1.0 = full).",
    )
    return p


def write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    if args.ask or not args.input or not args.label:
        questions()
        if not args.input:
            die("pass a WAV path as the first argument", 2)
        if not args.label:
            die("--label is required so the EQ recipe is not guessed", 2)
        return 2

    preflight()

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        die(f"file not found: {src}")

    recipe = RECIPES["labels"][args.label]

    out_dir = (args.out_dir or src.parent / f"{src.stem}_fieldaudio").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"source: {src}")
    print(f"out:    {out_dir}")
    print(f"label:  {args.label}   place: {args.place}   long: {args.long}")
    print(f"recipes: {RECIPES_PATH}")
    print("note:   no stem isolation; bands only; source is never deleted")
    if recipe.get("untested"):
        print(
            f"WARNING: the {args.label!r} recipe is marked \"untested\" in {RECIPES_PATH.name} -- "
            "A/B it against a few real takes before trusting it unattended."
        )

    probe = subprocess.check_output(["ffprobe", "-hide_banner", "-i", str(src)], stderr=subprocess.STDOUT, text=True)
    (out_dir / "00_probe.txt").write_text(probe)
    print(probe)

    vd0 = volumedetect(src)
    print(f"source max_volume={vd0['max_volume']} dB  mean={vd0['mean_volume']} dB")

    analysis = analyze(src, out_dir / "01_windows.txt", recipe)
    print(f"duration={analysis['dur']:.3f}s  clean_runs={len(analysis.get('clean_runs') or [])}")

    if args.start is not None and args.end is not None:
        start, end, why = args.start, args.end, "manual --start/--end"
    else:
        start, end, why = pick_bed(args.label, analysis, args.long, recipe)
        if args.start is not None:
            start = args.start
            why += " (start overridden)"
        if args.end is not None:
            end = args.end
            why += " (end overridden)"

    if end <= start:
        die(f"invalid bed {start}-{end}")

    print(f"bed: {start:.2f}-{end:.2f}s  ({why})")

    trim_path = out_dir / "02_trim.wav"
    copy_trim(src, trim_path, start, end)

    pre_eq_notes: list[str] = []
    have_profile = (
        args.denoise_profile_start is not None
        or args.denoise_profile_end is not None
        or args.denoise_profile_file is not None
    )
    if have_profile and not args.plan_only:
        if (args.denoise_profile_start is None) != (args.denoise_profile_end is None):
            die("pass both --denoise-profile-start and --denoise-profile-end, or neither")
        if args.denoise_profile_file is not None and args.denoise_profile_start is not None:
            die("pass --denoise-profile-start/-end OR --denoise-profile-file, not both")
        preflight_denoise_profile()
        denoised = out_dir / "02a_denoise_profile.wav"
        cmd = [
            "python3", str(DENOISE_PROFILE), str(trim_path), "--out", str(denoised),
            "--prop-decrease", str(args.denoise_prop_decrease),
        ]
        if args.denoise_non_stationary:
            cmd.append("--non-stationary")
        if args.denoise_profile_file is not None:
            cmd.extend(["--noise-file", str(args.denoise_profile_file)])
        else:
            cmd.extend([
                "--noise-start", f"{args.denoise_profile_start:.3f}",
                "--noise-end", f"{args.denoise_profile_end:.3f}",
            ])
        print(f"+ denoise-profile: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        trim_path = denoised
        noise_desc = (
            str(args.denoise_profile_file)
            if args.denoise_profile_file is not None
            else f"{args.denoise_profile_start:.2f}-{args.denoise_profile_end:.2f}s of trimmed bed"
        )
        pre_eq_notes.append(
            f"denoise_profile=yes noise={noise_desc} "
            f"non_stationary={args.denoise_non_stationary} prop_decrease={args.denoise_prop_decrease}"
        )

    if args.remove_voice and not args.plan_only:
        preflight_voice_removal()
        voice_removed = out_dir / "02b_voice_removed.wav"
        cmd = [
            "python3", str(REMOVE_FOREGROUND), str(trim_path),
            "--out", str(voice_removed), "--model", args.voice_model,
        ]
        print(f"+ remove-voice: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        trim_path = voice_removed
        pre_eq_notes.append(f"remove_voice=yes model={args.voice_model}")

    vd_trim = volumedetect(trim_path)
    max_vol = vd_trim["max_volume"] if vd_trim["max_volume"] is not None else -12.0

    # Linear gain if the take is quiet. Pad before EQ if already hot.
    pre_pad = max_vol > -6.0
    gain_db = 0.0
    if max_vol < -8.0:
        gain_db = abs(max_vol) - 1.0
        pre_pad = False

    chain = eq_chain(recipe, denoise=args.denoise, repair=args.repair, pre_pad=pre_pad)
    if gain_db:
        chain = f"volume={gain_db}dB," + chain
    print(f"eq: {chain}")

    if args.plan_only:
        long_i, long_tp, long_lra = targets(recipe, "long")
        short_i, short_tp, short_lra = targets(recipe, "short")
        loop_mode = args.loop
        if loop_mode == "auto" and args.label in CORRELATED_WATER:
            loop_mode = "never"
        print("\n=== PLAN ONLY (no rendering) ===")
        print(f"bed: {start:.2f}-{end:.2f}s ({why})")
        print(f"eq chain: {chain}")
        print(f"long targets:  I={long_i} TP={long_tp} LRA={long_lra}")
        print(f"short targets: I={short_i} TP={short_tp} LRA={short_lra}")
        print(f"short loop mode resolved to: {loop_mode}")
        print(f"correlated_water(no free wrap): {args.label in CORRELATED_WATER}")
        if have_profile:
            print("denoise-profile: would run (skipped for --plan-only)")
        if args.remove_voice:
            print("remove-voice: would run (skipped for --plan-only)")
        return 0

    eq_path = out_dir / "03_eq.wav"
    ffmpeg_wav(chain, trim_path, eq_path)

    products = []
    report = [
        f"run_at_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"ffmpeg={ffmpeg_version()}",
        f"recipes_file={RECIPES_PATH}",
        f"source={src}",
        f"label={args.label}",
        f"place={args.place}",
        f"bed={start:.3f}-{end:.3f} ({why})",
        f"source_max_volume={vd0['max_volume']}",
        f"trim_max_volume={vd_trim['max_volume']}",
        f"eq={chain}",
        "did_not=fake_LRA,delete_source",
    ]
    if not args.remove_voice:
        report.append("did_not_stem_split=true (pass --remove-voice to enable)")
    report.extend(pre_eq_notes)
    if recipe.get("untested"):
        report.append(f"untested_recipe={args.label} (A/B before trusting unattended)")

    def finish_master(kind: str, audio_src: Path, fade_in: float, fade_out_start_from_end: float) -> Path:
        i, tp, lra = targets(recipe, kind)
        meas = loudnorm_measure(audio_src, i, tp, lra)
        dur = duration_seconds(audio_src)
        fade_out_at = max(0.0, dur - fade_out_start_from_end)
        ln = (
            f"loudnorm=I={i}:TP={tp}:LRA={lra}"
            f":measured_I={meas['input_i']}:measured_LRA={meas['input_lra']}"
            f":measured_TP={meas['input_tp']}:measured_thresh={meas['input_thresh']}"
            f":offset={meas['target_offset']}:linear=true"
        )
        # loudnorm true-peak path can emit 192 kHz and 0 dBFS sample peaks.
        # Force 48 kHz stereo in ffmpeg_wav. Pad then limiter (lotus_lake 2026-08-30).
        af = (
            f"{ln},"
            f"afade=t=in:st=0:d={fade_in},"
            f"afade=t=out:st={fade_out_at:.3f}:d={fade_out_start_from_end},"
            f"volume=-1.6dB,"
            f"alimiter=limit=-1.5dB:level=false:attack=0.5:release=50"
        )
        dest = out_dir / f"master_{kind}.wav"
        ffmpeg_wav(af, audio_src, dest)
        vd = volumedetect(dest)
        report.append(
            f"{kind}: dur={dur:.3f} I_target={i} measured_I={meas.get('input_i')} "
            f"input_lra={meas.get('input_lra')} max_volume={vd['max_volume']}"
        )
        return dest

    long_master = None
    if not args.skip_long:
        long_src = eq_path
        long_dur = duration_seconds(eq_path)
        if args.loop_target is not None:
            target = args.loop_target
            if target <= 0:
                die("--loop-target must be positive")
            loop_mode = args.loop
            if loop_mode == "auto" and args.label in CORRELATED_WATER:
                loop_mode = "never"
            if long_dur >= target:
                clipped = out_dir / "03b_long_clip.wav"
                copy_trim(eq_path, clipped, 0.0, target)
                long_src = clipped
                report.append(
                    f"long_looped=no bed_already_{long_dur:.3f}s_trimmed_to_{target:.3f}s"
                )
            elif loop_mode == "force":
                looped = out_dir / "03b_long_splice.wav"
                if long_dur < 8.0:
                    die("clean bed too short to splice the long master -- pass --start/--end")
                max_repeats = max(10, math.ceil(target / 60))
                interior_len, repeats = hard_splice_to_target(
                    eq_path, looped, target=target, edge=8.0,
                    splice=max(0.008, args.splice), max_repeats=max_repeats,
                )
                long_src = looped
                report.append(
                    f"long_looped=hard_splice interior={interior_len:.3f} repeats={repeats} "
                    f"splice={args.splice:.3f} target={target:.3f} (tiled, no equal-power wrap)"
                )
            elif loop_mode == "never":
                report.append(
                    f"long_looped=no kept_native_{long_dur:.3f}s "
                    f"(correlated {args.label}; equal-power wrap dips -- pass --loop force to hard-splice)"
                )
            else:
                looped = out_dir / "03b_long_wrap.wav"
                slice_start = start
                slice_end = end
                if (slice_end - slice_start) < 4.0:
                    die("clean slice shorter than 4s; cannot loop the long master -- pass --start/--end")
                subprocess.check_call(
                    [
                        "python3", str(LOOP), str(src),
                        "--start", f"{slice_start:.3f}",
                        "--end", f"{slice_end:.3f}",
                        "--target", f"{target:.3f}",
                        "--out", str(looped),
                    ]
                )
                looped_eq = out_dir / "03b_long_wrap_eq.wav"
                ffmpeg_wav(chain, looped, looped_eq)
                long_src = looped_eq
                report.append(
                    f"long_looped=yes slice={slice_start:.3f}-{slice_end:.3f} target={target:.3f}"
                )
            long_dur = duration_seconds(long_src)
        fade = 4.0 if long_dur > 120 else 1.0
        long_master = finish_master("long", long_src, fade, fade)
        products.append(long_master)

    short_master = None
    if not args.skip_short:
        short_src = eq_path
        short_dur = duration_seconds(eq_path)
        loop_mode = args.loop
        if loop_mode == "auto" and args.label in CORRELATED_WATER:
            loop_mode = "never"
        if short_dur < 180 and loop_mode == "force":
            looped = out_dir / "04_short_splice.wav"
            if short_dur < 8.0:
                die("clean slice too short to splice a Short -- pass --start/--end")
            interior_len, repeats = hard_splice_to_target(
                eq_path, looped, target=180.0, edge=8.0, splice=max(0.008, args.splice)
            )
            short_src = looped
            report.append(
                f"short_looped=hard_splice interior={interior_len:.3f} repeats={repeats} "
                f"splice={args.splice:.3f} (tiled, no esin wrap)"
            )
        elif short_dur < 180 and loop_mode == "never":
            short_src = eq_path
            report.append(
                f"short_looped=no kept_native_{short_dur:.3f}s "
                f"(correlated {args.label}; XY wrap dips)"
            )
        elif short_dur < 180:
            looped = out_dir / "04_short_loop.wav"
            slice_start = start
            slice_end = end
            if (slice_end - slice_start) < 4.0:
                die("clean slice shorter than 4 s; cannot loop a Short -- pass --start/--end")
            subprocess.check_call(
                [
                    "python3", str(LOOP), str(src),
                    "--start", f"{slice_start:.3f}",
                    "--end", f"{slice_end:.3f}",
                    "--target", "180",
                    "--out", str(looped),
                ]
            )
            looped_eq = out_dir / "04_short_loop_eq.wav"
            ffmpeg_wav(chain, looped, looped_eq)
            short_src = looped_eq
            report.append(f"short_looped=yes slice={slice_start:.3f}-{slice_end:.3f}")
        else:
            # Trim the first 180 s of the EQ bed for the Short.
            clipped = out_dir / "04_short_clip.wav"
            copy_trim(eq_path, clipped, 0.0, 180.0)
            short_src = clipped
            report.append("short_looped=no used_first_180s_of_eq_bed")
        short_master = finish_master("short", short_src, 0.4, 0.4)
        products.append(short_master)

    stills = {}
    if args.still_16x9 and args.still_16x9.exists():
        still_1080 = out_dir / "still_1920x1080.jpg"
        thumb_720 = out_dir / "thumb_1280x720.jpg"
        crop_still(args.still_16x9, still_1080, 1920, 1080)
        crop_still(args.still_16x9, thumb_720, 1280, 720)
        stills["16x9"] = still_1080
        stills["thumb"] = thumb_720
    if args.still_9x16 and args.still_9x16.exists():
        still_vert = out_dir / "still_1080x1920.jpg"
        crop_still(args.still_9x16, still_vert, 1080, 1920)
        stills["9x16"] = still_vert
        shutil.copyfile(still_vert, out_dir / "thumb_1080x1920.jpg")

    if long_master and stills.get("16x9"):
        mux = out_dir / "long.mp4"
        mux_still(stills["16x9"], long_master, mux, short=False)
        products.append(mux)
    if short_master and stills.get("9x16"):
        mux = out_dir / "short.mp4"
        mux_still(stills["9x16"], short_master, mux, short=True)
        products.append(mux)

    if not stills:
        report.append(
            "stills=skipped (pass --still-16x9 and --still-9x16 to mux still-frame videos)"
        )
        print(
            "stills not generated here. Make frames that match place+weather, then re-run "
            "with --still-16x9 and --still-9x16. Default look is covered porch only after you say so."
        )

    report.append("products:")
    for pth in products:
        report.append(f"  {pth}")
    write_report(out_dir / "REPORT.txt", report)

    print("\n=== REPORT ===")
    print("\n".join(report))
    print(f"\nWrote {out_dir / 'REPORT.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
