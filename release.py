#!/usr/bin/env python3
"""One command from a finished Bluebox mix to upload-ready YouTube and
Instagram files.

  1. master the mix            process_music_wav.py  -> audio/master_24bit_48k.wav (+ formats)
                                                        audio/music_analysis.json
  2. pick the Short/Reel clip  pick_clip.py          -> clips/clip_<N>s.wav (+ grid)
  3. full-length video         visualize_wav.py      -> <slug>_youtube_16x9.mp4
  4. Short / Reel video        visualize_wav.py      -> <slug>_short_9x16.mp4
     (optional square)                               -> <slug>_feed_1x1.mp4
  5. captions                                        -> captions.txt

Every step is one of the standalone scripts, run as-is -- this file only
wires them together, so anything it does can be redone by hand with the
same flags. The visuals are beat-locked to the analyzed grid (--grid) and
use the genre's default style from music_recipes.json (--style, --emoji,
--symmetry override it).

--outputs picks the deliverables (default: master,clip,video_16x9,
video_9x16,captions -- plus video_1x1 with --square, minus video_16x9 with
--no-landscape):
  master      the mastered audio, in every --formats format (wav24, wav16, flac, mp3)
  clip        the Short/Reel excerpt as a WAV
  video_16x9  full-length YouTube video
  video_9x16  Short / Reel video
  video_1x1   square Instagram feed video
  captions    captions.txt
  analysis    music_analysis.json + a one-page PNG chart
Steps run only when a requested output needs them (the clip is made for
the 9:16/1:1 videos even if the clip WAV itself isn't requested).

Every mastering switch of process_music_wav.py passes straight through
(--no-eq, --no-glue, --mono-bass-hz, --fade-in, ...). --progress-json
emits machine-readable progress, ending in a manifest of the deliverables;
--plan-only --json prints the steps as JSON.

Video rendering is the slow part: roughly a few seconds of render time per
second of audio at 1920x1080. --max-seconds renders short previews.

Usage:
  python3 release.py bluebox_mix.wav --genre techno --title "Night Drive"
  python3 release.py mix.wav --genre trap --title "Low Smoke" --clip-length 15 --loop-clip
  python3 release.py mix.wav --genre techno --outputs master --formats flac,mp3
  python3 release.py master.wav --genre ambient --title "Peaceful Sunrise" --already-mastered --square
  python3 release.py mix.wav --genre psytrance --title "Fractal" --max-seconds 20   # quick preview
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from music_common import (
    SCRIPT_DIR, die, ffmpeg_to, fmt_time, genre_recipe, load_recipes,
    parse_timestamp, preflight, probe,
)
from pipeline_io import (
    AUDIO_FORMATS, PARENT_STEP_ENV, emit, emit_manifest, emit_plan, enable_progress,
    export_formats, format_path, json_mode, parse_formats, print_json,
)
from video_render import (
    BEAT_STYLES, FFMPEG_STYLES, VIZ_PASSTHROUGH, add_viz_group, video_command,
)

DEFAULT_METHOD = "Hardware synths, recorded DAWless and mixed live on a 1010music Bluebox."
OUTPUTS = {
    "master": "Mastered audio",
    "clip": "Short/Reel clip (WAV)",
    "video_16x9": "YouTube video (16:9)",
    "video_9x16": "Short / Reel video (9:16)",
    "video_1x1": "Instagram feed video (1:1)",
    "captions": "Captions",
    "analysis": "Analysis (JSON + chart)",
}
# process_music_wav.py switches release.py passes straight through:
# (argparse dest, flag, kind)
MASTER_PASSTHROUGH = (
    ("target_i", "--target-i", float), ("target_tp", "--target-tp", float),
    ("width", "--width", float), ("mono_bass_hz", "--mono-bass-hz", float),
    ("highpass_hz", "--highpass-hz", float), ("softclip_threshold", "--softclip-threshold", float),
    ("fade_in", "--fade-in", float), ("fade_out", "--fade-out", float),
    ("no_mono_bass", "--no-mono-bass", bool), ("no_eq", "--no-eq", bool),
    ("no_dynamic_eq", "--no-dynamic-eq", bool), ("no_glue", "--no-glue", bool),
    ("softclip", "--softclip", bool), ("no_softclip", "--no-softclip", bool),
)


def parse_outputs(value: str) -> tuple[str, ...]:
    out: list[str] = []
    for o in (x.strip().lower() for x in value.split(",")):
        if not o:
            continue
        if o not in OUTPUTS:
            raise argparse.ArgumentTypeError(f"unknown output {o!r}; choose from {', '.join(OUTPUTS)}")
        if o not in out:
            out.append(o)
    if not out:
        raise argparse.ArgumentTypeError("--outputs needs at least one output")
    return tuple(out)


def run_step(name: str, label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===\n+ {shlex.join(cmd)}", flush=True)
    emit("step_start", step=name, label=label)
    env = dict(os.environ, **{PARENT_STEP_ENV: name})
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        emit("step_error", step=name, message=f"exit {rc}")
        die(f"{label} failed (exit {rc}) -- see the output above; outputs from earlier steps are kept")
    emit("step_end", step=name)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "track"


def captions(title: str, artist: str | None, genre: str, recipe: dict, a: dict, clip_len: float | None, method: str) -> str:
    t, k = a["tempo"], a["key"]
    facts = [genre]
    if t.get("has_grid"):
        facts.append(f"{t['bpm']:g} BPM")
    if k.get("name"):
        facts.append(f"{k['name']} ({k['camelot']})")
    tags = recipe["hashtags"][:7]  # youtube-mux.md convention: under 8, at the bottom
    by = f" -- {artist}" if artist else ""
    yt = [
        "YOUTUBE (full track)",
        f"Title: {title}{by} | {genre.capitalize()}",
        "",
        "Description:",
        f"{title} -- {', '.join(facts)}.",
        method,
        f"Length: {fmt_time(a['duration'])}",
        "",
        " ".join(tags),
    ]
    short = [
        "YOUTUBE SHORT / INSTAGRAM REEL",
        f"Title: {title}{by} #{genre} #shorts",
        f"Caption: {title} -- {', '.join(facts[1:]) or genre}. {method}",
        " ".join(tags),
    ]
    if clip_len is not None:
        short.append(f"(clip is {clip_len:.0f}s -- check the platforms' current length limits before posting)")
    return "\n".join(yt + ["", "-" * 40, ""] + short) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Bluebox stereo mix (or a finished master with --already-mastered)")
    p.add_argument("--genre", help="ambient | trap | techno | psytrance (required -- not guessed)")
    p.add_argument("--title", help="Track title (video overlay, tags, captions; default: file name)")
    p.add_argument("--artist", help="Artist tag/captions")
    p.add_argument("--outputs", type=parse_outputs, help=f"Comma-separated deliverables: {', '.join(OUTPUTS)}")
    p.add_argument("--formats", type=parse_formats, default=("wav24", "wav16"),
                   help=f"Audio formats for the master output: {', '.join(AUDIO_FORMATS)} (default: wav24,wav16)")
    p.add_argument("--preset", default="youtube", help="Loudness preset for the master (default: youtube)")
    p.add_argument("--start", type=parse_timestamp, help="Trim the mix: start (seconds or mm:ss)")
    p.add_argument("--end", type=parse_timestamp, help="Trim the mix: end (seconds or mm:ss)")
    p.add_argument("--already-mastered", action="store_true", help="Input is already a master: convert/analyze it, don't re-master")
    m = p.add_argument_group("mastering switches (passed to process_music_wav.py)")
    for dest, flag, kind in MASTER_PASSTHROUGH:
        if kind is bool:
            m.add_argument(flag, dest=dest, action="store_true")
        else:
            m.add_argument(flag, dest=dest, type=kind)
    p.add_argument("--clip-length", type=float, default=30.0, help="Short/Reel length in seconds (default 30)")
    p.add_argument("--loop-clip", action="store_true", help="Cut the Short/Reel as a seamless bar-exact loop")
    p.add_argument("--drop-at", type=parse_timestamp, help="Override the drop the clip is built around")
    p.add_argument("--clip-start", type=parse_timestamp, help="Pin the clip's start instead")
    p.add_argument("--style", choices=BEAT_STYLES + FFMPEG_STYLES,
                   help="Visualizer style: beat-locked (radial | bars | glowburst | wormhole, visualize_wav.py) "
                        "or ffmpeg (cqt | spectrum | waves | vectorscope | freqs | histogram | spatial, "
                        "ffmpeg_visualize.py); default: the genre's style")
    p.add_argument("--emoji", help="Center emoji (radial style); overrides the genre default")
    p.add_argument("--no-emoji", action="store_true", help="Drop the genre's default emoji")
    p.add_argument("--symmetry", type=int, help="Radial symmetry (1 = off); overrides the genre default")
    add_viz_group(p)
    p.add_argument("--square", action="store_true", help="Legacy: same as adding video_1x1 to the default outputs")
    p.add_argument("--no-landscape", action="store_true", help="Legacy: drop video_16x9 from the default outputs")
    p.add_argument("--max-seconds", type=float, help="Render only the first N seconds of each video (preview)")
    p.add_argument("--method", default=DEFAULT_METHOD, help="How-it-was-made line for the captions")
    p.add_argument("--out-dir", type=Path, help="Output folder (default: <stem>_release/)")
    p.add_argument("--plan-only", action="store_true", help="Print the steps and exit")
    p.add_argument("--json", action="store_true", help="With --plan-only: print the steps as one JSON document")
    p.add_argument("--progress-json", action="store_true", help="Emit machine-readable progress lines (see pipeline_io.py)")
    return p


def visual_args(args, recipe: dict) -> tuple[str, list[str]]:
    """Style + its knobs: the genre default unless overridden. --style
    alone drops the genre's style-specific knobs (an emoji only means
    something on radial)."""
    style = args.style or recipe["visual_style"]
    extra = [] if args.style else list(recipe["visual_args"])

    def drop(flag):
        while flag in extra:
            i = extra.index(flag)
            del extra[i:i + 2]

    if args.no_emoji or args.emoji:
        drop("--emoji")
    if args.emoji:
        extra += ["--emoji", args.emoji]
    if args.symmetry is not None:
        drop("--symmetry")
        extra += ["--symmetry", str(args.symmetry)]
    return style, extra


def main() -> int:
    args = build_parser().parse_args()
    if args.json and not args.plan_only:
        die("--json only applies to --plan-only", 2)
    if args.json:
        json_mode()
    if args.progress_json:
        enable_progress()
    recipes = load_recipes()
    recipe = genre_recipe(recipes, args.genre)
    src = Path(args.input).expanduser().resolve()
    if not args.plan_only and not src.exists():
        die(f"file not found: {src}")
    outputs = args.outputs
    if outputs is None:
        outputs = ("master", "clip") + (() if args.no_landscape else ("video_16x9",)) + ("video_9x16",) \
            + (("video_1x1",) if args.square else ()) + ("captions",)
    elif args.square or args.no_landscape:
        die("--square/--no-landscape are shorthands for the default outputs; with --outputs, list what you want", 2)
    if args.loop_clip and not recipe["expect_grid"]:
        print(f"note: {args.genre} usually has no beat grid -- --loop-clip will fail if analysis finds none")

    title = args.title or src.stem
    slug = slugify(title)
    out = (args.out_dir or src.parent / f"{src.stem}_release").resolve()
    audio, clips = out / "audio", out / "clips"
    py = sys.executable
    style, vis_extra = visual_args(args, recipe)
    clip_tag = f"{int(round(args.clip_length))}s" + ("_loop" if args.loop_clip else "")
    master = audio / "master_24bit_48k.wav"
    analysis = audio / "music_analysis.json"
    clip = clips / f"clip_{clip_tag}.wav"
    clip_grid = clips / f"clip_{clip_tag}_grid.json"
    need_clip = any(o in outputs for o in ("clip", "video_9x16", "video_1x1"))
    formats = args.formats if "master" in outputs else ("wav24",)

    steps: list[tuple[str, str, list[str] | None]] = []
    if args.already_mastered:
        steps.append(("master", "Preparing the master", None))
        steps.append(("analyze", "Analyzing the master",
                      [py, str(SCRIPT_DIR / "music_analyze.py"), str(master), "--genre", args.genre,
                       "--out-dir", str(audio)]))
    else:
        cmd = [py, str(SCRIPT_DIR / "process_music_wav.py"), str(src), "--genre", args.genre,
               "--preset", args.preset, f"--title={title}", "--out-dir", str(audio),
               "--formats", ",".join(formats)]
        if args.artist:
            cmd += [f"--artist={args.artist}"]  # one token: a leading "-" can't read as a flag
        if args.start is not None:
            cmd += ["--start", f"{args.start}"]
        if args.end is not None:
            cmd += ["--end", f"{args.end}"]
        for dest, flag, kind in MASTER_PASSTHROUGH:
            val = getattr(args, dest)
            if kind is bool and val:
                cmd.append(flag)
            elif kind is not bool and val is not None:
                cmd += [flag, f"{val:g}"]
        steps.append(("master", "Mastering", cmd))
    if need_clip:
        cmd = [py, str(SCRIPT_DIR / "pick_clip.py"), str(master), "--genre", args.genre,
               "--length", f"{args.clip_length:g}", "--analysis", str(analysis), "--out-dir", str(clips)]
        if args.loop_clip:
            cmd.append("--loop")
        if args.drop_at is not None:
            cmd += ["--drop-at", f"{args.drop_at}"]
        if args.clip_start is not None:
            cmd += ["--start", f"{args.clip_start}"]
        steps.append(("clip", "Picking the Short/Reel clip", cmd))
    if style not in FFMPEG_STYLES and any(getattr(args, d) not in (None, False) for d, _f, _k in VIZ_PASSTHROUGH):
        print(f"note: --viz-* options only apply to ffmpeg styles; ignoring them for {style}")
    videos = {
        "video_16x9": (out / f"{slug}_youtube_16x9.mp4", "landscape", master, analysis, "Rendering the YouTube video (16:9)"),
        "video_9x16": (out / f"{slug}_short_9x16.mp4", "shorts", clip, clip_grid, "Rendering the Short / Reel video (9:16)"),
        "video_1x1": (out / f"{slug}_feed_1x1.mp4", "square", clip, clip_grid, "Rendering the square video (1:1)"),
    }
    for key, (dst, fmt, audio_in, grid, label) in videos.items():
        if key in outputs:
            # ffmpeg styles follow the sound itself; the beat-locked ones get the grid
            steps.append((key, label, video_command(args, style, audio_in, dst, key, title,
                                                    python_extra=vis_extra,
                                                    grid=None if style in FFMPEG_STYLES else grid)))
    if "captions" in outputs:
        steps.append(("captions", "Writing captions", None))
    if "analysis" in outputs:
        steps.append(("analysis", "Drawing the analysis chart", None))

    plan_doc = {
        "pipeline": "music", "script": "release.py", "genre": args.genre, "title": title,
        "untested_recipe": bool(recipe.get("untested")), "outputs": list(outputs),
        "formats": list(formats) if "master" in outputs else [], "style": style, "visual_args": vis_extra,
        "out_dir": str(out),
        "steps": [{"step": n, "label": lbl, "command": c} for n, lbl, c in steps],
    }
    if args.plan_only:
        if args.json:
            print_json(plan_doc)
            return 0
        print(f"release: {title} ({args.genre}{', recipe untested' if recipe.get('untested') else ''})  ->  {out}")
        print(f"outputs: {', '.join(outputs)}" + (f"   formats: {', '.join(formats)}" if "master" in outputs else ""))
        for name, label, cmd in steps:
            print(f"\n[{label}]\n  " + (shlex.join(cmd) if cmd else "(in release.py)"))
        return 0

    preflight()
    out.mkdir(parents=True, exist_ok=True)
    emit_plan([(n, lbl) for n, lbl, _ in steps])
    print(f"release: {title} ({args.genre})  outputs: {', '.join(outputs)}  ->  {out}")
    for name, label, cmd in steps:
        if cmd is not None:
            run_step(name, label, cmd)
        elif name == "master":
            # already mastered: normalize the container to the 24-bit/48k
            # master every later step expects -- no processing -- then the
            # requested formats and an analysis
            emit("step_start", step=name, label=label)
            audio.mkdir(parents=True, exist_ok=True)
            tmp = master.with_suffix(".tmp.wav")
            meta = {"title": title, "artist": args.artist or "", "genre": args.genre}
            meta_args = [a for k, v in meta.items() if v for a in ("-metadata", f"{k}={v}")]
            ffmpeg_to(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                       "-ar", "48000", "-c:a", "pcm_s24le", *meta_args, str(tmp)], tmp, master)
            export_formats(master, "master", formats, meta)
            emit("step_end", step=name)
        elif name == "captions":
            emit("step_start", step=name, label=label)
            a = json.loads(analysis.read_text())
            clip_len = probe(clip)["duration"] if need_clip else None
            (out / "captions.txt").write_text(captions(title, args.artist, args.genre, recipe, a, clip_len, args.method))
            emit("step_end", step=name)
        elif name == "analysis":
            emit("step_start", step=name, label=label)
            from music_analyze import write_png
            write_png(json.loads(analysis.read_text()), audio / "music_analysis.png")
            emit("step_end", step=name)

    files: list[dict] = []
    if "master" in outputs:
        files += [{"kind": "master", "format": f, "path": format_path(master, "master", f)} for f in formats]
    if "clip" in outputs:
        files.append({"kind": "clip", "format": "wav", "path": clip})
    for key, (dst, *_rest) in videos.items():
        if key in outputs:
            files.append({"kind": key, "format": "mp4", "path": dst})
    if "captions" in outputs:
        files.append({"kind": "captions", "format": "txt", "path": out / "captions.txt"})
    if "analysis" in outputs:
        files += [{"kind": "analysis", "format": "json", "path": analysis},
                  {"kind": "analysis", "format": "png", "path": audio / "music_analysis.png"}]
    missing = [str(f["path"]) for f in files if not Path(f["path"]).exists()]
    if missing:
        die("expected outputs are missing: " + ", ".join(missing))

    report = [
        f"run_at_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"source={src}",
        f"genre={args.genre} title={title!r} style={style} {' '.join(vis_extra)}".rstrip(),
        f"outputs={','.join(outputs)}" + (f" formats={','.join(formats)}" if "master" in outputs else ""),
        f"master={master}",
    ] + ([f"clip={clip} ({probe(clip)['duration']:.2f}s)"] if need_clip else []) \
      + ([f"preview: videos limited to the first {args.max_seconds:g}s"] if args.max_seconds else []) \
      + [f"product={f['path']}" for f in files]
    (out / "REPORT.txt").write_text("\n".join(report) + "\n")
    files.append({"kind": "report", "format": "txt", "path": out / "REPORT.txt"})
    emit_manifest(files)
    print("\n=== RELEASE ===")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
