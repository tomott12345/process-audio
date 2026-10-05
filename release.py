#!/usr/bin/env python3
"""One command from a finished Bluebox mix to upload-ready YouTube and
Instagram files.

  1. master the mix            process_music_wav.py  -> audio/master_24bit_48k.wav
                                                        audio/music_analysis.json
  2. pick the Short/Reel clip  pick_clip.py          -> clips/clip_<N>s.wav (+ grid)
  3. full-length video         visualize_wav.py      -> <slug>_youtube_16x9.mp4
  4. Short / Reel video        visualize_wav.py      -> <slug>_short_9x16.mp4
     (optional square)                               -> <slug>_feed_1x1.mp4
  5. captions                                        -> captions.txt

Every step is one of the standalone scripts, run as-is -- this file only
wires them together, so anything it does can be redone by hand with the
same flags. The visuals are beat-locked to the analyzed grid (--grid) and
use the genre's default style from music_recipes.json (--style overrides).

Video rendering is the slow part: roughly a few seconds of render time per
second of audio at 1920x1080. --max-seconds renders short previews of both
videos first; --no-landscape skips the full-length video entirely.

Usage:
  python3 release.py bluebox_mix.wav --genre techno --title "Night Drive"
  python3 release.py mix.wav --genre trap --title "Low Smoke" --clip-length 15 --loop-clip
  python3 release.py master.wav --genre ambient --title "Peaceful Sunrise" --already-mastered --square
  python3 release.py mix.wav --genre psytrance --title "Fractal" --max-seconds 20   # quick preview
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from music_common import (
    SCRIPT_DIR, die, fmt_time, genre_recipe, load_recipes, parse_timestamp,
    preflight, probe,
)

DEFAULT_METHOD = "Hardware synths, recorded DAWless and mixed live on a 1010music Bluebox."


def step(title: str, cmd: list[str]) -> None:
    print(f"\n=== {title} ===\n+ {shlex.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        die(f"{title} failed (exit {rc}) -- see the output above; outputs from earlier steps are kept")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "track"


def captions(title: str, artist: str | None, genre: str, recipe: dict, a: dict, clip_len: float, method: str) -> str:
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
        f"(clip is {clip_len:.0f}s -- check the platforms' current length limits before posting)",
    ]
    return "\n".join(yt + ["", "-" * 40, ""] + short) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Bluebox stereo mix (or a finished master with --already-mastered)")
    p.add_argument("--genre", help="ambient | trap | techno | psytrance (required -- not guessed)")
    p.add_argument("--title", help="Track title (video overlay, tags, captions; default: file name)")
    p.add_argument("--artist", help="Artist tag/captions")
    p.add_argument("--preset", default="youtube", help="Loudness preset for the master (default: youtube)")
    p.add_argument("--start", type=parse_timestamp, help="Trim the mix: start (seconds or mm:ss)")
    p.add_argument("--end", type=parse_timestamp, help="Trim the mix: end (seconds or mm:ss)")
    p.add_argument("--already-mastered", action="store_true", help="Input is already a master: analyze it, don't re-master")
    p.add_argument("--clip-length", type=float, default=30.0, help="Short/Reel length in seconds (default 30)")
    p.add_argument("--loop-clip", action="store_true", help="Cut the Short/Reel as a seamless bar-exact loop")
    p.add_argument("--drop-at", type=parse_timestamp, help="Override the drop the clip is built around")
    p.add_argument("--clip-start", type=parse_timestamp, help="Pin the clip's start instead")
    p.add_argument("--style", help="Visualizer style override (radial | bars | glowburst | wormhole)")
    p.add_argument("--square", action="store_true", help="Also render a 1:1 square clip for an Instagram feed post")
    p.add_argument("--no-landscape", action="store_true", help="Skip the full-length 16:9 video")
    p.add_argument("--max-seconds", type=float, help="Render only the first N seconds of each video (preview)")
    p.add_argument("--method", default=DEFAULT_METHOD, help="How-it-was-made line for the captions")
    p.add_argument("--out-dir", type=Path, help="Output folder (default: <stem>_release/)")
    p.add_argument("--plan-only", action="store_true", help="Print the steps and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    recipes = load_recipes()
    recipe = genre_recipe(recipes, args.genre)
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        die(f"file not found: {src}")
    title = args.title or src.stem
    slug = slugify(title)
    out = (args.out_dir or src.parent / f"{src.stem}_release").resolve()
    audio, clips = out / "audio", out / "clips"
    py = sys.executable
    style = args.style or recipe["visual_style"]
    vis_extra = [] if args.style else list(recipe["visual_args"])
    clip_tag = f"{int(round(args.clip_length))}s" + ("_loop" if args.loop_clip else "")
    master = src if args.already_mastered else audio / "master_24bit_48k.wav"
    analysis = audio / "music_analysis.json"
    clip = clips / f"clip_{clip_tag}.wav"
    clip_grid = clips / f"clip_{clip_tag}_grid.json"
    preview = ["--max-seconds", f"{args.max_seconds:g}"] if args.max_seconds else []

    steps: list[tuple[str, list[str]]] = []
    if args.already_mastered:
        steps.append(("analyze master", [py, str(SCRIPT_DIR / "music_analyze.py"), str(src), "--genre", args.genre, "--out-dir", str(audio)]))
    else:
        cmd = [py, str(SCRIPT_DIR / "process_music_wav.py"), str(src), "--genre", args.genre,
               "--preset", args.preset, "--title", title, "--out-dir", str(audio)]
        if args.artist:
            cmd += ["--artist", args.artist]
        if args.start is not None:
            cmd += ["--start", f"{args.start}"]
        if args.end is not None:
            cmd += ["--end", f"{args.end}"]
        steps.append(("master", cmd))
    cmd = [py, str(SCRIPT_DIR / "pick_clip.py"), str(master), "--genre", args.genre,
           "--length", f"{args.clip_length:g}", "--analysis", str(analysis), "--out-dir", str(clips)]
    if args.loop_clip:
        cmd.append("--loop")
    if args.drop_at is not None:
        cmd += ["--drop-at", f"{args.drop_at}"]
    if args.clip_start is not None:
        cmd += ["--start", f"{args.clip_start}"]
    steps.append(("pick Short/Reel clip", cmd))
    vis = [py, str(SCRIPT_DIR / "visualize_wav.py")]
    look = ["--style", style, "--title", title] + vis_extra + preview
    if not args.no_landscape:
        steps.append(("YouTube video (16:9, full track)",
                      vis + [str(master), str(out / f"{slug}_youtube_16x9.mp4"), "--format", "landscape", "--grid", str(analysis)] + look))
    steps.append(("Short / Reel video (9:16)",
                  vis + [str(clip), str(out / f"{slug}_short_9x16.mp4"), "--format", "shorts", "--grid", str(clip_grid)] + look))
    if args.square:
        steps.append(("Instagram feed video (1:1)",
                      vis + [str(clip), str(out / f"{slug}_feed_1x1.mp4"), "--format", "square", "--grid", str(clip_grid)] + look))

    print(f"release: {title} ({args.genre}{', recipe untested' if recipe.get('untested') else ''})  ->  {out}")
    if args.plan_only:
        for name, cmd in steps:
            print(f"\n[{name}]\n  {shlex.join(cmd)}")
        print("\n[captions]\n  captions.txt from the master's analysis")
        return 0

    preflight()
    out.mkdir(parents=True, exist_ok=True)
    for name, cmd in steps:
        step(name, cmd)

    a = json.loads(analysis.read_text())
    clip_len = probe(clip)["duration"]
    (out / "captions.txt").write_text(captions(title, args.artist, args.genre, recipe, a, clip_len, args.method))

    products = sorted(p for p in out.iterdir() if p.suffix in (".mp4", ".txt") and p.name != "REPORT.txt")
    report = [
        f"run_at_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"source={src}",
        f"genre={args.genre} title={title!r} style={style} {' '.join(vis_extra)}".rstrip(),
        f"master={master}",
        f"clip={clip} ({clip_len:.2f}s)",
    ] + ([f"preview: videos limited to the first {args.max_seconds:g}s"] if args.max_seconds else []) + [
        f"product={p}" for p in products
    ]
    (out / "REPORT.txt").write_text("\n".join(report) + "\n")
    print("\n=== RELEASE ===")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
