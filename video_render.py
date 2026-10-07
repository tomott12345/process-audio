"""Visualizer videos for any pipeline: the style list, the --viz-* look
options, and the code that renders a video from a finished master -- shared
by release.py (music), process_field_wav.py (nature), and
process_speech_wav.py (speech) so all three render videos the same way.

Two renderers:
  beat-locked / Python-drawn  radial | bars | glowburst | wormhole  -> visualize_wav.py
  ffmpeg filters              cqt | spectrum | waves | vectorscope | freqs | histogram | spatial
                                                                 -> ffmpeg_visualize.py
The ffmpeg styles render several times faster than real time; the Python
styles run at roughly a quarter of real time, which matters for an
hour-long nature master.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from pipeline_io import PARENT_STEP_ENV, emit

SCRIPT_DIR = Path(__file__).resolve().parent

BEAT_STYLES = ["radial", "bars", "glowburst", "wormhole"]
FFMPEG_STYLES = ["cqt", "spectrum", "waves", "vectorscope", "freqs", "histogram", "spatial"]
ALL_STYLES = BEAT_STYLES + FFMPEG_STYLES

# output id -> (visualizer --format, file suffix, label)
VIDEO_FORMATS = {
    "video_16x9": ("landscape", "youtube_16x9", "YouTube video (16:9)"),
    "video_9x16": ("shorts", "short_9x16", "Short / Reel video (9:16)"),
    "video_1x1": ("square", "feed_1x1", "Instagram feed video (1:1)"),
}

# the --viz-* look options: (argparse dest, flag, kind). Each maps to the
# ffmpeg_visualize.py flag of the same name without the "viz-" prefix.
VIZ_PASSTHROUGH = (
    ("viz_palette", "--viz-palette", str), ("viz_glow", "--viz-glow", bool),
    ("viz_glow_strength", "--viz-glow-strength", float), ("viz_trails", "--viz-trails", bool),
    ("viz_trails_decay", "--viz-trails-decay", float), ("viz_cqt_axis", "--viz-cqt-axis", bool),
    ("viz_no_cqt_sonogram", "--viz-no-cqt-sonogram", bool), ("viz_spectrum_color", "--viz-spectrum-color", str),
    ("viz_spectrum_slide", "--viz-spectrum-slide", str), ("viz_spectrum_speed", "--viz-spectrum-speed", str),
    ("viz_spectrum_scale", "--viz-spectrum-scale", str), ("viz_spectrum_orientation", "--viz-spectrum-orientation", str),
    ("viz_waves_mode", "--viz-waves-mode", str), ("viz_waves_split", "--viz-waves-split", bool),
    ("viz_vectorscope_mode", "--viz-vectorscope-mode", str), ("viz_vectorscope_draw", "--viz-vectorscope-draw", str),
    ("viz_vectorscope_zoom", "--viz-vectorscope-zoom", float), ("viz_freqs_mode", "--viz-freqs-mode", str),
    ("viz_freqs_fscale", "--viz-freqs-fscale", str), ("viz_histogram_slide", "--viz-histogram-slide", str),
    ("viz_histogram_dmode", "--viz-histogram-dmode", str), ("viz_spatial_win", "--viz-spatial-win", str),
)


def viz_args(args) -> list[str]:
    """--viz-* flags -> ffmpeg_visualize.py flags (same names without the
    prefix; --viz-no-cqt-sonogram -> --no-cqt-sonogram)."""
    out: list[str] = []
    for dest, flag, kind in VIZ_PASSTHROUGH:
        val = getattr(args, dest, None)
        target = "--" + flag[len("--viz-"):]
        if kind is bool and val:
            out.append(target)
        elif kind is not bool and val is not None:
            out.append(f"{target}={val:g}" if kind is float else f"{target}={val}")
    return out


def add_viz_group(p: argparse.ArgumentParser) -> None:
    v = p.add_argument_group("ffmpeg visualizer look (ffmpeg styles only)")
    for dest, flag, kind in VIZ_PASSTHROUGH:
        if kind is bool:
            v.add_argument(flag, dest=dest, action="store_true")
        else:
            v.add_argument(flag, dest=dest, type=kind)


def add_video_args(p: argparse.ArgumentParser, default_style: str) -> None:
    """Video outputs and their look, for the nature and speech pipelines
    (release.py has its own --outputs and genre-driven defaults)."""
    g = p.add_argument_group("visualizer videos")
    g.add_argument("--video-16x9", action="store_true", help="Render a 16:9 YouTube video")
    g.add_argument("--video-9x16", action="store_true", help="Render a 9:16 Short / Reel video")
    g.add_argument("--video-1x1", action="store_true", help="Render a 1:1 Instagram feed video")
    g.add_argument("--style", choices=ALL_STYLES, default=default_style,
                   help=f"Visualizer style (default {default_style}); ffmpeg styles "
                        f"({', '.join(FFMPEG_STYLES)}) are much faster than the Python-drawn ones")
    g.add_argument("--emoji", help="Center emoji (radial style)")
    g.add_argument("--symmetry", type=int, help="Radial symmetry (radial style)")
    g.add_argument("--max-seconds", type=float, help="Render only the first N seconds of each video (preview)")
    add_viz_group(p)


def requested_videos(args) -> list[str]:
    return [k for k in VIDEO_FORMATS if getattr(args, k, False)]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "audio"


def video_command(args, style: str, audio: Path, dst: Path, out_id: str, title: str,
                  python_extra: list[str] | None = None, grid: Path | None = None) -> list[str]:
    """The renderer command for one video. Values travel as single
    --flag=value tokens so user text starting with "-" can't read as a flag."""
    fmt = VIDEO_FORMATS[out_id][0]
    preview = [f"--max-seconds={args.max_seconds:g}"] if getattr(args, "max_seconds", None) else []
    if style in FFMPEG_STYLES:
        return [sys.executable, str(SCRIPT_DIR / "ffmpeg_visualize.py"), str(audio), str(dst),
                f"--style={style}", f"--format={fmt}", f"--title={title}", *viz_args(args), *preview]
    cmd = [sys.executable, str(SCRIPT_DIR / "visualize_wav.py"), str(audio), str(dst),
           f"--style={style}", f"--format={fmt}", f"--title={title}", *(python_extra or []), *preview]
    if grid is not None:
        cmd.append(f"--grid={grid}")
    return cmd


def python_style_extra(args) -> list[str]:
    extra = []
    if getattr(args, "emoji", None):
        extra.append(f"--emoji={args.emoji}")
    if getattr(args, "symmetry", None):
        extra.append(f"--symmetry={args.symmetry}")
    return extra


def run_video(step: str, label: str, cmd: list[str]) -> None:
    """Run one render as a step: progress events from the renderer are
    attributed to `step`, and a failure stops the pipeline with the
    renderer's exit code."""
    print(f"\n=== {label} ===\n+ {shlex.join(cmd)}", flush=True)
    emit("step_start", step=step, label=label)
    rc = subprocess.call(cmd, env=dict(os.environ, **{PARENT_STEP_ENV: step}))
    if rc != 0:
        emit("step_error", step=step, message=f"exit {rc}")
        print(f"error: {label} failed (exit {rc}) -- outputs from earlier steps are kept", file=sys.stderr)
        raise SystemExit(1)
    emit("step_end", step=step)
