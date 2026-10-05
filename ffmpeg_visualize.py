#!/usr/bin/env python3
"""Render an audio visualization video with ffmpeg's built-in visualization
filters -- no per-frame Python, so it runs many times faster than real time.
This script only builds one ffmpeg command; ffmpeg does all the rendering.

Styles (one per ffmpeg filter):
  cqt          showcqt       constant-Q spectrum: musical bars + a scrolling sonogram
  spectrum     showspectrum  scrolling (or full-frame) spectrogram, 15 colour maps
  waves        showwaves     oscilloscope-style waveform
  vectorscope  avectorscope  stereo field: lissajous / polar goniometer
  freqs        showfreqs     live frequency response: bars, lines, or dots
  histogram    ahistogram    level histogram over time
  spatial      showspatial   stereo placement of each frequency

Shared look options:
  --palette   gold-violet | neon | fire | ice | mono -- colours for cqt, waves,
              freqs, vectorscope, and histogram (spectrum has its own
              --spectrum-color; spatial uses the filter's own colours)
  --glow      soft bloom (screen-blended blur)
  --trails    motion trails (ffmpeg lagfun)
  --title     text along the bottom (drawn as a PNG overlay -- this ffmpeg
              build has no drawtext; needs Pillow, skipped with a note if absent)

Formats: shorts 1080x1920, landscape 1920x1080, square 1080x1080; 30 fps,
H.264 + AAC 320k/48k with +faststart -- the same upload-safe encode as
visualize_wav.py, on Apple's hardware encoder when available (--encoder). --max-seconds renders a preview.

These visuals follow the audio itself; they don't use the beat grid from
music_analyze.py (visualize_wav.py --grid does that).

Usage:
  python3 ffmpeg_visualize.py track.wav out.mp4 --style cqt --format landscape
  python3 ffmpeg_visualize.py track.wav out.mp4 --style spectrum --spectrum-color magma --trails
  python3 ffmpeg_visualize.py clip.wav short.mp4 --style vectorscope --format shorts --glow --title "Night Drive"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from music_common import die, probe
from pipeline_io import PARENT_STEP_ENV, emit, enable_progress

import os

FORMATS = {"shorts": (1080, 1920), "landscape": (1920, 1080), "square": (1080, 1080)}
FPS = 30
STYLES = ("cqt", "spectrum", "waves", "vectorscope", "freqs", "histogram", "spatial")

# palette -> (left colour, right colour) as 0xRRGGBB
PALETTES = {
    "gold-violet": ("f2b64a", "9a6bff"),
    "neon": ("00f0ff", "ff2bd6"),
    "fire": ("ffcf3f", "ff3b1f"),
    "ice": ("bfe9ff", "3d7bff"),
    "mono": ("f2f2f2", "9aa3b5"),
}
SPECTRUM_COLORS = ("channel", "intensity", "rainbow", "moreland", "nebulae", "fire", "fiery", "fruit",
                   "cool", "magma", "green", "viridis", "plasma", "cividis", "terrain")


def rgb(hex6: str) -> tuple[int, int, int]:
    return int(hex6[0:2], 16), int(hex6[2:4], 16), int(hex6[4:6], 16)


def gradient(palette: str) -> str:
    """Map brightness onto the palette: black -> second colour -> first
    colour -> near-white (a heat-map ramp). Applied to filters whose own
    colours mix to grey or white -- showcqt and showfreqs add the left and
    right channel colours, so centred (mono) content, most of a mix, comes
    out white whatever colours you give them. The image is made grey first
    (R = G = B), so a per-channel lookup on brightness is an exact gradient."""
    c1, c2 = (rgb(x) for x in PALETTES[palette])
    stops = [(0, (0, 0, 0)), (89, c2), (191, c1), (255, tuple(min(255, int(v + (255 - v) * 0.7)) for v in c1))]
    exprs = []
    for ch in range(3):
        e = f"{stops[-1][1][ch]}"
        for (x0, v0), (x1, v1) in reversed(list(zip(stops, stops[1:]))):
            seg = f"{v0[ch]}+(val-{x0})*{(v1[ch] - v0[ch]) / (x1 - x0):.5f}"
            e = f"if(lt(val\\,{x1})\\,{seg}\\,{e})"
        exprs.append(e)
    return f"format=gray,format=gbrp,lutrgb=r={exprs[0]}:g={exprs[1]}:b={exprs[2]}"


def vis_filter(a, w: int, h: int) -> str:
    """The audio->video filter for the chosen style, at output size."""
    left, right = PALETTES[a.palette]
    s = f"s={w}x{h}"
    if a.style == "cqt":
        sono = "" if a.cqt_sonogram else ":sono_h=0"
        axis = "" if a.cqt_axis else ":axis_h=0"
        cqt = f"showcqt={s}:fps={FPS}:bar_g=2:sono_g=4:bar_v=9:sono_v=17{sono}{axis}"
        if a.cqt_axis:
            return cqt  # keep the axis's own note colours readable
        return f"{cqt},{gradient(a.palette)}"
    if a.style == "spectrum":
        orient = a.spectrum_orientation
        # a new spectrum column per FFT hop: more overlap = faster scroll.
        # Measured: with no overlap the scroll crosses ~1% of a landscape
        # frame per second (1.7% square); pick the overlap that crosses the
        # whole frame in the chosen time, so a 30s Short isn't half empty
        cross = {"slow": 20, "medium": 10, "fast": 5}[a.spectrum_speed]
        base = {(1920, 1080): 0.010, (1080, 1920): 0.0104, (1080, 1080): 0.017}.get((w, h), 0.01)
        overlap = min(0.95, max(0.0, 1 - base * cross))
        return (f"showspectrum={s}:fps={FPS}:slide={a.spectrum_slide}:mode=combined:color={a.spectrum_color}"
                f":scale={a.spectrum_scale}:fscale=log:orientation={orient}:legend=0:saturation=1.2"
                f":overlap={overlap:.3f},scale={w}:{h}")
    if a.style == "waves":
        return (f"showwaves={s}:rate={FPS}:mode={a.waves_mode}:scale=sqrt:draw=full"
                f":split_channels={1 if a.waves_split else 0}:colors=0x{left}|0x{right}")
    if a.style == "vectorscope":
        (r1, g1, b1) = rgb(left)
        return (f"avectorscope={s}:rate={FPS}:mode={a.vectorscope_mode}:draw={a.vectorscope_draw}"
                f":scale=sqrt:zoom={a.vectorscope_zoom:g}:rc={r1}:gc={g1}:bc={b1}:ac=255:rf=8:gf=8:bf=8:af=12")
    if a.style == "freqs":
        return (f"showfreqs={s}:rate={FPS}:mode={a.freqs_mode}:ascale=log:fscale={a.freqs_fscale}"
                f":win_size=4096:averaging=2:cmode=combined,{gradient(a.palette)}")
    if a.style == "histogram":
        # ahistogram has no colour options
        return (f"ahistogram={s}:rate={FPS}:dmode={a.histogram_dmode}:slide={a.histogram_slide}:scale=log:ascale=log"
                f":rheight=0.12,{gradient(a.palette)}")
    if a.style == "spatial":
        return f"showspatial={s}:rate={FPS}:win_size={a.spatial_win}"
    raise ValueError(a.style)


def post_chain(a) -> list[str]:
    """Video effects after the visualization, as filter-graph fragments."""
    fx = []
    if a.trails:
        fx.append(f"lagfun=decay={a.trails_decay:g}")
    return fx


def title_png(text: str, w: int, h: int, path: Path) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("note: Pillow isn't installed -- skipping the title overlay", file=sys.stderr)
        return False
    size = max(28, int(min(w, h) * 0.035))
    font = None
    for cand in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(cand, size)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    img = Image.new("RGBA", (w, int(size * 2.4)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=font)
    x, y = (w - tw) / 2, size * 0.6
    d.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 180))
    d.text((x, y), text, font=font, fill=(235, 236, 242, 255))
    img.save(path)
    return True


def has_videotoolbox() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True).stdout
    except OSError:
        return False
    return " h264_videotoolbox " in out


def video_codec(a, w: int, h: int) -> list[str]:
    """Apple's hardware H.264 encoder when available (the filters are fast;
    a software x264 encode is most of the render time), else x264. The
    bitrate is generous on purpose: YouTube/Instagram re-encode anyway, and
    spectrograms are detail-heavy."""
    if a.encoder == "videotoolbox" or (a.encoder == "auto" and has_videotoolbox()):
        rate = "20M" if w * h >= 1920 * 1080 else "14M"
        return ["-c:v", "h264_videotoolbox", "-b:v", rate, "-maxrate", rate, "-profile:v", "high",
                "-allow_sw", "1", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", a.preset, "-crf", "18", "-pix_fmt", "yuv420p"]


def build_command(a, w: int, h: int, title: Path | None) -> list[str]:
    # effects run in planar RGB: blending or decaying YUV chroma planes
    # tints black backgrounds (a screen blend lifts U/V off neutral)
    graph = [f"[0:a]aformat=channel_layouts=stereo,{vis_filter(a, w, h)},format=gbrp[v0]"]
    cur = "v0"
    fx = post_chain(a)
    if fx:
        graph.append(f"[{cur}]{','.join(fx)}[v1]")
        cur = "v1"
    if a.glow:
        # blur a quarter-size copy and scale it back up: the same soft bloom
        # for a fraction of the cost of blurring every 1080p frame
        sigma = max(2, int(min(w, h) * 0.004))
        graph.append(f"[{cur}]split[g0][g1];[g1]scale={w // 4}:{h // 4},gblur=sigma={sigma},scale={w}:{h}[gb];"
                     f"[g0][gb]blend=all_mode=screen:all_opacity={a.glow_strength:g}[v2]")
        cur = "v2"
    inputs = ["-i", str(a.input)]
    if title is not None:
        inputs += ["-i", str(title)]
        graph.append(f"[{cur}][1:v]overlay=x=0:y=H-h-{int(h * 0.06)}:format=auto[v3]")
        cur = "v3"
    graph.append(f"[{cur}]format=yuv420p[vout]")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-y"]
    if a.max_seconds:
        cmd += ["-t", f"{a.max_seconds:g}"]
    cmd += inputs + [
        "-filter_complex", ";".join(graph),
        "-map", "[vout]", "-map", "0:a",
        "-r", str(FPS),
        *video_codec(a, w, h),
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-movflags", "+faststart", "-shortest",
        "-progress", "pipe:1",
        str(a.output),
    ]
    return cmd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--style", choices=STYLES, default="cqt")
    p.add_argument("--format", choices=FORMATS.keys(), default="landscape")
    p.add_argument("--title", default=None)
    p.add_argument("--max-seconds", type=float, default=None, help="render only the first N seconds (preview)")
    p.add_argument("--palette", choices=PALETTES.keys(), default="gold-violet")
    p.add_argument("--glow", action="store_true", help="soft bloom around bright parts")
    p.add_argument("--glow-strength", type=float, default=0.6, help="0.1..1 (default 0.6)")
    p.add_argument("--trails", action="store_true", help="motion trails")
    p.add_argument("--trails-decay", type=float, default=0.92, help="0.5..0.99 -- higher = longer trails (default 0.92)")
    p.add_argument("--cqt-axis", action=argparse.BooleanOptionalAction, default=False, help="note-name axis (cqt)")
    p.add_argument("--cqt-sonogram", action=argparse.BooleanOptionalAction, default=True, help="scrolling sonogram under the bars (cqt)")
    p.add_argument("--spectrum-color", choices=SPECTRUM_COLORS, default="plasma")
    p.add_argument("--spectrum-slide", choices=("scroll", "rscroll", "replace", "fullframe"), default="scroll")
    p.add_argument("--spectrum-scale", choices=("log", "sqrt", "cbrt", "lin", "4thrt", "5thrt"), default="log")
    p.add_argument("--spectrum-orientation", choices=("vertical", "horizontal"), default="vertical")
    p.add_argument("--spectrum-speed", choices=("slow", "medium", "fast"), default="medium",
                   help="scroll speed: crosses the frame in ~20 / 10 / 5 seconds")
    p.add_argument("--waves-mode", choices=("cline", "line", "p2p", "point"), default="cline")
    p.add_argument("--waves-split", action="store_true", help="one lane per channel (waves)")
    p.add_argument("--vectorscope-mode", choices=("lissajous", "lissajous_xy", "polar"), default="lissajous")
    p.add_argument("--vectorscope-draw", choices=("aaline", "line", "dot"), default="aaline")
    p.add_argument("--vectorscope-zoom", type=float, default=1.5, help="1..10 (default 1.5)")
    p.add_argument("--freqs-mode", choices=("bar", "line", "dot"), default="bar")
    p.add_argument("--freqs-fscale", choices=("log", "lin", "rlog"), default="log")
    p.add_argument("--histogram-slide", choices=("replace", "scroll"), default="scroll")
    p.add_argument("--histogram-dmode", choices=("single", "separate"), default="single")
    p.add_argument("--spatial-win", type=int, choices=(1024, 2048, 4096), default=4096)
    p.add_argument("--encoder", choices=("auto", "videotoolbox", "x264"), default="auto",
                   help="auto = Apple hardware H.264 if available, else x264")
    p.add_argument("--preset", default="veryfast", help="x264 preset (default veryfast)")
    p.add_argument("--progress-json", action="store_true", help="emit machine-readable progress lines (see pipeline_io.py)")
    p.add_argument("--print-command", action="store_true", help="print the ffmpeg command and exit")
    return p


def main() -> int:
    a = build_parser().parse_args()
    if a.progress_json:
        enable_progress()
    for name, val, lo, hi in (("glow-strength", a.glow_strength, 0.1, 1.0), ("trails-decay", a.trails_decay, 0.5, 0.99),
                              ("vectorscope-zoom", a.vectorscope_zoom, 1.0, 10.0)):
        if not lo <= val <= hi:
            die(f"--{name} {val:g} is out of range ({lo:g}..{hi:g})", 2)
    if shutil.which("ffmpeg") is None:
        die("ffmpeg not found on PATH")
    a.input = Path(a.input).expanduser().resolve()
    a.output = Path(a.output).expanduser().resolve()
    if not a.input.exists():
        die(f"file not found: {a.input}")
    w, h = FORMATS[a.format]
    dur = probe(a.input)["duration"]
    if a.max_seconds:
        dur = min(dur, a.max_seconds)

    with tempfile.TemporaryDirectory() as td:
        title = None
        if a.title:
            tp = Path(td) / "title.png"
            if title_png(a.title, w, h, tp):
                title = tp
        cmd = build_command(a, w, h, title)
        if a.print_command:
            print(" ".join(cmd))
            return 0
        print(f"Rendering {a.style} at {w}x{h}, {dur:.1f}s ...")
        a.output.parent.mkdir(parents=True, exist_ok=True)
        total = max(1, int(dur * FPS))
        step = os.environ.get(PARENT_STEP_ENV, "video")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        last = -1
        for line in proc.stdout:
            key, _, val = line.strip().partition("=")
            if key == "out_time_us" and val.isdigit():
                done = min(total, int(int(val) / 1e6 * FPS))
                if done - last >= FPS or done == total:
                    emit("progress", step=step, done=done, total=total, detail="frames")
                    last = done
        err = proc.stderr.read()
        if proc.wait() != 0:
            print(err[-3000:], file=sys.stderr)
            die(f"ffmpeg failed (exit {proc.returncode})")
        emit("progress", step=step, done=total, total=total, detail="frames")
    print(f"Done: {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
