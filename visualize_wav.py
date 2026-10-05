#!/usr/bin/env python3
"""
visualize_wav.py — render an audio-reactive visualization video from a WAV file.

Optional feature (see requirements-optional.txt): needs librosa + Pillow on
top of this repo's core numpy/scipy requirements, plus the ffmpeg binary on
PATH. Nothing else in the repo depends on this script.

Pipeline:
  1. Load audio with librosa, compute a mel spectrogram (drives radial bars
     or bar-equalizer heights), an RMS envelope (drives a pulsing center
     circle / overall brightness), and an onset envelope (drives particle
     spawning and the beat-punch zoom).
  2. Render each frame's new light (bars/circle, glowburst rays/core/rings,
     wormhole tunnel rings,
     and for --symmetry>1 one wedge rotated into N copies) onto a black Pillow layer at a
     supersampled resolution, then downsample that layer once to the
     target resolution with the BOX filter (exact area averaging -- clean,
     fast anti-aliasing for an integer supersample factor, unlike Pillow's
     native jagged line/ellipse drawing at 1x).
  3. Spawn/update/draw particles at bar tips (output resolution).
  4. Composite that anti-aliased layer into a persistent float32
     accumulation buffer at output resolution: the previous buffer is
     faded (motion trails) and the new layer plus a box-blurred, additive
     copy of it (bloom/glow) are added on top. Trails/bloom run at output
     resolution rather than supersampled resolution on purpose -- blur
     cost scales with pixel count, so this is the difference between a
     blur that costs tens of milliseconds a frame and one that costs
     hundreds.
  5. On a detected onset, briefly zoom the composited frame (beat punch).
  6. Draw the title crisply on top (it does not get blurred, trailed, or
     zoomed), and stream the raw RGB bytes into an ffmpeg subprocess that
     muxes them with the original audio into a single .mp4 (no
     intermediate frame files).

Usage:
  python3 visualize_wav.py input.wav output.mp4 --format shorts --style radial
  python3 visualize_wav.py input.wav output.mp4 --format landscape --style bars
  python3 visualize_wav.py input.wav output.mp4 --format landscape --style glowburst
  python3 visualize_wav.py input.wav output.mp4 --format landscape --style wormhole
  python3 visualize_wav.py input.wav output.mp4 --format shorts --title "Contemplation"
  python3 visualize_wav.py input.wav output.mp4 --format shorts --symmetry 6
  python3 visualize_wav.py input.wav output.mp4 --format shorts --no-particles --no-beat-punch --symmetry 1
  python3 visualize_wav.py master.wav output.mp4 --format landscape --grid music_analysis.json
    (--grid: lock beat punch / emoji pulse to a music_analyze.py beat grid
    and flash on drops; see release.py for the one-command music flow)

Styles:
  radial     spectrum bars pulsing outward from a center ring (default)
  bars       classic bar-graph equalizer
  glowburst  sunburst -- a soft glowing core that breathes with loudness,
             mirrored tapered light rays sized by band energy, and
             expanding rings on onsets; suits slow/beatless ambient
             material
  wormhole   a tunnel of spectrum-shaped rings flying toward the viewer,
             curving and twisting with depth; travel speed follows
             loudness and surges on onsets
  (--symmetry and --emoji are radial-only)

Formats:
  shorts     1080x1920 (9:16, YouTube Shorts / Reels / TikTok)
  landscape  1920x1080 (16:9, standard YouTube)
  square     1080x1080 (1:1)

Look/style knobs (all optional, sensible defaults on):
  --supersample N      render at Nx resolution internally, then downsample
                        with the BOX filter for anti-aliased edges (default
                        2; use 1 to disable and render fastest)
  --trail-decay F       0..1, how much of the previous frame survives into
                        the next (default 0.85; 1.0 = trails never fade,
                        0 = no trails at all -- equivalent to --no-trails)
  --glow-strength F     intensity of the additive bloom around bright
                        elements (default 0.55; 0 = equivalent to --no-glow)
  --glow-radius PX      blur radius for the bloom, in output pixels
                        (default: scales with frame size)
  --no-trails           disable motion trails (same as --trail-decay 0)
  --no-glow             disable bloom (same as --glow-strength 0)
  --particles           spawn small drifting sparks at active bar tips
                        (default on)
  --no-particles        disable particles
  --particle-rate F      spawn probability scale per band per frame,
                        proportional to that band's amplitude (default 0.5)
  --max-particles N      cap on simultaneous particles (default 260)
  --beat-punch           brief zoom/flash on detected onsets (default on)
  --no-beat-punch        disable beat punch
  --punch-strength F      max zoom fraction on a strong onset (default 0.045)
  --symmetry N           radial style only: draw one 1/N wedge and rotate
                        it into N copies for a kaleidoscope/mandala look
                        (default 1 = off; try 6 or 8)

Requires: librosa, numpy, pillow, and the ffmpeg binary on PATH.
"""

import argparse
import colorsys
import os
import math
import random
import subprocess
import sys
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from pipeline_io import PARENT_STEP_ENV, emit, enable_progress

FORMATS = {
    "shorts": (1080, 1920),
    "landscape": (1920, 1080),
    "square": (1080, 1080),
}


def build_pulse_envelope(onset_frames, n_frames, decay=0.72):
    """1.0 on a detected onset frame, decaying exponentially until the next
    one -- turns a sparse list of hit times into a continuous per-frame
    "how hard are we mid-beat right now" signal, suitable for driving a
    zoom, a flash, or a heartbeat-style pulse."""
    env = np.zeros(n_frames, dtype=np.float32)
    for f in onset_frames:
        if f < n_frames:
            env[f] = 1.0
    for i in range(1, n_frames):
        env[i] = max(env[i], env[i - 1] * decay)
    return env


def grid_envelopes(grid_path, n_frames, fps):
    """Beat-locked replacements for the onset-detected envelopes, from a
    music_analysis.json / clip_*_grid.json (music_analyze.py, pick_clip.py).
    Onset detection jitters on busy trap hats and psy 16th-note basslines;
    the analyzed grid doesn't. Returns (punch_env, kick_env, flash_env) or
    None when the file has no beat grid (ambient) -- the caller then keeps
    onset detection.

    punch: a pulse on every beat (downbeats 1.0, other beats 0.6), scaled by
    section -- full in drops, calmer in intros/breakdowns. kick: the same
    beats with a slower decay (drives the emoji). flash: a white flash on
    each drop's first downbeat."""
    import json

    with open(grid_path) as f:
        a = json.load(f)
    if not a.get("tempo", {}).get("has_grid"):
        return None
    g = a["grid"]
    downs = set(round(t, 3) for t in g.get("downbeats", []))
    scale = {"drop": 1.0, "build": 0.85, "groove": 0.75, "intro": 0.6, "outro": 0.6, "breakdown": 0.35}

    def section_scale(t):
        for sec in a.get("sections", []):
            if sec["start"] <= t < sec["end"]:
                return scale.get(sec["label"], 0.7)
        return 0.7

    def pulses(times, amps, decay):
        env = np.zeros(n_frames, dtype=np.float32)
        for t, amp in zip(times, amps):
            fi = int(round(t * fps))
            if 0 <= fi < n_frames:
                env[fi] = max(env[fi], amp)
        for i in range(1, n_frames):
            env[i] = max(env[i], env[i - 1] * decay)
        return env

    beats = g.get("beats", [])
    amps = [(1.0 if round(t, 3) in downs else 0.6) * section_scale(t) for t in beats]
    punch = pulses(beats, amps, 0.72)
    kick = pulses(beats, [min(1.0, x / 0.85) for x in amps], 0.8)
    flash = pulses(a.get("drops", []), [1.0] * len(a.get("drops", [])), 0.85)
    # the drop hit itself gets a bigger zoom than an ordinary downbeat
    punch = np.maximum(punch, 1.6 * flash)
    return punch, kick, flash


def load_features(wav_path, fps, n_bands=48, sr_target=22050, kick_fmax=200.0):
    """Load audio and compute per-frame band energies, RMS envelope, a
    full-band onset "punch" envelope (drives beat-punch zoom/particles),
    and a bass-restricted "kick" envelope (drives the center emoji pulse)
    -- all aligned to the same per-video-frame grid since they share
    hop_length. Restricting the kick envelope's onset detection to
    low frequencies (below kick_fmax Hz, the kick drum's fundamental +
    near harmonics) is what keeps it responding mainly to kick/bass hits
    rather than every snare or hi-hat too."""
    import librosa

    y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    duration = len(y) / sr
    hop_length = int(sr / fps)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_bands, hop_length=hop_length, fmax=sr / 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db - mel_db.min()
    if mel_db.max() > 0:
        mel_db = mel_db / mel_db.max()

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if rms.max() > 0:
        rms = rms / rms.max()

    n_frames = mel_db.shape[1]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames"
    )
    punch_env = build_pulse_envelope(onset_frames, n_frames, decay=0.72)

    # fewer, wider mel bins for this one -- the default 128 mel bins packed
    # into a narrow 0-kick_fmax Hz range leaves many of them empty (librosa
    # warns about it) and buys nothing extra for a single low-frequency
    # onset envelope
    bass_onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, fmax=kick_fmax, n_mels=24
    )
    kick_frames = librosa.onset.onset_detect(
        onset_envelope=bass_onset_env, sr=sr, hop_length=hop_length, units="frames"
    )
    kick_env = build_pulse_envelope(kick_frames, n_frames, decay=0.8)

    return mel_db, rms, punch_env, kick_env, n_frames, duration, y, sr


def smooth(prev, target, attack=0.55, release=0.15):
    """Exponential smoothing with faster attack than release (punchy but not jittery)."""
    out = np.empty_like(target)
    for i in range(len(target)):
        a = attack if target[i] > prev[i] else release
        out[i] = prev[i] + (target[i] - prev[i]) * a
    return out


def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def load_font(size):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


EMOJI_FONT_CANDIDATES = (
    # Linux (this repo's own containers, most Linux desktops with a
    # noto-color-emoji package installed)
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    # macOS
    "/System/Library/Fonts/Apple Color Emoji.ttc",
)

_HEART_CHARS = {"❤️", "❤", "♥️", "♥", "heart", "red heart"}


def render_emoji_glyph(emoji, px=256):
    """Render a single emoji/character to a tightly-cropped RGBA image using
    whatever color-capable system emoji font is available, then resize to
    px -- rendered once, resized per-frame for the beat pulse from that one
    render (cheap: resizing a small bitmap, not re-rendering text every
    frame). Returns None if no usable font is found or the glyph comes back
    blank (e.g. a tofu box), so the caller can fall back.

    Color emoji fonts are typically fixed-strike bitmap fonts (CBDT/CBLC)
    that only support specific embedded pixel sizes -- e.g. Noto Color
    Emoji ships exactly one strike, 109px -- so ImageFont.truetype(path, N)
    raises OSError("invalid pixel size") for any other N. We try a handful
    of known strike sizes rather than assume one."""
    for path in EMOJI_FONT_CANDIDATES:
        for try_px in (109, 128, 136, 160, 96, px):
            try:
                font = ImageFont.truetype(path, try_px)
            except Exception:
                continue
            try:
                canvas_px = try_px * 2
                canvas = Image.new("RGBA", (canvas_px, canvas_px), (0, 0, 0, 0))
                draw = ImageDraw.Draw(canvas)
                draw.text((try_px // 2, try_px // 2), emoji, font=font, embedded_color=True)
                bbox = canvas.getbbox()
                if not bbox:
                    continue
                glyph = canvas.crop(bbox)
                # pad to square before the final resize so a non-square
                # glyph bbox (e.g. a flame taller than it is wide) doesn't
                # get stretched when main() later treats it as size_px x
                # size_px
                side = max(glyph.size)
                square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
                square.paste(glyph, ((side - glyph.width) // 2, (side - glyph.height) // 2), glyph)
                return square.resize((px, px), Image.LANCZOS)
            except Exception:
                continue
    return None


def render_vector_heart(px=256, fill=(230, 30, 60, 255), outline=(255, 120, 140, 255)):
    """Hand-drawn heart as a fallback for when no color emoji font is
    available (or for a cleaner, more 'neon' look than a flat bitmap emoji
    anyway) -- two overlapping circles plus a triangle, the classic
    construction, anti-aliased by drawing at 4x and downsampling."""
    ss = 4
    size = px * ss
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    cx, r = size / 2, size * 0.26
    lobe_dx = r * 0.95
    draw.ellipse([cx - lobe_dx - r, size * 0.20, cx - lobe_dx + r, size * 0.20 + 2 * r], fill=fill)
    draw.ellipse([cx + lobe_dx - r, size * 0.20, cx + lobe_dx + r, size * 0.20 + 2 * r], fill=fill)
    draw.polygon(
        [
            (cx - lobe_dx - r, size * 0.20 + r * 0.9),
            (cx + lobe_dx + r, size * 0.20 + r * 0.9),
            (cx, size * 0.92),
        ],
        fill=fill,
    )
    canvas = canvas.resize((px, px), Image.LANCZOS)
    bbox = canvas.getbbox()
    return canvas.crop(bbox) if bbox else canvas


def get_emoji_glyph(emoji):
    """Try the system emoji font first; fall back to the hand-drawn heart
    for any heart-like request so that's always reliable regardless of
    what fonts happen to be installed. Returns None for a non-heart emoji
    with no usable font (caller should warn and skip)."""
    glyph = render_emoji_glyph(emoji)
    if glyph is not None:
        return glyph
    if emoji.strip().lower() in _HEART_CHARS:
        return render_vector_heart()
    return None


def radial_bar_geometry(w, h, bands, rms_val, angle_offset=0.0, angle_span=2 * math.pi):
    """Shared geometry for the radial style: yields (x1,y1,x2,y2,color,width,tip_x,tip_y,amp)
    for each band, and returns base_r too (needed for the center ring)."""
    cx, cy = w // 2, h // 2
    n = len(bands)
    base_r = min(w, h) * 0.16 * (1 + 0.08 * rms_val)
    max_extra = min(w, h) * 0.30
    bars = []
    for i, amp in enumerate(bands):
        frac = i / max(1, n - 1) if n > 1 else 0.0
        angle = angle_offset + frac * angle_span - math.pi / 2
        r1 = base_r
        r2 = base_r + max_extra * (0.08 + amp)
        x1, y1 = cx + r1 * math.cos(angle), cy + r1 * math.sin(angle)
        x2, y2 = cx + r2 * math.cos(angle), cy + r2 * math.sin(angle)
        hue = frac * 0.6 + 0.55
        color = hsv_to_rgb(hue % 1.0, 0.65, 0.95)
        width = max(2, int(min(w, h) * 0.006))
        bars.append((x1, y1, x2, y2, color, width, x2, y2, amp))
    return bars, cx, cy, base_r


def rotate_point(x, y, cx, cy, angle_deg):
    a = math.radians(angle_deg)
    dx, dy = x - cx, y - cy
    return (cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a))


def base_ring_radius(w, h, rms_val):
    return min(w, h) * 0.16 * (1 + 0.08 * rms_val)


def draw_radial_layer(hi_w, hi_h, out_w, out_h, bands, rms_val, symmetry=1):
    """New light for this frame only, on a pure black background -- no
    title, no persistent state, no particles. Returns an image already at
    OUTPUT resolution (hi_w/hi_h are used only for anti-aliasing the initial
    draw) plus spawn_points in output-resolution coordinates.

    When symmetry<=1: draw once at hi-res, downsample once with BOX. Cheap.

    When symmetry>1 (kaleidoscope/mandala): draw one 1/symmetry wedge at
    hi-res, downsample THAT wedge once with BOX, then do the `symmetry`
    rotations at output resolution instead of hi-res -- rotating a full
    hi-res frame N times is the single most expensive thing this script can
    do (each rotate scales with pixel count), so doing it post-downsample
    is what keeps --symmetry usably fast."""
    ss_x, ss_y = hi_w / out_w, hi_h / out_h

    if symmetry <= 1:
        img = Image.new("RGB", (hi_w, hi_h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        bars, cx, cy, base_r = radial_bar_geometry(hi_w, hi_h, bands, rms_val)
        for x1, y1, x2, y2, color, width, tip_x, tip_y, amp in bars:
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        ring_r = base_r * 0.9
        draw.ellipse(
            [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
            outline=(230, 230, 255),
            width=max(2, int(min(hi_w, hi_h) * 0.003)),
        )
        out_img = img.resize((out_w, out_h), Image.BOX) if (hi_w, hi_h) != (out_w, out_h) else img
        spawn_points = [(x2 / ss_x, y2 / ss_y, color, amp) for x1, y1, x2, y2, color, width, _, _, amp in bars]
        return out_img, spawn_points

    n = len(bands)
    wedge_n = max(2, n // symmetry)
    # sample the wedge's bands spread across the whole spectrum (stride)
    # rather than just the low end, so each wedge reflects bass through
    # treble, not only bass
    stride = max(1, n // wedge_n)
    wedge_bands = bands[::stride][:wedge_n]
    span = 2 * math.pi / symmetry

    wedge_img_hi = Image.new("RGB", (hi_w, hi_h), (0, 0, 0))
    wedge_draw = ImageDraw.Draw(wedge_img_hi)
    bars_hi, cx_hi, cy_hi, base_r_hi = radial_bar_geometry(
        hi_w, hi_h, wedge_bands, rms_val, angle_offset=0.0, angle_span=span
    )
    for x1, y1, x2, y2, color, width, *_ in bars_hi:
        wedge_draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    wedge_out = wedge_img_hi.resize((out_w, out_h), Image.BOX) if (hi_w, hi_h) != (out_w, out_h) else wedge_img_hi

    # tip coordinates computed directly at output resolution -- cheap
    # (pure arithmetic, no drawing) and avoids scaling rounding
    bars_out, cx, cy, base_r = radial_bar_geometry(
        out_w, out_h, wedge_bands, rms_val, angle_offset=0.0, angle_span=span
    )

    acc = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    step_deg = 360.0 / symmetry
    spawn_points = []
    for k in range(symmetry):
        angle = step_deg * k
        # NEAREST here, not BILINEAR: this gets box-blurred (bloom) and
        # trail-blended immediately after, which hides NEAREST's jaggies
        # completely while cutting rotation cost several times over --
        # rotation is the single most expensive part of --symmetry mode.
        rotated = wedge_out.rotate(angle, resample=Image.NEAREST, center=(cx, cy))
        acc = np.maximum(acc, np.asarray(rotated, dtype=np.uint8))
        for x1, y1, x2, y2, color, width, tip_x, tip_y, amp in bars_out:
            rx, ry = rotate_point(tip_x, tip_y, cx, cy, angle)
            spawn_points.append((rx, ry, color, amp))

    img = Image.fromarray(acc, mode="RGB")
    draw = ImageDraw.Draw(img)
    ring_r = base_r * 0.9
    draw.ellipse(
        [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
        outline=(230, 230, 255),
        width=max(2, int(min(out_w, out_h) * 0.003)),
    )
    return img, spawn_points


def draw_bars_layer(hi_w, hi_h, out_w, out_h, bands, rms_val, symmetry=1):
    """symmetry is accepted (and ignored) for interface parity with
    draw_radial_layer -- --symmetry only applies to --style radial."""
    w, h = hi_w, hi_h
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    n = len(bands)
    margin = w * 0.04
    gap = 4
    bar_w = (w - 2 * margin - gap * (n - 1)) / n
    baseline = h * 0.72
    max_h = h * 0.5

    spawn_points_hi = []
    for i, amp in enumerate(bands):
        bh = max(0.0, max_h * float(amp))
        x0 = margin + i * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = baseline - bh
        y1 = baseline
        hue = 0.55 + 0.35 * (i / n)
        color = hsv_to_rgb(hue, 0.7, 0.9 + 0.1 * min(1.0, amp))
        draw.rectangle([x0, y0, x1, y1], fill=color)
        refl_h = bh * 0.35
        if refl_h > 0.5:
            draw.rectangle([x0, y1, x1, y1 + refl_h], fill=tuple(c // 4 for c in color))
        spawn_points_hi.append(((x0 + x1) / 2, y0, color, amp))

    out_img = img.resize((out_w, out_h), Image.BOX) if (hi_w, hi_h) != (out_w, out_h) else img
    ss_x, ss_y = hi_w / out_w, hi_h / out_h
    spawn_points = [(x / ss_x, y / ss_y, c, a) for x, y, c, a in spawn_points_hi]
    return out_img, spawn_points


def glowburst_ray_color(frac, amp):
    """Warm sunrise ramp by frequency: amber/gold lows through rose to a
    soft violet at the top, brighter and slightly less saturated as a band
    gets louder so peaks read as hot-white-ish rather than just longer."""
    hue = (0.12 - 0.30 * frac) % 1.0
    return hsv_to_rgb(hue, 0.75 - 0.25 * min(1.0, amp), 0.55 + 0.45 * min(1.0, amp))


class BurstRings:
    """Expanding shockwave rings launched from the glow core on detected
    onsets -- the "burst" in glowburst. Each ring is [radius_frac, alpha];
    radius is a fraction of min(w,h) so the same state draws correctly at
    supersampled and output resolution."""

    def __init__(self, fps, max_rings=8, min_gap_s=1.5):
        self.fps = fps
        self.max_rings = max_rings
        # onset detection fires ~2x/sec even on beatless pads; a minimum gap
        # keeps the rings reading as occasional swells, not a ripple tank
        self.min_gap = int(min_gap_s * fps)
        self.since_last = self.min_gap
        self.rings = []

    def update(self, onset, core_frac):
        # onset is the punch envelope, which is exactly 1.0 on a detected
        # onset frame and decays after -- launch only on the onset itself
        self.since_last += 1
        if onset >= 0.999 and self.since_last >= self.min_gap and len(self.rings) < self.max_rings:
            self.rings.append([core_frac, 1.0])
            self.since_last = 0
        alive = []
        for r, a in self.rings:
            r += 0.35 / self.fps
            a *= 0.94
            if a > 0.03 and r < 0.75:
                alive.append([r, a])
        self.rings = alive

    def draw(self, draw, cx, cy, scale):
        for r, a in self.rings:
            rad = r * scale
            color = hsv_to_rgb(0.09, 0.45, a)
            draw.ellipse(
                [cx - rad, cy - rad, cx + rad, cy + rad],
                outline=color,
                width=max(2, int(scale * 0.004 * (0.5 + a))),
            )


_glow_dist_cache = {}


def glow_core(out_w, out_h, radius_frac, intensity):
    """Soft Gaussian sun at the center, as a float32 RGB array at output
    resolution. The normalized distance map is computed once per frame size
    and cached -- per frame it's just an exp and a multiply."""
    key = (out_w, out_h)
    d2 = _glow_dist_cache.get(key)
    if d2 is None:
        yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        s = float(min(out_w, out_h))
        d2 = ((xx - out_w / 2) ** 2 + (yy - out_h / 2) ** 2) / (s * s)
        _glow_dist_cache[key] = d2
    falloff = np.exp(-d2 / (radius_frac * radius_frac)) * intensity
    warm = np.array([255.0, 214.0, 150.0], dtype=np.float32)
    return falloff[..., None] * warm


def draw_glowburst_layer(hi_w, hi_h, out_w, out_h, bands, rms_val, t=0.0, bursts=None):
    """Sunburst: a soft glowing core whose size/brightness follows RMS,
    tapered light rays radiating from it (length = band amplitude, mirrored
    left/right so the burst is symmetric, slowly rotating), and expanding
    rings on onsets (see BurstRings). Designed for slow, beatless material
    -- ambient pads read as a breathing sun rather than a twitchy meter."""
    img = Image.new("RGB", (hi_w, hi_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = hi_w / 2, hi_h / 2
    s = min(hi_w, hi_h)
    n = len(bands)
    core_frac = 0.05 * (1 + 0.4 * rms_val)
    r0 = s * core_frac * 0.8
    max_len = s * 0.30  # keeps peak rays clear of the title line
    rot = t * 0.04  # radians/sec -- barely-there drift
    half_w = math.pi / n * 0.55  # angular half-width of each ray at its base

    tips = []
    for i, amp in enumerate(bands):
        frac = i / max(1, n - 1)
        color = glowburst_ray_color(frac, amp)
        length = max_len * (0.05 + float(amp) ** 1.4)
        for side in (1, -1):
            a = rot - math.pi / 2 + side * (frac * math.pi + half_w)
            r1 = r0 + length
            base_l = (cx + r0 * math.cos(a - half_w), cy + r0 * math.sin(a - half_w))
            base_r = (cx + r0 * math.cos(a + half_w), cy + r0 * math.sin(a + half_w))
            tip = (cx + r1 * math.cos(a), cy + r1 * math.sin(a))
            draw.polygon([base_l, tip, base_r], fill=color)
            tips.append((tip[0], tip[1], color, float(amp)))

    if bursts is not None:
        bursts.draw(draw, cx, cy, s)

    out_img = img.resize((out_w, out_h), Image.BOX) if (hi_w, hi_h) != (out_w, out_h) else img
    core = glow_core(out_w, out_h, core_frac, 0.3 + 0.35 * rms_val)
    layer = np.minimum(np.asarray(out_img, dtype=np.float32) + core, 255.0)
    out_img = Image.fromarray(layer.astype(np.uint8), mode="RGB")

    ss_x, ss_y = hi_w / out_w, hi_h / out_h
    spawn_points = [(x / ss_x, y / ss_y, c, a) for x, y, c, a in tips]
    return out_img, spawn_points


class WormholeTunnel:
    """Travel state for the wormhole style: how far we've flown down the
    tunnel (phase, in ring-spacing units) and the clock that bends it.
    Speed follows loudness plus a kick on onsets, so swells feel like
    acceleration rather than just brighter rings."""

    def __init__(self, fps, n_rings=20):
        self.fps = fps
        self.n_rings = n_rings
        self.phase = 0.0

    def advance(self, rms_val, punch):
        speed = 0.5 + 2.2 * rms_val + 2.5 * punch  # rings/sec
        self.phase += speed / self.fps


def draw_wormhole_layer(hi_w, hi_h, out_w, out_h, bands, rms_val, t=0.0, tunnel=None):
    """Wormhole: concentric rings receding to a vanishing point, flying
    toward the viewer. Each ring is a closed polygon whose radius is pushed
    out by the spectrum (bands mirrored around the circumference, twisted
    with depth so bumps spiral down the tunnel); ring centers drift with
    depth so the tunnel curves. Rings keep a stable hue as they approach
    (keyed to ring identity, not screen slot), and fade out both far away
    and right before they pass the camera."""
    img = Image.new("RGB", (hi_w, hi_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx0, cy0 = hi_w / 2, hi_h / 2
    s = min(hi_w, hi_h)
    n = len(bands)
    n_rings = tunnel.n_rings if tunnel else 20
    phase = tunnel.phase if tunnel else t
    frac_phase = phase % 1.0
    base_id = int(phase)
    r_scale = s * 0.16
    n_pts = 96
    bend = s * 0.05

    # mirrored spectrum around the ring: angle 0 (top) = lows, bottom = highs
    ring_amp = np.empty(n_pts)
    for j in range(n_pts):
        f = abs(((j / n_pts) * 2.0) - 1.0)  # 1 at top, 0 at bottom, 1 at top again
        ring_amp[j] = bands[min(n - 1, int((1.0 - f) * (n - 1)))]
    # circular smoothing -- raw per-band steps make the rings look jagged
    kern = np.hanning(9)
    kern /= kern.sum()
    ring_amp = np.convolve(np.concatenate([ring_amp[-4:], ring_amp, ring_amp[:4]]), kern, mode="valid")

    rings = []
    for k in range(n_rings):
        z = k + 1.0 - frac_phase  # depth; shrinks as we fly forward
        if z < 0.35:
            continue
        r = r_scale / z
        if r < 2:
            continue
        ring_id = base_id + n_rings - k  # stable identity as the ring approaches
        depth_frac = z / n_rings
        # far rings bend away from center more -> curved tunnel; sqrt so
        # the bend is spread along the tunnel instead of whipping the tail
        ox = bend * math.sqrt(depth_frac) * math.sin(0.23 * t + 0.12 * z)
        oy = bend * math.sqrt(depth_frac) * math.cos(0.17 * t + 0.10 * z)
        cx, cy = cx0 + ox, cy0 + oy
        twist = 0.18 * z + 0.15 * t
        bump = 0.22 / (1.0 + 0.15 * z)
        bright = max(0.0, 1.0 - depth_frac) * min(1.0, (z - 0.35) / 0.8)
        if bright <= 0.02:
            continue
        hue = (0.55 + 0.28 * ((ring_id * 0.071) % 1.0)) % 1.0
        color = hsv_to_rgb(hue, 0.7, bright * (0.55 + 0.45 * rms_val))
        width = max(1, min(int(s * 0.007), int(s * 0.0022 / z) + 1))
        pts = []
        for j in range(n_pts):
            th = twist + 2 * math.pi * j / n_pts - math.pi / 2
            rr = r * (1.0 + bump * float(ring_amp[j]))
            pts.append((cx + rr * math.cos(th), cy + rr * math.sin(th)))
        rings.append((z, pts, color, width, cx, cy, bright))

    # far to near, so near rings draw over the far ones
    rings.sort(key=lambda rg: -rg[0])
    prev_pts = None
    for z, pts, color, width, cx, cy, _ in rings:
        draw.line(pts + [pts[0]], fill=color, width=width, joint="curve")
        # tunnel "struts": short lines linking matching points on adjacent rings
        if prev_pts is not None:
            dim = tuple(c // 3 for c in color)
            for j in range(0, n_pts, 12):
                draw.line([prev_pts[j], pts[j]], fill=dim, width=max(1, width // 2))
        prev_pts = pts

    # the light at the end of the tunnel -- anchored to the farthest ring
    # that's actually visible, so it sits where the eye reads the tunnel end
    visible = [rg for rg in rings if rg[6] > 0.2]
    if visible:
        fx, fy = visible[0][4], visible[0][5]
        glow_r = s * (0.006 + 0.012 * rms_val)
        draw.ellipse([fx - glow_r, fy - glow_r, fx + glow_r, fy + glow_r], fill=(235, 225, 255))

    out_img = img.resize((out_w, out_h), Image.BOX) if (hi_w, hi_h) != (out_w, out_h) else img
    ss_x, ss_y = hi_w / out_w, hi_h / out_h
    spawn_points = []
    if rings:
        # sparks peel off the nearest ring at its loudest points
        _, pts, color, _, _, _, _ = rings[-1]
        for j in range(0, n_pts, 8):
            spawn_points.append((pts[j][0] / ss_x, pts[j][1] / ss_y, color, float(ring_amp[j])))
    return out_img, spawn_points


class ParticleSystem:
    """Small drifting sparks spawned at active bar tips. Cheap Python-object
    list rather than numpy arrays -- particle counts here (a few hundred at
    most) are far below where vectorization would pay for its own overhead,
    and it keeps spawn/kill logic simple."""

    def __init__(self, max_particles, spawn_rate, fps):
        self.max_particles = max_particles
        self.spawn_rate = spawn_rate
        self.fps = fps
        self.particles = []  # each: [x, y, vx, vy, age, life, color]

    def spawn(self, spawn_points, threshold=0.35):
        if len(self.particles) >= self.max_particles:
            return
        for x, y, color, amp in spawn_points:
            if amp < threshold:
                continue
            if random.random() > amp * self.spawn_rate:
                continue
            if len(self.particles) >= self.max_particles:
                break
            angle = random.uniform(0, 2 * math.pi)
            speed = (20 + 60 * amp) / self.fps * 30  # px/sec-ish, scaled by fps below
            vx = math.cos(angle) * speed / self.fps
            vy = math.sin(angle) * speed / self.fps - (10 / self.fps)  # slight upward bias
            life = random.uniform(0.4, 0.9) * self.fps  # frames
            self.particles.append([x, y, vx, vy, 0.0, life, color])

    def update_and_draw(self, draw):
        alive = []
        for p in self.particles:
            x, y, vx, vy, age, life, color = p
            age += 1
            if age >= life:
                continue
            x += vx
            y += vy
            vy += 0.02  # gentle drag toward outward drift settling
            life_frac = 1.0 - (age / life)
            r = max(1, 3.5 * life_frac)
            faded = tuple(int(c * (0.4 + 0.6 * life_frac)) for c in color)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=faded)
            alive.append([x, y, vx, vy, age, life, color])
        self.particles = alive


class FrameCompositor:
    """Owns the persistent trail buffer and applies trails + bloom.

    Supersampling is used only to anti-alias each frame's freshly drawn
    lines/circles (drawn hi-res, downsampled once to output resolution
    before reaching here). Trails and bloom then run entirely at output
    resolution -- blur cost scales with pixel count, so doing it post-
    downsample instead of at 2x-4x the pixels is the difference between
    a blur that costs tens of milliseconds and one that costs hundreds.
    """

    def __init__(self, out_w, out_h, trail_decay, glow_strength, glow_radius):
        self.out_w, self.out_h = out_w, out_h
        self.trail_decay = trail_decay
        self.glow_strength = glow_strength
        self.glow_radius = glow_radius
        self.buffer = np.zeros((out_h, out_w, 3), dtype=np.float32)

    def composite(self, new_layer_img):
        new_layer = np.asarray(new_layer_img, dtype=np.float32)

        if self.trail_decay > 0:
            self.buffer *= self.trail_decay
        else:
            self.buffer[:] = 0

        if self.glow_strength > 0:
            glow_img = new_layer_img.filter(ImageFilter.BoxBlur(self.glow_radius))
            glow = np.asarray(glow_img, dtype=np.float32) * self.glow_strength
            self.buffer += new_layer + glow
        else:
            self.buffer += new_layer

        np.clip(self.buffer, 0, 255, out=self.buffer)
        return Image.fromarray(self.buffer.astype(np.uint8), mode="RGB")


def apply_beat_punch(img, punch, max_zoom_frac, w, h):
    """Brief zoom toward center on a detected onset. Skipped entirely when
    punch is near zero (most frames) so it costs nothing on average."""
    if punch < 0.03:
        return img
    scale = 1.0 + max_zoom_frac * punch
    new_w, new_h = int(w * scale), int(h * scale)
    big = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return big.crop((left, top, left + w, top + h))


def draw_title(img, title, font, out_w, out_h):
    draw = ImageDraw.Draw(img)
    tw = draw.textlength(title, font=font)
    draw.text((out_w / 2 - tw / 2, out_h * 0.86), title, fill=(220, 220, 230), font=font)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav_path")
    ap.add_argument("out_path")
    ap.add_argument("--format", choices=FORMATS.keys(), default="shorts")
    ap.add_argument("--style", choices=["radial", "bars", "glowburst", "wormhole"], default="radial")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bands", type=int, default=48)
    ap.add_argument("--title", default=None)
    ap.add_argument("--max-seconds", type=float, default=None, help="render only the first N seconds (quick preview)")
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--trail-decay", type=float, default=0.85)
    ap.add_argument("--glow-strength", type=float, default=0.55)
    ap.add_argument("--glow-radius", type=float, default=None)
    ap.add_argument("--no-trails", action="store_true")
    ap.add_argument("--no-glow", action="store_true")
    ap.add_argument("--no-particles", action="store_true")
    ap.add_argument("--particle-rate", type=float, default=0.5)
    ap.add_argument("--max-particles", type=int, default=260)
    ap.add_argument("--no-beat-punch", action="store_true")
    ap.add_argument("--punch-strength", type=float, default=0.045)
    ap.add_argument("--symmetry", type=int, default=1, help="radial style only: draw a 1/N wedge rotated into N copies (default 1 = off)")
    ap.add_argument("--emoji", default=None, help='emoji/character to place in the center (radial style only), e.g. "❤️", "\U0001f525", "⭐". Rendered via a system color-emoji font when available; falls back to a hand-drawn glowing heart if the font is missing and the character is heart-like.')
    ap.add_argument("--emoji-beat", choices=["kick", "rms", "off"], default="kick", help="what drives the emoji's pulse size: kick-drum-restricted onsets (default), overall loudness, or a static size")
    ap.add_argument("--emoji-size", type=float, default=0.22, help="base emoji size as a fraction of min(width,height) (default 0.22)")
    ap.add_argument("--emoji-pulse", type=float, default=0.5, help="extra size fraction at peak beat strength (default 0.5, i.e. up to +50%% on a hard hit)")
    ap.add_argument("--kick-fmax", type=float, default=200.0, help="Hz cutoff for what counts as 'kick drum' when --emoji-beat kick (default 200)")
    ap.add_argument("--grid", default=None, help="music_analysis.json (or a pick_clip.py clip_*_grid.json) for THIS audio: lock beat punch / emoji pulse to the analyzed beat grid and flash on drops. Falls back to onset detection when the file has no grid (ambient).")
    ap.add_argument("--progress-json", action="store_true", help="emit machine-readable progress lines (see pipeline_io.py)")
    args = ap.parse_args()
    if args.progress_json:
        enable_progress()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) and retry.")

    w, h = FORMATS[args.format]
    ss = max(1, args.supersample)
    hi_w, hi_h = w * ss, h * ss

    trail_decay = 0.0 if args.no_trails else max(0.0, min(1.0, args.trail_decay))
    glow_strength = 0.0 if args.no_glow else max(0.0, args.glow_strength)
    glow_radius = args.glow_radius if args.glow_radius is not None else min(w, h) * 0.012
    particles_on = not args.no_particles
    beat_punch_on = not args.no_beat_punch
    symmetry = max(1, args.symmetry)
    if symmetry > 1 and args.style != "radial":
        print(f"Note: --symmetry only applies to --style radial; ignoring for {args.style}.", file=sys.stderr)
        symmetry = 1

    print(f"Loading audio and computing features ({args.bands} bands @ {args.fps}fps)...")
    mel_db, rms, punch_env, kick_env, n_frames, duration, y, sr = load_features(
        args.wav_path, args.fps, args.bands, kick_fmax=args.kick_fmax
    )

    flash_env = None
    if args.grid:
        envs = grid_envelopes(args.grid, n_frames, args.fps)
        if envs is None:
            print(f"Note: {args.grid} has no beat grid (beatless track) -- keeping onset detection.", file=sys.stderr)
        else:
            punch_env, kick_env, flash_env = envs
            print(f"Beat-locked to {args.grid}")

    if args.max_seconds:
        n_frames = min(n_frames, int(args.max_seconds * args.fps))

    if args.style in ("glowburst", "wormhole"):
        # rescale each band to its own 2nd-98th percentile range: mel_db is
        # normalized against the whole spectrogram, so on dark material
        # (pads, ambience) the upper bands sit near zero all track long and
        # the burst collapses to a narrow cone of bass rays
        lo = np.percentile(mel_db, 2, axis=1, keepdims=True)
        hi = np.percentile(mel_db, 98, axis=1, keepdims=True)
        mel_db = np.clip((mel_db - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)

    font = load_font(int(h * 0.03))
    compositor = FrameCompositor(w, h, trail_decay, glow_strength, glow_radius)
    particles = ParticleSystem(args.max_particles, args.particle_rate, args.fps) if particles_on else None
    bursts = BurstRings(args.fps) if args.style == "glowburst" else None
    tunnel = WormholeTunnel(args.fps) if args.style == "wormhole" else None

    emoji_glyph = None
    if args.emoji:
        if args.style != "radial":
            print(f"Note: --emoji only applies to --style radial; ignoring for {args.style}.", file=sys.stderr)
        else:
            emoji_glyph = get_emoji_glyph(args.emoji)
            if emoji_glyph is None:
                print(
                    f"Warning: no usable emoji font found for {args.emoji!r} and it's not "
                    "heart-like (no built-in fallback for it) -- skipping the emoji.",
                    file=sys.stderr,
                )
    emoji_base_px = min(w, h) * args.emoji_size

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(args.fps),
        "-i", "-",
        "-i", args.wav_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
        "-r", str(args.fps),
        # upload-safe: 48k/320k AAC (YouTube and Instagram both re-encode,
        # so start from the best source), moov atom up front for streaming
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-movflags", "+faststart",
        "-shortest",
        args.out_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    prev_bands = np.zeros(args.bands)
    print(
        f"Rendering {n_frames} frames at {w}x{h} (supersample {ss}x, "
        f"trails={'on' if trail_decay > 0 else 'off'}, glow={'on' if glow_strength > 0 else 'off'}, "
        f"particles={'on' if particles_on else 'off'}, beat-punch={'on' if beat_punch_on else 'off'}, "
        f"symmetry={symmetry}, emoji={'on (' + args.emoji_beat + ')' if emoji_glyph is not None else 'off'})..."
    )
    try:
        for i in range(n_frames):
            target = mel_db[:, i]
            prev_bands = smooth(prev_bands, target)
            t = i / args.fps
            rms_val = rms[i] if i < len(rms) else 0.0

            if args.style == "radial":
                layer, spawn_points = draw_radial_layer(hi_w, hi_h, w, h, prev_bands, rms_val, symmetry=symmetry)
            elif args.style == "wormhole":
                tunnel.advance(rms_val, float(punch_env[i]) if i < len(punch_env) else 0.0)
                layer, spawn_points = draw_wormhole_layer(hi_w, hi_h, w, h, prev_bands, rms_val, t=t, tunnel=tunnel)
            elif args.style == "glowburst":
                bursts.update(float(punch_env[i]) if i < len(punch_env) else 0.0, 0.05 * (1 + 0.4 * rms_val))
                layer, spawn_points = draw_glowburst_layer(hi_w, hi_h, w, h, prev_bands, rms_val, t=t, bursts=bursts)
            else:
                layer, spawn_points = draw_bars_layer(hi_w, hi_h, w, h, prev_bands, rms_val)

            if emoji_glyph is not None:
                if args.emoji_beat == "kick":
                    beat_val = float(kick_env[i]) if i < len(kick_env) else 0.0
                elif args.emoji_beat == "rms":
                    beat_val = rms_val
                else:
                    beat_val = 0.0
                size_px = max(4, int(emoji_base_px * (1.0 + args.emoji_pulse * beat_val)))
                sized = emoji_glyph.resize((size_px, size_px), Image.LANCZOS)
                paste_x = w // 2 - size_px // 2
                paste_y = h // 2 - size_px // 2
                layer.paste(sized, (paste_x, paste_y), sized)

            if particles is not None:
                particles.spawn(spawn_points)
                pdraw = ImageDraw.Draw(layer)
                particles.update_and_draw(pdraw)

            frame_img = compositor.composite(layer)

            if beat_punch_on:
                frame_img = apply_beat_punch(frame_img, float(punch_env[i]) if i < len(punch_env) else 0.0, args.punch_strength, w, h)

            if flash_env is not None and i < len(flash_env) and flash_env[i] > 0.02:
                frame_img = Image.blend(frame_img, Image.new("RGB", (w, h), (255, 255, 255)), 0.35 * float(flash_env[i]))

            if args.title:
                frame_img = draw_title(frame_img, args.title, font, w, h)

            proc.stdin.write(frame_img.tobytes())
            if i % (args.fps * 5) == 0:
                print(f"  {t:6.1f}s / {duration:.1f}s")
            if i % args.fps == 0 or i == n_frames - 1:
                emit("progress", step=os.environ.get(PARENT_STEP_ENV, "video"), done=i + 1, total=n_frames,
                     detail="frames")
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        ret = proc.wait()
        if ret != 0:
            print(err[-3000:], file=sys.stderr)
            sys.exit(f"ffmpeg exited with code {ret}")

    print(f"Done: {args.out_path}")


if __name__ == "__main__":
    main()
