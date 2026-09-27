#!/usr/bin/env python3
"""
visualize_wav.py — render an audio-reactive visualization video from a WAV file.

Optional feature (see requirements-optional.txt): needs librosa + Pillow on
top of this repo's core numpy/scipy requirements, plus the ffmpeg binary on
PATH. Nothing else in the repo depends on this script.

Pipeline:
  1. Load audio with librosa, compute a mel spectrogram (drives radial bars)
     and an RMS envelope (drives a pulsing center circle).
  2. Render each frame with Pillow (fast — no matplotlib per-frame overhead).
  3. Stream raw RGB frames into an ffmpeg subprocess that muxes them with the
     original audio into a single .mp4 (no intermediate frame files).

Usage:
  python3 visualize_wav.py input.wav output.mp4 --format shorts --style radial
  python3 visualize_wav.py input.wav output.mp4 --format landscape --style bars
  python3 visualize_wav.py input.wav output.mp4 --format shorts --title "Contemplation"

Formats:
  shorts     1080x1920 (9:16, YouTube Shorts / Reels / TikTok)
  landscape  1920x1080 (16:9, standard YouTube)
  square     1080x1080 (1:1)

Requires: librosa, numpy, pillow, and the ffmpeg binary on PATH.
"""

import argparse
import math
import subprocess
import sys
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FORMATS = {
    "shorts": (1080, 1920),
    "landscape": (1920, 1080),
    "square": (1080, 1080),
}


def load_features(wav_path, fps, n_bands=48, sr_target=22050):
    """Load audio and compute per-frame band energies + RMS envelope."""
    import librosa

    y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    duration = len(y) / sr
    hop_length = int(sr / fps)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_bands, hop_length=hop_length, fmax=sr / 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # normalize each band to 0..1 across the whole track
    mel_db = mel_db - mel_db.min()
    if mel_db.max() > 0:
        mel_db = mel_db / mel_db.max()

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if rms.max() > 0:
        rms = rms / rms.max()

    n_frames = mel_db.shape[1]
    return mel_db, rms, n_frames, duration, y, sr


def smooth(prev, target, attack=0.55, release=0.15):
    """Exponential smoothing with faster attack than release (punchy but not jittery)."""
    out = np.empty_like(target)
    for i in range(len(target)):
        a = attack if target[i] > prev[i] else release
        out[i] = prev[i] + (target[i] - prev[i]) * a
    return out


def draw_radial_frame(w, h, bands, rms_val, t, title=None, font=None):
    img = Image.new("RGB", (w, h), (8, 8, 14))
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    n = len(bands)
    base_r = min(w, h) * 0.16 * (1 + 0.08 * rms_val)
    max_extra = min(w, h) * 0.30

    for i, amp in enumerate(bands):
        angle = (i / n) * 2 * math.pi - math.pi / 2
        r1 = base_r
        r2 = base_r + max_extra * (0.08 + amp)
        x1, y1 = cx + r1 * math.cos(angle), cy + r1 * math.sin(angle)
        x2, y2 = cx + r2 * math.cos(angle), cy + r2 * math.sin(angle)
        hue = (i / n) * 0.6 + 0.55  # blue -> violet -> pink sweep
        color = hsv_to_rgb(hue % 1.0, 0.65, 0.95)
        width = max(2, int(min(w, h) * 0.006))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    glow_r = base_r * 0.9
    draw.ellipse(
        [cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r],
        outline=(230, 230, 255),
        width=3,
    )

    if title:
        tw = draw.textlength(title, font=font)
        draw.text((cx - tw / 2, h * 0.86), title, fill=(220, 220, 230), font=font)

    return img


def draw_bars_frame(w, h, bands, rms_val, t, title=None, font=None):
    img = Image.new("RGB", (w, h), (10, 10, 16))
    draw = ImageDraw.Draw(img)
    n = len(bands)
    margin = w * 0.04
    gap = 4
    bar_w = (w - 2 * margin - gap * (n - 1)) / n
    baseline = h * 0.72
    max_h = h * 0.5

    for i, amp in enumerate(bands):
        bh = max(0.0, max_h * float(amp))
        x0 = margin + i * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = baseline - bh
        y1 = baseline
        hue = 0.55 + 0.35 * (i / n)
        color = hsv_to_rgb(hue, 0.7, 0.9 + 0.1 * min(1.0, amp))
        draw.rectangle([x0, y0, x1, y1], fill=color)
        # mirrored reflection, faded
        refl_h = bh * 0.35
        if refl_h > 0.5:
            draw.rectangle([x0, y1, x1, y1 + refl_h], fill=tuple(c // 4 for c in color))

    if title:
        tw = draw.textlength(title, font=font)
        draw.text((w / 2 - tw / 2, h * 0.85), title, fill=(220, 220, 230), font=font)

    return img


def hsv_to_rgb(h, s, v):
    import colorsys
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav_path")
    ap.add_argument("out_path")
    ap.add_argument("--format", choices=FORMATS.keys(), default="shorts")
    ap.add_argument("--style", choices=["radial", "bars"], default="radial")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bands", type=int, default=48)
    ap.add_argument("--title", default=None)
    ap.add_argument("--max-seconds", type=float, default=None, help="render only the first N seconds (quick preview)")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) and retry.")

    w, h = FORMATS[args.format]
    print(f"Loading audio and computing features ({args.bands} bands @ {args.fps}fps)...")
    mel_db, rms, n_frames, duration, y, sr = load_features(args.wav_path, args.fps, args.bands)

    if args.max_seconds:
        n_frames = min(n_frames, int(args.max_seconds * args.fps))

    font = load_font(int(h * 0.03))
    draw_fn = draw_radial_frame if args.style == "radial" else draw_bars_frame

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(args.fps),
        "-i", "-",
        "-i", args.wav_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        args.out_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    prev_bands = np.zeros(args.bands)
    print(f"Rendering {n_frames} frames at {w}x{h}...")
    try:
        for i in range(n_frames):
            target = mel_db[:, i]
            prev_bands = smooth(prev_bands, target)
            t = i / args.fps
            img = draw_fn(w, h, prev_bands, rms[i] if i < len(rms) else 0.0, t, args.title, font)
            proc.stdin.write(img.tobytes())
            if i % (args.fps * 5) == 0:
                print(f"  {t:6.1f}s / {duration:.1f}s")
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
