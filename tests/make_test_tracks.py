#!/usr/bin/env python3
"""Synthesize test audio with known answers, so the pipelines can be
checked without real recordings:

  techno_test.wav     128 BPM, A minor, intro/breakdown/build/drop/outro, stereo-detuned bass
  trap_test.wav       70 BPM (half-time), F minor, pure-sine 808 (fails the phone check)
  psytrance_test.wav  145 BPM, E minor, kick + rolling 16th bass, breakdown
  ambient_test.wav    beatless D major pads, wide stereo low end
  rain_test.wav       70 s of steady pink noise (a "clean bed")
  speech_test.wav     12 s of voiced bursts with pauses and silent edges

Usage:
  python3 tests/make_test_tracks.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000
rng = np.random.default_rng(1)

def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))

def kick(len_s=0.35, f0=150, f1=45):
    n = int(len_s * SR); t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * env_exp(n, 0.12)

def hat(len_s=0.05):
    n = int(len_s * SR); x = rng.standard_normal(n)
    x = np.diff(np.concatenate([[0], x]))  # crude highpass
    return 0.25 * x * env_exp(n, 0.012)

def snare(len_s=0.2):
    n = int(len_s * SR); t = np.arange(n) / SR
    return (0.4 * rng.standard_normal(n) * env_exp(n, 0.05) + 0.3 * np.sin(2*np.pi*190*t) * env_exp(n, 0.04))

def tone(freq, len_s, tau=None, saw=False, detune=0.0):
    n = int(len_s * SR); t = np.arange(n) / SR
    f = freq * (1 + detune)
    x = (2 * ((t * f) % 1) - 1) if saw else np.sin(2 * np.pi * f * t)
    e = env_exp(n, tau) if tau else np.ones(n)
    a = min(n, int(0.003 * SR)); e[:a] *= np.linspace(0, 1, a)
    return x * e

def add(buf, x, at, ch=None, gain=1.0):
    i = int(at * SR)
    if i >= buf.shape[0]: return
    x = x[: buf.shape[0] - i] * gain
    if ch is None:
        buf[i:i+len(x), 0] += x; buf[i:i+len(x), 1] += x
    else:
        buf[i:i+len(x), ch] += x

def lowpass(x, fc):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, fc, fs=SR, output="sos"), x)

def pad(buf, freqs, t0, t1, gain=0.08):
    n = int((t1 - t0) * SR); t = np.arange(n) / SR
    for ch, det in ((0, -0.004), (1, 0.004)):
        x = sum(lowpass(tone(f, t1 - t0, saw=True, detune=det), 2500) for f in freqs)
        sw = 0.5 - 0.5 * np.cos(2 * np.pi * t / max(1e-3, (t1 - t0)))
        add(buf, x * sw, t0, ch, gain)

def norm(buf, peak_db=-6):
    return buf / np.abs(buf).max() * 10 ** (peak_db / 20)

def techno():
    bpm = 128; b = 60 / bpm; bar = 4 * b
    plan = [("intro", 8), ("break", 8), ("build", 4), ("drop", 16), ("outro", 8)]
    total = sum(n for _, n in plan) * bar
    buf = np.zeros((int(total * SR) + SR, 2)); t = 0.0
    A = 55.0
    for name, nb in plan:
        for i in range(nb * 4):
            bt = t + i * b
            if name in ("intro", "drop", "outro"):
                add(buf, kick(), bt, gain=0.9)
            if name in ("intro", "drop"):
                add(buf, hat(), bt + b / 2, ch=1, gain=0.8); add(buf, hat(), bt + b / 2, ch=0, gain=0.4)
            if name == "build":
                for k in range(4):
                    add(buf, snare(), bt + k * b / 4, gain=0.2 + 0.6 * i / (nb * 4))
            if name == "drop":
                for k in (1, 2, 3):  # offbeat 16ths, stereo-detuned on purpose
                    add(buf, tone(A, b / 4 * 0.9, tau=0.06, saw=True, detune=-0.01), bt + k * b / 4, 0, 0.35)
                    add(buf, tone(A, b / 4 * 0.9, tau=0.06, saw=True, detune=0.01), bt + k * b / 4, 1, 0.35)
        if name in ("break", "build"):
            pad(buf, [220, 261.63, 329.63], t, t + nb * bar, 0.12)
        t += nb * bar
    return norm(buf), bpm, "A minor"

def trap():
    bpm = 70; b = 60 / bpm; bar = 4 * b
    plan = [("intro", 4), ("drop", 8), ("break", 2), ("drop", 4)]
    total = sum(n for _, n in plan) * bar
    buf = np.zeros((int(total * SR) + SR, 2)); t = 0.0
    notes = [43.65, 43.65, 51.91, 38.89]  # F1 F1 Ab1 Eb1  (F minor)
    for name, nb in plan:
        for i in range(nb * 4):
            bt = t + i * b
            if name != "break":
                for k in range(4):
                    add(buf, hat(), bt + k * b / 4, gain=0.6)
            if name == "drop":
                if i % 4 == 0:
                    add(buf, kick(0.3, 120, 50), bt, gain=0.8)
                    add(buf, tone(notes[(i // 4) % 4], 1.8 * b, tau=1.5), bt, gain=0.9)  # pure-sine 808
                if i % 4 == 2:
                    add(buf, snare(), bt, gain=0.9)
                if i % 4 == 3:
                    add(buf, kick(0.3, 120, 50), bt + b / 2, gain=0.6)
        pad(buf, [349.23, 415.30, 523.25], t, t + nb * bar, 0.07)
        t += nb * bar
    return norm(buf), bpm, "F minor"

def psy():
    bpm = 145; b = 60 / bpm; bar = 4 * b
    plan = [("intro", 8), ("drop", 16), ("break", 8), ("drop", 8)]
    total = sum(n for _, n in plan) * bar
    buf = np.zeros((int(total * SR) + SR, 2)); t = 0.0
    E = 41.2
    for name, nb in plan:
        for i in range(nb * 4):
            bt = t + i * b
            if name != "break":
                add(buf, kick(0.2, 160, 50), bt, gain=0.9)
                add(buf, hat(), bt + b / 2, gain=0.6)
            if name == "drop":
                for k in (1, 2, 3):
                    add(buf, lowpass(tone(E * 2, b / 4 * 0.85, tau=0.04, saw=True), 900), bt + k * b / 4, gain=0.6)
        if name == "break":
            pad(buf, [164.81, 196.0, 246.94], t, t + nb * bar, 0.12)
        t += nb * bar
    return norm(buf), bpm, "E minor"

def ambient():
    total = 90.0
    buf = np.zeros((int(total * SR), 2))
    chords = [[146.83, 220.0, 293.66, 369.99], [196.0, 246.94, 293.66, 392.0], [146.83, 220.0, 293.66, 440.0]]
    for k, ch in enumerate(chords):
        t0 = k * 28.0
        pad(buf, ch, t0, t0 + 34.0, 0.10 + 0.05 * k)
    return norm(buf, -10), None, "D major"

def rain():
    n = int(70 * SR)
    white = rng.standard_normal((n, 2))
    # pink-ish: sum of octave-spaced one-pole lowpasses
    from scipy.signal import lfilter
    pink = sum(lfilter([1 - a], [1, -a], white, axis=0) / (k + 1) for k, a in enumerate((0.5, 0.9, 0.99, 0.997)))
    return norm(pink, -12), None, None


def speech():
    total = 12.0
    buf = np.zeros((int(total * SR), 2))
    t = 1.0
    for k in range(8):
        d = 0.6 + 0.15 * (k % 3)
        n = int(d * SR)
        tt = np.arange(n) / SR
        f0 = 120 + 20 * np.sin(2 * np.pi * 3 * tt)
        ph = 2 * np.pi * np.cumsum(f0) / SR
        voiced = sum(np.sin(h * ph) / h for h in range(1, 12))
        hiss = 0.05 * rng.standard_normal(n)
        env = np.sin(np.pi * tt / d) ** 0.5
        add(buf, (voiced + hiss) * env, t, gain=0.3)
        t += d + (0.4 if k % 3 else 0.9)
    return norm(buf, -8), None, None


TRACKS = {"techno": techno, "trap": trap, "psytrance": psy, "ambient": ambient, "rain": rain, "speech": speech}


def make_all(out_dir: Path, names=None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in names or TRACKS:
        path = out_dir / f"{name}_test.wav"
        if not path.exists():
            buf, _bpm, _key = TRACKS[name]()
            sf.write(path, buf.astype(np.float32), SR, subtype="PCM_24")
        paths[name] = path
    return paths


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for name, path in make_all(Path(sys.argv[1])).items():
        print(f"{name:10s} {path}")
