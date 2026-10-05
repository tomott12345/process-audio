#!/usr/bin/env python3
"""Analyze a finished track or Bluebox mix: tempo + beat/bar grid, key
(with Camelot), loudness, stereo/mono-bass checks, a phone-speaker check,
spectral tilt, and an intro/build/drop/breakdown/outro section map.
Read-only -- it never changes the audio. Writes music_analysis.json, which
process_music_wav.py, pick_clip.py, release.py, and visualize_wav.py --grid
all read.

--genre is required (not guessed): it sets the tempo range the beat
tracker folds into -- the classic beat-tracker failure is landing on half
or double the real tempo. Trap is reported at its half-time feel (~70 BPM)
while the double-time grid is kept internally as beats_fullres, so hat
rolls and visual sync still land. Ambient often has no reliable beat; it's
reported as "no grid" and everything downstream falls back to seconds
instead of inventing bars.

Tempo is assumed constant (hardware sequencers on one clock don't drift)
and fitted by linear regression over every tracked beat, which is what
makes +/-0.1 BPM achievable on a few minutes of audio.

Usage:
  python3 music_analyze.py mix.wav --genre techno
  python3 music_analyze.py mix.wav --genre trap --png
  python3 music_analyze.py mix.wav --genre ambient --out-dir ./analysis
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from music_common import (
    astats, die, ebur128, fmt_time, genre_recipe, load_recipes, preflight,
    probe, require_python, write_json,
)

SR = 22050
HOP = 512
ANALYSIS_VERSION = 1

# Krumhansl-Kessler key profiles
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PITCHES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
CAMELOT_MAJOR = {"C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B", "B": "1B",
                 "F#": "2B", "Db": "3B", "Ab": "4B", "Eb": "5B", "Bb": "6B", "F": "7B"}
CAMELOT_MINOR = {"A": "8A", "E": "9A", "B": "10A", "F#": "11A", "Db": "12A", "Ab": "1A",
                 "Eb": "2A", "Bb": "3A", "F": "4A", "C": "5A", "G": "6A", "D": "7A"}

# a beat grid is trusted when the folded onset histogram's peak is this
# many times its mean (see _comb_score); per-genre override: grid_score_min
GRID_SCORE_MIN = 3.0


def _db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-20))


def _onset_env(y, hop: int, fmax: float | None = None):
    """Onset strength with its slow-moving baseline removed, so sustained
    energy (pads, a held 808) doesn't masquerade as periodicity."""
    import librosa
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    kw = {"fmax": fmax, "n_mels": 24} if fmax else {}
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop, **kw)
    base = uniform_filter1d(env, size=max(3, int(0.5 * SR / hop)))
    return np.maximum(0.0, env - base)


def _comb_hist(env, fps: float, bpm: float, nb: int, i0: int = 0):
    """Fold the onset envelope at one beat period into an nb-bin phase
    histogram. A constant tempo stacks every beat into the same bins; a
    wrong tempo smears them out."""
    import numpy as np

    period = fps * 60.0 / bpm
    idx = np.arange(len(env)) + i0  # absolute frame index: chunks share one phase reference
    ph = ((idx % period) / period * nb).astype(int) % nb
    counts = np.bincount(ph, minlength=nb)
    hist = np.bincount(ph, weights=env, minlength=nb) / np.maximum(counts, 1)
    return 0.25 * np.roll(hist, 1) + 0.5 * hist + 0.25 * np.roll(hist, -1)


def _comb_score(env, fps: float, bpm: float, nbins: int | None = None):
    """Returns (peakiness, phase in seconds of the strongest bin)."""
    import numpy as np

    period = fps * 60.0 / bpm
    nb = nbins or max(4, int(round(period)))
    hist = _comb_hist(env, fps, bpm, nb)
    mean = hist.mean()
    if mean <= 0:
        return 0.0, 0.0
    k = int(np.argmax(hist))
    return float(hist[k] / mean), (k + 0.5) / nb * period / fps


def _best_bpm(env, fps: float, center: float, half_width: float, step: float) -> float:
    import numpy as np

    bpms = np.arange(center - half_width, center + half_width + step / 2, step)
    scores = [_comb_score(env, fps, float(b))[0] for b in bpms]
    return float(bpms[int(np.argmax(scores))])


def _snap_lag(y, beats, period: float) -> float:
    """Median offset (seconds) from grid beats to where the attacks actually
    start in the waveform. The onset envelope peaks a few ms after a
    transient begins (more on a soft kick than a click), and a late grid
    puts every bar cut just after the kick's attack -- chopping it and
    bleeding it into the previous bar. Measured on 1 ms log-energy frames:
    the steepest rise within -40..+25 ms of each beat."""
    import numpy as np

    fr = max(1, int(SR * 0.001))
    n = len(y) // fr
    e = np.log10(np.add.reduceat(y[: n * fr] ** 2, np.arange(0, n * fr, fr)) + 1e-12)
    rise = np.zeros_like(e)
    rise[3:] = e[3:] - e[:-3]
    fsec = fr / SR  # frame length in seconds (~1 ms)
    offs = []
    for t in beats[:: max(1, len(beats) // 400)]:
        c = int(round(t / fsec))
        a, b = c - int(0.040 / fsec), c + int(0.025 / fsec)
        if a < 3 or b >= n:
            continue
        k = int(np.argmax(rise[a:b]))
        if rise[a + k] > 0.5:  # a real attack (>5 dB in 3 ms), not noise
            offs.append((a + k - 1 - c) * fsec)  # rise[k] spans frames k-3..k; the jump lands near k-1
    if len(offs) < 8:
        return 0.0
    lag = float(np.median(offs))
    return lag if abs(lag) < min(0.04, period / 4) else 0.0


def estimate_grid(y, recipe: dict, flags: list[str]) -> tuple[dict, dict]:
    """Constant-tempo comb search inside the genre's tempo range.

    1. coarse: the most onset-dense <=90s excerpt folded into 16 phase bins
       per beat -- tolerant of tempo error, so a coarse step finds the
       right neighborhood.
    2. excerpt, full frame resolution, +/-0.6 BPM at 0.01 steps.
    3. whole track, full resolution, a narrow window at a step matched to
       the track length: over a long track even a hundredth of a BPM
       smears the folded beats, which is what pins the tempo down.
    Searching only inside the genre range is what keeps trap at ~70
    instead of locking onto the 140 double-time."""
    import numpy as np

    lo, hi = recipe["tempo_range"]
    lo_s, hi_s = lo / 1.04, hi * 1.04
    hop = 128
    fps = SR / hop
    duration = len(y) / SR
    no_grid = {"has_grid": False, "bpm": None, "bpm_fullres": None, "subdivision": 1, "grid_score": None}
    if duration < 8.0:
        flags.append("too short to fit a beat grid -- treating as no grid")
        return no_grid, {}

    env = _onset_env(y, hop)
    # coarse: most onset-dense <=90s excerpt
    win = int(min(duration, 90.0) * fps)
    if len(env) > win:
        dens = np.convolve(env, np.ones(win), mode="valid")
        i0 = int(np.argmax(dens[:: max(1, int(fps))]) * max(1, int(fps)))
        ex = env[i0:i0 + win]
    else:
        ex = env
    n_beats_ex = len(ex) / fps * hi_s / 60.0
    step = max(0.005, hi_s * (1 / 16) / max(n_beats_ex, 1) / 2)
    cands = np.arange(lo_s, hi_s + step, step)
    coarse = [(_comb_score(ex, fps, b, nbins=16)[0], b) for b in cands]
    best_c = max(coarse)[1]

    # fine: measure how the beat phase drifts across the track at the coarse
    # tempo (per 20s chunk), and correct the tempo by that drift rate --
    # a tempo error shows up as a steady phase slide, which a linear fit
    # over every beat-carrying chunk measures to well under 0.1 BPM
    bpm = _best_bpm(ex, fps, float(best_c), 0.6, 0.01)
    if len(env) > 1.5 * len(ex):
        nb = fps * 60.0 / bpm
        width = bpm / (nb * duration * bpm / 60.0)
        bpm = _best_bpm(env, fps, bpm, 0.03, max(0.0002, width / 2))
    score, _ = _comb_score(env, fps, bpm)
    score_min = recipe.get("grid_score_min", GRID_SCORE_MIN)
    if score < score_min:
        flags.append(f"no steady beat (grid score {score:.2f} < {score_min}) -- times in seconds, not bars")
        out = dict(no_grid)
        out["grid_score"] = round(score, 2)
        return out, {}

    # the beat is where the kick is: take the beat phase from the low-band
    # onset histogram, not the full-band one (offbeat hats and 16th-note
    # basslines can out-peak the kick there)
    bass_env = _onset_env(y, hop, fmax=150)
    _, phase = _comb_score(bass_env, fps, bpm)
    period = 60.0 / bpm
    first = phase - math.floor(phase / period) * period
    beats = first + period * np.arange(int((duration - first) / period) + 1)
    lag = _snap_lag(y, beats, period)
    # a beat snapped a few ms before t=0 is the track's first beat, not a
    # reason to start the grid a whole beat later
    first = max(0.0, first + lag) if first + lag > -0.02 else (first + lag) % period
    beats = first + period * np.arange(int((duration - first) / period) + 1)

    # downbeats: the bar phase with the most kick energy on the one

    def bass_at(times):
        f = np.clip(np.round(np.asarray(times) * fps).astype(int), 0, len(bass_env) - 1)
        hits = np.maximum.reduce([bass_env[np.clip(f + d, 0, len(bass_env) - 1)] for d in (-2, -1, 0, 1, 2)])
        return float(hits.mean()) if len(hits) else 0.0

    # Where the one is: parts enter and drop out on bar/phrase starts, so
    # level jumps between beats pile up on the downbeat phase -- decisive
    # whenever the arrangement changes at all. A kick accent (trap's kick on
    # the one) is the fallback; four-on-the-floor kicks are equal on every
    # beat and can't tell the phases apart on their own.
    level = np.array([
        _db(float(np.mean(y[int(t * SR):int((t + period) * SR)] ** 2))) if int(t * SR) < len(y) else -200.0
        for t in beats
    ])
    jumps = np.abs(np.diff(level, prepend=level[0]))
    jump_by_phase = [float(jumps[j::4].sum()) for j in range(4)]
    kick_by_phase = [bass_at(beats[j::4]) for j in range(4)]

    def decisive(scores, ratio):
        r = sorted(scores, reverse=True)
        return r[1] > 0 and r[0] / r[1] >= ratio

    if decisive(jump_by_phase, 1.5):
        bar_phase, downbeat_method = int(np.argmax(jump_by_phase)), "arrangement changes"
    elif decisive(kick_by_phase, 1.15):
        bar_phase, downbeat_method = int(np.argmax(kick_by_phase)), "kick accent"
    else:
        bar_phase, downbeat_method = int(np.argmax(jump_by_phase)), "arrangement changes (weak)"
    downbeats = beats[bar_phase::4]
    sub = int(recipe.get("grid_subdivision", 1))
    beats_fullres = first + (period / sub) * np.arange(int((duration - first) / (period / sub)) + 1)

    tempo_info = {
        "has_grid": True,
        "bpm": round(bpm, 2),
        "bpm_fullres": round(bpm * sub, 2),
        "subdivision": sub,
        "grid_score": round(score, 2),
        "downbeat_method": downbeat_method,
        "attack_snap_ms": round(lag * 1000, 1),
    }
    grid = {
        "beats_per_bar": 4,
        "bar_seconds": round(4 * period, 6),
        "first_downbeat": round(float(downbeats[0]), 4) if len(downbeats) else None,
        "beats": [round(float(t), 4) for t in beats],
        "downbeats": [round(float(t), 4) for t in downbeats],
        "beats_fullres": [round(float(t), 4) for t in beats_fullres],
    }
    return tempo_info, grid


def estimate_key(y) -> dict:
    import librosa
    import numpy as np

    harm = librosa.effects.harmonic(y, margin=2.0)
    # start at C3: below that, bass fundamentals and their overtone series
    # (a saw bass's 5th harmonic is a major third) pull minor keys to major
    chroma = librosa.feature.chroma_cqt(
        y=harm, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C3"), n_octaves=5
    ).mean(axis=1)
    if chroma.sum() <= 0:
        return {"name": None, "camelot": None, "confidence": 0.0}
    scores = []
    for i in range(12):
        for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = float(np.corrcoef(chroma, np.roll(prof, i))[0, 1])
            scores.append((r, PITCHES[i], mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    tonic, mode = best[1], best[2]
    camelot = (CAMELOT_MAJOR if mode == "major" else CAMELOT_MINOR)[tonic]
    return {
        "name": f"{tonic} {mode}",
        "camelot": camelot,
        "confidence": round(best[0] - second[0], 3),
        "runner_up": f"{second[1]} {second[2]}",
    }


def stereo_and_spectrum(y_st, recipe: dict, flags: list[str]) -> tuple[dict, dict, dict]:
    import numpy as np
    from scipy.signal import butter, sosfilt

    if y_st.ndim == 1:
        left = right = y_st
        is_mono = True
    else:
        left, right = y_st[0], y_st[1]
        is_mono = bool(np.allclose(left, right))
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    def corr(a, b):
        den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
        return float(np.dot(a, b) / den) if den > 0 else 1.0

    # low band = below this genre's mono-bass crossover: the band the
    # mastering step folds to mono, so before/after numbers line up
    low_hz = recipe["mono_bass_hz"]
    lp = butter(4, low_hz, fs=SR, output="sos")
    lo_l, lo_r = sosfilt(lp, left), sosfilt(lp, right)
    lo_mid, lo_side = 0.5 * (lo_l + lo_r), 0.5 * (lo_l - lo_r)
    e_lr = 0.5 * (float(np.dot(left, left)) + float(np.dot(right, right)))
    stereo = {
        "is_mono": is_mono,
        "correlation": round(corr(left, right), 3),
        "low_band_hz": low_hz,
        "low_band_correlation": round(corr(lo_l, lo_r), 3),
        "low_side_to_mid_db": round(_db(float(np.dot(lo_side, lo_side))) - _db(float(np.dot(lo_mid, lo_mid))), 1),
        "mono_sum_loss_db": round(_db(float(np.dot(mid, mid))) - _db(e_lr), 2),
    }
    if not is_mono and stereo["low_band_correlation"] < 0.9:
        flags.append(
            f"stereo content below {low_hz} Hz (low-band correlation {stereo['low_band_correlation']}) -- "
            "the mono-bass step in process_music_wav.py folds this band to mono"
        )
    if stereo["mono_sum_loss_db"] < -3.5:
        flags.append(f"mix loses {-stereo['mono_sum_loss_db']:.1f} dB summed to mono -- check for out-of-phase elements")

    # average power spectrum (frames sub-sampled to keep long jams cheap)
    n_fft = 8192
    hop = n_fft * 2
    frames = [mid[i:i + n_fft] for i in range(0, max(1, len(mid) - n_fft), hop)]
    frames = [f for f in frames if len(f) == n_fft] or [np.pad(mid, (0, max(0, n_fft - len(mid))))[:n_fft]]
    win = np.hanning(n_fft)
    specs = np.array([np.abs(np.fft.rfft(f * win)) ** 2 for f in frames])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    avg = specs.mean(axis=0)
    total = float(avg[freqs > 20].sum()) or 1e-20

    centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    bands = []
    for c in centers:
        m = (freqs >= c / math.sqrt(2)) & (freqs < c * math.sqrt(2))
        bands.append(_db(float(avg[m].sum())))
    slope = float(np.polyfit(np.arange(len(centers)), bands, 1)[0])
    ref = recipe.get("tilt_reference_db_per_octave")
    tilt = {
        "octave_centers_hz": centers,
        "octave_levels_db": [round(b - max(bands), 1) for b in bands],
        "slope_db_per_octave": round(slope, 2),
        "reference_db_per_octave": ref,
    }
    if ref is not None and abs(slope - ref) > 1.5:
        flags.append(f"spectral tilt {slope:.1f} dB/oct vs genre reference {ref} -- mix is {'darker' if slope < ref else 'brighter'} than reference")

    # phone check: what survives a ~150 Hz phone-speaker rolloff, and does
    # the bass have harmonics a phone can reproduce? Harmonics are measured
    # on bass-heavy frames only (top 30% of low-band energy)
    sub = float(avg[(freqs > 20) & (freqs < 150)].sum())
    phone_loss = _db(total - sub) - _db(total)
    low_e = specs[:, (freqs > 20) & (freqs < 150)].sum(axis=1)
    heavy = specs[low_e >= np.percentile(low_e, 70)].mean(axis=0)
    lm = (freqs >= 30) & (freqs <= 150)
    f0 = float(freqs[lm][int(np.argmax(heavy[lm]))])
    df = freqs[1]

    def peak_near(f):
        m = (freqs > f - 3 * df) & (freqs < f + 3 * df)
        return float(heavy[m].max()) if m.any() else 1e-20

    fund = peak_near(f0)
    harm_db = _db(sum(peak_near(f0 * h) for h in (2, 3, 4))) - _db(fund)
    phone = {
        "sub_share_pct": round(100.0 * sub / total, 1),
        "phone_speaker_loss_db": round(phone_loss, 1),
        "bass_fundamental_hz": round(f0, 1),
        "bass_harmonics_db": round(harm_db, 1),
    }
    if phone_loss < -6.0 and harm_db < -18.0:
        flags.append(
            f"phone speakers lose {-phone_loss:.1f} dB and the bass (~{f0:.0f} Hz) is close to a pure sine "
            f"(harmonics {harm_db:.0f} dB) -- it will mostly vanish on a phone; add saturation/harmonics to the bass"
        )
    elif phone_loss < -6.0:
        flags.append(f"phone speakers lose {-phone_loss:.1f} dB of this mix (it's sub-heavy); bass harmonics present ({harm_db:.0f} dB)")
    return stereo, tilt, phone


def _merge_segments(feats, lengths, threshold_db: float, min_units: int) -> list[list[int]]:
    """Agglomerative merge of adjacent units (bars, or 2s windows) by their
    per-band levels: repeatedly join the most similar neighbours until every
    neighbouring pair differs by more than threshold_db in some band, then
    fold runs shorter than min_units into their closer neighbour."""
    import numpy as np

    segs = [[i, i] for i in range(len(feats))]

    def mean(seg):
        w = lengths[seg[0]:seg[1] + 1]
        return np.average(feats[seg[0]:seg[1] + 1], axis=0, weights=w)

    def dist(s1, s2):
        return float(np.max(np.abs(mean(s1) - mean(s2))))

    while len(segs) > 1:
        ds = [dist(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
        i = int(np.argmin(ds))
        if ds[i] > threshold_db:
            break
        segs[i] = [segs[i][0], segs[i + 1][1]]
        segs.pop(i + 1)

    # fold short runs: always merge the closest neighbouring pair that
    # involves a short run, so a breakdown's bars gather with each other
    # instead of being swallowed one by one by the loud drop next door
    def short(sg):
        return sg[1] - sg[0] + 1 < min_units

    while len(segs) > 1 and any(short(sg) for sg in segs):
        pairs = [(dist(segs[i], segs[i + 1]), i) for i in range(len(segs) - 1)
                 if short(segs[i]) or short(segs[i + 1])]
        _, i = min(pairs)
        segs[i] = [segs[i][0], segs[i + 1][1]]
        segs.pop(i + 1)
    return segs


def detect_sections(y, tempo: dict, grid: dict, recipe: dict) -> tuple[list[dict], list[float], float | None]:
    """Split the track into sections from low (<150 Hz) / mid / high band
    levels per bar (per 2s window with no grid), then label them by how full
    they are relative to the fullest section: a drop is where the most bands
    are near their loudest at once -- a kick-only intro is loud, but its mids
    and highs aren't. Intro/build/outro come from context."""
    import numpy as np
    from scipy.signal import butter, sosfilt

    duration = len(y) / SR
    # steep (8th-order) splits: a loud 808 leaking through a gentle filter
    # would otherwise make the mid band follow the bassline's notes
    bands = [
        sosfilt(butter(8, 150, fs=SR, output="sos"), y),
        sosfilt(butter(8, [150, 2000], btype="band", fs=SR, output="sos"), y),
        sosfilt(butter(8, 2000, btype="high", fs=SR, output="sos"), y),
    ]
    if tempo["has_grid"]:
        edges = [t for t in grid["downbeats"] if t < duration - 0.5]
        if edges[0] > 0.05:
            edges = [0.0] + edges
        min_units, threshold = 2, 2.0
    else:
        edges = list(np.arange(0.0, duration - 1.0, 2.0))
        min_units, threshold = 4, 3.0
    edges.append(duration)

    feats, lengths = [], []
    for s0, e0 in zip(edges[:-1], edges[1:]):
        i0, i1 = int(s0 * SR), max(int(e0 * SR), int(s0 * SR) + 1)
        feats.append([max(_db(float(np.mean(b[i0:i1] ** 2))), -90.0) for b in bands])
        lengths.append(e0 - s0)
    feats = np.array(feats)
    feats -= feats.max(axis=0)  # dB below each band's loudest unit
    lengths = np.array(lengths)
    # leading/trailing near-silence (a recorder left running, a fade tail)
    # is not a section: keep it out of the merge so it can't drag the
    # neighbouring section's level down, then give its time back at the end
    quiet = feats.mean(axis=1) < -40.0
    lead = 0
    while lead < len(feats) - 1 and quiet[lead]:
        lead += 1
    tail = len(feats)
    while tail > lead + 1 and quiet[tail - 1]:
        tail -= 1
    segs = [[a + lead, b + lead] for a, b in _merge_segments(feats[lead:tail], lengths[lead:tail], threshold, min_units)]
    segs[0][0], segs[-1][1] = 0, len(feats) - 1
    core = slice(lead, tail)

    sections = []
    for i0, i1 in segs:
        j0, j1 = max(i0, core.start), min(i1, core.stop - 1)
        m = np.average(feats[j0:j1 + 1], axis=0, weights=lengths[j0:j1 + 1])
        sections.append({
            "start": round(float(edges[i0]), 3),
            "end": round(float(edges[i1 + 1]), 3),
            "label": "",
            "_fill": float(m.mean()),
            "_low": float(m[0]),
            "_rise": float(feats[j1].mean() - feats[j0].mean()),
        })
    top = max(sc["_fill"] for sc in sections)
    top_low = max(sc["_low"] for sc in sections)
    for sc in sections:
        sc["energy"] = round(float(np.clip(1.0 + sc["_fill"] / 20.0, 0.0, 1.0)), 3)

    drops: list[float] = []
    main_drop = None
    if tempo["has_grid"]:
        for sc in sections:
            if sc["_fill"] >= top - 1.0 and sc["_low"] >= top_low - 3.0:
                sc["label"] = "drop"
            elif sc["_low"] < top_low - 10.0:
                sc["label"] = "breakdown"
            else:
                sc["label"] = "groove"
        for i, sc in enumerate(sections[:-1]):
            if sc["label"] != "drop" and sections[i + 1]["label"] == "drop" and sc["_rise"] > 2.0:
                sc["label"] = "build"
        idx = [i for i, sc in enumerate(sections) if sc["label"] == "drop"]
        if idx:
            for sc in sections[:idx[0]]:
                if sc["label"] in ("groove", "breakdown"):
                    sc["label"] = "intro"
            for sc in sections[idx[-1] + 1:]:
                if sc["label"] in ("groove", "breakdown"):
                    sc["label"] = "outro"
        # main drop: the biggest entrance (largest jump in fullness); a later
        # drop has to be clearly bigger to beat the first one
        best = None
        for i in idx:
            jump = sections[i]["_fill"] - (sections[i - 1]["_fill"] if i > 0 else -40.0)
            if best is None or jump > best + 1.5:
                best, main_drop = jump, sections[i]["start"]
        drops = [sections[i]["start"] for i in idx]
    else:
        for sc in sections:
            sc["label"] = "peak" if sc["_fill"] >= top - 1.5 else ("quiet" if sc["_fill"] < top - 8.0 else "swell")
        peak = max(sections, key=lambda sc: sc["_fill"])
        main_drop = peak["start"]

    # adjacent sections that ended up with the same label read as one
    merged: list[dict] = []
    for sc in sections:
        if merged and merged[-1]["label"] == sc["label"]:
            prev = merged[-1]
            w0, w1 = prev["end"] - prev["start"], sc["end"] - sc["start"]
            prev["energy"] = round((prev["energy"] * w0 + sc["energy"] * w1) / (w0 + w1), 3)
            prev["end"] = sc["end"]
        else:
            merged.append(sc)
    for sc in merged:
        for k in ("_fill", "_low", "_rise"):
            sc.pop(k, None)
    if tempo["has_grid"]:
        drops = [sc["start"] for sc in merged if sc["label"] == "drop"]
        if main_drop is not None and main_drop not in drops:
            main_drop = max((d for d in drops if d <= main_drop), default=main_drop)
    return merged, drops, main_drop


def analyze(path: Path, genre: str, recipe: dict) -> dict:
    require_python({"numpy": "numpy", "scipy": "scipy", "librosa": "librosa"})
    import librosa
    import numpy as np

    info = probe(path)
    flags: list[str] = []
    print(f"analyzing {path.name} ({fmt_time(info['duration'])}, {info['sample_rate']} Hz, {info['channels']} ch)...")
    y_st, _ = librosa.load(str(path), sr=SR, mono=False)
    y = np.mean(y_st, axis=0) if y_st.ndim == 2 else y_st

    tempo, grid = estimate_grid(y, recipe, flags)
    if not recipe["expect_grid"] and tempo["has_grid"]:
        flags.append(f"found a steady {tempo['bpm']} BPM grid in a genre that usually has none -- using it")
    key = estimate_key(y)
    if key["confidence"] is not None and key["confidence"] < 0.05:
        flags.append(f"key is ambiguous ({key['name']} vs {key.get('runner_up')}) -- treat the key tag as a guess")

    loud = ebur128(path)
    st = astats(path)
    loud.update({
        "sample_peak_dbfs": st["sample_peak_dbfs"],
        "plr_db": round(loud["true_peak_dbtp"] - loud["integrated_lufs"], 1) if loud["integrated_lufs"] not in (None, float("-inf")) else None,
        "dc_offset": st["dc_offset"],
        "samples_at_peak": st["peak_count"],
    })
    if st["sample_peak_dbfs"] is not None and st["sample_peak_dbfs"] > -0.1 and (st["peak_count"] or 0) > 8:
        flags.append(f"{int(st['peak_count'])} samples pinned at {st['sample_peak_dbfs']} dBFS -- likely clipping in the recording")
    if st["dc_offset"] is not None and abs(st["dc_offset"]) > 0.001:
        flags.append(f"DC offset {st['dc_offset']:.4f} -- the mastering high-pass removes it")

    stereo, tilt, phone = stereo_and_spectrum(y_st, recipe, flags)
    sections, drops, main_drop = detect_sections(y, tempo, grid, recipe)

    return {
        "version": ANALYSIS_VERSION,
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(path),
        "genre": genre,
        "duration": round(info["duration"], 3),
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "bits": info["bits"],
        "tempo": tempo,
        "grid": grid,
        "key": key,
        "loudness": loud,
        "stereo": stereo,
        "phone": phone,
        "tilt": tilt,
        "sections": sections,
        "drops": drops,
        "main_drop": main_drop,
        "flags": flags,
    }


def summary_lines(a: dict) -> list[str]:
    t, k, l, s, p = a["tempo"], a["key"], a["loudness"], a["stereo"], a["phone"]
    lines = [f"genre:     {a['genre']}   duration {fmt_time(a['duration'])}"]
    if t["has_grid"]:
        sub = f" (double-time grid {t['bpm_fullres']} kept internally)" if t["subdivision"] > 1 else ""
        lines.append(f"tempo:     {t['bpm']} BPM{sub}  grid score {t['grid_score']}")
        lines.append(f"grid:      bar = {a['grid']['bar_seconds']:.3f}s, first downbeat {a['grid']['first_downbeat']}s")
    else:
        lines.append(f"tempo:     no grid (times in seconds; grid score {t['grid_score']})")
    lines.append(f"key:       {k['name']} ({k['camelot']})  confidence {k['confidence']}")
    lines.append(f"loudness:  {l['integrated_lufs']} LUFS  LRA {l['lra_lu']} LU  true peak {l['true_peak_dbtp']} dBTP  PLR {l['plr_db']} dB")
    lines.append(f"stereo:    corr {s['correlation']}  <{s['low_band_hz']} Hz corr {s['low_band_correlation']}  mono-sum loss {s['mono_sum_loss_db']} dB")
    lines.append(f"phone:     loses {p['phone_speaker_loss_db']} dB  bass ~{p['bass_fundamental_hz']} Hz, harmonics {p['bass_harmonics_db']} dB")
    lines.append(f"tilt:      {a['tilt']['slope_db_per_octave']} dB/octave")
    lines.append("sections:")
    for sec in a["sections"]:
        mark = "  <- main drop" if a["main_drop"] is not None and abs(sec["start"] - a["main_drop"]) < 1e-3 else ""
        lines.append(f"  {fmt_time(sec['start']):>7} - {fmt_time(sec['end']):>7}  {sec['label']:<9} energy {sec['energy']:.2f}{mark}")
    if a["flags"]:
        lines.append("flags:")
        lines.extend(f"  ! {f}" for f in a["flags"])
    return lines


def write_png(a: dict, path: Path) -> None:
    require_python({"matplotlib": "matplotlib"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"intro": "#8aa", "build": "#e9b949", "drop": "#d64545", "breakdown": "#5b8def",
              "outro": "#8aa", "groove": "#7a7", "peak": "#d64545", "swell": "#e9b949", "quiet": "#5b8def"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [1, 1.4]})
    for s in a["sections"]:
        ax1.axvspan(s["start"], s["end"], color=colors.get(s["label"], "#999"), alpha=0.6)
        ax1.text((s["start"] + s["end"]) / 2, 0.5, s["label"], ha="center", va="center", fontsize=8)
    if a["main_drop"] is not None:
        ax1.axvline(a["main_drop"], color="black", lw=2)
    ax1.set_xlim(0, a["duration"]); ax1.set_yticks([]); ax1.set_xlabel("seconds")
    t = a["tempo"]
    ax1.set_title(
        f"{Path(a['source']).name} -- {a['genre']}, "
        + (f"{t['bpm']} BPM" if t["has_grid"] else "no grid")
        + f", {a['key']['name']} ({a['key']['camelot']}), {a['loudness']['integrated_lufs']} LUFS"
    )
    tl = a["tilt"]
    ax2.bar([str(c) for c in tl["octave_centers_hz"]], tl["octave_levels_db"], color="#5b8def")
    ax2.set_ylabel("dB (rel. loudest octave)"); ax2.set_xlabel("octave band (Hz)")
    ax2.set_title(
        f"tilt {tl['slope_db_per_octave']} dB/oct   low-band corr {a['stereo']['low_band_correlation']}   "
        f"phone loss {a['phone']['phone_speaker_loss_db']} dB"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Track or Bluebox mix (WAV or anything ffmpeg reads)")
    p.add_argument("--genre", help="ambient | trap | techno | psytrance (required -- not guessed)")
    p.add_argument("--out-dir", type=Path, help="Output folder (default: <stem>_music/)")
    p.add_argument("--png", action="store_true", help="Also write a one-page analysis PNG (needs matplotlib)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    recipes = load_recipes()
    recipe = genre_recipe(recipes, args.genre)
    preflight(("ebur128", "astats"))
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        die(f"file not found: {src}")
    out_dir = (args.out_dir or src.parent / f"{src.stem}_music").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    a = analyze(src, args.genre, recipe)
    out = out_dir / "music_analysis.json"
    write_json(out, a)
    print("\n" + "\n".join(summary_lines(a)))
    print(f"\nWrote {out}")
    if args.png:
        png = out_dir / "music_analysis.png"
        write_png(a, png)
        print(f"Wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
