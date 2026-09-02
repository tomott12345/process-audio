# process-audio

A small ffmpeg/Python pipeline that turns raw field recordings (rain, thunder,
crickets/insects, birds, brooks, waterfalls, mixed ambience) into cleaned,
loudness-normalized "long" and "Short" masters for YouTube -- optionally
muxed with a still image into ready-to-upload MP4s.

It never deletes your source file, never guesses an EQ recipe for content it
doesn't recognize, and keeps the tunable parts (EQ chains, loudness targets,
clean-run detection thresholds) in `recipes.json` rather than buried in code.

## Requirements

- `ffmpeg` / `ffprobe` on your PATH (with `afftdn`, `adeclick`, `adeclip`,
  `acrossfade`, `alimiter`, `loudnorm`, `equalizer` -- a normal full-featured
  build has all of these)
- Python 3 with `numpy` and `scipy`

The script checks all of this itself at startup and fails with a clear
message (and an install hint) if anything is missing.

## Quick start

```bash
./process_field_wav.sh take.wav --label rain --place porch --long clean
```

`--start`/`--end` override the automatic bed selection. `--start` is seconds to trim from the beginning (e.g. `5`); `--end` takes either seconds or an `mm:ss` / `hh:mm:ss` timestamp to cut off at:

```bash
./process_field_wav.sh take.wav --label thunder --place porch --start 5 --end 3:45
```

If you skip `--label`/`--place` (or pass `--ask`), it prints the questions it
needs answered instead of guessing:

```bash
./process_field_wav.sh take.wav --ask
```

Preview the chosen bed, EQ chain, and loudness targets without rendering
anything (handy before committing to a long render):

```bash
./process_field_wav.sh take.wav --label birds --place woods --plan-only
```

Loop the long master to a specific length -- an hour, say, for a background
video -- instead of leaving it at native bed length:

```bash
./process_field_wav.sh take.wav --label rain --place porch \
  --loop force --loop-target 1:00:00
```

`--loop-target` accepts seconds or an `mm:ss`/`hh:mm:ss` timestamp. It reuses
the same two loop-construction methods as the Short, resolved by `--loop`
the same way: hard-splice tiling (short ~12ms linear splices, no audible
dip) for correlated water content -- rain/thunder/water/brook -- or whenever
you pass `--loop force`; the equal-power wrap otherwise. If the clean bed is
already at or past the target length, it's just trimmed to exactly that
length rather than extended. Loudness normalization, fades, and the limiter
are applied once to the finished full-length file, not per repeat.

Process a whole folder or a mixed-content manifest at once:

```bash
./batch_process.py --dir ./card_dump --label insects --place woods
./batch_process.py --manifest takes.csv
```

## Content types

`rain | thunder | insects | mixed | water | brook | birds | waterfall` --
each with its own EQ chain, loudness targets, and bed-selection strategy in
`recipes.json`. Add a new content type by adding an entry there; the script
refuses to run (with a clear error) if a label is missing a required key,
rather than silently reusing another label's recipe.

`birds` and `waterfall` are marked `"untested": true` in `recipes.json` --
they're a reasonable starting point but haven't been validated against real
takes yet. A/B them against a few real recordings before trusting them
unattended, and tighten the recipe (or the `clean_run_detector` factors) as
needed.

## How bed selection works

`analyze_windows.py` scores the recording in 1-second windows and flags a
window "dirty" (handling noise, wind rumble, a stray thump) *relative to that
file's own measured baseline* rather than a fixed absolute loudness -- so a
continuously loud waterfall or a recording full of legitimate bird chirps
doesn't fail to find a usable "clean" bed just because it isn't a quiet porch
recording. Genuine digital clipping is still flagged on an absolute basis.

`process_field_wav.py` then picks a bed from the clean runs using the
strategy in the label's recipe: the longest clean run (weather/water), the
highest-activity run (insects/birds), or a comparison of the two (mixed).

## Removing unwanted noise or a voice/foreground

Two optional, narrower alternatives to full "stem isolation" (which isn't
realistic for nature-sound content -- separating rain from crickets from
birds with off-the-shelf tools produces artifact-heavy results, so this
pipeline doesn't attempt it). These solve two specific, tractable problems
instead:

**A known, steady noise (hum, hiss, distant traffic, an AC unit)** --
`denoise_profile.py` samples a few seconds of *just that noise* (a stretch
with no wanted ambience) and spectrally subtracts its profile from the rest
of the file, via the `noisereduce` library. Point it at a span within your
already-trimmed bed:

```bash
./process_field_wav.sh take.wav --label rain --place porch \
  --denoise-profile-start 0 --denoise-profile-end 3
```

`--denoise-profile-start`/`--denoise-profile-end` are timestamps *within the
trimmed bed* (0 = start of the bed you end up with, after `--start`/`--end`
or automatic bed selection), not the raw source file. If the noise-only
sample lives outside the bed (or in a separate recording entirely), point at
it directly instead:

```bash
./process_field_wav.sh take.wav --label rain --place porch \
  --denoise-profile-file hum_only.wav
```

Add `--denoise-non-stationary` if the noise's character drifts over time
(default assumes it's steady), and `--denoise-prop-decrease 0.6` (0-1) to
subtract less aggressively if full removal (`1.0`, the default) eats into
the wanted ambience. Only needs `noisereduce`
(`pip3 install noisereduce --break-system-packages`); the script checks for
it and tells you if it's missing, only when you actually use this flag.

Can also be run standalone: `python3 denoise_profile.py --help`.

**An unwanted foreground that behaves like speech (a voice, footsteps, a
passing conversation)** -- `remove_foreground.py` runs Demucs' pretrained
vocal-separation model on the trimmed bed and keeps the non-vocal stem. This
is a real source-separation model, so unlike EQ or trimming it can remove
foreground content that overlaps in *time* with the ambience you want to
keep. It was trained to split music into vocals vs. everything else, not
built for field recordings, so results vary -- best on a clear, close
voice/footsteps against a quieter bed; worst when the unwanted sound is
soft or blends into the ambience:

```bash
./process_field_wav.sh take.wav --label rain --place porch --remove-voice
```

This is a genuinely heavy, optional dependency: `torch` + `demucs` (a GB+
download) plus pretrained model weights (~80MB, fetched on first run), and
it can take a couple of minutes per file on a CPU. It's not part of the
normal preflight check -- only checked when you actually pass
`--remove-voice`:

```bash
pip3 install torch --break-system-packages
pip3 install demucs --break-system-packages
```

`--voice-model` picks a different Demucs model if you want to experiment
(default `htdemucs`). Can also be run standalone:
`python3 remove_foreground.py --help`.

Both flags run against the already-trimmed bed, before EQ, and REPORT.txt
records what ran (noise span or file used, prop_decrease, model name) so a
month from now you can see exactly what was applied to a given master.
`--plan-only` prints what *would* run without actually invoking either
(no heavy processing, no dependency check).

## Layout

- `process_field_wav.py` / `process_field_wav.sh` -- the main pipeline
- `analyze_windows.py` -- clean-run / bed-selection analysis
- `loop_crossfade.py` -- equal-power seamless loop builder
- `denoise_profile.py` -- noise-profile subtraction for a known steady noise
  (optional; needs `noisereduce`)
- `remove_foreground.py` -- Demucs-based voice/foreground removal (optional,
  heavy; needs `torch` + `demucs`)
- `batch_process.py` -- run the pipeline over a folder or CSV manifest
- `recipes.json` -- per-label EQ chains, loudness targets, loop policy,
  clean-run detector tuning
- `youtube-mux.md` -- listing/thumbnail conventions
- `ROBUSTNESS_PLAN.md` -- the analysis this rewrite was built from
