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
  `acrossfade`, `alimiter`, `loudnorm`, `equalizer`, `deesser`,
  `acompressor`, `silenceremove` -- a normal full-featured build has all of
  these)
- Python 3, plus the packages in `requirements.txt`:

  ```bash
  pip3 install -r requirements.txt --break-system-packages
  ```

  That's `numpy`/`scipy` for the nature pipeline's bed analysis and looping
  (`process_speech_wav.py` doesn't need either -- it's pure ffmpeg). Two
  features have their own heavy, genuinely optional dependencies, listed
  separately in `requirements-optional.txt` rather than bundled in here so
  installing the repo doesn't pull in a multi-GB `torch` download by
  default: `--denoise-profile-*` needs `noisereduce`, and
  `--remove-voice`/`--remove-voice-at` needs `torch` + `demucs`. Install
  either only if you're using the flag that needs it:

  ```bash
  pip3 install -r requirements-optional.txt --break-system-packages
  ```

Each script checks its own dependencies at startup (or lazily, only when a
flag that needs an optional one is passed) and fails with a clear message
-- including an install hint -- if anything is missing, rather than a raw
traceback.

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

`rain | thunder | insects | mixed | water | brook | birds | waterfall | wind |
fire | ocean | frogs` -- each with its own EQ chain, loudness targets, and
bed-selection strategy in `recipes.json`. Add a new content type by adding an
entry there; the script refuses to run (with a clear error) if a label is
missing a required key, rather than silently reusing another label's recipe.

`birds`, `waterfall`, `wind`, `fire`, `ocean`, and `frogs` are marked
`"untested": true` in `recipes.json` -- they're a reasoned starting point
(EQ bands picked for each content type's actual spectral shape, not copied
from a neighbor) but haven't been validated against real takes yet. A/B them
against a few real recordings before trusting them unattended, and tighten
the recipe (or the `clean_run_detector` factors) as needed.

A label's `activity_band: [lo_hz, hi_hz]` (optional; defaults to 2500-9000,
tuned for insects/birds) controls what frequency range counts as "activity"
for `bed_selection: activity`. `frogs` overrides this to `[300, 2500]`,
since typical frog/toad calls sit well below cricket/cicada chirp range --
without this override, activity-based bed selection would just find
whatever run has the most incidental high-frequency insect energy, not the
best frog chorus. Species vary a lot in call pitch, so treat this as a
starting point to tune per recording, same as the EQ.

## How bed selection works

`analyze_windows.py` scores the recording in 1-second windows and flags a
window "dirty" (handling noise, wind rumble, a stray thump) *relative to that
file's own measured baseline* rather than a fixed absolute loudness -- so a
continuously loud waterfall or a recording full of legitimate bird chirps
doesn't fail to find a usable "clean" bed just because it isn't a quiet porch
recording. Genuine digital clipping is still flagged on an absolute basis.

`process_field_wav.py` then picks a bed from the clean runs using the
strategy in the label's recipe: the longest clean run (weather/water/wind/
ocean), the highest-activity run (insects/birds/fire/frogs -- each over its
own `activity_band`), or a comparison of the two (mixed).

## Spoken word: interviews and podcasts

`process_speech_wav.py` is a separate, standalone script for dialogue --
interviews, panels, solo podcast episodes -- not a mode of the nature
pipeline above. That's deliberate: this pipeline's whole design (bed
selection, looping) is built around *ambient texture* that's repeat-tolerant
and doesn't need every second kept. Speech is the opposite -- every word
matters and it isn't repeat-tolerant -- so the speech script skips bed
selection and looping entirely and just processes the recording (or your
`--start`/`--end` span) once, start to finish.

```bash
python3 process_speech_wav.py interview.wav --preset apple
```

Chain: highpass (rumble/plosives) -> gentle presence EQ -> de-esser (tame
sibilance) -> gentle compressor (evens out level swings between mic
distance/energy/speakers) -> two-pass loudnorm to a podcast loudness target
-> short fades -> true-peak limiter. `--no-compress`/`--no-deess` skip
either step if you'd rather handle it yourself.

`--preset` picks a loudness target: `apple` (-16 LUFS), `spotify` (-19
LUFS), `youtube` (-14 LUFS), or `general` (-16 LUFS, the default). These
are commonly-cited platform targets, not fetched from a live spec -- confirm
against each platform's current published loudness guidance before assuming
they haven't moved. `--target-i`/`--target-tp`/`--target-lra` override
individual values if you need something else. `--mono` downmixes before
normalizing, since several platforms define their loudness target for mono.

`--trim-silence` trims only leading/trailing silence (never touches
mid-file pauses -- it uses ffmpeg's `silenceremove` in "leading edge" mode
on the file and its reverse, specifically because the naive stop-at-silence
approach mistakes an ordinary conversational pause for the end of the
recording and truncates the file there). `--silence-threshold` (default
-45dB) tunes what counts as silence.

`--plan-only` prints the resolved chain and targets without rendering.

## Removing unwanted noise or a voice/foreground

Two optional, narrower alternatives to full "stem isolation" (which isn't
realistic for nature-sound content -- separating rain from crickets from
birds with off-the-shelf tools produces artifact-heavy results, so this
pipeline doesn't attempt it). These solve two specific, tractable problems
instead. Both live in `requirements-optional.txt` (see Requirements above)
if you'd rather install everything for a feature at once than run the
individual `pip3 install` commands below:

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

To clean only specific moments instead of the whole bed -- footsteps at a
known timestamp, a cough, a passing conversation -- use `--remove-voice-at`
instead of `--remove-voice`. It only runs Demucs on a short padded window
around each moment, then splices the cleaned span back in with a short
crossfade; everything else in the file is left completely untouched, and
it's much faster than processing the whole recording:

```bash
./process_field_wav.sh take.wav --label rain --place street --long full \
  --remove-voice-at 15 --remove-voice-at end --remove-voice-pad 4
```

Repeatable for multiple moments; `end` means the tail of the chosen bed.
Timestamps are in the same timeline as `--start`/`--end` -- the original
source file -- not the trimmed bed, since bed selection (or `--long full`'s
own 2s lead-trim) can already shift where the bed's t=0 actually falls
relative to your source recording. `--remove-voice-pad` (default 4s) is
how much padding to include on each side of the timestamp -- give the
footsteps room to actually sit inside the window with clean margin at the
edges for the splice. `--remove-voice` and `--remove-voice-at` are mutually
exclusive.

Both flags run against the already-trimmed bed, before EQ, and REPORT.txt
records what ran (noise span or file used, prop_decrease, model name) so a
month from now you can see exactly what was applied to a given master.
`--plan-only` prints what *would* run without actually invoking either
(no heavy processing, no dependency check).

## Audio visualization

`visualize_wav.py` renders an audio-reactive visualization video from a WAV
file -- a radial spectrum (bars pulsing outward from a center circle) or a
classic bar-graph equalizer -- and muxes it with the original audio into a
single mp4 via ffmpeg. Useful for turning a track into a YouTube Short/Reel
or a longer landscape upload without a separate video editor.

```bash
python3 visualize_wav.py input.wav output.mp4 --format shorts --style radial --title "Track Name"
python3 visualize_wav.py input.wav output.mp4 --format landscape --style bars
```

`--format` is `shorts` (1080x1920), `landscape` (1920x1080), or `square`
(1080x1080). `--max-seconds N` renders only the first N seconds, useful for
a quick preview before committing to a full render. This is optional and
not wired into the main pipeline -- see requirements-optional.txt for its
dependencies (librosa, Pillow) and install with:

```bash
pip3 install librosa pillow --break-system-packages
```

The radial style has a stack of look/style options, all on by sensible
defaults: anti-aliasing (`--supersample`), motion trails (`--trail-decay`,
`--no-trails`), bloom/glow (`--glow-strength`, `--glow-radius`,
`--no-glow`), particles at active bar tips (`--particles`,
`--particle-rate`, `--max-particles`), a beat-reactive zoom on detected
onsets (`--beat-punch`, `--punch-strength`), and kaleidoscope/mandala
symmetry (`--symmetry N`, e.g. `--symmetry 6`). A center emoji that pulses
with the kick drum is also available for the radial style:

```bash
python3 visualize_wav.py input.wav output.mp4 --format shorts --emoji "❤️"
```

`--emoji-beat` picks what drives its pulse size -- `kick` (default,
bass-restricted onset detection so it responds to the kick drum rather
than every hit), `rms` (overall loudness), or `off` (static size).
`--emoji-size` and `--emoji-pulse` control base size and how much it grows
on a hit. Rendering uses a system color-emoji font when one is available
(Noto Color Emoji on Linux, Apple Color Emoji on macOS); a heart-like
request (`"❤️"`, `"♥"`, `"heart"`) falls back to a hand-drawn glowing
vector heart if no such font is found, so that case always works. Other
emoji need a color font present on the machine running the script -- it
warns and skips the emoji rather than failing if none is found.

Run `python3 visualize_wav.py --help` for the full flag list and defaults.

## Layout

- `process_field_wav.py` / `process_field_wav.sh` -- the main nature-ambience
  pipeline
- `process_speech_wav.py` -- standalone dialogue/podcast processing (no bed
  selection, no looping)
- `analyze_windows.py` -- clean-run / bed-selection analysis
- `loop_crossfade.py` -- equal-power seamless loop builder
- `denoise_profile.py` -- noise-profile subtraction for a known steady noise
  (optional; needs `noisereduce`)
- `remove_foreground.py` -- Demucs-based voice/foreground removal (optional,
  heavy; needs `torch` + `demucs`)
- `batch_process.py` -- run the pipeline over a folder or CSV manifest
- `visualize_wav.py` -- audio-reactive visualization video (radial or bar
  spectrum), muxed with the source WAV into an mp4 (optional; needs
  `librosa` + `pillow`)
- `recipes.json` -- per-label EQ chains, loudness targets, loop policy,
  clean-run detector tuning
- `requirements.txt` -- core Python dependencies (numpy/scipy)
- `requirements-optional.txt` -- heavy, feature-specific dependencies
  (noisereduce, torch, demucs) -- install only what you need
- `youtube-mux.md` -- listing/thumbnail conventions
- `ROBUSTNESS_PLAN.md` -- the analysis this rewrite was built from
