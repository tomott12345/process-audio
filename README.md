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

## Layout

- `process_field_wav.py` / `process_field_wav.sh` -- the main pipeline
- `analyze_windows.py` -- clean-run / bed-selection analysis
- `loop_crossfade.py` -- equal-power seamless loop builder
- `batch_process.py` -- run the pipeline over a folder or CSV manifest
- `recipes.json` -- per-label EQ chains, loudness targets, loop policy,
  clean-run detector tuning
- `youtube-mux.md` -- listing/thumbnail conventions
- `ROBUSTNESS_PLAN.md` -- the analysis this rewrite was built from
