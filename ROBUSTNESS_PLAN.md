# Field-Audio Pipeline — Robustness & Improvement Plan

Analysis of `process_field_wav.py`, `process_field_wav.sh`, `analyze_windows.py`, `loop_crossfade.py`, and `youtube-mux.md` in `process-audio/`, prepared 2026-09-01.

The pipeline is well designed in its bones: it separates analysis (`analyze_windows.py`) from EQ/loudness (`process_field_wav.py`) from loop construction (`loop_crossfade.py`), it refuses to guess a content label, it never deletes source files, and the `REPORT.txt` / comment trail (e.g. the `lotus_lake` notes) show real lessons already baked in — the "no 1.5s equal-power wrap on correlated water," "loudnorm can emit 192kHz / 0dBFS peaks," and "`-shortest` alone leaves dead air" notes are all correct and worth keeping. The gaps below are things I could confirm by reading the code and, in one case, by tracing the arithmetic — not guesses.

## 1. Confirmed bugs

**1a. `--loop force` silently produces a Short far shorter than the requested 180s, with no error or warning.**
This is the highest-priority fix. I traced `hard_splice_to_target()` numerically for a range of bed lengths (`edge=8`, `target=180`, `splice=0.012`):

| clean bed length | output length actually produced | shortfall |
|---|---|---|
| 23s | 14.0s | 166s short |
| 40s | 48.0s | 132s short |
| 90s | 148.0s | 32s short |
| 100s | 168.0s | 12s short |
| 120s | 180.0s | none — works correctly |

The cause: when the requested extension (`need`) exceeds the interior bed length, the code clamps `b_start` back to `interior_start`, which makes slice B identical to slice A. The function then splices a clip to itself and returns whatever that adds up to — silently — instead of iterating to reach the target or raising an error. The failure threshold is a clean bed of roughly **106 seconds** (≈90s of usable interior after the 8s edge trim on each side); anything shorter than that and `--loop force` under-delivers. Nothing downstream catches it: `mux_still()`'s duration check compares the video against the (already-too-short) audio file, not against the 180s the user asked for, so a Short can ship at, say, 30 seconds with no error anywhere in the run. Given that crickets/insects and short takes are exactly the case `--loop force` exists for, this is worth fixing before anything else — either loop-tile the interior until the target is met (more consistent with how `loop_crossfade.py` already tiles a unit), or `die()` with a clear message when the bed can't reach target with a single splice.

**1b. `EQ_THUNDER_SHORT` is defined but never used.** `eq_chain()` routes `label == "thunder"` to `EQ_RAIN`, not `EQ_THUNDER_SHORT`. Either the thunder-specific curve was abandoned on purpose (fine, but delete the dead constant so it doesn't look load-bearing) or this is a regression — worth a decision either way.

**1c. Adding a new label without updating `eq_chain()` fails silently, not loudly.** The function's final `else` falls back to `EQ_RAIN` for any label it doesn't recognize. Today that branch is unreachable because argparse's `choices=LABELS` prevents an unmapped value from ever arriving — but the moment someone adds `"birds"` or `"waterfall"` to `LABELS` (see §2) without also adding a branch in `eq_chain()`, `targets()`, and `CORRELATED_WATER`, the script will quietly EQ a bird recording like rain instead of erroring. That directly contradicts the script's own stated principle ("EQ is band work only... does not guess"). Recommend replacing the final `else` with a `raise` so a missing recipe fails at the command line, not in the output file.

**1d. Leftover temp files on mid-splice failure.** In `hard_splice_to_target()`, if the `acrossfade` ffmpeg call fails, `tmp_a`/`tmp_b` are created but the `unlink()` calls are never reached (they're after the `check_call`, not in a `finally`). Minor, but it clutters the output folder after any failed run and makes retries confusing.

**1e. Vestigial path.** `SKILL_SCRIPTS = Path("/home/workdir/.grok/skills/field-audio/scripts")` is a fallback lookup location that doesn't exist on this machine (it looks like a leftover from wherever this pipeline was originally scaffolded). Harmless today because `SCRIPT_DIR` always wins, but it's dead weight worth deleting so the script doesn't imply a dependency that isn't real.

## 2. The bigger gap: birds and waterfalls aren't actually supported

You said the source material is often crickets, birds, and waterfalls, not just rain/thunder/insects/water/brook. Right now:

- **There is no `birds` label**, and no bird-appropriate EQ curve. A bird take has to be forced into `insects`, `mixed`, or `water`, none of which fit its spectral shape.
- **There is no `waterfall` label either**, distinct from `water`/`brook`. Continuous, broadband, loud rushing water behaves very differently from a quiet brook or intermittent rain.
- **`analyze_windows.py`'s "clean" thresholds are fixed absolute amplitudes, tuned to a specific quiet Zoom-recorder gain level**, not relative to each file's own loudness. I converted the constants to dBFS to check this:
  - `rumble` dirty above 0.00022 ≈ **‑73 dB**
  - `thump` dirty above 0.00018 ≈ **‑75 dB**
  - `mid` dirty above 0.00035 ≈ **‑69 dB**
  - `peak` (any sample) dirty above 0.012 ≈ **‑38 dBFS**
  - `rms` ("LOUD") dirty above 0.0018 ≈ **‑55 dBFS RMS**

  These are all very low thresholds — consistent with the "quiet window, peak ‑38dB" example already in the script's own comments. That's fine for a quiet porch-rain take, but it means: a **waterfall** recording, which is continuously loud broadband noise, will trip `LOUD`/`PEAK`/`MID` almost everywhere and likely register **zero clean runs**, silently falling back to "use the whole file after a 2s start trim" — not an actual best-bed selection. A **bird** recording will trip `PEAK` on essentially every chirp (bird calls routinely exceed ‑38 dBFS), so bird takes will also mostly fail clean-run detection. And the "insect" band (2500–9000 Hz) overlaps a lot of bird call range, so under `--label mixed` a birdy passage can get misread as an insect run and bias bed selection toward the wrong span.

  The fix isn't just "add more labels" — it's making the dirty/clean thresholds **relative to the file's own measured noise floor / percentile levels** (e.g., flag windows that are N dB above the file's own 10th-percentile RMS, rather than an absolute number), so the same detector works whether the source is a quiet brook at ‑50 dBFS or a loud waterfall at ‑15 dBFS. That single change would make the pipeline actually robust across recorders and gain levels, which is the core of what "more robust" should mean here.

**Recommended new recipes to add**, once the detector is fixed:
- `birds` label: an EQ that protects 2–8kHz transient detail (gentle, not the insect curve's presence boost, since bird calls are already forward and shouldn't be pushed further), and a `--repair`-style declick that's careful not to shave chirp attacks.
- `waterfall` label: distinct from `brook`/`water` — likely wants more low-mid control (waterfalls carry more low rumble than a shallow brook) and probably shouldn't be lumped into `CORRELATED_WATER`'s "never loop" rule by default, since a waterfall's broadband noise is usually much more loop-friendly than a stereo brook's transient splashes. Worth testing rather than assuming.

## 3. Engineering robustness (things that will bite you on some future file, not every file)

- **No preflight checks.** The script assumes `ffmpeg`/`ffprobe` are on `PATH` and that the ffmpeg build has `afftdn`, `adeclick`, `adeclip`, `acrossfade`, `alimiter`, `loudnorm`. If any are missing, the failure is a raw `CalledProcessError` traceback with no hint what went wrong. A one-time `ffmpeg -filters` / `which` check at startup with a clear error message would save real debugging time.
- **Whole-file in-memory decode for long recordings.** Both `analyze_windows.py` and `loop_crossfade.py` decode the *entire* source to float32 in RAM (`np.fromfile`). At 48kHz/stereo/float32 that's about **1.3 GB per hour** of audio. For a short clip that's nothing; for an all-night rain or thunderstorm recording several hours long, that's several GB just to look at 1-second windows or to pull a 30-second slice. `loop_crossfade.py` in particular decodes the whole file just to extract a short `--start`/`--end` slice — it should `ffmpeg -ss/-t` trim to the slice first, then decode only that.
- **Channel-count inconsistency.** `analyze_windows.py`'s decoder always forces `-ac 2` for measurement, but the actual render path (`copy_trim`/`ffmpeg_wav`) doesn't force a channel count — so a source with an unusual layout (mono lav mic, 4-channel ambisonic rig) could produce a master with a different channel count than what was analyzed, or than YouTube expects. Worth forcing `-ac 2` explicitly in the render chain too.
- **Hardcoded, un-tunable constants everywhere.** EQ curves, clean-run thresholds, loudness targets (`I`/`TP`/`LRA`), denoise strength (`nr=8` fixed regardless of content) all live as Python literals inside the script. As you add more content types (birds, waterfalls, wind, maybe owls or frogs down the line), editing code each time is more error-prone than editing data. A small JSON/YAML "recipe" file per label (thresholds + EQ chain + loudness targets + loop policy) would make new content types a config change, not a code change, and would make it much easier to keep `eq_chain()`/`targets()`/`CORRELATED_WATER` in sync (fixing 1c for good).
- **`youtube-mux.md` duplicates logic that already lives in `process_field_wav.py`.** The markdown's hand-written ffmpeg commands (`mux_still`, still-cropping) use slightly different flags than the Python versions (e.g. `-n` no-clobber in the doc vs `-y` overwrite in code, no `-threads 4` in the doc). They'll drift apart the next time one gets edited and not the other. Recommend trimming the doc to usage notes + the listing/title conventions (which are genuinely useful and not duplicated in code) and pointing at the script for the actual mux commands, rather than keeping two implementations.
- **No batch mode.** Every take is processed with one invocation. If you're regularly clearing a card with several field recordings on it, a thin wrapper that walks a folder and prompts (or reads a small manifest of `filename,label,place`) would remove the main repetitive friction.
- **No dry-run / analysis-only mode for long files.** Right now the only way to see the bed selection and EQ chain before committing to a multi-minute render is to run the whole pipeline. A `--plan-only` flag that prints the chosen bed, EQ chain, and loudness targets without rendering would speed up iterating on long-file settings.
- **No reproducibility metadata.** `REPORT.txt` records what was done but not when, or with which ffmpeg version — useful context to have a month later when comparing two takes that came out sounding different.

## 4. Suggested order of work

1. **Fix `hard_splice_to_target` (1a)** — this is the one that ships a wrong deliverable today with zero indication anything went wrong.
2. **Make clean-run detection relative, not absolute (§2)** — this is the change that actually unlocks birds and waterfalls; everything else about those content types is downstream of getting real bed selection working.
3. **Add `birds` and `waterfall` labels** with their own EQ/loudness recipes, once (2) makes bed selection trustworthy for them. Test against a few real takes of each before trusting it unattended.
4. **Move the fail-silent `else` in `eq_chain()`/`targets()` to a hard error (1c)**, so future label additions can't ship half-wired.
5. **Clean up dead weight**: unused `EQ_THUNDER_SHORT` (1b) or wire it up, vestigial `SKILL_SCRIPTS` path (1e), temp-file cleanup on failure (1d).
6. **Fix the memory/performance issues for long files** (§3) — matters more once you're running this on multi-hour ambient recordings.
7. **Extract a recipe config file** and add preflight checks, batch mode, and a dry-run flag — quality-of-life work that makes the first six items easier to maintain going forward.

Items 1 and 4 are correctness fixes (things silently produce wrong output today). Item 2 and 3 are the ones that actually address "crickets, birds, waterfalls" being real, common inputs rather than edge cases. The rest is durability work that pays off the more you run this pipeline.
