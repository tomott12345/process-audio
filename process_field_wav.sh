#!/usr/bin/env bash
# Field-audio wrapper. Label is required so EQ is not guessed.
#
# Usage:
#   process_field_wav.sh take.wav --label brook --place brook --long clean
#   process_field_wav.sh take.wav --label rain --place porch --loop never
#   process_field_wav.sh take.wav --label birds --place woods --plan-only
#   process_field_wav.sh take.wav --ask
#
# Labels and places come from recipes.json (edit that file to add a new
# content type instead of editing this script). As of 2026-09 that includes:
#   rain | thunder | insects | mixed | water | brook | birds | waterfall
# `birds` and `waterfall` are marked "untested" in recipes.json -- A/B them
# against a few real takes before trusting them unattended.
#
# Long:    --long clean  (longest clean span)  |  --long full
# Short:   --loop auto|never|force
#          auto  = never wrap correlated water/brook/rain/thunder (lotus_lake)
#          never = keep native bed length
#          force = short linear splices tiled up to the target length (never
#                  a 1.5 s esin wrap); refuses rather than under-delivering
#                  when the clean bed is too short to reach the target sanely
# --plan-only prints the chosen bed / EQ chain / loudness targets and exits
#          without rendering -- useful before committing to a long render.
#
# For a whole folder or card of takes at once, see batch_process.py.
#
# Lessons baked into the Python pipeline (do not re-learn per take):
# - Brook/water uses the rain-band EQ. Never the cricket 180 Hz high-pass.
# - Quiet window (example peak -38 dB) gets linear gain abs(peak)-1 before EQ.
# - loudnorm can emit 192 kHz and 0 dBFS sample peaks. Outputs are forced
#   to 48 kHz pcm_s24le stereo, then volume=-1.6dB + alimiter -1.5 dB.
# - XY water is correlated. A 1.5-2.5 s equal-power acrossfade dips at the seam.
#   Prefer a Short that is the native clean bed (2:27 is a valid Short).
# - Clean-run detection is relative to each file's own measured baseline, not
#   a fixed absolute level -- a loud waterfall or a bird take full of chirps
#   no longer fails to find a clean run just because it isn't a quiet porch
#   take. See recipes.json's clean_run_detector section to retune.
# - Never delete the source.
# - Mux: -t AUDIO_DUR on the still and the MP4. -shortest alone leaves dead air.
#
# Example from lotus_lake.WAV (Zoom H4essential XY, 297 s, 2026-08-30):
#   process_field_wav.sh lotus_lake.WAV \
#     --label brook --place brook --long clean \
#     --start 90 --end 237 --loop never \
#     --still-16x9 still_1920x1080.jpg \
#     --still-9x16 still_1080x1920.jpg
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/process_field_wav.py" "$@"
