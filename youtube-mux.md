# YouTube mux & listing notes

Still-frame + audio muxing and thumbnail cropping are implemented in
`process_field_wav.py` (`mux_still()`, `crop_still()`) -- pass `--still-16x9`
and `--still-9x16` and the script builds `long.mp4` / `short.mp4` plus
`thumb_1280x720.jpg` / `thumb_1080x1920.jpg` automatically. This file used to
duplicate those ffmpeg commands by hand; the two copies had drifted (different
overwrite flags, thread count, etc.), so this file no longer carries a second
implementation -- if you need the exact commands, read the script.

For a whole card of takes at once, see `batch_process.py --dir` or `--manifest`.

## Copy vs picture

Match the still to the recording. Rain-from-porch is not rain-on-glass.
Correct the frame rather than keep a prettier mismatch.

## Listing defaults from this workflow

Long rain
- Title shape -- place + object + weather (`Rain from the Porch | Two Cups and Distant Thunder`)
- Description first lines -- what you hear, then method, then length
- Hashtags at the bottom only, under 8

Insect Short
- Title shape -- `Night Insects from the Porch | Two Cups, No Talk`
- Search variant adds `Soft Cricket Sounds for Sleep (3 Minutes)`

Thumbnails -- 1280x720 landscape, 1080x1920 Shorts. Keep under 2 MB. Upload manually in Studio.
