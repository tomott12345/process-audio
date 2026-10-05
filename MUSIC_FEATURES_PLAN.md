# EDM / DAWless Music Features Plan

Plan for extending `process-audio/` from field recordings and speech into
music made on hardware synths and DAWless rigs: grooveboxes, drum machines,
and a mixer or multitrack recorder in place of a DAW. Prepared 2026-10-05.

## Where the repo stands today

The repo covers three things well: the nature-ambience pipeline
(`process_field_wav.py`), dialogue (`process_speech_wav.py`), and the
audio-reactive visualizer (`visualize_wav.py`). The only music-aware code is
in the visualizer: kick-restricted onset detection and per-band rescaling.
Nothing treats a track as *music*. Nothing knows about tempo, bars, key,
stems, or club loudness, and there's no mastering chain for a finished
track. The speech script's chain (de-esser, podcast LUFS targets) is wrong
for EDM, and the nature pipeline's bed-selection and loop logic assumes
ambience that tolerates repeats.

Music gets the same treatment speech did: **separate scripts, not a mode of
the nature pipeline.** The repo's conventions carry over unchanged:

- never delete or overwrite the source
- don't guess: ask for the genre/preset the same way `--label` is asked for
- tunables live in JSON (`music_recipes.json`), not Python literals
- `--plan-only` on everything, plus a `REPORT.txt` that records what ran
- heavy dependencies are optional and checked only when their flag is used
- a startup preflight that names any missing ffmpeg filter or binary

## Your rig: 1010music Bluebox (multitrack + live mix)

The Bluebox records each input as its own track and you mix on it, so
every session gives you two things: **per-track stems** and **the stereo
mix you performed**. This changes the plan:

- **Phase 4 becomes core.** Stems are the main input, not an edge case.
- **No alignment or drift correction is needed.** Every track comes
  through one device's converters on one clock, so the tracks are
  sample-aligned. You've confirmed there's no drift between devices on long
  jams, so the beat grid can assume a constant tempo, and time-stretching
  (`rubberband`) is dropped from the plan.
- **There are two workflows, and both need support:**
  1. **Master the Bluebox mix.** The performance is good and only needs
     finishing. This goes straight to Phase 2.
  2. **Re-mix from stems (repair mode).** The live mix had a problem: a
     channel too hot in the drop, a hat that clipped, a bass that should
     have been ducked. The script rebuilds the mix offline from the stems,
     fixes only what's wrong, then masters it.
- **Bluebox project ingest** (new, first step of Phase 4). Read a project
  folder off the SD card, map each track file to a stem role (kick, bass,
  hats, pads, lead, fx), and record sample rate, bit depth, and length.

**To verify on a real project before building ingest** (no Bluebox files
are on this Mac yet, so these are unconfirmed):
- folder layout and track file naming on the SD card
- whether per-track recordings are pre- or post-channel-EQ/FX. The Bluebox
  reverb/delay sends are probably only in the stereo mix, which decides
  whether repair mode needs its own reverb/delay or must blend in the
  original mix's effects.
- whether stereo inputs land as one stereo file or two mono files
- recording format (bit depth / sample rate, possibly 32-bit float)
- whether channel names set on the device appear in the files or in a
  project/metadata file

## What's already available on this machine

ffmpeg 9.0.1 here has everything needed for an all-ffmpeg mastering chain:
`acrossover` (multiband split), `acompressor`, `sidechaincompress`,
`adynamicequalizer`, `asoftclip`, `alimiter`, `loudnorm`, `ebur128`,
`astats`, `stereotools`, `extrastereo`, `firequalizer`, `aexciter`.

The `rubberband` time-stretch filter is missing from this build, but with
no drift on your rig nothing in the plan needs it.

`librosa` 0.11.0 is already installed for the visualizer. It handles beat
tracking, key estimation, and segmentation, so Phases 1–3 add no new heavy
dependencies.

---

## Phase 1: Analysis foundation (`music_analyze.py`)

Almost every later feature needs a beat grid, so this comes first.
It's read-only and makes no audio changes.

1. **Tempo + beat/downbeat grid.** Use `librosa.beat.beat_track` with a
   tempo prior set by genre, because the beat trackers' most common error
   is landing on half or double the real tempo:
   - techno ~125–140
   - psytrance ~138–148
   - trap ~65–80, reported as the half-time feel (e.g. 70). If the tracker
     locks onto double time (~140), halve it. The beat grid stays at full
     resolution internally, so hat rolls and visual sync still land
     correctly.
   - ambient: often there is no reliable beat. Report "no grid", and every
     later step falls back to seconds instead of bars rather than
     inventing a grid.

   Your gear shows no drift on long jams, so assume a constant tempo and
   report confidence. Estimate downbeats with bass-band onset strength on
   4/4 boundaries.
2. **Key detection.** Correlate a chroma profile against Krumhansl-Schmuckler
   templates. Output the key in standard notation *and* Camelot (e.g.
   `8A`), the notation DJs mix by.
3. **Loudness/QC report.** Integrated + short-term LUFS (from ffmpeg
   `ebur128`), LRA, true peak, PLR/crest factor, DC offset (`astats`), and
   clipped-sample count.
4. **Mix checks:**
   - **Mono compatibility:** level lost when summed to mono, and phase
     correlation overall and in the low band (<150 Hz).
   - **Low-end phase:** flag stereo content below ~120 Hz, which is common
     with stereo hardware synths, chorus/ensemble on bass patches, and
     wide reverbs.
   - **Spectral tilt:** compare against a reference curve per genre.
   - **Phone-speaker check:** most YouTube Shorts and Instagram plays happen
     on phone speakers, which reproduce almost nothing below ~150 Hz. Report
     how much of the track's energy sits only in the sub band. A trap 808
     or psy bassline that's pure sine sub disappears on a phone. The fix is
     harmonics/saturation on the bass stem (Phase 4), not more low end.
5. **Output:** a `music_analysis.json` that later scripts read, plus an
   optional one-page PNG: a loudness-over-time strip, the spectrum against
   the genre curve, and a correlation meter.

**Why it matters for DAWless:** with no DAW there are no meters or
analyzers on the master bus. This script plays that role after the fact.

## Phase 2: Mastering chain (`process_music_wav.py` + `music_recipes.json`)

A music sibling of `process_speech_wav.py`. Chain, in order (each step can
be turned off and each is set by the recipe):

1. DC-offset removal plus a subsonic high-pass (~25–30 Hz). Hardware synths,
   especially analog ones, often carry DC or infrasonic content.
2. **Mono-bass ("elliptical EQ")** below a crossover set per genre
   (default ~120 Hz). This is the most useful EDM-specific step for a
   hardware rig: `acrossover` splits the signal, the low band is summed to
   mono, then the bands are recombined.
3. Corrective/tonal EQ from the recipe (low shelf, mud cut ~250–400 Hz,
   air shelf).
4. Optional **dynamic EQ** (`adynamicequalizer`) to tame resonant synth
   peaks only when they spike: an acid 303 squelch, a resonant filter
   sweep.
5. **Glue compression**: a gentle bus compressor (2:1, slow attack so kick
   transients pass through).
6. Optional **multiband compression** (3-band `acrossover` → per-band
   `acompressor` → `amix`), off by default.
7. Optional stereo width on mids/highs only (`stereotools`), never on the
   low band.
8. **Soft clip** (`asoftclip`) before the limiter. EDM masters commonly
   shave kick peaks this way to gain loudness without limiter pumping.
9. Loudness: measure, apply one static gain, soft-clip, then a true-peak
   limiter at 4x oversampling; re-measure and correct. (As built, this
   replaced the planned linear `loudnorm`: linear mode silently falls back
   to dynamic, which pumps, whenever the gain would exceed the TP ceiling.)
10. **Deliverables.** One render produces:
    - a 24-bit/48k WAV master (48k is what YouTube/Instagram video uses,
      so the video mux doesn't resample)
    - a 16-bit WAV with triangular high-pass dither
      (`aresample=dither_method=triangular_hp`) for archiving or anything
      that needs 16-bit
    - BPM/key/genre/title tags from Phase 1 written into the files and the
      `REPORT.txt`
    - the video formats are covered in Phase 5

**Presets** (`--preset`), stored in `music_recipes.json`:

| preset | target LUFS | TP | mono-bass Hz | notes |
|---|---|---|---|---|
| `youtube` | -14 | -1.0 | genre | YouTube turns louder uploads down, so there's no gain from going hotter |
| `instagram` | -14 | -1.0 | genre | treated like YouTube until measured; see below |

Both platforms re-encode the audio to lossy AAC, so -1 dBTP leaves room for
inter-sample peaks the encoder adds. The -14 LUFS figure for YouTube is the
commonly cited normalization target. Instagram doesn't publish one, so
measure a few of your own uploads (download them back and run
`music_analyze.py`) before trusting either number. These are starting
points, not a spec.

**Genre recipes** (`--genre`) set the sound. The preset sets only the
loudness target:

| genre | mono-bass Hz | glue / dynamics | soft clip | notes |
|---|---|---|---|---|
| `ambient` | ~100 | light glue, keep dynamics; may finish quieter than -14 on purpose | off | no grid needed; wide stereo is fine above the crossover |
| `trap` | ~100 | moderate; fast enough to catch the 808 | on, gentle | protect hat/snare transients; add 808 harmonics for phones |
| `techno` | ~120 | slow-attack glue so the kick punches through | on | dynamic EQ on resonant acid/303 peaks |
| `psytrance` | ~150 | tight; little pumping | on | kick and rolling bassline alternate 16ths, so keep the low end tight and mono. No sidechain (the arrangement already separates them) |

Like the nature `untested` flag, ship the recipes marked
`"untested": true` until they've been A/B'd on real tracks.

**Optional reference matching:** `--reference pro_track.wav`, using the
`matchering` library (optional dependency), matches EQ curve and loudness
to a commercial track in the same genre.

## Phase 3: Live-jam workflow

DAWless sessions are usually long continuous recordings: a 45-minute
Bluebox jam. This phase turns that into usable material.

1. **Jam splitter (`split_jam.py`).** Find track boundaries in a long
   recording, using silences plus novelty/self-similarity segmentation
   (librosa) for jams that flow from one track into the next without
   stopping. Snap each boundary to the nearest downbeat (Phase 1 grid) and
   export each section as its own file. Ambient jams with no grid are
   split by novelty and silence alone, at seconds instead of bars. It asks
   before cutting, like
   `--ask`: it prints the proposed splits with timestamps, and you accept or
   override them with `--split-at`.
2. **YouTube chapters / tracklist.** Generate a `00:00 Track 1 (128 BPM,
   8A)` chapter list from the splits, for uploading a full live set
   alongside the visualizer.
3. **Bar-aligned trimming.** Use `--start-bar`/`--end-bar` instead of
   seconds, so cuts land on the "one" instead of mid-bar.
4. **Loop/sample export (`export_loops.py`).** Cut 1/2/4/8/16-bar loops at
   exact grid positions, with zero-crossing-snapped edges and a ~2 ms
   micro-fade. Name them `kit_128bpm_8A_bar033-040.wav`, ready to load back
   into a sampler.
5. **Drop/breakdown detection.** Label sections (intro / build / drop /
   breakdown / outro) from energy and bass-band envelopes. For ambient, the
   label becomes "swell" or "peak". This feeds the short-clip picker
   below, the chapter names, and the visualizer (Phase 5).
6. **Short-clip picker (`pick_clip.py`).** This is the most useful step
   for posting to Shorts and Reels: pick the best excerpt of a
   finished track and export it ready to post.
   - **Lengths:** 15/30/60/90s.
   - **Where it starts:** by default, a few bars before the main drop, so
     the build lands inside the clip. For ambient it uses the peak swell.
   - **Bar-aligned cuts** with a fast fade-in and a phrase-ending
     fade-out.
   - **Loop-friendly option:** cut on an exact 8/16-bar phrase so the
     clip loops seamlessly when the platform auto-repeats it. This reuses
     `loop_crossfade.py` ideas, but cuts on bars.
   - **`--plan-only`** prints the chosen window and the reasoning before
     rendering.
   - **Check limits at build time:** platform length caps for Shorts and
     Reels change often, so check current limits rather than hard-coding
     them.

## Phase 4: Multitrack and hardware-rig problems

This is the core of the Bluebox workflow, since every session produces
per-track stems.

1. **Bluebox project ingest (`ingest_bluebox.py`).** Point it at a project
   folder (or the whole SD card). It lists the tracks, guesses each stem
   role from the channel name, and asks you to confirm any it can't match,
   consistent with the repo's ask-don't-guess rule. It saves the mapping to
   a `stems.json` beside the project so later runs don't ask again. It also
   flags empty or silent tracks and clipped tracks.
2. **Repair-mode re-mix (`remix_stems.py`).** Rebuild the mix from stems
   with per-stem gain, plus time-ranged fixes:
   `--gain hats=-3dB@2:10-3:40`, `--mute fx@5:00-5:16`. You fix the one
   bad moment in the performance and keep everything else.
   - **Bluebox mix as the default reference.** Start from a gain match to
     the Bluebox stereo mix (least-squares fit of stem gains to the mix),
     so the offline re-mix sounds like your performance before you change
     anything.
   - **Effects.** If per-track files turn out to be pre-FX, blend in the
     original mix's ambience or add your own reverb/delay send.
3. **Per-stem cleanup presets** (`--stem kick|bass|hats|pads|lead|fx`):
   - high-pass everything except kick and bass
   - remove DC
   - gate noise between hits on drum stems
   - hum-notch for ground loops: auto-detect a 50 or 60 Hz fundamental
     and notch its harmonics (common with mixed USB/mains-powered gear)
   - remove the steady noise floor with the existing `denoise_profile.py`
     (sample a few seconds of the synth idling)
   - **bass harmonics for phones:** saturate a copy of the 808/bass stem,
     high-pass it at ~150 Hz, and blend it under the clean sub. The bass
     then reads on phone speakers without changing the sub. This is the
     main fix for trap and psy posted to Shorts and Reels.
4. **Sidechain ducking** (`sidechaincompress`). Duck bass/pads from the
   kick stem: classic EDM pump, done after the fact when the hardware rig
   couldn't do it live. On by default for `techno` and `trap`, off for
   `psytrance` and `ambient`.
5. **Stem → premaster sum.** Gain-stage and sum stems to a stereo
   premaster with headroom (-6 dBFS peak), then hand off to Phase 2.

## Phase 5: YouTube and Instagram output

You release to YouTube and Instagram, so finished output means a video.
`visualize_wav.py` already renders `shorts` (1080x1920), `landscape`, and
`square` formats. This phase connects it to the music pipeline.

1. **Beat-synced visualizer.** Pass `music_analysis.json` to
   `visualize_wav.py --grid` so beat punch and emoji pulse lock to the real
   beat grid instead of onset detection, and drops trigger a special
   effect (bigger shockwave, palette shift). Onset detection jitters on
   busy trap hats and psy 16th-note basslines; a grid doesn't. Ambient
   tracks (no grid) keep using `glowburst`/`wormhole`, which already
   follow loudness.
2. **Stem-driven visuals (stems you already have).** Drive separate
   visual elements from separate Bluebox stems: kick → beat punch,
   bass → ring/core size, hats → particles, lead → color. This is much
   cleaner than pulling bands out of a full mix, and it's something
   almost no off-the-shelf visualizer can do.
3. **Default style per genre:**
   - ambient → `glowburst` or `wormhole`
   - techno → `bars` or radial with `--symmetry`
   - psytrance → radial with high symmetry, mandala-style
   - trap → radial with the kick emoji
4. **One-command publish set (`release.py`).** A finished master produces:
   - a full-length landscape video for YouTube, with chapters if it came
     from a jam
   - a 9:16 Short/Reel cut from the `pick_clip.py` excerpt, sized for both
     YouTube Shorts and Instagram Reels
   - optionally a 1:1 square for an Instagram feed post
   - a description/caption text file with title, BPM, key, genre hashtags,
     and the chapter list. It follows the existing `youtube-mux.md`
     conventions.
5. **Upload-safe video encoding.** H.264 + AAC 48k at a high bitrate
   (320k AAC), `+faststart`, constant frame rate, with the audio taken
   from the -1 dBTP master so the platforms' own re-encode doesn't clip it.
6. **Batch.** Extend `batch_process.py` (or add a manifest column) so a
   card dump of jams can be split, analyzed, mastered, and rendered to
   videos in one pass.

---

## Suggested order and effort

Ordered for a Bluebox rig releasing ambient, trap, techno, and psytrance
to YouTube and Instagram:

| # | Item | Effort | Depends on | Value for you |
|---|---|---|---|---|
| 1 | `music_analyze.py`: BPM (genre priors, trap at ~70), key, grid, mix/phone checks | M | — | High: everything builds on it |
| 2 | `process_music_wav.py`: 4 genre recipes + youtube/instagram presets | M | 1 | High: masters the Bluebox mix |
| 3 | Mono-bass + soft-clip + linear loudnorm | S | 2 | High: the biggest audible gains |
| 4 | Drop detection + `pick_clip.py` for Shorts/Reels | M | 1 | High: the clip *is* the post |
| 5 | `release.py`: beat-synced visualizer (`--grid`) + publish set + captions | M | 1, 4 | High: one command from master to upload-ready videos |
| 6 | `ingest_bluebox.py` + `stems.json` | S–M | a real project to inspect | High: unlocks stems |
| 7 | Stem-driven visuals | S–M | 5, 6 | Medium–High: a distinctive look |
| 8 | `remix_stems.py` repair mode + per-stem presets (808/bass harmonics, sidechain) | M | 2, 6 | High for trap/psy on phones |
| 9 | Jam splitter + chapters (mix and stems cut together) | M | 1, 6 | Medium–High for long jams |
| 10 | Bar-aligned loop export from stems | M | 1, 6 | Medium |
| 11 | Reference matching (matchering) | S | 2 | Medium |

Dropped: tempo-drift correction (your gear doesn't drift), and the DJ
edit, club, SoundCloud, and Bandcamp presets (not where you release). The
Demucs 4-stem idea is also dropped: you have real stems.

**Milestone 1 (items 1–5) — built 2026-10-05** on `feature/music-milestone-1` (`music_analyze.py`, `process_music_wav.py`, `pick_clip.py`, `release.py`, `visualize_wav.py --grid`; see README).

**Milestone 1 (items 1–5):** finished Bluebox mix → mastered audio →
landscape video + Short/Reel + caption in one run. **Milestone 2
(items 6–9):** the stem workflow: ingest, stem-driven visuals, repair
re-mixes, phone-audible bass, and jam splitting with matching stems.

## Testing approach

- Keep a small `test_audio/` set built from your own Bluebox projects:
  - one track per genre (ambient, trap, techno, psytrance)
  - a jam with a hard stop between tracks
  - a jam with a seamless transition
  - a bass patch with deliberately stereo low end
- For each, check the measured values against known truth: BPM within
  ±0.1, splits within one bar, post-master true peak ≤ target, mono-bass
  correlation ≈ 1.0 below the cutoff, and trap reported at
  ~70 BPM, not ~140.
- Upload a test Short/Reel privately, download it back, and measure it to
  confirm the platforms' actual loudness handling.
- A/B every genre recipe against a commercial reference at matched
  loudness before removing the `untested` flag.

## Open questions for you

1. ~~**Your rig's recording path.**~~ Answered: 1010music Bluebox, a
   multitrack recording mixed on the device. Still needed: one real
   Bluebox project folder copied to this Mac, to check the details
   listed under "Your rig" before writing ingest.
2. ~~**Main genres.**~~ Answered: ambient, trap, classic techno,
   psytrance.
3. ~~**Release targets.**~~ Answered: YouTube and Instagram.
4. ~~**Drift.**~~ Answered: none; drift handling dropped.
5. ~~**Trap BPM convention.**~~ Answered: half-time, ~70 BPM.
