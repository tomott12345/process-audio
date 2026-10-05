#!/usr/bin/env python3
"""Describe every pipeline this repo can run -- music, nature, speech -- as
one JSON document: the content types, every option with its per-type
default, which options apply to which type, how each maps to a command-line
flag, which outputs and audio formats exist, and which optional
dependencies are missing. A UI (the Go web app) builds its form and its
command lines from this instead of hard-coding them, so the recipes and
scripts stay the single source of truth.

Usage:
  python3 pipeline_capabilities.py            # pretty JSON
  python3 pipeline_capabilities.py --compact

How a caller turns form values into a command line (all of it is data
below, nothing is implied):
  argv = [python, <script>, <input>]
  + [type.param, <type>]                       if the pipeline has types
  + for each option the user CHANGED from its default (per-type default
    first, then the plain default), skipping options whose applies_to /
    requires_option / requires_output don't hold:
        bool    value true  -> flag_true  (if set)   value false -> flag_false (if set)
        number / choice / time / text  -> [flag, value]
        text with flag_empty, set to ""  -> [flag_empty]
        time_list -> [flag, v] repeated per value
  + outputs: outputs_param  -> [outputs_param, "a,b,c"]
             per-output flag_when_off -> that flag for each output NOT chosen
  + formats: [formats_param, "wav24,mp3"]  when any chosen output has formats
  + run.progress_flag, and --out-dir <job dir>
A plan (no rendering) is the same argv + plan_flags.

Option fields:
  id, group, label, help, kind (bool | number | choice | time | text | time_list),
  advanced (true = behind the collapsed "Advanced" disclosure),
  default, defaults_by_type {type: value}, detail_by_type {type: text},
  min / max / step / unit, choices [{value, label}],
  applies_to [types]           -- absent = every type
  requires_option {id: value}  -- only meaningful when another option has that value
  requires_output [outputs]    -- only meaningful when one of these outputs is chosen
  requires_analysis "has_grid" -- depends on the uploaded file (music analysis)
  requires_dependency name     -- disabled, with disabled_reason, if it isn't installed
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from music_common import load_recipes as load_music_recipes  # noqa: E402
from pipeline_io import AUDIO_FORMATS, PROGRESS_PREFIX  # noqa: E402

CAPABILITIES_VERSION = 1


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def dependencies() -> dict:
    try:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True).stdout
    except OSError:
        enc = ""
    deps = {
        "ffmpeg": bool(enc),
        "mp3_encoder": " libmp3lame " in enc,
        "flac_encoder": " flac " in enc,
        "numpy": _has("numpy"), "scipy": _has("scipy"),
        "librosa": _has("librosa"), "pillow": _has("PIL"), "matplotlib": _has("matplotlib"),
        "noisereduce": _has("noisereduce"),
        "demucs": _has("torch") and _has("demucs"),
    }
    return deps


INSTALL_HINTS = {
    "librosa": "pip3 install librosa pillow --break-system-packages",
    "pillow": "pip3 install pillow --break-system-packages",
    "matplotlib": "pip3 install matplotlib --break-system-packages",
    "noisereduce": "pip3 install noisereduce --break-system-packages",
    "demucs": "pip3 install torch demucs --break-system-packages (GB+ download)",
    "mp3_encoder": "install an ffmpeg build with libmp3lame (e.g. brew install ffmpeg)",
    "flac_encoder": "install an ffmpeg build with FLAC support",
}


def opt(id, group, label, kind, **kw) -> dict:
    o = {"id": id, "group": group, "label": label, "kind": kind, "advanced": kw.pop("advanced", False)}
    o.update(kw)
    return o


TIME_HELP = "seconds, or mm:ss / hh:mm:ss"


# ---------------------------------------------------------------- music

def music_pipeline(deps: dict) -> dict:
    rec = load_music_recipes()
    genres = rec["genres"]
    presets = rec["presets"]

    def by_type(fn):
        return {g: fn(r) for g, r in genres.items()}

    def visual_arg(r, flag, default):
        a = r["visual_args"]
        return a[a.index(flag) + 1] if flag in a else default

    types = []
    for g, r in genres.items():
        lo, hi = r["tempo_range"]
        bits = [f"{lo}-{hi} BPM" if r["expect_grid"] else "usually beatless",
                f"mono bass < {r['mono_bass_hz']} Hz",
                "soft clip" if r.get("softclip") else "no soft clip, keeps dynamics",
                f"{r['visual_style']} visuals"]
        if r.get("grid_subdivision", 1) > 1:
            bits[0] += " (half-time feel)"
        types.append({
            "id": g, "label": g.capitalize() if g != "psytrance" else "Psytrance",
            "untested": bool(r.get("untested")), "summary": ", ".join(bits),
            "facts": {"tempo_range": r["tempo_range"], "expect_grid": r["expect_grid"],
                      "visual_style": r["visual_style"], "hashtags": r["hashtags"]},
        })
    videos = ["video_16x9", "video_9x16", "video_1x1"]
    clip_users = ["clip", "video_9x16", "video_1x1"]
    dyn_types = [g for g, r in genres.items() if r.get("dynamic_eq")]
    options = [
        # basic
        opt("title", "tags", "Title", "text", flag="--title", default="",
            help="Video overlay, file tags, and captions (default: the file name)"),
        opt("artist", "tags", "Artist", "text", flag="--artist", default=""),
        opt("preset", "loudness", "Loudness target", "choice", flag="--preset", default="youtube",
            choices=[{"value": k, "label": f"{k.capitalize()} ({v['target_i']:g} LUFS)"
                      + (" -- unverified" if v.get("untested") else "")} for k, v in presets.items()],
            help="Ambient lands 2 LU under the preset on purpose"),
        opt("clip_length", "clip", "Short/Reel length", "choice", flag="--clip-length", default="30",
            choices=[{"value": v, "label": f"{v} s"} for v in ("15", "30", "60", "90")],
            requires_output=clip_users,
            help="Cut on whole bars, never longer than this. Check current platform limits."),
        opt("loop_clip", "clip", "Seamless loop", "bool", flag_true="--loop-clip", default=False,
            requires_output=clip_users, requires_analysis="has_grid",
            help="An exact 4/8/16-bar phrase from the drop, so the platform's auto-repeat never breaks the groove"),
        opt("start", "trim", "Start", "time", flag="--start", advanced=True, help=TIME_HELP),
        opt("end", "trim", "End", "time", flag="--end", advanced=True, help=TIME_HELP),
        # advanced: loudness
        opt("target_i", "loudness", "Custom loudness", "number", flag="--target-i", advanced=True,
            min=-30, max=-6, step=0.5, unit="LUFS", help="Overrides the preset"),
        opt("target_tp", "loudness", "True-peak ceiling", "number", flag="--target-tp", advanced=True,
            min=-3, max=-0.1, step=0.1, unit="dBTP"),
        # advanced: processing
        opt("highpass_hz", "processing", "Subsonic high-pass", "number", flag="--highpass-hz", advanced=True,
            min=10, max=60, step=1, unit="Hz", defaults_by_type=by_type(lambda r: r["highpass_hz"]),
            help="Also removes DC offset"),
        opt("mono_bass", "processing", "Mono bass", "bool", flag_false="--no-mono-bass", default=True, advanced=True,
            help="Fold the low end to mono -- tighter on big systems, survives phones"),
        opt("mono_bass_hz", "processing", "Mono-bass crossover", "number", flag="--mono-bass-hz", advanced=True,
            min=60, max=250, step=5, unit="Hz", defaults_by_type=by_type(lambda r: r["mono_bass_hz"]),
            requires_option={"mono_bass": True}),
        opt("width", "processing", "Stereo width (above the crossover)", "number", flag="--width", advanced=True,
            min=0.5, max=2.0, step=0.05, unit="x", defaults_by_type=by_type(lambda r: r["width"])),
        opt("eq", "processing", "Tonal EQ", "bool", flag_false="--no-eq", default=True, advanced=True,
            detail_by_type=by_type(lambda r: r["eq"] or "none")),
        opt("dynamic_eq", "processing", "Dynamic EQ", "bool", flag_false="--no-dynamic-eq", default=True,
            advanced=True, applies_to=dyn_types,
            detail_by_type={g: genres[g]["dynamic_eq"] for g in dyn_types},
            help="Tames resonant synth peaks only when they spike"),
        opt("glue", "processing", "Glue compression", "bool", flag_false="--no-glue", default=True, advanced=True,
            detail_by_type=by_type(lambda r: "{ratio}:1 at {threshold_db} dB, attack {attack_ms} ms".format(**r["glue"]))),
        opt("softclip", "processing", "Soft clip", "bool", flag_true="--softclip", flag_false="--no-softclip",
            advanced=True, defaults_by_type=by_type(lambda r: bool(r.get("softclip"))),
            help="Rounds off the peaks the loudness gain pushes over, before the limiter"),
        opt("softclip_threshold", "processing", "Soft-clip threshold", "number", flag="--softclip-threshold",
            advanced=True, min=-6, max=0, step=0.5, unit="dBFS",
            defaults_by_type=by_type(lambda r: (r.get("softclip") or {}).get("threshold_db", 0.0)),
            requires_option={"softclip": True}),
        opt("fade_in", "processing", "Fade in", "number", flag="--fade-in", advanced=True,
            min=0, max=30, step=0.01, unit="s", defaults_by_type=by_type(lambda r: r["fade_in"])),
        opt("fade_out", "processing", "Fade out", "number", flag="--fade-out", advanced=True,
            min=0, max=60, step=0.1, unit="s", defaults_by_type=by_type(lambda r: r["fade_out"])),
        # advanced: clip
        opt("drop_at", "clip", "Build the clip around a drop at", "time", flag="--drop-at", advanced=True,
            requires_output=clip_users, help="Overrides the detected main drop; " + TIME_HELP),
        opt("clip_start", "clip", "Pin the clip start at", "time", flag="--clip-start", advanced=True,
            requires_output=clip_users, help=TIME_HELP),
        # video
        opt("style", "video", "Visual style", "choice", flag="--style", advanced=True,
            choices=[{"value": v, "label": v} for v in ("radial", "bars", "glowburst", "wormhole")],
            defaults_by_type=by_type(lambda r: r["visual_style"]), requires_output=videos),
        opt("emoji", "video", "Center emoji", "text", flag="--emoji", flag_empty="--no-emoji", advanced=True,
            defaults_by_type=by_type(lambda r: visual_arg(r, "--emoji", "")),
            requires_output=videos, requires_option={"style": "radial"}),
        opt("symmetry", "video", "Symmetry", "number", flag="--symmetry", advanced=True, min=1, max=12, step=1,
            defaults_by_type=by_type(lambda r: int(visual_arg(r, "--symmetry", 1))),
            requires_output=videos, requires_option={"style": "radial"}),
        opt("max_seconds", "video", "Preview length", "number", flag="--max-seconds", advanced=True,
            min=1, max=600, step=1, unit="s", requires_output=videos,
            help="Render only the first N seconds of each video -- a quick look before the full render"),
        opt("method", "tags", "How it was made (captions)", "text", flag="--method", advanced=True,
            default="Hardware synths, recorded DAWless and mixed live on a 1010music Bluebox.",
            requires_output=["captions"]),
    ]
    outputs = [
        {"id": "master", "label": "Mastered audio", "formats": True, "default": True},
        {"id": "clip", "label": "Short/Reel clip (WAV)", "default": False},
        {"id": "video_16x9", "label": "YouTube video (16:9, full track)", "default": True,
         "requires_dependency": "librosa", "slow": True},
        {"id": "video_9x16", "label": "Short / Reel video (9:16)", "default": True, "requires_dependency": "librosa"},
        {"id": "video_1x1", "label": "Instagram feed video (1:1)", "default": False, "requires_dependency": "librosa"},
        {"id": "captions", "label": "Captions (YouTube + Instagram)", "default": True},
        {"id": "analysis", "label": "Analysis (JSON + chart)", "default": False, "requires_dependency": "matplotlib"},
    ]
    return {
        "id": "music", "label": "Music",
        "description": "Bluebox mixes and finished tracks: genre mastering for YouTube/Instagram, "
                       "a Short/Reel cut around the drop, beat-locked visualizer videos, captions.",
        "script": "release.py",
        "requires_dependency": "librosa",
        "type": {"param": "--genre", "label": "Music type", "required": True},
        "types": types,
        "groups": [
            {"id": "tags", "label": "Title & tags"}, {"id": "loudness", "label": "Loudness"},
            {"id": "trim", "label": "Trim"}, {"id": "processing", "label": "Processing"},
            {"id": "clip", "label": "Short / Reel clip"}, {"id": "video", "label": "Video"},
        ],
        "options": options,
        "outputs": outputs, "outputs_param": "--outputs",
        "formats_param": "--formats", "default_formats": ["wav24", "wav16"],
        "analyze": {"script": "music_analyze.py",
                    "args": ["{input}", "--genre", "{type}", "--out-dir", "{dir}"],
                    "result_file": "music_analysis.json",
                    "provides": ["has_grid", "bpm", "key", "sections", "main_drop", "loudness", "flags"]},
    }


# ---------------------------------------------------------------- nature

def nature_pipeline(deps: dict) -> dict:
    rec = json.loads((SCRIPT_DIR / "recipes.json").read_text())
    labels = rec["labels"]
    bed_words = {"longest": "longest clean run", "activity": "most active run", "mixed": "longest vs most active"}
    types = []
    for name, r in labels.items():
        bits = [f"bed: {bed_words.get(r['bed_selection'], r['bed_selection'])}"]
        if r.get("correlated_water"):
            bits.append("never free-looped (correlated water)")
        lt = r["targets"]["long"]
        bits.append(f"long master {lt[0]:g} LUFS")
        types.append({"id": name, "label": name.capitalize(), "untested": bool(r.get("untested")),
                      "summary": ", ".join(bits),
                      "facts": {"bed_selection": r["bed_selection"],
                                "correlated_water": bool(r.get("correlated_water")),
                                "targets": r["targets"]}})
    correlated = [n for n, r in labels.items() if r.get("correlated_water")]
    options = [
        opt("place", "basics", "Where was it recorded?", "choice", flag="--place", default="unknown",
            choices=[{"value": p, "label": p} for p in rec["places"]]),
        opt("long", "basics", "Long master", "choice", flag="--long", default="clean",
            choices=[{"value": "clean", "label": "Best clean bed only"},
                     {"value": "full", "label": "Whole file after cleanup"}],
            requires_output=["master_long"]),
        opt("title", "basics", "Title", "text", flag="--title", default=""),
        opt("start", "trim", "Start", "time", flag="--start", advanced=True,
            help="Overrides automatic bed selection; " + TIME_HELP),
        opt("end", "trim", "End", "time", flag="--end", advanced=True, help=TIME_HELP),
        opt("loop", "looping", "Short looping", "choice", flag="--loop", default="auto", advanced=True,
            choices=[{"value": "auto", "label": "auto"}, {"value": "never", "label": "never"},
                     {"value": "force", "label": "force (hard splice)"}],
            help="auto never loops correlated water: " + ", ".join(correlated)),
        opt("splice", "looping", "Splice overlap", "number", flag="--splice", default=0.012, advanced=True,
            min=0.008, max=0.05, step=0.001, unit="s", requires_option={"loop": "force"}),
        opt("loop_target", "looping", "Extend the long master to", "time", flag="--loop-target", advanced=True,
            requires_output=["master_long"], help="e.g. 1:00:00 for an hour; " + TIME_HELP),
        opt("denoise", "cleanup", "Gentle broadband denoise", "bool", flag_true="--denoise", default=False,
            advanced=True, detail=rec["denoise_chain"]),
        opt("repair", "cleanup", "Declick / declip", "bool", flag_true="--repair", default=False,
            advanced=True, detail=rec["repair_chain"]),
        opt("denoise_profile_start", "cleanup", "Noise-only span start (in the bed)", "time",
            flag="--denoise-profile-start", advanced=True, requires_dependency="noisereduce",
            help="A stretch with only the unwanted steady noise (hum, hiss, traffic); 0 = start of the bed"),
        opt("denoise_profile_end", "cleanup", "Noise-only span end", "time", flag="--denoise-profile-end",
            advanced=True, requires_dependency="noisereduce"),
        opt("denoise_non_stationary", "cleanup", "Noise drifts over time", "bool",
            flag_true="--denoise-non-stationary", default=False, advanced=True, requires_dependency="noisereduce"),
        opt("denoise_prop_decrease", "cleanup", "Noise reduction strength", "number",
            flag="--denoise-prop-decrease", default=1.0, min=0, max=1, step=0.05, advanced=True,
            requires_dependency="noisereduce"),
        opt("remove_voice", "cleanup", "Remove voices/footsteps (whole bed)", "bool", flag_true="--remove-voice",
            default=False, advanced=True, requires_dependency="demucs",
            help="Demucs source separation -- slow on a CPU"),
        opt("remove_voice_at", "cleanup", "Remove voices/footsteps at", "time_list", flag="--remove-voice-at",
            advanced=True, requires_dependency="demucs",
            help="Specific moments in the source file (or 'end'); only a padded window around each is touched"),
        opt("remove_voice_pad", "cleanup", "Padding around each moment", "number", flag="--remove-voice-pad",
            default=4.0, min=1, max=20, step=0.5, unit="s", advanced=True, requires_dependency="demucs"),
        opt("voice_model", "cleanup", "Demucs model", "choice", flag="--voice-model", default="htdemucs",
            advanced=True, requires_dependency="demucs",
            choices=[{"value": m, "label": m} for m in ("htdemucs", "htdemucs_ft", "mdx_extra")]),
        opt("artist", "basics", "Artist", "text", flag="--artist", default="", advanced=True),
    ]
    return {
        "id": "nature", "label": "Nature",
        "description": "Field recordings (rain, insects, birds, water...): clean-bed selection, "
                       "per-type EQ, a long master and a 3-minute Short master.",
        "script": "process_field_wav.py",
        "type": {"param": "--label", "label": "What is it a recording of?", "required": True},
        "types": types,
        "groups": [{"id": "basics", "label": "Basics"}, {"id": "trim", "label": "Trim"},
                   {"id": "looping", "label": "Looping"}, {"id": "cleanup", "label": "Cleanup"}],
        "options": options,
        "outputs": [
            {"id": "master_long", "label": "Long master", "formats": True, "default": True, "flag_when_off": "--skip-long"},
            {"id": "master_short", "label": "Short master (3 min)", "formats": True, "default": True,
             "flag_when_off": "--skip-short"},
        ],
        "outputs_param": None,
        "formats_param": "--formats", "default_formats": ["wav24"],
        "analyze": {"script": "process_field_wav.py",
                    "args": ["{input}", "--label", "{type}", "--plan-only", "--json", "--out-dir", "{dir}"],
                    "result_file": None, "result": "stdout",
                    "provides": ["bed", "clean_runs", "short_loop_mode"]},
    }


# ---------------------------------------------------------------- speech

def speech_pipeline(deps: dict) -> dict:
    from process_speech_wav import PRESETS

    options = [
        opt("preset", "loudness", "Platform", "choice", flag="--preset", default="general",
            choices=[{"value": k, "label": f"{k.capitalize()} ({v[0]:g} LUFS)"} for k, v in PRESETS.items()]),
        opt("title", "tags", "Title", "text", flag="--title", default=""),
        opt("artist", "tags", "Show / artist", "text", flag="--artist", default=""),
        opt("trim_silence", "trim", "Trim leading/trailing silence", "bool", flag_true="--trim-silence",
            default=False, help="Never touches pauses in the middle"),
        opt("start", "trim", "Start", "time", flag="--start", advanced=True, help=TIME_HELP),
        opt("end", "trim", "End", "time", flag="--end", advanced=True, help=TIME_HELP),
        opt("silence_threshold", "trim", "Silence threshold", "number", flag="--silence-threshold", default=-45.0,
            min=-70, max=-20, step=1, unit="dB", advanced=True, requires_option={"trim_silence": True}),
        opt("compress", "processing", "Leveling compressor", "bool", flag_false="--no-compress", default=True,
            advanced=True),
        opt("deess", "processing", "De-esser", "bool", flag_false="--no-deess", default=True, advanced=True),
        opt("mono", "processing", "Mono", "bool", flag_true="--mono", default=False, advanced=True,
            help="Many podcast loudness specs are defined for mono"),
        opt("target_i", "loudness", "Custom loudness", "number", flag="--target-i", advanced=True,
            min=-30, max=-10, step=0.5, unit="LUFS"),
        opt("target_tp", "loudness", "True-peak ceiling", "number", flag="--target-tp", advanced=True,
            min=-3, max=-0.1, step=0.1, unit="dBTP"),
        opt("target_lra", "loudness", "Loudness range", "number", flag="--target-lra", advanced=True,
            min=3, max=20, step=0.5, unit="LU"),
    ]
    return {
        "id": "speech", "label": "Speech",
        "description": "Interviews and podcasts: the whole recording, cleaned and leveled to a platform target.",
        "script": "process_speech_wav.py",
        "type": None, "types": [],
        "groups": [{"id": "tags", "label": "Title & tags"}, {"id": "loudness", "label": "Loudness"},
                   {"id": "trim", "label": "Trim"}, {"id": "processing", "label": "Processing"}],
        "options": options,
        "outputs": [{"id": "master", "label": "Master", "formats": True, "default": True}],
        "outputs_param": None,
        "formats_param": "--formats", "default_formats": ["wav24"],
        "analyze": None,
    }


def gate(pipe: dict, deps: dict) -> dict:
    """Mark anything whose dependency is missing as disabled, with why."""
    def reason(dep):
        return f"needs {dep} -- {INSTALL_HINTS.get(dep, 'not installed')}"

    for item in [pipe] + pipe["options"] + pipe["outputs"]:
        dep = item.get("requires_dependency")
        item["available"] = bool(deps.get(dep, False)) if dep else True
        if not item["available"]:
            item["disabled_reason"] = reason(dep)
    return pipe


def capabilities() -> dict:
    deps = dependencies()
    formats = []
    for fid, f in AUDIO_FORMATS.items():
        dep = {"mp3": "mp3_encoder", "flac": "flac_encoder"}.get(fid)
        ok = deps.get(dep, True) if dep else True
        formats.append({"id": fid, "label": f["label"], "available": ok,
                        **({} if ok else {"disabled_reason": f"needs {dep} -- {INSTALL_HINTS[dep]}"})})
    return {
        "version": CAPABILITIES_VERSION,
        "python": sys.executable,
        "repo": str(SCRIPT_DIR),
        "dependencies": deps,
        "formats": formats,
        "run": {"progress_flag": "--progress-json", "progress_prefix": PROGRESS_PREFIX.strip(),
                "plan_flags": ["--plan-only", "--json"], "out_dir_flag": "--out-dir",
                "peaks": {"script": "audio_peaks.py", "args": ["{input}", "--points", "{points}"]}},
        "pipelines": [gate(p(deps), deps) for p in (music_pipeline, nature_pipeline, speech_pipeline)],
    }


def option_default(o: dict, type_id: str | None):
    if type_id is not None and type_id in o.get("defaults_by_type", {}):
        return o["defaults_by_type"][type_id]
    return o.get("default")


def option_active(o: dict, type_id: str | None, values: dict, outputs: list[str], pipe: dict) -> bool:
    """Whether an option means anything for this type / these outputs /
    the other options' values. Inactive options are never sent."""
    if "applies_to" in o and type_id not in o["applies_to"]:
        return False
    if o.get("requires_output") and not set(o["requires_output"]) & set(outputs):
        return False
    for dep_id, want in (o.get("requires_option") or {}).items():
        dep = next(x for x in pipe["options"] if x["id"] == dep_id)
        have = values.get(dep_id, option_default(dep, type_id))
        if have != want:
            return False
    return o.get("available", True)


def build_argv(caps: dict, pipeline_id: str, type_id: str | None, values: dict, outputs: list[str],
               formats: list[str], input_path: str, out_dir: str, *, plan: bool = False,
               progress: bool = True) -> list[str]:
    """Reference implementation of the form -> command line mapping
    described in the module docstring. The Go web app ports this; its tests
    compare against this function's output. Raises ValueError on anything
    the capabilities don't allow."""
    pipe = next((p for p in caps["pipelines"] if p["id"] == pipeline_id), None)
    if pipe is None:
        raise ValueError(f"unknown pipeline {pipeline_id!r}")
    argv = [caps["python"], str(Path(caps["repo"]) / pipe["script"]), input_path]
    if pipe["type"]:
        if type_id not in {t["id"] for t in pipe["types"]}:
            raise ValueError(f"{pipe['type']['label']}: unknown type {type_id!r}")
        argv += [pipe["type"]["param"], type_id]
    elif type_id:
        raise ValueError(f"pipeline {pipeline_id!r} has no types")
    out_ids = {o["id"]: o for o in pipe["outputs"]}
    for o in outputs:
        if o not in out_ids:
            raise ValueError(f"unknown output {o!r}")
        if not out_ids[o].get("available", True):
            raise ValueError(f"output {o!r} unavailable: {out_ids[o].get('disabled_reason')}")
    if not outputs:
        raise ValueError("choose at least one output")
    opts = {o["id"]: o for o in pipe["options"]}
    for oid in values:
        if oid not in opts:
            raise ValueError(f"unknown option {oid!r}")
    for o in pipe["options"]:
        if o["id"] not in values:
            continue
        v = values[o["id"]]
        if not option_active(o, type_id, values, outputs, pipe):
            continue
        if v == option_default(o, type_id) and o["kind"] != "time_list":
            continue
        kind = o["kind"]
        if kind == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"{o['id']}: expected true/false")
            flag = o.get("flag_true") if v else o.get("flag_false")
            if flag:
                argv.append(flag)
        elif kind == "number":
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{o['id']}: expected a number")
            if ("min" in o and v < o["min"]) or ("max" in o and v > o["max"]):
                raise ValueError(f"{o['id']}: {v} is outside {o.get('min')}..{o.get('max')}")
            argv += [o["flag"], f"{v:g}"]
        elif kind == "choice":
            if str(v) not in {c["value"] for c in o["choices"]}:
                raise ValueError(f"{o['id']}: {v!r} is not one of the choices")
            argv += [o["flag"], str(v)]
        elif kind == "time":
            if v in (None, ""):
                continue
            argv += [o["flag"], str(v)]
        elif kind == "time_list":
            for item in v or []:
                argv += [o["flag"], str(item)]
        elif kind == "text":
            if v == "" and o.get("flag_empty"):
                argv.append(o["flag_empty"])
            elif v != "":
                argv += [o["flag"], str(v)]
    if pipe["outputs_param"]:
        argv += [pipe["outputs_param"], ",".join(outputs)]
    for o in pipe["outputs"]:
        if o.get("flag_when_off") and o["id"] not in outputs:
            argv.append(o["flag_when_off"])
    if any(out_ids[o].get("formats") for o in outputs):
        known = {f["id"]: f for f in caps["formats"]}
        for f in formats:
            if f not in known or not known[f]["available"]:
                raise ValueError(f"format {f!r} unavailable")
        if not formats:
            raise ValueError("choose at least one audio format")
        argv += [pipe["formats_param"], ",".join(formats)]
    argv += [caps["run"]["out_dir_flag"], out_dir]
    if plan:
        argv += caps["run"]["plan_flags"]
    elif progress:
        argv.append(caps["run"]["progress_flag"])
    return argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compact", action="store_true", help="One-line JSON")
    args = ap.parse_args()
    data = capabilities()
    if args.compact:
        json.dump(data, sys.stdout, separators=(",", ":"), ensure_ascii=False)
    else:
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
