"""Contract tests: pipeline_capabilities.py must describe the scripts
truthfully, and the scripts must keep their promises (formats, progress
events, manifests, JSON plans) -- this is the interface the web app is
built on.

  pytest tests/                 # everything (renders audio; a few minutes)
  pytest tests/ -m "not slow"   # contract only, no renders (~1 minute)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import pipeline_capabilities as pc  # noqa: E402
from make_test_tracks import make_all  # noqa: E402
from pipeline_io import PROGRESS_PREFIX  # noqa: E402

KINDS = {"bool", "number", "choice", "time", "text", "time_list"}


@pytest.fixture(scope="session")
def caps():
    return pc.capabilities()


@pytest.fixture(scope="session")
def tracks(tmp_path_factory):
    return make_all(tmp_path_factory.mktemp("tracks"))


def pipe(caps, pid):
    return next(p for p in caps["pipelines"] if p["id"] == pid)


def run(argv, **kw):
    return subprocess.run(argv, cwd=REPO, text=True, capture_output=True, **kw)


def events(stdout: str) -> list[dict]:
    return [json.loads(line[len(PROGRESS_PREFIX):]) for line in stdout.splitlines() if line.startswith(PROGRESS_PREFIX)]


def test_capabilities_cli_is_json():
    r = run([sys.executable, "pipeline_capabilities.py", "--compact"])
    assert r.returncode == 0, r.stderr
    assert {p["id"] for p in json.loads(r.stdout)["pipelines"]} == {"music", "nature", "speech"}


@pytest.mark.parametrize("pid", ["music", "nature", "speech"])
def test_options_are_well_formed(caps, pid):
    p = pipe(caps, pid)
    type_ids = {t["id"] for t in p["types"]}
    opt_ids = {o["id"] for o in p["options"]}
    out_ids = {o["id"] for o in p["outputs"]}
    group_ids = {g["id"] for g in p["groups"]}
    assert len(opt_ids) == len(p["options"]), "duplicate option ids"
    for o in p["options"]:
        assert o["kind"] in KINDS, o
        assert o["group"] in group_ids, o
        flags = [o.get(k) for k in ("flag", "flag_true", "flag_false", "flag_empty") if o.get(k)]
        assert flags, f"{o['id']} maps to no flag"
        assert set(o.get("applies_to", [])) <= type_ids, o
        assert set(o.get("defaults_by_type", {})) <= type_ids, o
        assert set(o.get("requires_output", [])) <= out_ids, o
        assert set(o.get("requires_option", {})) <= opt_ids, o
        if o["kind"] == "choice":
            vals = {c["value"] for c in o["choices"]}
            for d in [o.get("default")] + list(o.get("defaults_by_type", {}).values()):
                assert d is None or str(d) in vals, (o["id"], d)
        if o["kind"] == "number":
            for d in [o.get("default")] + list(o.get("defaults_by_type", {}).values()):
                assert d is None or o["min"] <= d <= o["max"], (o["id"], d)
    if p["type"]:
        assert type_ids, "typed pipeline with no types"


def sample_value(o, type_id):
    """A valid, non-default value for an option."""
    d = pc.option_default(o, type_id)
    k = o["kind"]
    if k == "bool":
        return not bool(d)
    if k == "number":
        for cand in (o["max"], o["min"], (o["min"] + o["max"]) / 2):
            if cand != d:
                return int(cand) if o.get("step", 1) >= 1 and float(cand).is_integer() else cand
    if k == "choice":
        return next(c["value"] for c in o["choices"] if c["value"] != str(d))
    if k == "time":
        return {"end": "60", "drop_at": "0:20", "clip_start": "0:10", "loop_target": "200",
                "denoise_profile_end": "3"}.get(o["id"], "1")
    if k == "time_list":
        return ["10", "end"]
    if k == "text":
        if o.get("flag_empty") and d:
            return ""  # clearing a non-empty default is the interesting case
        return "⭐" if o["id"] == "emoji" else "Test Value"
    raise AssertionError(k)


def option_cases(caps):
    for p in caps["pipelines"]:
        for o in p["options"]:
            yield pytest.param(p["id"], o["id"], id=f"{p['id']}-{o['id']}")


@pytest.mark.parametrize("pid,oid", list(option_cases(pc.capabilities())))
def test_every_option_flag_is_accepted(caps, tracks, tmp_path, pid, oid):
    """Each option, set to a non-default value, must produce a command line
    the real script accepts (checked with --plan-only --json)."""
    p = pipe(caps, pid)
    o = next(x for x in p["options"] if x["id"] == oid)
    type_id = (o.get("applies_to") or [t["id"] for t in p["types"]] or [None])[0]
    values = {oid: sample_value(o, type_id)}
    for dep, want in (o.get("requires_option") or {}).items():
        values[dep] = want
    if o.get("requires_option", {}).get("style") == "radial":
        type_id = "trap"
    outputs = list(o.get("requires_output") or [x["id"] for x in p["outputs"] if x["default"] and x.get("available", True)])
    o_avail = dict(o, available=True)  # parse-check flags even if the dependency is missing
    p_check = dict(p, options=[o_avail if x["id"] == oid else x for x in p["options"]])
    caps_check = dict(caps, pipelines=[p_check if x["id"] == pid else x for x in caps["pipelines"]])
    inp = {"music": tracks["techno"], "nature": tracks["rain"], "speech": tracks["speech"]}[pid]
    argv = pc.build_argv(caps_check, pid, type_id, values, outputs, list(p["default_formats"]),
                         str(inp), str(tmp_path / "out"), plan=True)
    flags = [x.split("=", 1)[0] for x in argv if x.startswith("--")]
    assert any(o.get(k) in flags for k in ("flag", "flag_true", "flag_false", "flag_empty")), argv
    r = run(argv, timeout=300)
    assert r.returncode == 0, f"{argv}\n{r.stderr[-2000:]}"
    json.loads(r.stdout)  # --json: stdout is exactly one JSON document


def test_dash_leading_text_is_one_token(caps, tmp_path):
    argv = pc.build_argv(caps, "speech", None, {"title": "-Intro-"}, ["master"], ["wav24"], "x.wav",
                         str(tmp_path), plan=True)
    assert "--title=-Intro-" in argv
    r = run(argv)
    assert r.returncode == 0, r.stderr


def test_build_argv_rejects_bad_input(caps):
    with pytest.raises(ValueError):
        pc.build_argv(caps, "music", "house", {}, ["master"], ["wav24"], "i", "o")
    with pytest.raises(ValueError):
        pc.build_argv(caps, "music", "techno", {"width": 9}, ["master"], ["wav24"], "i", "o")
    with pytest.raises(ValueError):
        pc.build_argv(caps, "music", "techno", {"preset": "spotify"}, ["master"], ["wav24"], "i", "o")
    with pytest.raises(ValueError):
        pc.build_argv(caps, "music", "techno", {"nope": 1}, ["master"], ["wav24"], "i", "o")
    with pytest.raises(ValueError):
        pc.build_argv(caps, "speech", None, {}, ["master"], ["ogg"], "i", "o")
    with pytest.raises(ValueError):
        pc.build_argv(caps, "speech", None, {}, [], ["wav24"], "i", "o")


def test_inactive_options_are_not_sent(caps):
    # dynamic EQ exists only for techno; emoji only matters on radial style
    argv = pc.build_argv(caps, "music", "ambient", {"dynamic_eq": False, "emoji": "⭐"},
                         ["master", "video_9x16"], ["wav24"], "i", "o")
    assert "--no-dynamic-eq" not in argv and not any(a.startswith("--emoji") for a in argv)
    argv = pc.build_argv(caps, "music", "techno", {"dynamic_eq": False}, ["master"], ["wav24"], "i", "o")
    assert "--no-dynamic-eq" in argv
    # video-only options are dropped when no video is requested
    argv = pc.build_argv(caps, "music", "trap", {"symmetry": 6}, ["master"], ["wav24"], "i", "o")
    assert not any(a.startswith("--symmetry") for a in argv)


def test_release_outputs_select_steps(caps):
    argv = pc.build_argv(caps, "music", "techno", {}, ["master", "captions"], ["flac"], "x.wav", "/tmp/o", plan=True)
    plan = json.loads(run(argv).stdout)
    assert [s["step"] for s in plan["steps"]] == ["master", "captions"]
    argv = pc.build_argv(caps, "music", "techno", {}, ["video_9x16"], ["wav24"], "x.wav", "/tmp/o", plan=True)
    plan = json.loads(run(argv).stdout)
    assert [s["step"] for s in plan["steps"]] == ["master", "clip", "video_9x16"]
    assert plan["formats"] == []


# ------------------------------------------------------------- renders

def check_run(argv, timeout=900):
    r = run(argv, timeout=timeout)
    assert r.returncode == 0, r.stderr[-3000:]
    ev = events(r.stdout)
    assert ev and ev[0]["event"] == "plan", ev[:2]
    top = [e for e in ev if "parent" not in e]
    manifest = [e for e in top if e["event"] == "manifest"]
    assert len(manifest) == 1 and top[-1]["event"] == "manifest"
    planned = [s["step"] for s in ev[0]["steps"]]
    started = [e["step"] for e in top if e["event"] == "step_start"]
    ended = [e["step"] for e in top if e["event"] == "step_end"]
    assert started == planned and ended == planned, (planned, started, ended)
    files = manifest[0]["files"]
    for f in files:
        assert Path(f["path"]).exists(), f
    return files, ev


def loudness(path):
    from music_common import ebur128
    return ebur128(Path(path))


@pytest.mark.slow
def test_speech_formats_and_events(caps, tracks, tmp_path):
    argv = pc.build_argv(caps, "speech", None, {"trim_silence": True, "title": "Ep 1"}, ["master"],
                         ["wav24", "wav16", "flac", "mp3"], str(tracks["speech"]), str(tmp_path))
    files, _ = check_run(argv)
    fmts = {f["format"]: f["path"] for f in files if f["kind"] == "master"}
    assert set(fmts) == {"wav24", "wav16", "flac", "mp3"}
    for f, path in fmts.items():
        m = loudness(path)
        assert abs(m["integrated_lufs"] - (-16.0)) < 1.0, (f, m)


@pytest.mark.slow
def test_nature_formats_and_skips(caps, tracks, tmp_path):
    argv = pc.build_argv(caps, "nature", "rain", {"place": "porch"}, ["master_long"], ["wav24", "flac", "mp3"],
                         str(tracks["rain"]), str(tmp_path))
    assert "--skip-short" in argv
    files, _ = check_run(argv)
    kinds = {(f["kind"], f["format"]) for f in files}
    assert kinds >= {("master_long", "wav24"), ("master_long", "flac"), ("master_long", "mp3")}
    assert not any(k == "master_short" for k, _ in kinds)


@pytest.mark.slow
def test_music_release_toggles_formats_and_video(caps, tracks, tmp_path):
    values = {"title": "Low Smoke", "glue": False, "fade_out": 2.0, "max_seconds": 2, "clip_length": "15"}
    argv = pc.build_argv(caps, "music", "trap", values, ["master", "clip", "video_9x16", "captions"],
                         ["wav24", "flac", "mp3"], str(tracks["trap"]), str(tmp_path))
    files, ev = check_run(argv, timeout=1800)
    kinds = {(f["kind"], f["format"]) for f in files}
    assert kinds >= {("master", "wav24"), ("master", "flac"), ("master", "mp3"), ("clip", "wav"),
                     ("video_9x16", "mp4"), ("captions", "txt")}
    m = loudness(next(f["path"] for f in files if f["kind"] == "master" and f["format"] == "wav24"))
    assert abs(m["integrated_lufs"] - (-14.0)) <= 0.35 and m["true_peak_dbtp"] <= -0.95, m
    assert "acompressor" not in (tmp_path / "audio" / "REPORT.txt").read_text()  # --no-glue reached the chain
    frames = [e for e in ev if e.get("detail") == "frames" and e.get("parent") == "video_9x16"]
    assert frames and frames[-1]["done"] == frames[-1]["total"]


@pytest.mark.slow
def test_defaults_unchanged_without_new_flags(tracks, tmp_path):
    """Existing command lines keep producing exactly what they used to."""
    r = run([sys.executable, "process_music_wav.py", str(tracks["psytrance"]), "--genre", "psytrance",
             "--no-analysis", "--out-dir", str(tmp_path / "m")])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "@@progress" not in r.stdout
    assert sorted(p.name for p in (tmp_path / "m").glob("master*")) == ["master_16bit_44k1.wav", "master_24bit_48k.wav"]
    r = run([sys.executable, "process_speech_wav.py", str(tracks["speech"]), "--out-dir", str(tmp_path / "s")])
    assert r.returncode == 0, r.stderr[-2000:]
    assert sorted(p.name for p in (tmp_path / "s").glob("master*")) == ["master.wav"]
