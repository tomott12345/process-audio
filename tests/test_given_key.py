"""--key: a song's known key replaces the estimate everywhere the key is
reported (captions, tags, REPORT.txt, music_analysis.json). Without it,
nothing changes.

  pytest tests/test_given_key.py
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
from music_analyze import key_label, parse_key  # noqa: E402
from music_common import genre_recipe, load_recipes  # noqa: E402
from release import captions  # noqa: E402


def run(argv, **kw):
    return subprocess.run(argv, cwd=REPO, text=True, capture_output=True, **kw)


@pytest.mark.parametrize("text,label", [
    ("F dorian", "F dorian (5A)"),       # Eb major's notes -> the 5 pair, minor side
    ("A minor", "A minor (8A)"),
    ("E minor", "E minor (9A)"),
    ("C major", "C major (8B)"),
    ("G mixolydian", "G mixolydian (8B)"),
    ("F lydian", "F lydian (8B)"),
    ("E phrygian", "E phrygian (8A)"),
    ("D# minor_pentatonic", "D# minor pentatonic (2A)"),
    ("C# major_pentatonic", "C# major pentatonic (3B)"),
    ("D hirajoshi", "D hirajoshi (7A)"),
    ("Ebm", "Eb minor (2A)"),
    ("f# Minor", "F# minor (11A)"),
    ("Bb", "Bb major (6B)"),
    ("C whole tone", "C whole tone"),    # no Camelot code: name only
])
def test_parse_key(text, label):
    k = parse_key(text)
    assert key_label(k) == label
    assert k["source"] == "given"


@pytest.mark.parametrize("text", ["", "H minor", "dorian", "C## major"])
def test_parse_key_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_key(text)


def test_option_is_exposed():
    music = next(p for p in pc.capabilities()["pipelines"] if p["id"] == "music")
    o = next(o for o in music["options"] if o["id"] == "key")
    assert (o["group"], o["kind"], o["flag"], o["default"], o["advanced"]) == ("tags", "text", "--key", "", True)
    caps = pc.capabilities()
    argv = pc.build_argv(caps, "music", "ambient", {"key": "F dorian"}, ["captions"], ["wav24"], "x.wav", "/o")
    assert "--key=F dorian" in argv
    argv = pc.build_argv(caps, "music", "ambient", {"key": ""}, ["captions"], ["wav24"], "x.wav", "/o")
    assert not any(a.startswith("--key") for a in argv)


def plan(*extra):
    r = run([sys.executable, "release.py", "x.wav", "--genre", "ambient", "--plan-only", "--json", *extra])
    return r, (json.loads(r.stdout) if r.returncode == 0 else None)


def test_release_passes_key_to_every_analysis():
    _, doc = plan("--key", "F dorian")
    master = next(s for s in doc["steps"] if s["step"] == "master")
    assert "--key=F dorian" in master["command"]
    _, doc = plan("--key", "F dorian", "--already-mastered")
    analyze = next(s for s in doc["steps"] if s["step"] == "analyze")
    assert "--key=F dorian" in analyze["command"]
    _, doc = plan()
    assert not any("--key" in a for s in doc["steps"] for a in (s["command"] or []))


def test_release_rejects_unreadable_key():
    r, _ = plan("--key", "H minor")
    assert r.returncode == 2 and "can't read the key" in r.stderr


def test_captions_use_given_key():
    recipe = genre_recipe(load_recipes(), "ambient")
    a = {"tempo": {"has_grid": True, "bpm": 75.99}, "duration": 120.0,
         "key": dict(parse_key("F dorian"), detected={"name": "C minor", "camelot": "5A"})}
    text = captions("Docker Test", None, "ambient", recipe, a, None, "Made by a machine.")
    assert "Docker Test -- ambient, 75.99 BPM, F dorian (5A)." in text
    assert "C minor" not in text
    a["key"] = parse_key("C whole tone")
    assert "75.99 BPM, C whole tone." in captions("T", None, "ambient", recipe, a, None, "m")


@pytest.fixture(scope="module")
def tracks(tmp_path_factory):
    return make_all(tmp_path_factory.mktemp("tracks"))


def test_release_end_to_end(tracks, tmp_path):
    """An already-mastered track, captions only: the given key reaches
    the analysis JSON, captions.txt and REPORT.txt; the estimate is kept."""
    r = run([sys.executable, "release.py", str(tracks["techno"]), "--genre", "techno", "--title", "Docker Test",
             "--already-mastered", "--outputs", "captions", "--key", "F dorian", "--out-dir", str(tmp_path)],
            timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    a = json.loads((tmp_path / "audio" / "music_analysis.json").read_text())
    assert (a["key"]["name"], a["key"]["camelot"], a["key"]["source"]) == ("F dorian", "5A", "given")
    assert a["key"]["detected"]["name"]  # the estimate is still recorded
    assert not any("key is ambiguous" in f for f in a["flags"])
    assert "F dorian (5A)" in (tmp_path / "captions.txt").read_text()
    assert "key=F dorian (5A) (given)" in (tmp_path / "REPORT.txt").read_text()
