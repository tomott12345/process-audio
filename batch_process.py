#!/usr/bin/env python3
"""Batch-run process_field_wav.py over many takes without retyping flags each time.

Two ways to describe a batch:

1. One label/place applied to every WAV in a folder:
     batch_process.py --dir ./card_dump --label insects --place woods

2. A manifest CSV for mixed content on one card (one row per file; label and
   place are required, the rest optional):
     batch_process.py --manifest takes.csv

   takes.csv:
     file,label,place,long,loop,start,end
     porch_rain.wav,rain,porch,clean,auto,,
     brook_1.wav,brook,brook,clean,never,90,237

Any flags after `--` are forwarded to every process_field_wav.py call, e.g.:
     batch_process.py --dir ./card_dump --label birds --place woods -- --plan-only

Continues past a single file's failure rather than aborting the whole batch;
prints a pass/fail summary at the end and exits non-zero if anything failed.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROCESS = SCRIPT_DIR / "process_field_wav.py"

MANIFEST_FLAG_COLUMNS = {
    "label": "--label", "place": "--place", "long": "--long", "loop": "--loop",
    "start": "--start", "end": "--end", "out_dir": "--out-dir",
    "still_16x9": "--still-16x9", "still_9x16": "--still-9x16",
}


def run_one(wav: Path, extra_flags: list[str]) -> tuple[bool, str]:
    cmd = ["python3", str(PROCESS), str(wav), *extra_flags]
    print("+ " + " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode == 0, wav.name


def from_dir(directory: Path, label: str, place: str, extra_flags: list[str]) -> list[tuple[bool, str]]:
    wavs = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".wav", ".wave"))
    if not wavs:
        print(f"no .wav files found in {directory}", file=sys.stderr)
    results = []
    for wav in wavs:
        ok, name = run_one(wav, ["--label", label, "--place", place, *extra_flags])
        results.append((ok, name))
    return results


def from_manifest(manifest: Path, extra_flags: list[str]) -> list[tuple[bool, str]]:
    results = []
    with manifest.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "file" not in (reader.fieldnames or []):
            raise SystemExit(f"manifest {manifest} needs a 'file' column")
        for row in reader:
            wav = Path(row["file"]).expanduser()
            if not wav.is_absolute():
                wav = (manifest.parent / wav).resolve()
            flags = list(extra_flags)
            for col, flag in MANIFEST_FLAG_COLUMNS.items():
                val = (row.get(col) or "").strip()
                if val:
                    flags.extend([flag, val])
            ok, name = run_one(wav, flags)
            results.append((ok, name))
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, help="Folder of WAVs to process with one shared label/place")
    p.add_argument("--label", help="Required with --dir")
    p.add_argument("--place", default="unknown", help="Used with --dir")
    p.add_argument("--manifest", type=Path, help="CSV with per-file file,label,place[,long,loop,start,end,...]")
    p.add_argument(
        "extra", nargs=argparse.REMAINDER,
        help="Any remaining flags (after --) are forwarded to every process_field_wav.py call",
    )
    args = p.parse_args()

    if not args.dir and not args.manifest:
        raise SystemExit("pass --dir DIR --label X --place Y, or --manifest FILE.csv")
    if args.dir and args.manifest:
        raise SystemExit("pass either --dir or --manifest, not both")
    if args.dir and not args.label:
        raise SystemExit("--dir requires --label")

    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.dir:
        results = from_dir(args.dir, args.label, args.place, extra)
    else:
        results = from_manifest(args.manifest, extra)

    print("\n=== BATCH SUMMARY ===")
    failed = [name for ok, name in results if not ok]
    for ok, name in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    print(f"\n{len(results) - len(failed)}/{len(results)} succeeded")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
