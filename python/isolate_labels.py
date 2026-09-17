"""Separate the scorings (ground truth) from a session NWB.

The session ``.nwb`` holds both the recording and every scorer's labels, so it
cannot be handed to students as-is. This makes the two halves:

  * a **student copy** — the same recording with the scorings removed, so it
    opens in the scoring GUI with nothing pre-filled;
  * optionally the **labels on their own** as ``.npz`` files, kept wherever you
    keep the ground truth.

The original NWB is never modified.

Examples::

    # what's in there
    python isolate_labels.py --nwb Rat6_20260629.nwb --list

    # student copy next to the original, ground truth pulled out to ./truth/
    python isolate_labels.py --nwb Rat6_20260629.nwb --extract truth

    # keep one demo scoring in the student copy
    python isolate_labels.py --nwb Rat6_20260629.nwb --keep Sachuriga
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sleep_nwb_store as snwb


def extract_labels(nwb_path, out_dir):
    """Save every scoring as ``<out_dir>/<session>__<scorer>.npz``.

    Each file carries ``states`` (per 1 s bin), ``timestamps``, ``events`` and
    the scorer's name — enough to reload or compare against later.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(nwb_path).stem
    written = []
    for entry in snwb.list_scorings(nwb_path):
        states, timestamps, meta = snwb.read_scoring(nwb_path, name=entry["name"])
        if states is None:
            continue
        scorer = meta.get("scorer") or entry["name"]
        path = out_dir / f"{stem}__{snwb._slug(scorer)}.npz"
        np.savez(path,
                 states=np.asarray(states, dtype=int),
                 timestamps=np.asarray(timestamps, dtype=float),
                 events=np.asarray(meta.get("events", np.zeros((0, 2))), dtype=float),
                 labeled_by=np.array(scorer),
                 date=np.array(meta.get("date", "")))
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Split a session NWB into a label-free student copy and the "
                    "scorings kept separately.")
    ap.add_argument("--nwb", required=True, help="Session .nwb file.")
    ap.add_argument("--out", default=None,
                    help="Student copy (default: <stem>_nolabels.nwb beside the original).")
    ap.add_argument("--keep", action="append", default=[], metavar="SCORER",
                    help="Scorer to KEEP in the student copy (repeatable).")
    ap.add_argument("--extract", default=None, metavar="DIR",
                    help="Also write each scoring to DIR as .npz (the ground truth).")
    ap.add_argument("--list", action="store_true",
                    help="Only list the scorings in the file, change nothing.")
    args = ap.parse_args()

    nwb = Path(args.nwb)
    if not nwb.is_file():
        print(f"No such file: {nwb}")
        return 1

    entries = snwb.list_scorings(nwb)
    print(f"{nwb.name}: {len(entries)} scoring(s)")
    for e in entries:
        print(f"   · {e['label']}")
    if args.list:
        return 0
    if not entries:
        print("Nothing to isolate.")
        return 0

    if args.extract:
        written = extract_labels(nwb, args.extract)
        print(f"\nGround truth -> {args.extract}/")
        for p in written:
            print(f"   · {p.name}")

    out = Path(args.out) if args.out else nwb.with_name(nwb.stem + "_nolabels.nwb")
    removed = snwb.strip_scorings(nwb, out, keep=args.keep)
    kept = [e["label"] for e in snwb.list_scorings(out)]
    print(f"\nStudent copy -> {out}")
    print(f"   removed: {', '.join(removed) if removed else '(none)'}")
    print(f"   kept:    {', '.join(kept) if kept else '(no scorings — clean)'}")
    print(f"   original {nwb.name} untouched ({len(entries)} scoring(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
