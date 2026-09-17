"""Put scorings that live in ``results/`` files into the session NWB.

The counterpart of :mod:`isolate_labels`, which takes them back out. Scoring
done in the GUI is written into the NWB as it is saved, but sessions scored
before the NWB existed only have their ``results_<date>_<scorer>.npz`` (and the
MATLAB-compatible ``.mat`` beside it). This walks those files into
``processing/sleep/states_<scorer>`` so the session is one file again.

Re-running is safe: a scorer already in the file is replaced, not duplicated,
so importing the same results twice leaves one entry.

Depends only on numpy, scipy and pynwb — no GUI imports — so it runs in the
environment that has pynwb even when that one has no Qt.

Examples::

    # everything in <op>/LFP_Output/results into the session NWB found in <op>
    python import_labels.py --session C:/data/sleep/op1

    # one file, into a named NWB, previewing first
    python import_labels.py --nwb Rat6_20260707.nwb \
                            --labels results/results_2026-09-17_sachi.npz --dry_run

    # what the NWB holds now
    python import_labels.py --nwb Rat6_20260707.nwb --list
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sleep_nwb as snwb                                     # noqa: E402

RESULTS_GLOB = "results_*"
RESULTS_DIRNAME = "results"
# A scoring may start a bin before the first LFP sample or end a bin after the
# last; more than this and the two are not the same recording.
SPAN_TOLERANCE_S = 5.0


def _scorer_of(data, keys, path):
    """Scorer name: the file's own ``labeled_by``, else the one in its name.

    Same rule as the editor's ``_labeled_by_of`` (state_editor.py), kept here so
    this tool needs no GUI imports.
    """
    if "labeled_by" in keys:
        v = np.asarray(data["labeled_by"]).ravel()
        name = str(v[0] if v.size else data["labeled_by"]).strip()
        if name and name != "unknown":
            return name
    m = re.match(r"results_\d{4}-\d{2}-\d{2}_(.+)$", Path(path).stem)
    return m.group(1).replace("_", " ") if m else None


def _date_of(path):
    """The date in ``results_<date>_<scorer>``, else the file's own mtime."""
    m = re.match(r"results_(\d{4}-\d{2}-\d{2})_", Path(path).stem)
    if m:
        return m.group(1)
    ts = Path(path).stat().st_mtime
    return datetime.fromtimestamp(ts, timezone.utc).astimezone().date().isoformat()


def load_results(path):
    """Read a scoring file. Returns a dict of states/timestamps/events/scorer/date.

    Accepts the editor's ``.npz`` and the MATLAB-compatible ``.mat``. The legacy
    light/drowsy code 2 is folded into WAKE, as everywhere else in the toolkit.
    """
    path = Path(path)
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        data = loadmat(str(path))
        keys = set(data.keys())
    else:
        data = np.load(str(path), allow_pickle=True)
        keys = set(data.files)
    if "states" not in keys:
        raise ValueError(f"{path.name} has no 'states' array")

    states = np.asarray(data["states"]).ravel().astype(int)
    states[states == 2] = 1                       # legacy drowsy -> WAKE
    if "timestamps" in keys:
        ts = np.asarray(data["timestamps"]).ravel().astype(float)
    else:                                          # .mat stores no time base
        ts = np.arange(states.size, dtype=float)
    events = np.zeros((0, 2))
    if "events" in keys:
        ev = np.asarray(data["events"], dtype=float)
        if ev.size:
            events = ev.reshape(-1, 2)
    return {"states": states, "timestamps": ts, "events": events,
            "scorer": _scorer_of(data, keys, path), "date": _date_of(path),
            "source": path}


def find_results(folder):
    """Scoring files under ``folder``, preferring the ``.npz`` of each pair.

    Both formats are written for the same scoring, and they hold the same
    labels, but only the ``.npz`` carries the time base and the scorer's name.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    npz = sorted(folder.glob(RESULTS_GLOB + ".npz"))
    stems = {p.stem for p in npz}
    mat = [p for p in sorted(folder.glob(RESULTS_GLOB + ".mat"))
           if p.stem not in stems]
    return npz + mat


def recording_span(nwb_path):
    """``(first, last)`` second of the recording in the NWB, or None."""
    inputs = None
    try:
        inputs = snwb.read_sleep_inputs(nwb_path, lazy=True)
        ts = inputs.get("lfp_timestamps")
        if ts is None or not len(ts):
            return None
        return float(ts[0]), float(ts[-1])
    except Exception:
        return None
    finally:
        if inputs is not None:
            snwb.close_inputs(inputs)


def check_alignment(scoring, span):
    """Complaints about a scoring's time base against the recording's.

    A scoring written on a different time base would be stored happily and read
    back misaligned, which is worse than not importing it, so this is checked
    before anything is written rather than left to be noticed later.
    """
    problems = []
    ts = scoring["timestamps"]
    if ts.size == 0:
        return ["no timestamps"]
    if span is not None:
        lo, hi = span
        if ts[0] < lo - SPAN_TOLERANCE_S:
            problems.append(f"starts {lo - ts[0]:.0f}s before the recording")
        if ts[-1] > hi + SPAN_TOLERANCE_S:
            problems.append(f"ends {ts[-1] - hi:.0f}s after the recording "
                            f"({ts[-1]:.0f}s vs {hi:.0f}s)")
    if ts.size > 1:
        dt = np.diff(ts)
        if not np.allclose(dt, dt[0], atol=1e-6):
            problems.append("bin spacing is not constant")
    return problems


def import_one(nwb_path, scoring, dry_run=False):
    """Write one loaded scoring into the NWB. Returns the table name."""
    if not scoring["scorer"]:
        raise ValueError(f"no scorer name in {scoring['source'].name} or its "
                         f"filename — pass --scorer")
    if dry_run:
        return snwb.STATES_PREFIX + snwb._slug(scoring["scorer"])
    return snwb.write_scoring(
        nwb_path, scoring["states"], scoring["timestamps"],
        scorer=scoring["scorer"], events=scoring["events"], date=scoring["date"],
        extra={"imported_from": scoring["source"].name})


def describe(scoring):
    """One line about a scoring: who, when, how long, what is in it."""
    st, ts = scoring["states"], scoring["timestamps"]
    names = {0: "unscored", 1: "W", 3: "N", 4: "I", 5: "R"}
    parts = [f"{names.get(int(c), c)} {100.0 * n / st.size:.1f}%"
             for c, n in zip(*np.unique(st, return_counts=True))]
    span = f"{ts[0]:.0f}-{ts[-1]:.0f}s" if ts.size else "empty"
    return (f"{scoring['scorer']!r} · {scoring['date']} · {st.size} bins · "
            f"{span} · " + "  ".join(parts))


def main():
    ap = argparse.ArgumentParser(
        description="Import results_*.npz/.mat scorings into the session NWB.")
    ap.add_argument("--nwb", help="Session NWB (default: the one found in --session).")
    ap.add_argument("--session", help="Session op folder holding the NWB and "
                                      "LFP_Output/results.")
    ap.add_argument("--labels", action="append", default=[],
                    help="A scoring file to import (repeatable).")
    ap.add_argument("--results_dir",
                    help="Folder of results_* files (default: "
                         "<session>/LFP_Output/results).")
    ap.add_argument("--scorer", help="Override the scorer name (single --labels only).")
    ap.add_argument("--list", action="store_true",
                    help="List the scorings already in the NWB and exit.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Show what would be imported, write nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Import even if the time base does not match the recording.")
    args = ap.parse_args()

    nwb_path = args.nwb
    if nwb_path is None:
        if not args.session:
            ap.error("give --nwb or --session")
        nwb_path = snwb.find_session_nwb(args.session)
        if nwb_path is None:
            ap.error(f"no .nwb found in {args.session}")
    nwb_path = Path(nwb_path)
    if not nwb_path.is_file():
        ap.error(f"{nwb_path} does not exist")
    print(f"NWB: {nwb_path}")

    if args.list:
        found = snwb.list_scorings(nwb_path)
        print(f"{len(found)} scoring(s) stored:")
        for m in found:
            print(f"  {m['label']}   ({m['name']})")
        return 0

    paths = [Path(p) for p in args.labels]
    if not paths:
        results_dir = args.results_dir
        if results_dir is None:
            base = Path(args.session) if args.session else nwb_path.parent
            for cand in (base / "LFP_Output" / RESULTS_DIRNAME,
                         base / RESULTS_DIRNAME):
                if cand.is_dir():
                    results_dir = cand
                    break
        if results_dir is None:
            ap.error("no results folder found — pass --results_dir or --labels")
        paths = find_results(results_dir)
        print(f"Results folder: {results_dir}  ({len(paths)} file(s))")
    if not paths:
        print("Nothing to import.")
        return 0
    if args.scorer and len(paths) > 1:
        ap.error("--scorer applies to a single --labels file")

    span = recording_span(nwb_path)
    if span:
        print(f"Recording spans {span[0]:.0f}-{span[1]:.0f}s")

    written, skipped = [], []
    for path in paths:
        try:
            scoring = load_results(path)
        except Exception as exc:
            print(f"  {path.name}: SKIPPED ({type(exc).__name__}: {exc})")
            skipped.append(path)
            continue
        if args.scorer:
            scoring["scorer"] = args.scorer
        print(f"  {path.name}: {describe(scoring)}")
        problems = check_alignment(scoring, span)
        if problems:
            msg = "; ".join(problems)
            if not args.force:
                print(f"      SKIPPED — {msg}. Pass --force to import anyway.")
                skipped.append(path)
                continue
            print(f"      WARNING — {msg} (importing because --force)")
        try:
            name = import_one(nwb_path, scoring, dry_run=args.dry_run)
        except Exception as exc:
            print(f"      FAILED ({type(exc).__name__}: {exc})")
            skipped.append(path)
            continue
        print(f"      {'would write' if args.dry_run else 'wrote'} "
              f"processing/sleep/{name}")
        written.append(path)

    print(f"\n{len(written)} imported, {len(skipped)} skipped"
          + ("  (dry run — nothing written)" if args.dry_run else ""))
    if written and not args.dry_run:
        print("The NWB now holds:")
        for m in snwb.list_scorings(nwb_path):
            print(f"  {m['label']}")
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
