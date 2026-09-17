"""Shared NWB layout for sleep scoring — the contract between the tracker and
the scoring GUI.

ONE FILE PER SESSION: ``<op>/<Rat>_<YYYYMMDD>.nwb``. The tracker's **step 8**
creates it and writes everything the scorer needs; **step w** (behaviour /
trials) and **step u** (units) then append to that same file in ``r+`` mode.
Nothing rewrites it from scratch, so scorings stored inside survive the rest
of the pipeline.

Layout
------
``acquisition/``
    ``lfp``            TimeSeries ``(n_samples, n_channels)`` in uV, per-sample
                       timestamps (seconds). Written by step 8; step w skips it
                       when already present.
    ``emg_from_lfp``   TimeSeries ``(n,)`` normalised EMG-from-LFP (~5 Hz).
    ``motion``         TimeSeries ``(n,)`` accelerometer movement magnitude.

``processing/sleep/``
    ``sleep_channels``       TimeSeries carrying the per-rat cortex / sr / pyr
                             tetrode numbers as JSON in its ``description``.
    ``states_<scorer>``      TimeIntervals — one row per contiguous scored epoch
                             (``start_time``, ``stop_time``, ``state``,
                             ``state_code``); JSON metadata in ``description``.
    ``events_<scorer>``      TimeSeries of numbered event marks (optional).

A *scoring* is identified by its ``scorer`` name; re-saving under the same name
replaces that scorer's tables, so re-opening and continuing a scoring never
spawns a second copy. ``strip_scorings`` writes a copy with every scoring
removed — the version to hand to students, with no ground truth in it.

This file is kept byte-identical in both repos:
  * HM_Tracker_2025/src/nwb/sleep_nwb.py
  * HM_rat_sleep_score/python/sleep_nwb.py
Edit one, copy to the other.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- container names (the contract) ---------------------------------------- #
LFP_NAME = "lfp"
EMG_NAME = "emg_from_lfp"
MOTION_NAME = "motion"
SLEEP_MODULE = "sleep"
SLEEP_CHANNELS_NAME = "sleep_channels"
STATES_PREFIX = "states_"
EVENTS_PREFIX = "events_"

# HM state codes, shared with the editor (1 WAKE / 3 NREM / 5 REM; 0 unscored).
STATE_NAMES = {0: "none", 1: "awake", 3: "NREM", 5: "REM"}


def _slug(name: str) -> str:
    """Filesystem/NWB-safe token for a scorer name ('Sachu Riga' -> 'Sachu_Riga')."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "unknown")).strip("_")
    return s or "unknown"


# ---------------------------------------------------------------------------- #
#  Locating the session NWB
# ---------------------------------------------------------------------------- #
def session_nwb_name(prefix: str) -> str:
    """``Rat6_20260629_143022_`` (a step-8 file prefix) -> ``Rat6_20260629.nwb``.

    Matches the name ``create_nwb.py`` derives from the session's coordinate
    CSV, so step 8 and step w land on the same file.
    """
    m = re.match(r"^([A-Za-z]+\d+)_(\d{8})", str(prefix or ""))
    return f"{m.group(1)}_{m.group(2)}.nwb" if m else "session.nwb"


def find_session_nwb(folder):
    """The session NWB for an op folder, or None.

    Accepts the op folder itself or its ``LFP_Output`` subfolder (the scoring
    GUI is usually pointed at the latter), and ignores ``*.tmp.nwb`` leftovers.
    """
    folder = Path(folder)
    roots = [folder, folder.parent] if folder.name == "LFP_Output" else [folder]
    for root in roots:
        cands = [p for p in sorted(root.glob("*.nwb"))
                 if not p.name.endswith(".tmp.nwb")]
        if cands:
            return cands[0]
    return None


# ---------------------------------------------------------------------------- #
#  Reading the scoring inputs
# ---------------------------------------------------------------------------- #
def read_sleep_inputs(nwb_path, lazy=True):
    """Read everything the scorer needs from ``nwb_path``.

    Returns a dict with ``lfp`` (an h5py-backed ``(n_samples, n_channels)``
    array when ``lazy``, so channels are sliced without loading the session),
    ``lfp_timestamps``, ``fs``, ``emg``/``emg_timestamps``,
    ``motion``/``motion_timestamps``, ``sleep_channels`` and the open ``io``
    handle (close it via ``close_inputs``). Missing pieces come back as None.
    """
    from pynwb import NWBHDF5IO

    io = NWBHDF5IO(str(nwb_path), mode="r")
    nwbfile = io.read()
    out = {"io": io, "nwbfile": nwbfile, "path": str(nwb_path)}

    def _acq(name):
        try:
            return nwbfile.get_acquisition(name)
        except Exception:
            return None

    lfp = _acq(LFP_NAME)
    if lfp is not None:
        out["lfp"] = lfp.data if lazy else np.asarray(lfp.data)
        ts = np.asarray(lfp.timestamps[:]) if lfp.timestamps is not None else None
        out["lfp_timestamps"] = ts
        out["fs"] = _rate_from(ts, getattr(lfp, "rate", None))
    else:
        out["lfp"] = out["lfp_timestamps"] = out["fs"] = None

    for key, name in (("emg", EMG_NAME), ("motion", MOTION_NAME)):
        series = _acq(name)
        if series is None:
            out[key] = out[f"{key}_timestamps"] = None
            continue
        out[key] = np.asarray(series.data[:]).ravel()
        out[f"{key}_timestamps"] = (np.asarray(series.timestamps[:]).ravel()
                                    if series.timestamps is not None else None)

    out["sleep_channels"] = _read_sleep_channels(nwbfile)
    if not lazy:
        io.close()
        out["io"] = None
    return out


def close_inputs(inputs):
    """Close the handle returned by ``read_sleep_inputs(lazy=True)``."""
    io = (inputs or {}).get("io")
    if io is not None:
        try:
            io.close()
        except Exception:
            pass


def _rate_from(timestamps, rate=None):
    """Sampling rate from per-sample timestamps (median step), else ``rate``."""
    if timestamps is not None and np.size(timestamps) > 1:
        dt = float(np.median(np.diff(np.asarray(timestamps[:200000]))))
        if dt > 0:
            return 1.0 / dt
    return float(rate) if rate else None


def _sleep_module(nwbfile, create=False):
    mod = nwbfile.processing.get(SLEEP_MODULE)
    if mod is None and create:
        mod = nwbfile.create_processing_module(
            name=SLEEP_MODULE,
            description="Sleep scoring: per-rat channel choices and one "
                        "TimeIntervals table per scorer.")
    return mod


def _read_sleep_channels(nwbfile):
    """The cortex / sr / pyr tetrode numbers stored by step 8, or None."""
    mod = _sleep_module(nwbfile)
    if mod is None or SLEEP_CHANNELS_NAME not in mod.data_interfaces:
        return None
    try:
        return json.loads(mod[SLEEP_CHANNELS_NAME].description)
    except Exception:
        return None


# ---------------------------------------------------------------------------- #
#  Writing the scoring inputs (tracker step 8)
# ---------------------------------------------------------------------------- #
def add_sleep_inputs(nwbfile, lfp=None, lfp_timestamps=None, emg=None,
                     emg_timestamps=None, motion=None, motion_timestamps=None,
                     sleep_channels=None, lfp_unit="uV"):
    """Add the scorer's inputs to ``nwbfile``. Existing containers are left
    alone, so this is safe to re-run against a session that step w already
    populated. Returns the list of names actually added."""
    from pynwb import TimeSeries

    added = []

    def _add_acq(name, data, ts, unit, description):
        if data is None:
            return
        try:
            nwbfile.get_acquisition(name)
            return                      # already there — never clobber
        except Exception:
            pass
        ts = None if ts is None else np.asarray(ts).ravel()
        # A chunk iterator / H5DataIO streams straight to HDF5 (a multi-GB LFP
        # never lands in RAM); only a plain array can be length-reconciled here.
        if isinstance(data, (np.ndarray, list, tuple)):
            data = np.asarray(data)
            if ts is not None and data.shape[0] != ts.shape[0]:
                n = min(data.shape[0], ts.shape[0])
                data, ts = data[:n], ts[:n]
        kw = {"timestamps": ts} if ts is not None else {"rate": 1.0}
        nwbfile.add_acquisition(TimeSeries(name=name, data=data, unit=unit,
                                           description=description, **kw))
        added.append(name)

    _add_acq(LFP_NAME, lfp, lfp_timestamps, lfp_unit,
             "LFP voltage, one column per channel.")
    _add_acq(EMG_NAME, emg, emg_timestamps, "normalized",
             "EMG-from-LFP (Buzsaki cross-channel correlation), 0-1 normalised.")
    _add_acq(MOTION_NAME, motion, motion_timestamps, "g",
             "Accelerometer (IMU) movement magnitude.")

    if sleep_channels:
        mod = _sleep_module(nwbfile, create=True)
        if SLEEP_CHANNELS_NAME not in mod.data_interfaces:
            mod.add(TimeSeries(
                name=SLEEP_CHANNELS_NAME, data=[0], unit="n/a", rate=1.0,
                description=json.dumps({k: _jsonable(v)
                                        for k, v in dict(sleep_channels).items()})))
            added.append(SLEEP_CHANNELS_NAME)
    return added


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ---------------------------------------------------------------------------- #
#  Scorings
# ---------------------------------------------------------------------------- #
def list_scorings(nwb_path):
    """Every scoring in the file, newest first.

    Each entry is a dict with ``scorer``, ``date``, ``n_bins``, ``name`` (the
    NWB table name) and ``label`` (what the "State scored by" dropdown shows).
    """
    from pynwb import NWBHDF5IO

    if not Path(nwb_path).is_file():
        return []
    with NWBHDF5IO(str(nwb_path), mode="r") as io:
        mod = _sleep_module(io.read())
        if mod is None:
            return []
        out = []
        for name, obj in mod.data_interfaces.items():
            if not name.startswith(STATES_PREFIX):
                continue
            meta = _meta_of(obj)
            meta["name"] = name
            meta.setdefault("scorer", name[len(STATES_PREFIX):])
            out.append(meta)
    out.sort(key=lambda m: (m.get("date") or "", m.get("scorer") or ""), reverse=True)
    for m in out:
        date = m.get("date") or "?"
        n = m.get("n_bins")
        m["label"] = f"{m.get('scorer')} · {date}" + (f" · {n} bins" if n else "")
    return out


def _meta_of(table):
    try:
        return dict(json.loads(table.description))
    except Exception:
        return {}


def read_scoring(nwb_path, scorer=None, name=None, timestamps=None):
    """Read one scoring back as per-bin state codes.

    Picks the scoring by NWB table ``name`` or by ``scorer``; with neither it
    takes the newest. Returns ``(states, bin_timestamps, meta)`` — ``states``
    aligned to ``timestamps`` when given, else to the bin grid the scoring was
    saved on. Returns ``(None, None, {})`` when there is nothing to read.
    """
    from pynwb import NWBHDF5IO

    entries = list_scorings(nwb_path)
    if not entries:
        return None, None, {}
    if name:
        entries = [e for e in entries if e["name"] == name] or entries
    elif scorer:
        entries = [e for e in entries if e.get("scorer") == scorer] or entries
    target = entries[0]["name"]

    with NWBHDF5IO(str(nwb_path), mode="r") as io:
        nwbfile = io.read()
        mod = _sleep_module(nwbfile)
        table = mod[target]
        meta = _meta_of(table)
        starts = np.asarray(table["start_time"][:], dtype=float)
        stops = np.asarray(table["stop_time"][:], dtype=float)
        codes = np.asarray(table["state_code"][:], dtype=int)
        events = _read_events(mod, meta.get("scorer", ""))

    if timestamps is None:
        dt = float(meta.get("dt") or 1.0)
        t0 = float(meta.get("t0", starts[0] if starts.size else 0.0))
        n = int(meta.get("n_bins") or (round((stops[-1] - t0) / dt) if stops.size else 0))
        timestamps = t0 + np.arange(n) * dt
    timestamps = np.asarray(timestamps, dtype=float)

    states = np.zeros(timestamps.size, dtype=int)
    for a, b, c in zip(starts, stops, codes):
        states[(timestamps >= a) & (timestamps <= b)] = c
    meta["events"] = events
    meta["name"] = target
    return states, timestamps, meta


def _read_events(mod, scorer):
    name = EVENTS_PREFIX + _slug(scorer)
    if name not in mod.data_interfaces:
        return np.zeros((0, 2))
    ev = mod[name]
    nums = np.asarray(ev.data[:]).ravel()
    times = np.asarray(ev.timestamps[:]).ravel()
    return np.column_stack([nums, times]) if nums.size else np.zeros((0, 2))


def write_scoring(nwb_path, states, timestamps, scorer, events=None,
                  date=None, extra=None):
    """Save one scorer's result into the session NWB (in place, ``r+``).

    Stored as a ``TimeIntervals`` of contiguous epochs under
    ``processing/sleep/states_<scorer>``. Re-saving the same scorer REPLACES
    that scorer's tables rather than adding a second copy, so re-opening and
    continuing a scoring keeps one row per scorer. Returns the table name.
    """
    from pynwb import NWBHDF5IO, TimeSeries
    from pynwb.epoch import TimeIntervals

    states = np.asarray(states, dtype=int).ravel()
    timestamps = np.asarray(timestamps, dtype=float).ravel()
    n = min(states.size, timestamps.size)
    states, timestamps = states[:n], timestamps[:n]
    slug = _slug(scorer)
    table_name = STATES_PREFIX + slug
    events_name = EVENTS_PREFIX + slug
    dt = float(np.median(np.diff(timestamps))) if n > 1 else 1.0
    date = date or datetime.now(timezone.utc).astimezone().date().isoformat()

    meta = {"scorer": str(scorer), "date": date, "n_bins": int(n), "dt": dt,
            "t0": float(timestamps[0]) if n else 0.0,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "state_names": {str(k): v for k, v in STATE_NAMES.items()}}
    meta.update(extra or {})

    # drop any previous tables for this scorer first, at the HDF5 level: pynwb
    # can only add to an r+ file, so replacing means deleting before reopening.
    _h5_delete(nwb_path, [table_name, events_name])

    io = NWBHDF5IO(str(nwb_path), mode="r+")
    try:
        nwbfile = io.read()
        mod = _sleep_module(nwbfile, create=True)

        table = TimeIntervals(name=table_name, description=json.dumps(meta))
        table.add_column(name="state", description="Sleep state name.")
        table.add_column(name="state_code",
                         description="HM state code: 1 awake, 3 NREM, 5 REM.")
        for s, e, code in _runs(states):
            if code == 0:
                continue                       # unscored bins leave a gap
            stop = timestamps[e] if e < n else timestamps[-1] + dt
            table.add_row(start_time=float(timestamps[s]), stop_time=float(stop),
                          state=STATE_NAMES.get(int(code), str(code)),
                          state_code=int(code))
        mod.add(table)

        events = np.asarray(events if events is not None else
                            np.zeros((0, 2)), dtype=float).reshape(-1, 2)
        if events.size:
            mod.add(TimeSeries(name=events_name, data=events[:, 0],
                               timestamps=events[:, 1], unit="n/a",
                               description=f"Event marks placed by {scorer}."))
        io.write(nwbfile)
    finally:
        io.close()
    return table_name


def _h5_delete(nwb_path, names):
    """Delete containers from ``processing/sleep`` at the HDF5 level.

    pynwb can only add to a file opened ``r+`` — detaching an already-written
    container from its ProcessingModule corrupts the build on the next write.
    Removing the group with h5py instead is reliable: a ProcessingModule lists
    its children by group membership, so the container is simply gone when
    pynwb next reads the file. Missing names are ignored. Returns the names
    actually deleted.

    (HDF5 does not reclaim the freed bytes; the file keeps its size until it is
    rewritten. That is fine here — scoring tables are a few KB.)
    """
    import h5py

    names = [n for n in names if n]
    if not names or not Path(nwb_path).is_file():
        return []
    deleted = []
    with h5py.File(str(nwb_path), "r+") as f:
        mod = f.get(f"/processing/{SLEEP_MODULE}")
        if mod is None:
            return []
        for name in names:
            if name in mod:
                del mod[name]
                deleted.append(name)
    return deleted


def _runs(states):
    """Yield ``(start, end_exclusive, value)`` for each contiguous run."""
    n = len(states)
    if n == 0:
        return
    start = 0
    for i in range(1, n + 1):
        if i == n or states[i] != states[start]:
            yield start, i, int(states[start])
            start = i


# ---------------------------------------------------------------------------- #
#  Ground-truth isolation
# ---------------------------------------------------------------------------- #
def strip_scorings(nwb_path, out_path, keep=()):
    """Write a copy of ``nwb_path`` to ``out_path`` with scorings removed.

    This is the version to hand out: identical recording, no ground truth.
    ``keep`` optionally names scorers to retain (e.g. a demo scoring). Returns
    the list of scorer names that were removed.
    """
    from pynwb import NWBHDF5IO

    out_path = Path(out_path)
    if out_path.resolve() == Path(nwb_path).resolve():
        raise ValueError("out_path must differ from nwb_path")
    keep_slugs = {_slug(k) for k in keep}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nwb_path, out_path)

    doomed, removed = [], []
    for entry in list_scorings(out_path):
        slug = entry["name"][len(STATES_PREFIX):]
        if slug in keep_slugs:
            continue
        doomed += [entry["name"], EVENTS_PREFIX + slug]
        removed.append(entry.get("scorer") or slug)
    _h5_delete(out_path, doomed)
    return removed
