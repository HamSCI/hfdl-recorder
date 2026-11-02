"""Batch analysis of saved squitter windows into a local sink table.

``run_batch`` walks ``<window_dir>/<radiod_id>/<YYYYMMDD>/*.npz`` (the window
keeper's files), times each squitter with ``time_squitter``, and inserts one
``squitter_timing`` row per window into sigmond's local sink (table
``squitter_timing``, mode ``hfdl``, ``local=True``).  A ``.done`` marker
beside each window records that it has been analysed, so a rerun inserts
nothing.  One run handles at most ``max_windows`` windows.

Each row carries the window metadata, every ``Arrival`` field, and three
derived values:

- ``arrival_input_index``: the arrival in radiod input-sample index,
  ``input_index0 + (arrival_utc - t0_utc) * input_samprate``.
- ``delay_counter_s``: the arrival converted to UTC through hf-timestd's
  index registration nearest in time, minus the nominal slot start.
- ``label_diff_s``: the status-pair label of x[0] minus its counter label
  (``t0_utc`` minus the registration's UTC of ``input_index0``).  This is
  the status path's own noise.

A registration is one T3 shadow minute row: ``t3_shadow.t6_anchor_input_index``
and ``t3_shadow.t6_anchor_utc_ns`` name one instant in index and in UTC.  Its
UTC for index i is ``anchor_utc + (i - anchor_index) / input_samprate``.

Matching a registration to a window's radiod run.  The row's
``counter_epoch_id`` is ``ep-<int(epoch_offset_s)>``, a per-process name that
hf-timestd documents as unfit for comparison (counter_epoch_tracker, C1), and
the row holds no epoch offset.  So the match is a time-consistency check: the
anchor-implied epoch, ``anchor_utc - anchor_index / f_in``, must lie within
``EPOCH_TOL_S`` (2 s, the epoch tracker's skew allowance) of the window's ``input_epoch_gps_s`` converted to UTC.  Two
radiod runs differ by hours.  Among consistent rows, the one whose anchor UTC
lies nearest ``t0_utc`` wins, and only inside ``REG_MAX_AGE_S``.

The host clock labels nothing here.  Window files and minute rows carry every
time this module uses.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from hfdl_recorder.timing.analyzer import time_squitter

logger = logging.getLogger(__name__)

TABLE = "squitter_timing"
MODE = "hfdl"
DEFAULT_REGISTRATION_DIR = "/var/lib/timestd/t3_shadow"
DEFAULT_MAX_WINDOWS = 2000
FLUSH_EVERY = 100               # windows per durable commit
REG_MAX_AGE_S = 120.0            # registration must lie this close to the window
EPOCH_TOL_S = 2.0                # anchor-implied epoch vs the window's own; the
                                 # tracker's skew allowance (B4 pairs reached 0.8 s)
PLAUSIBLE_DELAY_S = (-0.030, 0.100)   # R12: outside this, a delay is not a squitter's

_GPS_UNIX_S = 315_964_800        # GPS epoch minus Unix epoch
_GPS_LEAP_S = 18                 # GPS minus UTC

_NAN_FIELDS = ("arrival_utc", "delay_s", "snr_db", "peak_width_s",
               "second_peak_ratio", "second_peak_delay_s", "doppler_hz",
               "lobe_distortion", "prekey_edge_offset_s")


def _clean(v):
    """JSON/SQL-friendly: NaN and inf become None, numpy scalars become Python."""
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return v if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


class Registrations:
    """Read-only view of hf-timestd's T3 shadow minute rows, cached per file."""

    def __init__(self, directory: Path):
        self._dir = Path(directory)
        self._cache: dict[str, list[tuple[int, float]]] = {}

    def _day(self, day: str) -> list[tuple[int, float]]:
        if day in self._cache:
            return self._cache[day]
        out: list[tuple[int, float]] = []
        try:
            text = (self._dir / f"minutes-{day}.jsonl").read_text(errors="replace")
        except OSError:
            text = ""
        for line in text.splitlines():
            try:
                shadow = json.loads(line)["t3_shadow"]
                idx = float(shadow["t6_anchor_input_index"])
                ns = int(shadow["t6_anchor_utc_ns"])
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            if math.isfinite(idx):
                out.append((ns, idx))
        self._cache[day] = out
        return out

    def nearest(self, t0_utc: float, epoch_utc: float,
                f_in: float) -> Optional[tuple[int, float]]:
        """(anchor_utc_ns, anchor_index) of the usable row nearest t0_utc."""
        t0 = datetime.fromtimestamp(t0_utc, timezone.utc)
        best = None
        for d in (-1, 0, 1):
            day = (t0 + timedelta(days=d)).strftime("%Y%m%d")
            for ns, idx in self._day(day):
                age = abs(ns / 1e9 - t0_utc)
                if age > REG_MAX_AGE_S:
                    continue
                whole, rem = divmod(ns, 10**9)
                implied = (whole - epoch_utc) + rem / 1e9 - idx / f_in
                if abs(implied) > EPOCH_TOL_S:
                    continue
                if best is None or age < best[0]:
                    best = (age, ns, idx)
        return None if best is None else (best[1], best[2])


def _exclusions(row: dict) -> str:
    out = []
    if row.get("utc_sync") is False:
        out.append("utc_sync_false")
    if row["found"]:
        d = row["delay_counter_s"]
        if d is None:
            d = row["delay_s"]
        if d is not None and not (PLAUSIBLE_DELAY_S[0] <= d <= PLAUSIBLE_DELAY_S[1]):
            out.append("implausible_delay")
    return ",".join(out)


def _row_for(path: Path, regs: Registrations) -> dict:
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["x"])
        meta = json.loads(str(z["meta"]))
    t0 = float(meta["t0_utc"])
    nominal = float(meta["nominal_utc"])
    fs = float(meta["fs"])
    arr = time_squitter(x, fs, t0, nominal)

    row = {k: _clean(v) for k, v in meta.items()
           if k not in ("counter_map", "decimation")}
    row["counter_map"] = (json.dumps(meta["counter_map"], sort_keys=True)
                          if meta.get("counter_map") is not None else None)
    row["decimation"] = (json.dumps(meta["decimation"], sort_keys=True)
                         if meta.get("decimation") is not None else None)
    row["radiod_id"] = path.parent.parent.name
    row["window_file"] = path.name
    for k, v in asdict(arr).items():
        row[k] = _clean(v)
    row["arrival_input_index"] = None
    row["delay_counter_s"] = None
    row["label_diff_s"] = None

    idx0, f_in = meta.get("input_index0"), meta.get("input_samprate")
    epoch_gps = meta.get("input_epoch_gps_s")
    if idx0 is not None and f_in and epoch_gps is not None:
        idx0, f_in = float(idx0), float(f_in)
        epoch_utc = float(epoch_gps) + _GPS_UNIX_S - _GPS_LEAP_S
        if arr.found:
            row["arrival_input_index"] = idx0 + (arr.arrival_utc - t0) * f_in
        reg = regs.nearest(t0, epoch_utc, f_in)
        if reg is not None:
            a_ns, a_idx = reg
            a_whole, a_rem = divmod(a_ns, 10**9)
            # UTC of index i, less a whole-second reference: keeps the
            # subtractions between nearby numbers, so float precision holds.
            ref = math.floor(t0)

            def utc_minus_ref(i: float) -> float:
                return (a_whole - ref) + a_rem / 1e9 + (i - a_idx) / f_in

            row["label_diff_s"] = (t0 - ref) - utc_minus_ref(idx0)
            if arr.found:
                row["delay_counter_s"] = (
                    utc_minus_ref(row["arrival_input_index"]) - (nominal - ref))
    row["excluded"] = _exclusions(row)
    return row


def _error_row(path: Path, exc: Exception) -> dict:
    row = {"radiod_id": path.parent.parent.name, "window_file": path.name,
           "found": False, "reason": f"batch_error:{type(exc).__name__}",
           "arrival_input_index": None, "delay_counter_s": None,
           "label_diff_s": None, "excluded": ""}
    for k in _NAN_FIELDS:
        row[k] = None
    row["m1_shift"] = None
    row["flags"] = ""
    # Best effort: the name is <gs_id>-<freq_khz>-<frame_index>.npz
    parts = path.name.split(".")[0].split("-")
    if len(parts) == 3:
        try:
            row["gs_id"], row["freq_khz"], row["frame_index"] = (int(p) for p in parts)
        except ValueError:
            pass
    return row


def _default_writer_factory(**kw):
    from sigmond.hamsci_sink import Writer
    return Writer.from_env(**kw)


def _windows(window_dir: Path, radiod_id: Optional[str] = None):
    """Window files in order: radiod, day, then name."""
    for p in sorted(window_dir.glob(f"{radiod_id or '*'}/*/*.npz")):
        if p.name.endswith(".tmp"):
            continue
        yield p


def _meta_nominal(path: Path) -> Optional[float]:
    """nominal_utc from the window's meta alone, None if unreadable."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return float(json.loads(str(z["meta"]))["nominal_utc"])
    except Exception:
        return None


def _done(path: Path) -> Path:
    return path.with_name(path.name + ".done")


def run_batch(window_dir, *, registration_dir=DEFAULT_REGISTRATION_DIR,
              since: Optional[float] = None,
              radiod_id: Optional[str] = None,
              max_windows: int = DEFAULT_MAX_WINDOWS,
              csv_path=None,
              writer_factory: Optional[Callable[..., object]] = None) -> dict:
    """Analyse windows not yet marked done; returns the run's counts."""
    window_dir = Path(window_dir)
    regs = Registrations(Path(registration_dir))
    counts = {"analysed": 0, "found": 0, "errors": 0, "skipped_done": 0,
              "skipped_since": 0, "deferred": 0}
    rows: list[dict] = []
    pending: list[Path] = []        # inserted, not yet durable: no .done yet
    writer = None
    stopped = False
    factory = writer_factory or _default_writer_factory

    def commit() -> bool:
        """Flush; mark the pending windows done only if nothing stays buffered."""
        try:
            writer.flush()
        except Exception as exc:
            logger.error("sink flush raised %s: %s", type(exc).__name__, exc)
        if getattr(writer, "buffered", 0):
            logger.error("sink flush left %d rows buffered; %d windows stay "
                         "unmarked and run again", writer.buffered, len(pending))
            return False
        for p in pending:
            try:
                _done(p).touch()
            except OSError as exc:
                logger.warning("cannot mark %s done: %s", p, exc)
        pending.clear()
        return True

    for path in _windows(window_dir, radiod_id):
        if _done(path).exists():
            counts["skipped_done"] += 1
            continue
        if counts["analysed"] >= max_windows:
            counts["deferred"] += 1
            continue
        if since is not None:
            nominal = _meta_nominal(path)
            if nominal is not None and nominal < since:
                counts["skipped_since"] += 1
                continue
        if writer is None:
            writer = factory(table=TABLE, mode=MODE, schema_version=1,
                             batch_rows=500, local=True)
            if getattr(writer, "is_noop", False):
                logger.error("sink writer is a no-op (sink path unwritable): "
                             "no rows stored, no windows marked done")
                stopped = True
                break
        try:
            row = _row_for(path, regs)
        except Exception as exc:     # one bad file never stalls the batch
            logger.warning("batch_error %s: %s: %s", path, type(exc).__name__, exc)
            row = _error_row(path, exc)
            counts["errors"] += 1
        try:
            writer.insert([row])
        except Exception as exc:     # BufferFull under a locked sink, say
            logger.error("sink insert raised %s: %s; stopping the run",
                         type(exc).__name__, exc)
            stopped = True
            break
        rows.append(row)
        pending.append(path)
        counts["analysed"] += 1
        counts["found"] += bool(row["found"])
        if len(pending) >= FLUSH_EVERY and not commit():
            stopped = True
            break
    if writer is not None:
        try:
            if pending and not stopped and not commit():
                stopped = True
        finally:
            writer.close()
    counts["stopped"] = stopped
    if csv_path and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    logger.info("squitter-analyze: %s", counts)
    return counts
