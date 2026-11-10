"""Batch analysis of saved squitter windows into a local sink table.

``run_batch`` walks ``<window_dir>/<radiod_id>/<YYYYMMDD>/*.npz`` (the window
keeper's files), times each squitter with ``time_squitter``, and inserts one
``squitter_timing`` row per window into sigmond's local sink (table
``squitter_timing``, mode ``hfdl``, ``local=True``).  A ``.done`` marker
beside each window records that it has been analysed, and by which
analyser: it holds ``ANALYZER_VERSION`` (an empty marker, as b72dc39 wrote,
counts as version 1).  A marker below the running analyser's version counts
as not done, so new timing code re-times old windows; every row carries
``analyzer_version`` to keep the two apart.  One run handles at most
``max_windows`` windows.

With the sink, the batch also writes sigmondb rows that hs-uploader ships to
wd30 (``timing/sigmondb_rows.py`` maps them).  Each window that gives an
arrival adds one ``arrival`` row; at the end of the run, one ``capture`` row
per (frequency, analyser version) covers the windows this run committed.
Both go to sink mode ``sigmondb``, schema 1, with ``producer``
``hfdl-recorder`` named on the writer: hs-uploader's SQL producer rule holds
no hfdl entry.  Neither is local; ``squitter_timing`` stays local.  Each row
carries the instance's ``reporter_id`` (``""`` when the config names none;
wd30 then takes the tar's site).  A commit flushes all three writers and
marks windows done only when none stays buffered, so no window counts as
done before its arrival is durable.  A run that stops on its store still
writes the captures of the windows it committed.  A row whose arrival cannot
be built (TypeError/ValueError: a found squitter with no SNR, say) gives no
arrival and counts in its capture as not found; the run logs one WARNING for
all such rows.  When every found squitter of a run fails so, and there were
at least two, the fault is the mapping's, not the rows': the run logs one ERROR and counts as stopped.
Any other exception from the mapping propagates.

Off the station (``sink=False``) the same code runs without sigmond: rows go
to ``jsonl_path`` (appended) and/or ``csv_path`` (this run's rows), and a
window gets its marker only after the file is written and fsynced.

A window whose registration has not arrived yet (hf-timestd writes the
minute row after the minute) gets no row and no marker while the newest
registration row lies before the window (+``REG_MAX_AGE_S``), so a later run
retries it.  After ``REG_WAIT_S`` past the window's nominal time (host clock:
a retry deadline only, it labels nothing) it is timed without one.

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
time this module uses; the host clock only bounds how long a window waits
for its registration.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from hfdl_recorder.timing import sigmondb_rows
from hfdl_recorder.timing.analyzer import ANALYZER_VERSION, time_squitter

logger = logging.getLogger(__name__)

TABLE = "squitter_timing"
MODE = "hfdl"
DEFAULT_REGISTRATION_DIR = "/var/lib/timestd/t3_shadow"
SCHEMA_VERSION = 2               # 2: analyzer_version, arrival_offset_s
DEFAULT_MAX_WINDOWS = 2000
FLUSH_EVERY = 100               # windows per durable commit
REG_MAX_AGE_S = 120.0            # registration must lie this close to the window
EPOCH_TOL_S = 2.0                # anchor-implied epoch vs the window's own; the
                                 # tracker's skew allowance (B4 pairs reached 0.8 s)
PLAUSIBLE_DELAY_S = (-0.030, 0.100)   # R12: outside this, a delay is not a squitter's
REG_WAIT_S = 7200.0              # wait this long past nominal for a registration
SIGMONDB_MODE = "sigmondb"
SIGMONDB_PRODUCER = "hfdl-recorder"   # named: hs-uploader's SQL rule has no hfdl entry

_GPS_UNIX_S = 315_964_800        # GPS epoch minus Unix epoch
_GPS_LEAP_S = 18                 # GPS minus UTC

_NAN_FIELDS = ("arrival_utc", "delay_s", "snr_db", "peak_width_s",
               "second_peak_ratio", "second_peak_delay_s", "doppler_hz",
               "lobe_distortion", "prekey_lead_s", "a1_utc",
               "arrival_offset_s")


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

    def newest_after(self, t: float) -> bool:
        """True when some row, any run, lies later than t (UTC seconds)."""
        t0 = datetime.fromtimestamp(t, timezone.utc)
        for d in (0, 1, 2):
            day = (t0 + timedelta(days=d)).strftime("%Y%m%d")
            if any(ns / 1e9 > t for ns, _ in self._day(day)):
                return True
        return False

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


def _row_for(path: Path, regs: Registrations) -> tuple[dict, bool]:
    """(row, registration_missing): the flag says a usable registration was
    wanted (the window carries an input index) but none matched."""
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
    row["analyzer_version"] = ANALYZER_VERSION
    for k, v in asdict(arr).items():
        row[k] = _clean(v)
    row["arrival_input_index"] = None
    row["delay_counter_s"] = None
    row["label_diff_s"] = None
    missing = False

    idx0, f_in = meta.get("input_index0"), meta.get("input_samprate")
    epoch_gps = meta.get("input_epoch_gps_s")
    if idx0 is not None and f_in and epoch_gps is not None:
        idx0, f_in = float(idx0), float(f_in)
        epoch_utc = float(epoch_gps) + _GPS_UNIX_S - _GPS_LEAP_S
        if arr.found:
            # The analyser's in-window offset, never arrival_utc - t0: two
            # epochs subtract to ~0.24 us grain, 15 input samples.
            row["arrival_input_index"] = idx0 + arr.arrival_offset_s * f_in
        reg = regs.nearest(t0, epoch_utc, f_in)
        missing = reg is None
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
    return row, missing


def _error_row(path: Path, exc: Exception) -> dict:
    row = {"radiod_id": path.parent.parent.name, "window_file": path.name,
           "analyzer_version": ANALYZER_VERSION, "found": False, "reason": f"batch_error:{type(exc).__name__}",
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
    try:
        listing = sorted(window_dir.glob(f"{radiod_id or '*'}/*/*.npz"))
    except FileNotFoundError:        # a day folder pruned while listing
        return
    for p in listing:
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


def _done_version(path: Path) -> Optional[int]:
    """The analyser version a window's marker names; None without a marker.
    An empty or unreadable marker (b72dc39 wrote empty ones) counts as 1."""
    try:
        text = _done(path).read_text().strip()
    except FileNotFoundError:
        return None
    except OSError:
        return 1
    try:
        return int(text) if text else 1
    except ValueError:
        return 1


def _mark_done(path: Path) -> None:
    tmp = _done(path).with_name(_done(path).name + ".tmp")
    tmp.write_text(f"{ANALYZER_VERSION}\n")
    os.replace(tmp, _done(path))


def _write_csv(csv_path, rows: list[dict]) -> None:
    """All of ``rows``, written to a temporary file, fsynced, then renamed."""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    csv_path = Path(csv_path)
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, csv_path)


def _append_jsonl(jsonl_path, rows: list[dict]) -> None:
    with open(jsonl_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


class _FileSink:
    """``sink=False``: rows go to files; a flush is durable or raises."""

    is_noop = False

    def __init__(self, csv_path, jsonl_path):
        self._csv, self._jsonl = csv_path, jsonl_path
        self._all: list[dict] = []
        self._buf: list[dict] = []

    @property
    def buffered(self) -> int:
        return len(self._buf)

    def insert(self, rows):
        self._buf.extend(rows)

    def flush(self):
        if not self._buf:
            return
        if self._jsonl:
            _append_jsonl(self._jsonl, self._buf)
        if self._csv:
            _write_csv(self._csv, self._all + self._buf)
        self._all.extend(self._buf)
        self._buf = []

    def close(self):
        pass


def run_batch(window_dir, *, registration_dir=DEFAULT_REGISTRATION_DIR,
              since: Optional[float] = None,
              radiod_id: Optional[str] = None,
              max_windows: int = DEFAULT_MAX_WINDOWS,
              csv_path=None, jsonl_path=None, sink: bool = True,
              writer_factory: Optional[Callable[..., object]] = None,
              now: Optional[float] = None,
              reporter_id: Optional[str] = None) -> dict:
    """Analyse windows not yet marked done (by this analyser version);
    returns the run's counts.  ``counts["stopped"]`` says the run gave up on
    its store; the CLI exits non-zero then.

    ``sink=False`` needs ``csv_path`` or ``jsonl_path`` and never imports
    sigmond.  With the sink, the files are an extra copy written at the end,
    and sigmondb arrival and capture rows go to the sink under
    ``reporter_id`` (``""`` when None)."""
    if not sink and not (csv_path or jsonl_path):
        raise ValueError("sink=False needs csv_path or jsonl_path")
    window_dir = Path(window_dir)
    regs = Registrations(Path(registration_dir))
    now = time.time() if now is None else float(now)
    counts = {"analysed": 0, "found": 0, "errors": 0, "skipped_done": 0,
              "skipped_since": 0, "deferred": 0, "awaiting_registration": 0,
              "vanished": 0, "arrivals": 0, "captures": 0}
    reporter = reporter_id or ""
    vanished_sample: Optional[Path] = None
    rows: list[dict] = []
    pending: list[Path] = []        # inserted, not yet durable: no .done yet
    pending_rows: list[dict] = []   # their rows, as the capture counts them
    committed_rows: list[dict] = [] # rows of windows marked done this run
    writer = None
    arrival_w = capture_w = None    # sigmondb writers, sink mode only
    no_arrival = 0                  # found rows the mapping refused
    found_mapped = 0                # found rows offered to the mapping
    no_arrival_sample: Optional[str] = None
    stopped = False
    if sink:
        factory = writer_factory or _default_writer_factory
    else:
        factory = lambda **kw: _FileSink(csv_path, jsonl_path)   # noqa: E731

    def writers():
        return [w for w in (writer, arrival_w, capture_w) if w is not None]

    def commit() -> bool:
        """Flush every writer; mark the pending windows done only if none
        stays buffered."""
        for w in writers():
            try:
                w.flush()
            except Exception as exc:
                logger.error("%s flush raised %s: %s", "sink" if sink else "file",
                             type(exc).__name__, exc)
                if not sink:
                    return False
        left = sum(getattr(w, "buffered", 0) for w in writers())
        if left:
            logger.error("flush left %d rows buffered; %d windows stay "
                         "unmarked and run again", left, len(pending))
            return False
        for p in pending:
            try:
                _mark_done(p)
            except OSError as exc:
                logger.warning("cannot mark %s done: %s", p, exc)
        pending.clear()
        committed_rows.extend(pending_rows)
        pending_rows.clear()
        return True

    def arrival_of(row: dict, path: Path):
        """(arrival or None, row as the capture counts it).  A row the
        mapping refuses (TypeError/ValueError: a found squitter with no SNR,
        say) gives no arrival and counts as not found.  Any other exception
        is a bug in the mapping and propagates."""
        nonlocal no_arrival, no_arrival_sample, found_mapped
        found_mapped += bool(row.get("found"))
        try:
            return sigmondb_rows.arrival_for(row, reporter), row
        except (TypeError, ValueError) as exc:
            no_arrival += 1
            no_arrival_sample = no_arrival_sample or f"{path}: {type(exc).__name__}: {exc}"
            return None, {**row, "found": False}

    # Whatever ends the loop -- its end, a stop, or an exception from it --
    # the committed windows still get their capture and the writers close;
    # an exception is re-raised once the run has reported.  A window not
    # yet committed never gets its marker.
    raised: Optional[BaseException] = None
    try:
        for path in _windows(window_dir, radiod_id):
            v = _done_version(path)
            if v is not None and v >= ANALYZER_VERSION:
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
                writer = factory(table=TABLE, mode=MODE, schema_version=SCHEMA_VERSION,
                                 batch_rows=500, local=True)
                if sink:
                    arrival_w, capture_w = (
                        factory(table=t, mode=SIGMONDB_MODE,
                                schema_version=sigmondb_rows.SCHEMA_VERSION,
                                producer=SIGMONDB_PRODUCER, batch_rows=500)
                        for t in ("arrival", "capture"))
                if any(getattr(w, "is_noop", False) for w in writers()):
                    logger.error("sink writer is a no-op (sink path unwritable): "
                                 "no rows stored, no windows marked done")
                    stopped = True
                    break
            try:
                row, missing = _row_for(path, regs)
            except FileNotFoundError:    # the keeper pruned it: expected, no row
                counts["vanished"] += 1
                vanished_sample = vanished_sample or path
                continue
            except Exception as exc:     # one bad file never stalls the batch
                logger.warning("batch_error %s: %s: %s", path, type(exc).__name__, exc)
                row, missing = _error_row(path, exc), False
                counts["errors"] += 1
            if missing and row.get("nominal_utc") is not None:
                nominal = float(row["nominal_utc"])
                if (not regs.newest_after(nominal + REG_MAX_AGE_S)
                        and now - nominal < REG_WAIT_S):
                    counts["awaiting_registration"] += 1   # retried next run
                    continue
            arrival, cap_row = arrival_of(row, path) if sink else (None, row)
            try:
                writer.insert([row])
                if arrival is not None:
                    arrival_w.insert([arrival])
            except Exception as exc:     # BufferFull under a locked sink, say
                logger.error("sink insert raised %s: %s; stopping the run",
                             type(exc).__name__, exc)
                stopped = True
                break
            rows.append(row)
            pending.append(path)
            pending_rows.append(cap_row)
            counts["arrivals"] += arrival is not None
            counts["analysed"] += 1
            counts["found"] += bool(row["found"])
            if len(pending) >= FLUSH_EVERY and not commit():
                stopped = True
                break
    except Exception as exc:
        raised = exc
        stopped = True
        logger.error("squitter-analyze raised %s: %s; writing the captures of "
                     "%d committed windows, marking no more", type(exc).__name__,
                     exc, len(committed_rows))
    if writer is not None:
        try:
            if pending and not stopped and not commit():
                stopped = True
            if capture_w is not None and committed_rows:
                # Even after a stop: these windows are done, so their
                # capture is written now or never.
                try:
                    caps = sigmondb_rows.captures_for(committed_rows, reporter)
                    capture_w.insert(caps)
                    capture_w.flush()
                    if getattr(capture_w, "buffered", 0):
                        raise RuntimeError(f"{capture_w.buffered} rows stay buffered")
                    counts["captures"] = len(caps)
                except Exception as exc:
                    logger.error("capture rows not stored (%s: %s); %d windows "
                                 "keep no capture", type(exc).__name__, exc,
                                 len(committed_rows))
                    stopped = True
        finally:
            for w in writers():
                w.close()
    if found_mapped >= 2 and no_arrival == found_mapped:
        # Every found squitter refused: a systematic mapping fault, not one
        # bad row.  The captures would claim "listened, not heard" for the
        # whole run, so the run fails loud.
        logger.error("all %d found squitters gave no arrival (the mapping "
                     "refused every one); first: %s", no_arrival, no_arrival_sample)
        stopped = True
    elif no_arrival:
        logger.warning("%d squitter rows gave no arrival (the mapping refused "
                       "them); e.g. %s", no_arrival, no_arrival_sample)
    counts["stopped"] = stopped
    if sink and rows:
        try:
            if csv_path:
                _write_csv(csv_path, rows)
            if jsonl_path:
                _append_jsonl(jsonl_path, rows)
        except OSError as exc:
            logger.error("cannot write the row copy: %s", exc)
    if counts["vanished"]:
        logger.info("%d windows vanished before they were read (pruned by the "
                    "window keeper); e.g. %s", counts["vanished"], vanished_sample)
    logger.info("squitter-analyze: %s", counts)
    if raised is not None:
        raise raised
    return counts
