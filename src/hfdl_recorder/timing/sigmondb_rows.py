"""Squitter rows to sigmondb arrival and capture rows.

The mapping follows wsprdaemon-server's sigmondb/backfill_hfdl.py, which wrote the rows already
in sigmondb.  A shared parity fixture (tests/fixtures/sigmondb/parity-arrivals.jsonl, also in
wsprdaemon-server) holds the two to the same arrivals.  This module imports nothing from
wsprdaemon-server, so ARRIVAL_COLUMNS and CAPTURE_COLUMNS copy ddl.COLUMNS, less received_at,
which the server stamps.

Three choices differ from the back-fill on purpose:
- One capture per batch run and (frequency, analyzer version), not per UTC hour.  Its time is
  the run's earliest nominal frame time; its duration runs to the last frame's end.  The host
  clock never enters a timing field.
- A label-timed arrival (no delay_counter_s) travels with the null provenance block and
  engineering delay_basis "label".
- A T6 counter arrival carries the assumed block: u_epoch_ns 230.0, k 2.0, provenance
  "assumed", origin "t6_anchor".

Rows arrive typed as batch._row_for builds them: counter_map a JSON string or None.
"""
import json
from datetime import datetime, timedelta
from typing import Optional

from hfdl_recorder.timing.frame_grid import FRAME_S

SCHEMA_VERSION = 1
U_EPOCH_NS = 230.0
K = 2.0
ORIGIN = "t6_anchor"

ARRIVAL_COLUMNS = (
    "time", "reporter_id", "tx_id", "group_key", "centre_hz", "rate_hz_per_s", "delay_ns",
    "delay_sigma_ns", "level_db", "margin_db", "half_band_db", "clean_tag", "doppler_hz",
    "phase_rate", "carrier_phase_rad", "s4", "sigma_phi_rad", "detector_version",
    "schema_version", "u_epoch_ns", "k", "p", "a_level", "a_level_provenance",
    "counter_epoch_id", "origin", "engineering")
CAPTURE_COLUMNS = (
    "time", "reporter_id", "centre_hz", "rate_hz_per_s", "duration_s", "n_sweeps", "expected",
    "null_max_db", "null_median_db", "n_expected", "n_found", "detector_version", "ring_ref",
    "schema_version", "u_epoch_ns", "k", "p", "a_level", "a_level_provenance",
    "counter_epoch_id", "origin", "engineering")

_EPOCH = datetime(1970, 1, 1)
_ENG_KEYS = ("frame_index", "frame_offset", "flags", "second_peak_ratio", "second_peak_delay_s",
             "peak_width_s", "lobe_distortion", "t_level_active")
_INT_ENG = ("frame_index", "frame_offset")


def utc_iso(epoch_s: float) -> str:
    """ISO-8601 UTC with microseconds and a Z, rounded as the back-fill rounds."""
    t = _EPOCH + timedelta(microseconds=round(epoch_s * 1e6))
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _null_provenance() -> dict:
    return dict(u_epoch_ns=None, k=None, p=None, a_level="", a_level_provenance="",
                counter_epoch_id="", origin="")


def _assumed_provenance(counter_epoch_id: str) -> dict:
    return dict(u_epoch_ns=U_EPOCH_NS, k=K, p=None, a_level="", a_level_provenance="assumed",
                counter_epoch_id=counter_epoch_id or "", origin=ORIGIN)


def _counter_map(v) -> Optional[dict]:
    if isinstance(v, dict):
        return v
    if not v:
        return None
    try:
        d = json.loads(v)
    except (TypeError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _provenance(row: dict) -> dict:
    """Assumed block for a T6 row with a counter_map, else null."""
    cm = _counter_map(row.get("counter_map"))
    if row.get("t_level_active") == "T6" and cm is not None:
        g = cm.get("input_epoch_gps_s")
        return _assumed_provenance(f"ep-{int(float(g))}" if g not in (None, "") else "")
    return _null_provenance()


def _detector(row: dict) -> str:
    return f"hfdl-analyzer-{row['analyzer_version']}"


def _eng(v, key: str) -> str:
    if v is None:
        return ""
    return str(int(v)) if key in _INT_ENG else str(v)


def arrival_for(row: dict, reporter_id: str) -> Optional[dict]:
    """The sigmondb arrival for one squitter row, or None.

    None unless the row is found, not excluded, and carries a delay.  delay_counter_s gives
    basis "counter"; without it, delay_s gives basis "label" and the null block.
    """
    if not row.get("found") or row.get("excluded") or row.get("nominal_utc") is None:
        return None
    counter = row.get("delay_counter_s") is not None
    delay = row["delay_counter_s"] if counter else row.get("delay_s")
    if delay is None:
        return None
    prov = _provenance(row) if counter else _null_provenance()
    eng = {k: _eng(row.get(k), k) for k in _ENG_KEYS}
    eng["delay_basis"] = "counter" if counter else "label"
    doppler = row.get("doppler_hz")
    a = dict(
        time=utc_iso(row["nominal_utc"]), reporter_id=reporter_id,
        tx_id=f"hfdl:{int(row['gs_id'])}", group_key=f"hfdl {int(row['freq_khz'])} kHz",
        centre_hz=row["freq_khz"] * 1e3, rate_hz_per_s=0, delay_ns=round(delay * 1e9),
        delay_sigma_ns=None, level_db=float(row["snr_db"]), margin_db=None, half_band_db=None,
        clean_tag="squitter", doppler_hz=None if doppler is None else float(doppler),
        phase_rate=None, carrier_phase_rad=None, s4=None, sigma_phi_rad=None,
        detector_version=_detector(row), schema_version=SCHEMA_VERSION, engineering=eng,
        **prov)
    return {c: a[c] for c in ARRIVAL_COLUMNS}


def captures_for(rows: list, reporter_id: str) -> list:
    """One capture per (frequency, analyzer version) over this run's rows that have a time.

    time is the earliest nominal frame time and duration_s runs to the last frame's end.
    expected maps "hfdl:<gs>" to that ground station's found fraction; n_expected counts the
    rows, n_found the rows that give an arrival.  A row without nominal_utc (a batch error)
    counts nowhere.
    """
    groups: dict = {}
    for r in rows:
        if r.get("nominal_utc") is None:
            continue
        groups.setdefault((int(r["freq_khz"]), r["analyzer_version"]), []).append(r)
    caps = []
    for (khz, _), rs in groups.items():
        times = [r["nominal_utc"] for r in rs]
        per_gs: dict = {}
        prov, n_found, n_anchored = None, 0, 0
        for r in rs:
            counts = per_gs.setdefault(f"hfdl:{int(r['gs_id'])}", [0, 0])
            counts[0] += 1
            p = _provenance(r)
            if p["origin"]:
                n_anchored += 1
                if prov is None:
                    prov = p
            if arrival_for(r, reporter_id) is not None:
                counts[1] += 1
                n_found += 1
        eng = {"source": "squitter-batch", "n_rows": str(len(rs)), "n_anchored": str(n_anchored)}
        for tx, (n, f) in per_gs.items():
            eng[f"n_rows:{tx}"] = str(n)
            eng[f"n_found:{tx}"] = str(f)
        c = dict(
            time=utc_iso(min(times)), reporter_id=reporter_id, centre_hz=khz * 1e3,
            rate_hz_per_s=0.0, duration_s=float(max(times) - min(times) + FRAME_S),
            n_sweeps=None, expected={tx: f / n for tx, (n, f) in per_gs.items()},
            null_max_db=None, null_median_db=None, n_expected=len(rs), n_found=n_found,
            detector_version=_detector(rs[0]), ring_ref="", schema_version=SCHEMA_VERSION,
            engineering=eng, **(prov or _null_provenance()))
        caps.append({k: c[k] for k in CAPTURE_COLUMNS})
    return caps
