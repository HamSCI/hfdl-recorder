#!/usr/bin/env python3
"""Offline squitter timing evaluation (squitter timing plan, Task 9).

Runs on the devbox only, on rows the daily pull lands in
``<data>/<station>/``.  Never on a station.

    squitter_eval.py report --data DIR --train-nights N --out report.md \
        [--stations stations.toml]

Inputs per station directory (searched recursively):

- ``squitter_timing`` rows (hfdl_recorder.timing.batch), as CSV
  (``squitter*.csv``, the ``squitter-analyze --csv`` export) or JSON lines
  (``squitter*.jsonl``), same columns either way.
- hf-timestd's T3 shadow record: ``minutes-<day>.jsonl`` (reference readings,
  the T6 anchor) and ``broadcasts-<channel>-<day>.jsonl`` (WWV and WWVH
  arrivals, one row per broadcast per minute).

The plane.  Every ``delay_counter_s`` gets its UTC through the minute row's
``t3_shadow.t6_anchor_*`` registration, so the reconstruction works in the
T6 anchor plane.  ``e(t)`` names that plane's error, reading minus truth, in
seconds.  The reference that measures the same plane: ``t6_offset_ms`` (the
T6 label of the PPS edge minus the GPS second), or T5 against T6
(``-t5_vs_adopted`` while ``vs_tier`` names T6).  T4 measures the host clock,
not this plane, so it serves only as a cross-check.

Every measurement estimates ``e``:

- HFDL:  ``delay_counter_s - prior(gs, rx) = e + b[gs, rx, band] + noise``.
  The prior follows a one-hop F-layer path at 300 km over the great circle;
  the nuisance ``b`` absorbs the ground station's offset, the receive chain
  and the prior's residue.  Rows of one key in one minute average first, so
  each (key, minute) counts once.
- WWV/WWVH: the broadcast row's ``arrival_input_index`` carried to UTC
  through the same minute's anchor, less the minute and the hop model's
  expected delay: ``y = e + b[channel, broadcaster] + noise``.  Rows without
  an index (``residual_ms`` only) live in T3's own registration frame, which
  T3 fits to those same arrivals; they stay out of check 4 and get counted.

Training nights learn each key's ``b`` (median) and demonstrated scatter
``sigma`` (1.4826 MAD) against a training truth: the station's reference
where the station may use it, else (``hide_reference``) the WWV-only
estimate with WWV's hop model taken at face value.  Scoring nights then
reconstruct ``e`` per minute by weighted least squares -- with the nuisances
fixed, the weighted mean of ``y - b`` with weights ``1/sigma**2`` -- once
from WWV and WWVH alone and once with HFDL added, and score both against the
reference.  The reconstruction never reads a reference column.

numpy only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:       # Python 3.10
    import tomli as tomllib

C_M_S = 299_792_458.0
R_EARTH_M = 6_371_000.0
H_F_M = 300_000.0                 # one-hop F layer, the prior
SIGMA_FLOOR_S = 10e-6
MIN_TRAIN_ROWS = 10
CLIP_SIGMAS = 5.0
MULTIFREQ_TOL_S = 0.3e-3          # path dispersion between bands, about
STEADY_S = 0.1e-3                 # check 1: night-to-night repeat for absolute time
HOUR_COVERAGE = 0.5               # an hour counts usable at this fraction of its minutes
EPOCH_TOL_S = 2.0                 # anchor-implied radiod epoch vs the rows' (batch.py)
_GPS_UNIX_S = 315_964_800
_GPS_LEAP_S = 18
# Analyser flags that keep a row out of every fit.  None today: on real data
# unresolved_multipath marks the strongest windows, so it is report-only.
EXCLUDING_FLAGS: tuple = ()
REPORT_FLAGS = ("unresolved_multipath",)   # counted as flagged_kept:<flag>, rows kept
HFDL_ARMS = ("wwv", "wwv+hfdl")

# HFDL ground stations, id -> (name, lat, lon).  Source: the HFDL system
# table (dumphfdl's systable.conf, "version = 52", as vendored in
# ka9q-radio/aux/systable.conf), read as data only.
GROUND_STATIONS = {
    1: ("San Francisco, California", 38.384587, -121.759647),
    2: ("Molokai, Hawaii", 21.184428, -157.186846),
    3: ("Reykjavik, Iceland", 63.847168, -22.455754),
    4: ("Riverhead, New York", 40.881922, -72.63762),
}

# The sign of every reference reading, as hf-timestd's minute rows state it
# (authority_manager.T3_SHADOW_OFFSET_SIGNS).  "Reading minus truth".
EXPECTED_OFFSET_SIGNS = {
    "t6_offset_ms": "T6 label of the PPS edge minus the GPS second",
    "t4_offset_ms": "host minus LAN GPS source (chrony local - source)",
    "t2_offset_ms": "host minus WAN source (chrony local - source)",
    "t2_error": "T2 error with T4 as truth = t4_offset_ms - t2_offset_ms",
    "t3_minus_t6_ms": "T3 label minus T6 label",
    "t3_vs_adopted": "T3 UTC estimate minus adopted bench's, ns",
    "t4_vs_adopted": "T4 UTC estimate minus adopted bench's, ns",
    "t5_vs_adopted": "T5 UTC estimate minus adopted bench's, ns",
    "t3_minus_t4": ("t3_vs_adopted.shadow_residual_ns - "
                    "t4_vs_adopted.shadow_residual_ns, same vs_tier"),
}
_SIGN_KEY = {"T6": "t6_offset_ms", "T4": "t4_offset_ms", "T5": "t5_vs_adopted"}
PLANE_TIERS = ("T6", "T5")        # readings of the T6 anchor plane's own error


class SignConventionError(ValueError):
    """A minute row states no sign convention, or one this tool does not expect."""


# ------------------------------------------------------------------ config

@dataclass
class StationCfg:
    name: str
    grid: Optional[str]
    latlon: Optional[tuple]
    reference: list = field(default_factory=lambda: ["T6"])
    cross_check: list = field(default_factory=list)
    hide_reference: bool = False
    require_adopted: list = field(default_factory=list)
    night_hours: Optional[tuple] = None
    input_samprate: Optional[float] = None
    night_start_hour: Optional[int] = None     # this station's night boundary, UTC


def grid_to_latlon(grid) -> Optional[tuple]:
    """Centre of a 4- or 6-character Maidenhead locator; None if not one."""
    if not isinstance(grid, str):
        return None
    g = grid.strip()
    if len(g) not in (4, 6):
        return None
    try:
        a, b = g[0].upper(), g[1].upper()
        if not ("A" <= a <= "R" and "A" <= b <= "R"):
            return None
        lon = -180.0 + (ord(a) - 65) * 20.0 + int(g[2]) * 2.0
        lat = -90.0 + (ord(b) - 65) * 10.0 + int(g[3]) * 1.0
        if len(g) == 6:
            c, d = g[4].lower(), g[5].lower()
            if not ("a" <= c <= "x" and "a" <= d <= "x"):
                return None
            lon += (ord(c) - 97) * (5.0 / 60) + 2.5 / 60
            lat += (ord(d) - 97) * (2.5 / 60) + 1.25 / 60
        else:
            lon += 1.0
            lat += 0.5
    except ValueError:
        return None
    return lat, lon


def load_stations(path) -> dict:
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    out = {}
    for name, s in (doc.get("stations") or {}).items():
        if "lat" in s and "lon" in s:
            latlon = (float(s["lat"]), float(s["lon"]))
        else:
            latlon = grid_to_latlon(s.get("grid"))
        ref = s.get("reference", ["T6"])
        ref = [ref] if isinstance(ref, str) else list(ref)
        for t in ref:
            if t not in PLANE_TIERS:
                raise ValueError(
                    f"{name}: reference {t!r} does not measure the T6 anchor plane "
                    f"the counter label lives in; use {PLANE_TIERS} and list "
                    f"T4 under cross_check")
        cross = s.get("cross_check", [])
        cross = [cross] if isinstance(cross, str) else list(cross)
        for t in cross:
            if t not in _SIGN_KEY:
                raise ValueError(f"{name}: unknown cross_check tier {t!r}")
        nh = s.get("night_hours")
        out[name] = StationCfg(
            name=name, grid=s.get("grid"), latlon=latlon, reference=ref,
            cross_check=cross,
            hide_reference=bool(s.get("hide_reference", False)),
            require_adopted=list(s.get("require_adopted", [])),
            night_hours=(int(nh[0]), int(nh[1])) if nh else None,
            input_samprate=(float(s["input_samprate"])
                            if s.get("input_samprate") else None),
            night_start_hour=(int(s["night_start_hour"])
                              if s.get("night_start_hour") is not None else None))
    return out


def station_night_hour(cfg: StationCfg, default: int = 12) -> int:
    """The station's night boundary: night_start_hour when set; else the
    default, unless the station's night window straddles it, when the
    boundary moves to the window's end so one window stays one night."""
    if cfg.night_start_hour is not None:
        return int(cfg.night_start_hour) % 24
    w = cfg.night_hours
    if w and w[0] != w[1] and _in_hours(default, w) and default != w[0]:
        return int(w[1]) % 24
    return default


# ------------------------------------------------------------------ loading

@dataclass
class StationData:
    rows: list = field(default_factory=list)
    minutes: list = field(default_factory=list)
    broadcasts: list = field(default_factory=list)


_BOOL = {"found", "utc_sync"}
_INT = {"gs_id", "freq_khz", "frame_index", "frame_offset", "m1_shift",
        "analyzer_version"}
_STR = {"flags", "excluded", "band", "gs_name", "reason", "radiod_id",
        "window_file", "t_level_active", "anchor_source", "anchor_id",
        "counter_map", "decimation"}
_BLANK_STR = {"flags", "excluded"}


def _coerce(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, str) and v == "":
            v = None
        if k in _BLANK_STR:
            out[k] = "" if v is None else str(v)
        elif v is None:
            out[k] = None
        elif k in _BOOL:
            out[k] = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1")
        elif k in _INT:
            out[k] = int(float(v))
        elif k in _STR:
            out[k] = str(v)
        elif isinstance(v, str):
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    for k in _BLANK_STR:
        out.setdefault(k, "")
    return out


def load_rows(path) -> list:
    path = Path(path)
    if path.suffix == ".csv":
        with open(path, newline="") as f:
            return [_coerce(r) for r in csv.DictReader(f)]
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(_coerce(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return rows


def _jsonl(path, kind) -> list:
    out = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("kind", kind) == kind:
            out.append(r)
    return out


def load_station_dir(directory) -> StationData:
    d = Path(directory)
    sd = StationData()
    for p in sorted(d.rglob("squitter*")):
        if p.suffix in (".csv", ".jsonl") and p.is_file():
            sd.rows.extend(load_rows(p))
    for p in sorted(d.rglob("minutes-*.jsonl")):
        sd.minutes.extend(_jsonl(p, "minute"))
    for p in sorted(d.rglob("broadcasts-*.jsonl")):
        sd.broadcasts.extend(_jsonl(p, "broadcast"))
    return sd


# ------------------------------------------------------------------ geometry

def path_prior_s(rx, gs) -> float:
    """One-hop F-layer (300 km) group delay over the great circle, seconds."""
    la1, lo1 = map(math.radians, rx)
    la2, lo2 = map(math.radians, gs)
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    theta = 2 * math.asin(min(1.0, math.sqrt(h)))
    r, rh = R_EARTH_M, R_EARTH_M + H_F_M
    leg = math.sqrt(r * r + rh * rh - 2 * r * rh * math.cos(theta / 2))
    return 2 * leg / C_M_S


def _gs_latlon(row) -> Optional[tuple]:
    gs = row.get("gs_id")
    if gs in GROUND_STATIONS:
        return GROUND_STATIONS[gs][1:]
    name = (row.get("gs_name") or "").lower()
    for n, lat, lon in GROUND_STATIONS.values():
        if name and name.split(",")[0] in n.lower():
            return lat, lon
    return None


def _gs_key(row):
    return row.get("gs_id") if row.get("gs_id") is not None else row.get("gs_name")


# ------------------------------------------------------------------ time

def night_of(t: float, start_hour: int = 12) -> str:
    """A night runs from start_hour UTC to start_hour the next day; it takes
    the date it starts on."""
    return datetime.fromtimestamp(t - start_hour * 3600, timezone.utc).strftime("%Y%m%d")


def split_nights(nights, n_train: int):
    """First n_train nights train, the rest score.  Never overlap."""
    ns = sorted(set(nights))
    if n_train < 1:
        raise ValueError("need at least one training night")
    if n_train >= len(ns):
        raise ValueError(f"{len(ns)} nights in the data, {n_train} for training: "
                         "none left to score")
    train, score = ns[:n_train], ns[n_train:]
    assert not set(train) & set(score)
    return train, score


def _minute(t: float) -> int:
    return int(math.floor(t / 60.0)) * 60


def _hour(t: float) -> int:
    return datetime.fromtimestamp(t, timezone.utc).hour


def _in_hours(h: int, window) -> bool:
    lo, hi = window
    return lo <= h < hi if lo <= hi else (h >= lo or h < hi)


# ------------------------------------------------------------------ reference

def _finite(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def reference_error_s(row: dict, tiers):
    """(error, reading minus truth, in seconds; tier) from the first tier in
    ``tiers`` the minute row holds, else (None, None).

    T6: t6_offset_ms, the T6 label minus the GPS second: the anchor plane's
    own error.  T5: t5_vs_adopted.shadow_residual_ns gives T5's UTC estimate
    minus the adopted bench's, so while the adopted bench is T6 its negative
    gives the anchor plane's error with T5 as truth; against any other bench
    it measures some other plane and does not count.  T4: t4_offset_ms, host
    minus the LAN GPS server -- the host clock, a cross-check only.

    Raises SignConventionError when the row states no convention for a tier
    asked for, or a different one.
    """
    signs = row.get("offset_signs")
    for tier in tiers:
        key = _SIGN_KEY[tier]
        if not isinstance(signs, dict) or key not in signs:
            raise SignConventionError(
                f"minute {row.get('utc_minute')}: no offset_signs[{key!r}]; "
                "refusing to guess the sign")
        if signs[key] != EXPECTED_OFFSET_SIGNS[key]:
            raise SignConventionError(
                f"{key}: row says {signs[key]!r}, this tool expects "
                f"{EXPECTED_OFFSET_SIGNS[key]!r}")
        if tier == "T5":
            blk = row.get("t5_vs_adopted")
            if not isinstance(blk, dict) or blk.get("vs_tier") != "T6":
                continue
            v = _finite(blk.get("shadow_residual_ns"))
            if v is not None:
                return -v / 1e9, tier
        else:
            v = _finite(row.get(key))
            if v is not None:
                return v / 1e3, tier
    return None, None


def _adopted_tier(row) -> Optional[str]:
    for k in ("t3_vs_adopted", "t4_vs_adopted", "t5_vs_adopted"):
        blk = row.get(k)
        if isinstance(blk, dict) and blk.get("vs_tier"):
            return blk["vs_tier"]
    return None


def _reference_series(sd: StationData, cfg: StationCfg, tiers=None) -> dict:
    """minute -> reference reading (s), where the station may score."""
    out = {}
    for m in sd.minutes:
        t = _finite(m.get("utc_minute"))
        if t is None:
            continue
        if cfg.require_adopted and _adopted_tier(m) not in cfg.require_adopted:
            continue
        v, _ = reference_error_s(m, tiers or cfg.reference)
        if v is not None:
            out[_minute(t)] = v
    return out


def _row_epochs(sd: StationData) -> list:
    """The radiod runs the HFDL rows name: input epoch, UTC seconds."""
    eps = set()
    for r in sd.rows:
        g = _finite(r.get("input_epoch_gps_s"))
        if g is not None:
            eps.add(round(g + _GPS_UNIX_S - _GPS_LEAP_S, 1))
    return sorted(eps)


def _anchor_series(sd: StationData, f_in=None, epochs=(), counts=None) -> dict:
    """minute -> (anchor_utc_ns, anchor_input_index) from the minute rows.

    The anchor and the broadcast's ``arrival_input_index`` come from one
    minute row of one hf-timestd, so they share a radiod run by
    construction (a restart inside the minute shows as
    ``wwv_index_off_minute``).  The anchors therefore name their own runs
    and all stay.  With the HFDL rows' runs known, an anchor whose implied
    epoch (anchor UTC - index / f_in) matches none of them gets counted as
    ``anchor_run_without_hfdl_rows``: WWV minutes with no HFDL beside them."""
    out = {}
    for m in sd.minutes:
        blk = m.get("t3_shadow")
        if not isinstance(blk, dict):
            continue
        idx, ns = _finite(blk.get("t6_anchor_input_index")), blk.get("t6_anchor_utc_ns")
        t = _finite(m.get("utc_minute"))
        if idx is None or ns is None or t is None:
            continue
        if f_in and epochs:
            implied = int(ns) / 1e9 - idx / f_in
            if min(abs(implied - e) for e in epochs) > EPOCH_TOL_S:
                if counts is not None:
                    counts["anchor_run_without_hfdl_rows"] += 1
        out[_minute(t)] = (int(ns), idx)
    return out


# ------------------------------------------------------------------ measurements

def _exclusion_reasons(r) -> list:
    out = []
    for reason in filter(None, (r.get("excluded") or "").split(",")):
        out.append(f"excluded:{reason.strip()}")
    if not r.get("found"):
        out.append("not_found")
    flags = set(filter(None, (r.get("flags") or "").split(",")))
    for fl in EXCLUDING_FLAGS:
        if fl in flags:
            out.append(f"flag:{fl}")
    return out


def _latest_analysis(rows, counts) -> list:
    """One row per window: the newest analyser's (``analyzer_version``; a
    row without one came from version 1).  Rows naming no window stay."""
    best = {}
    out = []
    for r in rows:
        key = (r.get("radiod_id"), r.get("window_file"))
        if key[1] is None:
            out.append(r)
            continue
        v = r.get("analyzer_version") or 1
        if key in best:
            counts["superseded_analysis"] += 1
            if v <= (best[key].get("analyzer_version") or 1):
                continue
        best[key] = r
    return out + list(best.values())


def _usable_rows(sd: StationData, cfg: StationCfg, counts: dict) -> list:
    """Rows fit for timing, each with ``_prior`` and ``_y``; counts the rest."""
    out = []
    for r in _latest_analysis(sd.rows, counts):
        flags = set(filter(None, (r.get("flags") or "").split(",")))
        for fl in REPORT_FLAGS:
            if fl in flags:
                counts[f"flagged_kept:{fl}"] += 1
        reasons = _exclusion_reasons(r)
        dc = _finite(r.get("delay_counter_s"))
        t = _finite(r.get("nominal_utc"))
        if not reasons and (dc is None or t is None):
            reasons = ["no_counter_label"]
        for why in reasons:
            counts[why] += 1
        if reasons:
            continue
        gsll = _gs_latlon(r)
        if cfg.latlon is not None and gsll is not None:
            prior = path_prior_s(cfg.latlon, gsll)
        else:
            prior = None
            counts["no_geometry"] += 1
        rr = dict(r)
        rr["_prior"] = prior
        rr["_y"] = dc - (prior or 0.0)
        out.append(rr)
        counts["used"] += 1
    return out


def _f_in(sd: StationData, cfg: StationCfg) -> Optional[float]:
    if cfg.input_samprate:
        return cfg.input_samprate
    v = [_finite(r.get("input_samprate")) for r in sd.rows]
    v = [x for x in v if x]
    return float(np.median(v)) if v else None


def _wwv_measurements(sd, cfg, anchors, counts) -> list:
    """(minute, key, y, row_sigma_s) per broadcast row, index route only."""
    f_in = _f_in(sd, cfg)
    out = []
    for b in sd.broadcasts:
        t = _finite(b.get("utc_minute"))
        if t is None:
            continue
        m = _minute(t)
        idx, exp = _finite(b.get("arrival_input_index")), _finite(b.get("expected_delay_ms"))
        a = anchors.get(m)
        y = None
        if idx is not None and exp is not None and a is not None and f_in:
            a_ns, a_idx = a
            whole, rem = divmod(a_ns, 10**9)
            # UTC of the arrival, less the minute, kept between nearby numbers
            arr = (whole - int(t)) + rem / 1e9 + (idx - a_idx) / f_in
            if abs(arr) < 2.0:
                y = arr - exp / 1e3
            else:
                counts["wwv_index_off_minute"] += 1
                continue
        if y is None:
            # residual_ms lives in T3's registration frame, fitted to these
            # same arrivals: it cannot score the anchor plane.
            if _finite(b.get("residual_ms")) is not None:
                counts["wwv_residual_route_excluded"] += 1
            else:
                counts["wwv_unusable"] += 1
            continue
        counts["wwv_index_route"] += 1
        sig = _finite(b.get("sigma_ms"))
        key = ("wwv", b.get("channel"), b.get("station"))
        out.append((m, key, y, sig / 1e3 if sig and sig > 0 else 1e-3))
    return out


def _hfdl_measurements(rows) -> list:
    """(minute, key, mean y, n rows): one entry per key per minute."""
    acc = defaultdict(list)
    for r in rows:
        key = ("hfdl", _gs_key(r), r.get("band") or r.get("freq_khz"))
        acc[(_minute(r["nominal_utc"]), key)].append(r["_y"])
    return [(m, k, float(np.mean(v)), len(v)) for (m, k), v in sorted(acc.items(), key=str)]


# ------------------------------------------------------------------ fitting

def _robust(x) -> tuple:
    x = np.asarray(x, float)
    med = float(np.median(x))
    sig = float(1.4826 * np.median(np.abs(x - med)))
    return med, max(sig, SIGMA_FLOOR_S)


def _wmean(pairs):
    """Weighted mean of (value, sigma) pairs, one clip pass at CLIP_SIGMAS."""
    if not pairs:
        return None
    v = np.array([p[0] for p in pairs])
    s = np.array([p[1] for p in pairs])
    w = 1.0 / s ** 2
    mu = float(np.sum(w * v) / np.sum(w))
    keep = np.abs(v - mu) <= CLIP_SIGMAS * np.maximum(s, 1.0 / math.sqrt(np.sum(w)))
    if keep.any() and not keep.all():
        w, v = w[keep], v[keep]
        mu = float(np.sum(w * v) / np.sum(w))
    return mu, float(1.0 / math.sqrt(np.sum(w)))


def _loo_scatter(by_m) -> dict:
    """key -> residuals of each key against the other keys of its minute."""
    res = defaultdict(list)
    for items in by_m.values():
        for y, _, key in items:
            others = [(yy, ss) for yy, ss, kk in items if kk != key]
            if others:
                res[key].append(y - _wmean(others)[0])
    return res


def _train(cfg, wwv, hfdl, ref, train_set, night_hour) -> dict:
    """key -> (b, sigma, n) from training nights only.  The HFDL sigma
    describes one key-minute mean."""
    nuis = {}
    tr = lambda m: night_of(m, night_hour) in train_set   # noqa: E731
    if cfg.hide_reference:
        by_m = defaultdict(list)
        for m, key, y, sig in wwv:
            if tr(m):
                by_m[m].append((y, sig, key))
        truth = {m: _wmean([(y, s) for y, s, _ in items])[0] for m, items in by_m.items()}
        # WWV keys: b = 0; scatter against the other keys of the same minute.
        res = _loo_scatter(by_m)
        row_sig = defaultdict(list)
        for items in by_m.values():
            for _, s, key in items:
                row_sig[key].append(s)
        for key in row_sig:
            sig = (_robust(res[key])[1] if len(res[key]) >= MIN_TRAIN_ROWS
                   else float(np.median(row_sig[key])))
            nuis[key] = (0.0, sig, len(row_sig[key]))
    else:
        truth = ref
        res = defaultdict(list)
        for m, key, y, _ in wwv:
            if tr(m) and m in truth:
                res[key].append(y - truth[m])
        for key, r in res.items():
            if len(r) >= MIN_TRAIN_ROWS:
                nuis[key] = (*_robust(r), len(r))
    res = defaultdict(list)
    for m, key, y, _n in hfdl:
        if tr(m) and m in truth:
            res[key].append(y - truth[m])
    for key, r in res.items():
        if len(r) >= MIN_TRAIN_ROWS:
            nuis[key] = (*_robust(r), len(r))
    if cfg.hide_reference:
        # Against WWV-at-face-value, an HFDL key's scatter carries WWV's
        # noise.  Its own shows against the other HFDL keys of the same
        # minute (their b removed); keep the WWV figure where too few.
        by_m = defaultdict(list)
        for m, key, y, _n in hfdl:
            if tr(m) and key in nuis:
                by_m[m].append((y - nuis[key][0], nuis[key][1], key))
        for key, r in _loo_scatter(by_m).items():
            if len(r) >= MIN_TRAIN_ROWS:
                b, _, n = nuis[key]
                nuis[key] = (b, _robust(r)[1], n)
    return nuis


def _reconstruct(wwv, hfdl, nuis, score_set, night_hour, with_hfdl, counts) -> dict:
    """minute -> (e_hat, sigma) on scoring nights.  Reads no reference."""
    by_m = defaultdict(list)
    items = [(m, k, y) for m, k, y, _ in wwv]
    if with_hfdl:
        items += [(m, k, y) for m, k, y, _n in hfdl]
    for m, key, y in items:
        if night_of(m, night_hour) not in score_set:
            continue
        if key not in nuis:
            if with_hfdl:
                counts["untrained_key_minutes"] += 1
            continue
        b, s, _ = nuis[key]
        by_m[m].append((y - b, s))
    return {m: _wmean(p) for m, p in sorted(by_m.items())}


# ------------------------------------------------------------------ scoring

def _stats(err) -> dict:
    e = np.abs(np.asarray(err, float))
    if not len(e):
        return {"n": 0, "rms_s": None, "p95_s": None}
    return {"n": int(len(e)), "rms_s": float(np.sqrt(np.mean(e ** 2))),
            "p95_s": float(np.percentile(e, 95))}


def _score_arm(rec, ref, minutes, night_hour, window, coverage) -> dict:
    """Score one arm over ``minutes`` (reference minutes on scoring nights)."""
    slots = defaultdict(lambda: [0, []])     # (night, hour) -> [ref minutes, errors]
    for m in minutes:
        slot = slots[(night_of(m, night_hour), _hour(m))]
        slot[0] += 1
        if rec.get(m) is not None:
            slot[1].append(rec[m][0] - ref[m])
    by_hour = defaultdict(lambda: {"err": [], "empty": 0, "slots": 0})
    over, empty, total = [], 0, 0
    for (night, h), (nref, errs) in slots.items():
        unusable = len(errs) < coverage * nref
        bh = by_hour[h]
        bh["err"] += errs
        bh["slots"] += 1
        bh["empty"] += unusable
        if _in_hours(h, window):
            over += errs
            total += 1
            empty += unusable
    out = _stats(over)
    out["empty_hours"] = empty
    out["hours"] = total
    out["by_hour"] = {h: dict(_stats(v["err"]), empty=v["empty"], slots=v["slots"])
                      for h, v in sorted(by_hour.items())}
    return out


def _score(recs, ref, score_set, night_hour, window, coverage) -> dict:
    """Both arms, plus like-for-like sets: minutes both arms reconstruct, and
    minutes only the HFDL arm fills."""
    minutes = [m for m in ref if night_of(m, night_hour) in score_set
               and _in_hours(_hour(m), window)]
    all_minutes = [m for m in ref if night_of(m, night_hour) in score_set]
    w, b = recs["wwv"], recs["wwv+hfdl"]
    out = {arm: _score_arm(recs[arm], ref, all_minutes, night_hour, window, coverage)
           for arm in HFDL_ARMS}
    common = [m for m in minutes if w.get(m) is not None and b.get(m) is not None]
    only = [m for m in minutes if w.get(m) is None and b.get(m) is not None]
    for arm in HFDL_ARMS:
        out[arm]["common"] = _stats([recs[arm][m][0] - ref[m] for m in common])
    out["wwv+hfdl"]["hfdl_only"] = _stats([b[m][0] - ref[m] for m in only])
    return out


def _cross_checks(sd, cfg, plane_ref) -> dict:
    """Each cross-check tier beside the anchor plane's error: reading and
    (reading - plane error) where both exist."""
    out = {}
    for tier in cfg.cross_check:
        if tier in cfg.reference:      # itself against itself: trivially zero
            continue
        ser = _reference_series(sd, cfg, [tier])
        both = [m for m in ser if m in plane_ref]
        d = np.array([ser[m] - plane_ref[m] for m in both])
        v = np.array(list(ser.values()))
        out[tier] = {
            "n": int(len(v)),
            "median_s": float(np.median(v)) if len(v) else None,
            "n_both": int(len(d)),
            "minus_plane_median_s": float(np.median(d)) if len(d) else None,
            "minus_plane_sigma_s": _robust(d)[1] if len(d) else None,
        }
    return out


# ------------------------------------------------------------------ checks 1-3

def _check1(used_by_station, refs, night_hours) -> dict:
    """Check 1.  Per (station, gs, band) and night: the mean of
    delay - prior - reference.  The verdict comes from each key's own
    night-to-night sd of those means, pooled across keys, so constant
    differences between keys (receive chains, the prior's residue) do not
    count.  Each ground station's nightly change splits into the part common
    to every key (the offset wandering) and the between-key remainder (the
    path).  The prior's error on a key enters its sum with the offset, so a
    failure stays ambiguous between offset and path unless the split names
    one.  The common part names offset wander only when keys from two sites
    or more carry it; at one site it mixes in that site's path."""
    per_key = {}
    for st, rows in used_by_station.items():
        night_hour = night_hours[st]
        acc = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["_prior"] is None:
                continue
            m = _minute(r["nominal_utc"])
            if m not in refs[st]:
                continue
            acc[(st, _gs_key(r), r.get("band"))][night_of(m, night_hour)].append(
                r["_y"] - refs[st][m])
        for key, nights in acc.items():
            per_key[key] = {}
            for night, v in sorted(nights.items()):
                v = np.asarray(v)
                med, s = _robust(v)
                v = v[np.abs(v - med) <= CLIP_SIGMAS * s]
                per_key[key][night] = (float(np.mean(v)), float(np.std(v)), int(len(v)))

    def pooled(sds):
        dof = sum(n - 1 for _, n in sds)
        return math.sqrt(sum(s * s * (n - 1) for s, n in sds) / dof) if dof > 0 else None

    per_gs, all_sds = {}, []
    by_gs = defaultdict(list)
    for key, nights in per_key.items():
        by_gs[key[1]].append((key, nights))
    for gs, keys in by_gs.items():
        sds, devs = [], defaultdict(dict)
        means = []
        multi_sites = set()
        for key, nights in keys:
            mu = np.array([o for o, _, _ in nights.values()])
            means.append(float(mu.mean()))
            if len(mu) >= 2:
                sds.append((float(np.std(mu, ddof=1)), len(mu)))
                multi_sites.add(key[0])
                for night, (o, _, _) in nights.items():
                    devs[night][key] = o - float(mu.mean())
        all_sds += sds
        common = {n: float(np.mean(list(d.values()))) for n, d in devs.items()}
        between = [dv - common[n] for n, d in devs.items() for dv in d.values()]
        # only keys with two nights or more carry a deviation
        nk = len({k for d in devs.values() for k in d})
        per_gs[gs] = {
            "offset_s": float(np.mean(means)),
            "within_key_sd_s": pooled(sds),
            "common_sd_s": (float(np.std(list(common.values()), ddof=1))
                            if len(common) >= 2 and len(multi_sites) >= 2 else None),
            "sites": len(multi_sites),
            # between-key spread, corrected for the mean it was measured against
            "between_key_sd_s": (float(np.sqrt(np.mean(np.square(between)) * nk / (nk - 1)))
                                 if between and nk > 1 else None),
            "keys": len(keys), "nights": len(devs),
        }
    overall = pooled(all_sds)
    return {"per_key": per_key, "per_gs": per_gs, "within_key_sd_s": overall,
            "steady": None if overall is None else overall <= STEADY_S}


def _check2(used_by_station, refs, night_hours) -> dict:
    """Single-arrival scatter against the reference, by hour, both labels."""
    out = {}
    for st, rows in used_by_station.items():
        night_hour = night_hours[st]
        res = {}
        for label, col in (("counter", "delay_counter_s"), ("status", "delay_s")):
            groups = defaultdict(list)
            for r in rows:
                v = _finite(r.get(col))
                m = _minute(r["nominal_utc"])
                if v is None or m not in refs[st]:
                    continue
                y = v - (r["_prior"] or 0.0) - refs[st][m]
                groups[(_gs_key(r), r.get("band"), night_of(m, night_hour))].append(
                    (y, _hour(r["nominal_utc"])))
            resid = []
            for g in groups.values():
                med = float(np.median([y for y, _ in g]))
                resid += [(y - med, h) for y, h in g]
            if not resid:
                res[label] = {"n": 0, "sigma1_s": None, "by_hour": {}}
                continue
            all_r = np.array([x for x, _ in resid])
            _, s1 = _robust(all_r)
            by_hour = {}
            for h in sorted({h for _, h in resid}):
                hv = np.array([x for x, hh in resid if hh == h])
                by_hour[h] = (_robust(hv)[1], int(len(hv)))
            res[label] = {"n": int(len(all_r)), "sigma1_s": s1, "by_hour": by_hour,
                          "n_for_0.1ms": math.ceil((s1 / 1e-4) ** 2),
                          "n_for_1ms": math.ceil((s1 / 1e-3) ** 2)}
        out[st] = res
    return out


def _utc_day(t: float) -> int:
    """frame_index wraps daily (0..2699 from 00:00Z), so a frame is named by
    its UTC day and index together.  A slot never crosses midnight."""
    return int(math.floor(t / 86400.0))


def _frame_key(r):
    return (_gs_key(r), r.get("freq_khz"), _utc_day(r["nominal_utc"]),
            r.get("frame_index"), r.get("frame_offset"))


def _check3(used_by_station, refs, night_hours) -> dict:
    """One squitter heard at two sites: (d_A - e_A) - (d_B - e_B) against the
    geometric prior difference."""
    idx = {}
    for st, rows in used_by_station.items():
        night_hour = night_hours[st]
        d = {}
        for r in rows:
            m = _minute(r["nominal_utc"])
            if r["_prior"] is None or m not in refs[st]:
                continue
            d[_frame_key(r)] = (r["_y"] - refs[st][m], night_of(m, night_hour))
        idx[st] = d
    out = {}
    sts = sorted(idx)
    for i, a in enumerate(sts):
        for b in sts[i + 1:]:
            pairs = defaultdict(list)
            for k, (ya, night) in idx[a].items():
                if k in idx[b]:
                    pairs[k[0]].append((ya - idx[b][k][0], night))
            for gs, v in pairs.items():
                x = np.array([d for d, _ in v])
                per_night = defaultdict(list)
                for d, n in v:
                    per_night[n].append(d)
                out[(gs, a, b)] = {
                    "n": int(len(x)), "median_s": float(np.median(x)),
                    "sigma_s": _robust(x)[1],
                    "per_night": {n: float(np.median(d)) for n, d in sorted(per_night.items())},
                }
    return out


def _multifreq(used_by_station, dup_counts) -> dict:
    """Same ground station, same frame, two frequencies: the gap.  A second
    row for one (gs, frame, frequency) gets counted and dropped."""
    out = {}
    for st, rows in used_by_station.items():
        frames = defaultdict(dict)
        for r in rows:
            fr = frames[(_gs_key(r), _utc_day(r["nominal_utc"]), r.get("frame_index"))]
            f = r.get("freq_khz")
            if f in fr:
                dup_counts[st] = dup_counts.get(st, 0) + 1
                continue
            fr[f] = r
        acc = defaultdict(lambda: {"counter": [], "status": []})
        for (gs, _, _), byf in frames.items():
            fs = sorted(f for f in byf if f is not None)
            for i, fa in enumerate(fs):
                for fb in fs[i + 1:]:
                    ra, rb = byf[fa], byf[fb]
                    a = acc[(st, gs, fa, fb)]
                    a["counter"].append(ra["delay_counter_s"] - rb["delay_counter_s"])
                    sa, sb = _finite(ra.get("delay_s")), _finite(rb.get("delay_s"))
                    if sa is not None and sb is not None:
                        a["status"].append(sa - sb)
        for key, a in acc.items():
            d = np.array(a["counter"])
            s = np.array(a["status"])
            out[key] = {
                "n": int(len(d)), "median_s": float(np.median(d)),
                "p95_abs_s": float(np.percentile(np.abs(d), 95)),
                "frac_over": float(np.mean(np.abs(d) > MULTIFREQ_TOL_S)),
                "status_p95_abs_s": (float(np.percentile(np.abs(s), 95))
                                     if len(s) else None),
            }
    return out


def _label_noise(sd: StationData) -> dict:
    v = np.array([x for x in (_finite(r.get("label_diff_s")) for r in sd.rows)
                  if x is not None])
    if not len(v):
        return {"n": 0, "median_s": None, "p95_abs_s": None, "max_abs_s": None}
    a = np.abs(v)
    return {"n": int(len(v)), "median_s": float(np.median(v)),
            "p95_abs_s": float(np.percentile(a, 95)), "max_abs_s": float(a.max())}


# ------------------------------------------------------------------ evaluate

@dataclass
class Evaluation:
    train_nights: list
    score_nights: list
    nuisances: dict
    recon: dict
    scores: dict
    check1: dict
    check2: dict
    check3: dict
    multifreq: dict
    multifreq_duplicates: dict
    exclusions: dict
    label_noise: dict
    cross_checks: dict
    notes: list
    overnight: tuple
    coverage: float
    windows: dict
    cfgs: dict


def evaluate(data: dict, cfgs: dict, train_nights: int, night_hour: int = 12,
             overnight=(0, 12), coverage: float = HOUR_COVERAGE) -> Evaluation:
    notes = []
    stations = [s for s in sorted(data) if s in cfgs]
    for s in sorted(set(data) - set(cfgs)):
        notes.append(f"{s}: no entry in the stations file; skipped")
    night_hours = {s: station_night_hour(cfgs[s], night_hour) for s in stations}
    nights = set()
    for s in stations:
        sd, nh = data[s], night_hours[s]
        nights |= {night_of(r["nominal_utc"], nh) for r in sd.rows
                   if _finite(r.get("nominal_utc")) is not None}
        nights |= {night_of(b["utc_minute"], nh) for b in sd.broadcasts
                   if _finite(b.get("utc_minute")) is not None}
    train, score = split_nights(nights, train_nights)
    train_set, score_set = set(train), set(score)

    nuisances, recon, scores, refs, used = {}, {}, {}, {}, {}
    exclusions, label_noise, cross, windows = {}, {}, {}, {}
    for s in stations:
        sd, cfg = data[s], cfgs[s]
        nh = night_hours[s]
        if nh != night_hour:
            notes.append(f"{s}: nights run {nh:02d}:00Z to {nh:02d}:00Z (its night "
                         "window straddles the default boundary, or night_start_hour)")
        dropped = [t for t in cfg.cross_check if t in cfg.reference]
        if dropped:
            notes.append(f"{s}: cross-check {'/'.join(dropped)} is also the reference; "
                         "dropped (it would compare the reference with itself)")
        counts = defaultdict(int)
        ref = _reference_series(sd, cfg)
        refs[s] = ref
        rows = _usable_rows(sd, cfg, counts)
        used[s] = rows
        if cfg.latlon is None:
            notes.append(f"{s}: no receiver location (grid TODO); checks 1 and 3 "
                         "skip it, the reconstruction runs without the prior")
        f_in = _f_in(sd, cfg)
        epochs = _row_epochs(sd)
        if not epochs:
            notes.append(f"{s}: no input_epoch_gps_s in the rows; no count of "
                         "WWV minutes from radiod runs without HFDL")
        anchors = _anchor_series(sd, f_in, epochs, counts)
        wwv = _wwv_measurements(sd, cfg, anchors, counts)
        hfdl = _hfdl_measurements(rows)
        nuis = _train(cfg, wwv, hfdl, ref, train_set, nh)
        nuisances[s] = nuis
        recs = {arm: _reconstruct(wwv, hfdl, nuis, score_set, nh,
                                  arm == "wwv+hfdl", counts) for arm in HFDL_ARMS}
        window = cfg.night_hours or tuple(overnight)
        windows[s] = window
        sc = _score(recs, ref, score_set, nh, window, coverage)
        for arm in HFDL_ARMS:
            recon[(s, arm)] = recs[arm]
            scores[(s, arm)] = sc[arm]
        exclusions[s] = dict(counts)
        exclusions[s].setdefault("used", 0)
        label_noise[s] = _label_noise(sd)
        cross[s] = _cross_checks(sd, cfg, ref)
        if not ref:
            notes.append(f"{s}: no reference minutes ({'/'.join(cfg.reference)}); "
                         "nothing scores")

    dups = {}
    mf = _multifreq(used, dups)
    for s, n in sorted(dups.items()):
        notes.append(f"{s}: {n} duplicate rows for one (ground station, frame, "
                     "frequency); the multi-frequency check kept the first")
    return Evaluation(
        train_nights=train, score_nights=score, nuisances=nuisances, recon=recon,
        scores=scores, check1=_check1(used, refs, night_hours),
        check2=_check2(used, refs, night_hours), check3=_check3(used, refs, night_hours),
        multifreq=mf, multifreq_duplicates=dups, exclusions=exclusions,
        label_noise=label_noise, cross_checks=cross, notes=notes,
        overnight=tuple(overnight), coverage=coverage, windows=windows, cfgs=cfgs)


# ------------------------------------------------------------------ report

def _ms(v, nd=3) -> str:
    return "-" if v is None else f"{v * 1e3:.{nd}f}"


def _gs_name(gs) -> str:
    return GROUND_STATIONS[gs][0] if gs in GROUND_STATIONS else str(gs)


def render_report(ev: Evaluation) -> str:
    L = []
    a = L.append
    a("# Squitter timing: offline evaluation\n")
    a(f"Training nights (nuisances learned here only): {', '.join(ev.train_nights)}  ")
    a(f"Scoring nights (held out): {', '.join(ev.score_nights)}  ")
    a("A night runs 12:00Z to 12:00Z (unless a station's note says otherwise) and "
      "takes the date it starts on.  "
      "All times in ms unless marked.\n")
    a("The counter label and the WWV index route both take UTC from the minute row's "
      "T6 anchor, so the reconstruction works in the T6 anchor plane, and the score "
      "uses that plane's own error: t6_offset_ms, or T5 against T6.  T4 measures the "
      "host clock and appears only as a cross-check.\n")
    for n, c in sorted(ev.cfgs.items()):
        w = ev.windows.get(n, ev.overnight)
        a(f"- {n}: reference {'/'.join(c.reference)}"
          f"{' (hidden from the reconstruction)' if c.hide_reference else ''}"
          f"{', while adopted ' + '/'.join(c.require_adopted) if c.require_adopted else ''}"
          f"; cross-check {'/'.join(c.cross_check) or 'none'}; night window "
          f"{w[0]:02d}-{w[1]:02d}Z")
    for n in ev.notes:
        a(f"- NOTE: {n}")
    a("")

    a("## Exclusions\n")
    a("Rows kept out of every timing fit: `excluded` set, `found` false, or no "
      "counter label; and a window's rows from an older analyser "
      "(`superseded_analysis`; the newest `analyzer_version` stays).  The flag "
      "`unresolved_multipath` only gets counted (`flagged_kept:unresolved_multipath`): "
      "on real data it marks the strongest windows, so those rows stay in.  "
      "`anchor_run_without_hfdl_rows` counts WWV minutes from radiod runs no HFDL "
      "row names; they still score.  WWV rows that carry only `residual_ms` (T3's own frame, fitted to those "
      "same arrivals) stay out of check 4: `wwv_residual_route_excluded`.\n")
    for s, c in sorted(ev.exclusions.items()):
        a(f"- {s}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    a("")

    a("## Status-path label noise\n")
    a("`label_diff_s` = status-pair label minus counter label of each window's first sample.\n")
    a("| station | n | median | p95 abs | max abs |")
    a("|---|---|---|---|---|")
    for s, ln in sorted(ev.label_noise.items()):
        a(f"| {s} | {ln['n']} | {_ms(ln['median_s'])} | {_ms(ln['p95_abs_s'])} | {_ms(ln['max_abs_s'])} |")
    a("")

    a("## Reference cross-checks\n")
    a("Each cross-check tier beside the anchor plane's error.  A steady "
      "(reading - plane) shows the anchor holding; T4 - T6 carries the host clock's "
      "own chrony residual.\n")
    a("| station | tier | n | median reading | n both | median (reading - plane) | sigma |")
    a("|---|---|---|---|---|---|---|")
    for s, cc in sorted(ev.cross_checks.items()):
        for tier, c in cc.items():
            a(f"| {s} | {tier} | {c['n']} | {_ms(c['median_s'])} | {c['n_both']} | "
              f"{_ms(c['minus_plane_median_s'])} | {_ms(c['minus_plane_sigma_s'])} |")
    a("")

    c1 = ev.check1
    a("## Check 1: does each ground station hold a steady offset?\n")
    verdict = {None: "no verdict (fewer than two nights per key)",
               True: "HOLDS", False: "FAILS"}[c1["steady"]]
    a(f"Verdict: {verdict}.  Pooled within-key night-to-night sd "
      f"{_ms(c1['within_key_sd_s'])} ms against {STEADY_S * 1e3:.1f} ms.\n")
    a("Each key (station, ground station, band) gives one mean per night of "
      "`delay_counter_s` - one-hop prior - reference.  The verdict pools each key's "
      "own sd over nights, so steady differences between keys (receive chains, the "
      "prior's residue) do not count.  Per ground station, the nightly change splits "
      "into the part common to every key (the offset wandering) and the between-key "
      "remainder (the path, e.g. hmF2); the common part needs keys from two sites "
      "or more, else it reads `-`.  The offset, the receive chain and the prior's "
      "error enter each key as one sum: a failure stays ambiguous between offset and "
      "path unless the split names one.  No hmF2 regression yet: the broadcast rows "
      "name their ionosonde source but carry no hmF2 value.\n")
    a("| ground station | mean offset | within-key sd | common sd | between-key sd | keys | sites | nights |")
    a("|---|---|---|---|---|---|---|---|")
    for gs, g in sorted(c1["per_gs"].items(), key=str):
        a(f"| {_gs_name(gs)} | {_ms(g['offset_s'])} | {_ms(g['within_key_sd_s'])} | "
          f"{_ms(g['common_sd_s'])} | {_ms(g['between_key_sd_s'])} | {g['keys']} | {g['sites']} | {g['nights']} |")
    a("")
    a("| station | ground station | band | night | offset | sd | n |")
    a("|---|---|---|---|---|---|---|")
    for (st, gs, band), nights in sorted(c1["per_key"].items(), key=str):
        for night, (o, sd, n) in nights.items():
            a(f"| {st} | {_gs_name(gs)} | {band} | {night} | {_ms(o)} | {_ms(sd)} | {n} |")
    a("")

    a("## Check 2: how tightly does one squitter time?\n")
    a("Single-arrival scatter (1.4826 MAD) against the reference, each (ground station, "
      "band, night) median removed.  Counter label (`delay_counter_s`, scored) beside the "
      "status-pair label (`delay_s`).\n")
    for st, c in sorted(ev.check2.items()):
        cc, ss = c["counter"], c["status"]
        a(f"- {st}: counter sigma1 {_ms(cc['sigma1_s'])} (n {cc['n']}; "
          f"squitters for 0.1 ms: {cc.get('n_for_0.1ms', '-')}, for 1 ms: "
          f"{cc.get('n_for_1ms', '-')}); status sigma1 {_ms(ss['sigma1_s'])} (n {ss['n']})")
        if cc["by_hour"]:
            a("")
            a("  | hour Z | counter sigma1 | n | status sigma1 |")
            a("  |---|---|---|---|")
            for h, (sg, n) in cc["by_hour"].items():
                st_h = ss["by_hour"].get(h, (None, 0))[0]
                a(f"  | {h:02d} | {_ms(sg)} | {n} | {_ms(st_h)} |")
            a("")
    a("")

    a("## Check 3: do the sites agree?\n")
    a("Same squitter (ground station, frequency, UTC day and frame) heard at two sites: "
      "(delay - prior - reference) at A minus the same at B.  Geometry and the "
      "references removed, what remains holds the two sites' chain difference plus the "
      "prior's residue; a night-to-night change points at labelling or the analyser.\n")
    if not ev.check3:
        a("No squitter heard at two sites with geometry and a reference at both.\n")
    else:
        a("| ground station | A | B | n | median | sigma | per night |")
        a("|---|---|---|---|---|---|---|")
        for (gs, sa, sb), c in sorted(ev.check3.items(), key=str):
            pn = ", ".join(f"{n} {_ms(v)}" for n, v in c["per_night"].items())
            a(f"| {_gs_name(gs)} | {sa} | {sb} | {c['n']} | {_ms(c['median_s'])} | "
              f"{_ms(c['sigma_s'])} | {pn} |")
        a("")

    a("## Check 4: with and without HFDL\n")
    a("WWVB gives no arrival rows (its record holds decoder rows only), so the "
      "without-HFDL arm uses WWV and WWVH alone; WWVB does not enter.  Each station's "
      "clock error gets reconstructed per minute on the scoring nights, by weighted "
      "least squares over the trained sources, then scored against the reference, over "
      "the station's night window.  The like-for-like comparison uses minutes both arms "
      "reconstruct; minutes only HFDL fills get their own line.  An hour counts empty "
      f"when the arm reconstructs under {ev.coverage:.0%} of its reference minutes.\n")
    a("| station | common minutes | WWV RMS | WWV p95 | +HFDL RMS | +HFDL p95 | "
      "HFDL-only minutes | HFDL-only RMS | HFDL-only p95 | WWV empty h | +HFDL empty h | hours |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st in sorted({k[0] for k in ev.scores}):
        w, b = ev.scores[(st, "wwv")], ev.scores[(st, "wwv+hfdl")]
        wc, bc, bo = w["common"], b["common"], b["hfdl_only"]
        a(f"| {st} | {wc['n']} | {_ms(wc['rms_s'])} | {_ms(wc['p95_s'])} | "
          f"{_ms(bc['rms_s'])} | {_ms(bc['p95_s'])} | {bo['n']} | {_ms(bo['rms_s'])} | "
          f"{_ms(bo['p95_s'])} | {w['empty_hours']} | {b['empty_hours']} | {w['hours']} |")
    a("")
    for st in sorted({k[0] for k in ev.scores}):
        a(f"### {st} by hour (all minutes each arm covers)\n")
        a("| hour Z | WWV RMS | WWV p95 | WWV empty | +HFDL RMS | +HFDL p95 | +HFDL empty |")
        a("|---|---|---|---|---|---|---|")
        w, b = ev.scores[(st, "wwv")]["by_hour"], ev.scores[(st, "wwv+hfdl")]["by_hour"]
        for h in sorted(set(w) | set(b)):
            ww, bb = w.get(h, {}), b.get(h, {})
            a(f"| {h:02d} | {_ms(ww.get('rms_s'))} | {_ms(ww.get('p95_s'))} | "
              f"{ww.get('empty', '-')}/{ww.get('slots', '-')} | {_ms(bb.get('rms_s'))} | "
              f"{_ms(bb.get('p95_s'))} | {bb.get('empty', '-')}/{bb.get('slots', '-')} |")
        a("")
    a("Trained nuisances (b, sigma per key-minute, n training key-minutes):\n")
    for st, nu in sorted(ev.nuisances.items()):
        for key, (bb, sg, n) in sorted(nu.items(), key=str):
            a(f"- {st} {key}: b {_ms(bb)}, sigma {_ms(sg)}, n {n}")
    a("")

    a("## Same-frame multi-frequency consistency\n")
    a(f"One ground station's squitters on two frequencies in the same frame; the gap "
      f"in `delay_counter_s` should stay within the path dispersion, about "
      f"{MULTIFREQ_TOL_S * 1e3:.1f} ms.  A larger gap points at the labels.\n")
    a("| station | ground station | kHz A | kHz B | n | median | p95 abs | over 0.3 ms | status-label p95 abs |")
    a("|---|---|---|---|---|---|---|---|---|")
    for (st, gs, fa, fb), m in sorted(ev.multifreq.items(), key=str):
        a(f"| {st} | {_gs_name(gs)} | {fa} | {fb} | {m['n']} | {_ms(m['median_s'])} | "
          f"{_ms(m['p95_abs_s'])} | {m['frac_over']:.1%} | {_ms(m['status_p95_abs_s'])} |")
    a("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="squitter_eval.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="write the evaluation report")
    r.add_argument("--data", required=True, help="directory of <station>/ subdirectories")
    r.add_argument("--train-nights", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--stations", help="stations TOML (default: <data>/stations.toml)")
    r.add_argument("--night-start-hour", type=int, default=12)
    r.add_argument("--overnight", default="0-12",
                   help="UTC hours lo-hi for stations without night_hours")
    r.add_argument("--hour-coverage", type=float, default=HOUR_COVERAGE,
                   help="fraction of an hour's reference minutes that makes it usable")
    args = ap.parse_args(argv)

    data_dir = Path(args.data)
    st_path = Path(args.stations) if args.stations else data_dir / "stations.toml"
    cfgs = load_stations(st_path)
    data = {p.name: load_station_dir(p) for p in sorted(data_dir.iterdir()) if p.is_dir()}
    lo, hi = (int(x) for x in args.overnight.split("-"))
    ev = evaluate(data, cfgs, args.train_nights, night_hour=args.night_start_hour,
                  overnight=(lo, hi), coverage=args.hour_coverage)
    Path(args.out).write_text(render_report(ev))
    print(f"wrote {args.out}: {len(ev.train_nights)} training nights, "
          f"{len(ev.score_nights)} scoring nights", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
