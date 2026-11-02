"""Offline squitter timing evaluation (tools/squitter_eval.py), on synthetic rows."""

import copy
import csv
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "squitter_eval.py"
_spec = importlib.util.spec_from_file_location("squitter_eval", _TOOL)
se = importlib.util.module_from_spec(_spec)
sys.modules["squitter_eval"] = se         # dataclasses look the module up
_spec.loader.exec_module(se)

OFFSET_S = 1.234e-3          # the ground station's offset the tests plant
SCATTER_S = 0.2e-3
NIGHT0 = datetime(2026, 11, 1, 12, tzinfo=timezone.utc).timestamp()   # night "20261101"
DAY_S = 86400.0

SIGNS = dict(se.EXPECTED_OFFSET_SIGNS)

CFG_TOML = """
[stations.RX-A]
grid = "EN16"
reference = ["T6"]

[stations.RX-H]
grid = "FN20mv"
reference = ["T4"]
hide_reference = true
"""


def _cfgs(tmp_path):
    p = tmp_path / "stations.toml"
    p.write_text(CFG_TOML)
    return se.load_stations(p)


def clock_error(t):
    """The label plane's error: slow wander of a few hundred microseconds."""
    return 0.3e-3 * math.sin(t / 7000.0) + 0.1e-3


def make_station(station, cfg, nights, seed, ref_field, wwv_gap_hours=(),
                 hours=range(0, 8), gs_ids=(1, 4), bands=((8927, "HFDL8"), (6559, "HFDL6"))):
    """rows, minutes and broadcasts for one receiving station."""
    rng = np.random.default_rng(seed)
    rows, minutes, broadcasts = [], [], []
    for n in range(nights):
        midnight = NIGHT0 + n * DAY_S + 12 * 3600       # 00:00Z after the night's evening
        for h in hours:
            for mi in range(60):
                m = midnight + h * 3600 + mi * 60
                e = clock_error(m)
                minutes.append({
                    "kind": "minute", "utc_minute": int(m),
                    "t6_offset_ms": e * 1e3 if ref_field == "t6" else None,
                    "t4_offset_ms": e * 1e3 if ref_field == "t4" else None,
                    "t2_offset_ms": None, "t5_vs_adopted": None,
                    "t3_vs_adopted": {"shadow_residual_ns": 0.0, "sigma_ns": 1e6,
                                      "vs_tier": "T6"},
                    "t3_shadow": None, "offset_signs": SIGNS,
                })
                if h not in wwv_gap_hours:
                    broadcasts.append({
                        "kind": "broadcast", "utc_minute": int(m),
                        "channel": "SHARED_10000", "station": "WWV",
                        "residual_ms": (e + 0.5e-3 + rng.normal(0, 1.0e-3)) * 1e3,
                        "sigma_ms": 1.0, "expected_delay_ms": 6.0,
                        "fold_snr_db": 10.0, "arrival_input_index": None,
                    })
                for gs in gs_ids:
                    gsll = se.GROUND_STATIONS[gs][1:]
                    prior = se.path_prior_s(cfg.latlon, gsll)
                    for fi, (freq, band) in enumerate(bands):
                        for k in range(2):
                            nominal = m + 32.0 * k + 0.5 * gs
                            dc = prior + OFFSET_S + clock_error(nominal) + rng.normal(0, SCATTER_S)
                            ld = 0.002 + rng.normal(0, 0.001)
                            rows.append({
                                "radiod_id": station, "gs_id": gs,
                                "gs_name": se.GROUND_STATIONS[gs][0],
                                "freq_khz": freq, "band": band,
                                "frame_index": int(nominal // 32), "frame_offset": 0,
                                "nominal_utc": nominal, "t_level_active": "T6",
                                "utc_sync": True, "found": True, "flags": "",
                                "excluded": "", "snr_db": 25.0, "doppler_hz": 0.1,
                                "delay_counter_s": dc, "delay_s": dc + ld,
                                "label_diff_s": ld,
                                "arrival_input_index": 1.0e9 + nominal * 64.8e6,
                            })
    return se.StationData(rows=rows, minutes=minutes, broadcasts=broadcasts)


def dataset(tmp_path, nights=4):
    cfgs = _cfgs(tmp_path)
    data = {
        "RX-A": make_station("RX-A", cfgs["RX-A"], nights, 1, "t6", wwv_gap_hours=(3, 4)),
        "RX-H": make_station("RX-H", cfgs["RX-H"], nights, 2, "t4", wwv_gap_hours=(5,)),
    }
    return data, cfgs


# ---------------------------------------------------------------- check 1

def test_check1_recovers_ground_station_offset(tmp_path):
    data, cfgs = dataset(tmp_path, nights=3)
    ev = se.evaluate(data, cfgs, train_nights=1)
    c1 = ev.check1
    assert c1, "check 1 produced nothing"
    for (station, gs, band), per_night in c1["per_key"].items():
        assert len(per_night) == 3
        for night, (off, sd, n) in per_night.items():
            assert abs(off - OFFSET_S) < 0.05e-3, (station, gs, band, night, off)
            assert abs(sd - SCATTER_S) < 0.05e-3
    for gs, (off, spread) in c1["per_gs"].items():
        assert abs(off - OFFSET_S) < 0.05e-3


# ---------------------------------------------------------------- nights

def test_split_nights_never_overlap():
    nights = ["20261101", "20261103", "20261102", "20261104"]
    train, score = se.split_nights(nights, 2)
    assert train == ["20261101", "20261102"]
    assert score == ["20261103", "20261104"]
    assert not set(train) & set(score)
    with pytest.raises(ValueError):
        se.split_nights(nights, 4)          # nothing left to score
    with pytest.raises(ValueError):
        se.split_nights(nights, 0)          # nothing to learn from


def test_night_boundary():
    assert se.night_of(NIGHT0 + 12 * 3600) == "20261101"           # 00:00Z 11-02
    assert se.night_of(NIGHT0 - 1) == "20261031"                   # 11:59:59Z 11-01
    assert se.night_of(NIGHT0) == "20261101"


def test_training_and_scoring_never_overlap(tmp_path):
    data, cfgs = dataset(tmp_path, nights=4)
    ev = se.evaluate(data, cfgs, train_nights=2)
    assert not set(ev.train_nights) & set(ev.score_nights)
    for (station, arm), rec in ev.recon.items():
        for m in rec:
            assert se.night_of(m) in ev.score_nights
    # The nuisances come from training nights only: wreck every scoring-night
    # row and the nuisances stay put.
    wrecked = copy.deepcopy(data)
    for sd in wrecked.values():
        for r in sd.rows:
            if se.night_of(r["nominal_utc"]) in ev.score_nights:
                r["delay_counter_s"] += 0.004
        for b in sd.broadcasts:
            if se.night_of(b["utc_minute"]) in ev.score_nights:
                b["residual_ms"] += 4.0
    ev2 = se.evaluate(wrecked, cfgs, train_nights=2)
    assert ev2.nuisances == ev.nuisances


# ---------------------------------------------------------------- reference

def _poison(data, station, nights=None):
    out = copy.deepcopy(data)
    for m in out[station].minutes:
        if nights is None or se.night_of(m["utc_minute"]) in nights:
            for k in ("t6_offset_ms", "t4_offset_ms"):
                m[k] = float("nan")
            m["t5_vs_adopted"] = {"shadow_residual_ns": float("nan"), "vs_tier": "T6"}
    return out


def _same(r1, r2):
    assert r1.keys() == r2.keys()
    for key in r1:
        assert r1[key].keys() == r2[key].keys(), key
        for m in r1[key]:
            assert r1[key][m] == r2[key][m], (key, m)


def test_reference_never_enters_reconstruction(tmp_path):
    data, cfgs = dataset(tmp_path, nights=4)
    ev = se.evaluate(data, cfgs, train_nights=2)
    assert ev.recon[("RX-A", "wwv+hfdl")], "nothing reconstructed"
    # Scoring nights: no station's reference reaches the reconstruction.
    poisoned = _poison(_poison(data, "RX-A", ev.score_nights), "RX-H", ev.score_nights)
    _same(ev.recon, se.evaluate(poisoned, cfgs, train_nights=2).recon)
    # The hidden station: its reference reaches nothing on any night.
    ev_h = se.evaluate(_poison(data, "RX-H"), cfgs, train_nights=2)
    assert {k: v for k, v in ev_h.recon.items() if k[0] == "RX-H"} == \
        {k: v for k, v in ev.recon.items() if k[0] == "RX-H"}
    # Power: RX-A learns its nuisances from its reference on training nights,
    # so shifting that reference must move RX-A's reconstruction.
    shifted = copy.deepcopy(data)
    for m in shifted["RX-A"].minutes:
        if se.night_of(m["utc_minute"]) in ev.train_nights:
            m["t6_offset_ms"] += 1.0
    ev_s = se.evaluate(shifted, cfgs, train_nights=2)
    key = ("RX-A", "wwv+hfdl")
    m0 = next(iter(ev.recon[key]))
    assert abs(ev_s.recon[key][m0][0] - ev.recon[key][m0][0] - 1e-3) < 1e-5


def test_reference_sign_convention():
    row = {"utc_minute": 0, "offset_signs": SIGNS, "t4_offset_ms": 2.0,
           "t6_offset_ms": -0.5, "t5_vs_adopted": {"shadow_residual_ns": 1.0e6,
                                                   "vs_tier": "T3"}}
    # reading minus truth, in seconds
    assert se.reference_error_s(row, ["T4"]) == (pytest.approx(2.0e-3), "T4")
    assert se.reference_error_s(row, ["T6"]) == (pytest.approx(-0.5e-3), "T6")
    # T5 minus adopted = +1 ms, so the adopted plane reads 1 ms early.
    assert se.reference_error_s(row, ["T5"]) == (pytest.approx(-1.0e-3), "T5")
    assert se.reference_error_s(row, ["T6", "T5"])[1] == "T6"
    row2 = dict(row, t6_offset_ms=None)
    assert se.reference_error_s(row2, ["T6", "T5"])[1] == "T5"
    assert se.reference_error_s(dict(row2, t5_vs_adopted=None), ["T6", "T5"]) == (None, None)
    # A changed convention refuses rather than scores with the wrong sign.
    bad = dict(SIGNS, t4_offset_ms="source minus host")
    with pytest.raises(se.SignConventionError):
        se.reference_error_s(dict(row, offset_signs=bad), ["T4"])


def test_reference_hfdl_sign_end_to_end(tmp_path):
    """A label plane running late makes delays longer; the reconstruction
    must then read late too, matching a reference that says late."""
    data, cfgs = dataset(tmp_path, nights=3)
    ev = se.evaluate(data, cfgs, train_nights=1)
    s = ev.scores[("RX-A", "wwv+hfdl")]
    assert s["p95_s"] < 0.2e-3          # a flipped sign would show ~2x the wander


# ---------------------------------------------------------------- exclusions

def test_exclusions_counted_and_kept_out_of_the_fit(tmp_path):
    data, cfgs = dataset(tmp_path, nights=3)
    rows = data["RX-A"].rows
    base = dict(rows[0])
    bad = [dict(base, excluded="implausible_delay"),
           dict(base, excluded="utc_sync_false"),
           dict(base, found=False, delay_counter_s=None),
           dict(base, flags="unresolved_multipath"),
           dict(base, flags="lobe_wide,prekey_edge_mismatch"),
           dict(base, delay_counter_s=None)]
    for r in bad:
        if r.get("delay_counter_s") is not None:
            r["delay_counter_s"] += 0.020         # would wreck the offset if fitted
    rows.extend(bad)
    ev = se.evaluate(data, cfgs, train_nights=1)
    ex = ev.exclusions["RX-A"]
    assert ex["excluded:implausible_delay"] == 1
    assert ex["excluded:utc_sync_false"] == 1
    assert ex["not_found"] == 1
    assert ex["flag:unresolved_multipath"] == 1
    assert ex["flag:prekey_edge_mismatch"] == 1
    assert ex["no_counter_label"] == 1
    assert ex["used"] == len(rows) - len(bad)
    for (station, gs, band), per_night in ev.check1["per_key"].items():
        for off, sd, n in per_night.values():
            assert abs(off - OFFSET_S) < 0.05e-3


# ---------------------------------------------------------------- other checks

def test_label_noise_distribution(tmp_path):
    data, cfgs = dataset(tmp_path, nights=2)
    ev = se.evaluate(data, cfgs, train_nights=1)
    ln = ev.label_noise["RX-A"]
    assert abs(ln["median_s"] - 0.002) < 0.0002
    assert ln["p95_abs_s"] > ln["median_s"]
    assert ln["max_abs_s"] >= ln["p95_abs_s"]
    assert ln["n"] == len(data["RX-A"].rows)


def test_check2_reports_both_labels(tmp_path):
    data, cfgs = dataset(tmp_path, nights=2)
    ev = se.evaluate(data, cfgs, train_nights=1)
    c2 = ev.check2["RX-A"]
    assert abs(c2["counter"]["sigma1_s"] - SCATTER_S) < 0.05e-3
    # The status-pair label carries its own 1 ms noise on top.
    assert c2["status"]["sigma1_s"] > 3 * c2["counter"]["sigma1_s"]
    assert c2["counter"]["by_hour"]


def test_multifrequency_consistency(tmp_path):
    data, cfgs = dataset(tmp_path, nights=2)
    ev = se.evaluate(data, cfgs, train_nights=1)
    mf = ev.multifreq
    assert mf, "no same-frame pairs found"
    for key, s in mf.items():
        assert s["frac_over"] < 0.5          # 0.28 ms rms difference of two 0.2 ms arrivals
    # A label fault on one band shows as a gap over 0.3 ms.
    for r in data["RX-A"].rows:
        if r["band"] == "HFDL6":
            r["delay_counter_s"] += 1.0e-3
    mf2 = se.evaluate(data, cfgs, train_nights=1).multifreq
    hit = [s for k, s in mf2.items() if k[0] == "RX-A"]
    assert hit and all(s["frac_over"] > 0.9 for s in hit)


def test_check3_sites_agree(tmp_path):
    data, cfgs = dataset(tmp_path, nights=2)
    ev = se.evaluate(data, cfgs, train_nights=1)
    c3 = ev.check3
    assert c3
    for key, s in c3.items():
        # same squitter heard at both sites: after geometry and each site's
        # reference, the planted offsets cancel
        assert abs(s["median_s"]) < 0.05e-3, key


def test_check4_hfdl_fills_wwv_gaps(tmp_path):
    data, cfgs = dataset(tmp_path, nights=3)
    ev = se.evaluate(data, cfgs, train_nights=1)
    wwv = ev.scores[("RX-A", "wwv")]
    both = ev.scores[("RX-A", "wwv+hfdl")]
    assert wwv["empty_hours"] == 2 * 2           # hours 3 and 4 on 2 scoring nights
    assert both["empty_hours"] == 0
    assert both["p95_s"] < wwv["p95_s"]
    # the hidden station still scores, against T4 alone
    assert ev.scores[("RX-H", "wwv+hfdl")]["n"] > 0


def test_path_prior_geometry():
    # zero ground range: straight up and down through 300 km
    assert se.path_prior_s((40.0, -100.0), (40.0, -100.0)) == pytest.approx(
        2 * 300e3 / 299_792_458.0, rel=1e-9)
    # longer paths take longer
    near = se.path_prior_s((40.0, -100.0), (40.0, -95.0))
    far = se.path_prior_s((40.0, -100.0), (40.0, -80.0))
    assert far > near > 2 * 300e3 / 299_792_458.0


def test_grid_decoding():
    lat, lon = se.grid_to_latlon("FN20mv")
    assert 40.8 < lat < 40.9 and -75.0 < lon < -74.9
    assert se.grid_to_latlon("TODO") is None


# ---------------------------------------------------------------- files

def _write_station(dirpath, sd, fmt):
    dirpath.mkdir(parents=True)
    keys = list(dict.fromkeys(k for r in sd.rows for k in r))
    if fmt == "csv":
        with open(dirpath / "squitter_timing.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in sd.rows:
                w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    else:
        with open(dirpath / "squitter_timing.jsonl", "w") as f:
            for r in sd.rows:
                f.write(json.dumps(r) + "\n")
    shadow = dirpath / "t3_shadow"
    shadow.mkdir()
    by_day = {}
    for m in sd.minutes:
        d = datetime.fromtimestamp(m["utc_minute"], timezone.utc).strftime("%Y%m%d")
        by_day.setdefault(("minutes", d), []).append(m)
    for b in sd.broadcasts:
        d = datetime.fromtimestamp(b["utc_minute"], timezone.utc).strftime("%Y%m%d")
        by_day.setdefault((f"broadcasts-{b['channel']}", d), []).append(b)
    for (stream, d), items in by_day.items():
        with open(shadow / f"{stream}-{d}.jsonl", "w") as f:
            f.write(json.dumps({"kind": "transition", "utc_minute": items[0]["utc_minute"]}) + "\n")
            for it in items:
                f.write(json.dumps(it) + "\n")


def test_csv_and_jsonl_load_alike(tmp_path):
    data, cfgs = dataset(tmp_path, nights=2)
    _write_station(tmp_path / "c" / "RX-A", data["RX-A"], "csv")
    _write_station(tmp_path / "j" / "RX-A", data["RX-A"], "jsonl")
    a = se.load_station_dir(tmp_path / "c" / "RX-A")
    b = se.load_station_dir(tmp_path / "j" / "RX-A")
    assert len(a.rows) == len(b.rows) == len(data["RX-A"].rows)
    for ra, rb in zip(a.rows, b.rows):
        for k in ("gs_id", "freq_khz", "found", "utc_sync", "delay_counter_s",
                  "flags", "excluded", "nominal_utc", "band"):
            assert ra[k] == rb[k], k
    assert isinstance(a.rows[0]["found"], bool) and isinstance(a.rows[0]["gs_id"], int)
    assert len(a.minutes) == len(data["RX-A"].minutes)       # transitions skipped
    assert len(a.broadcasts) == len(data["RX-A"].broadcasts)


def test_cli_report(tmp_path):
    data, cfgs = dataset(tmp_path, nights=3)
    root = tmp_path / "data"
    for st, sd in data.items():
        _write_station(root / st, sd, "csv" if st == "RX-A" else "jsonl")
    out = tmp_path / "report.md"
    rc = se.main(["report", "--data", str(root), "--train-nights", "1",
                  "--stations", str(tmp_path / "stations.toml"), "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    for head in ("Check 1", "Check 2", "Check 3", "Check 4",
                 "Same-frame multi-frequency", "Status-path label noise", "Exclusions"):
        assert head in text, head
    assert "WWVB" in text
    assert "20261101" in text and "train" in text.lower()


def test_wwv_index_route_shares_the_hfdl_plane():
    """A broadcast with an input index goes to UTC through the minute row's
    anchor, like the HFDL counter label; residual_ms only without one."""
    f_in = 64.8e6
    m = 1_793_000_040
    a_ns, a_idx = (m - 10) * 10**9 + 250_000, 5.0e12
    late = 0.7e-3                         # the plane reads 0.7 ms late
    arrival_utc_rel = 0.006 + late        # 6 ms hop, past the minute
    idx = a_idx + (10 + arrival_utc_rel - 250e-6) * f_in
    sd = se.StationData(rows=[], minutes=[{"utc_minute": m, "t3_shadow": {
        "t6_anchor_input_index": a_idx, "t6_anchor_utc_ns": a_ns}}],
        broadcasts=[{"utc_minute": m, "channel": "WWV_20000", "station": "WWV",
                     "arrival_input_index": idx, "expected_delay_ms": 6.0,
                     "residual_ms": 99.0, "sigma_ms": 2.0},
                    {"utc_minute": m, "channel": "WWV_20000", "station": "WWVH",
                     "arrival_input_index": None, "expected_delay_ms": 9.0,
                     "residual_ms": 1.5, "sigma_ms": 2.0}])
    cfg = se.StationCfg(name="X", grid=None, latlon=None, reference=["T6"],
                        input_samprate=f_in)
    from collections import defaultdict
    counts = defaultdict(int)
    out = se._wwv_measurements(sd, cfg, se._anchor_series(sd), counts)
    (m1, k1, y1, s1), (m2, k2, y2, s2) = out
    assert k1[-1] == "index" and y1 == pytest.approx(late, abs=1e-7)
    assert k2[-1] == "residual" and y2 == pytest.approx(1.5e-3)
    assert counts["wwv_route_index"] == 1 and counts["wwv_route_residual"] == 1
