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

F_IN = 64.8e6
EPOCH_UTC = NIGHT0 - 3600.0                       # the radiod run started an hour before
EPOCH_GPS = EPOCH_UTC - 315_964_800 + 18
HOST_MINUS_PLANE_S = 0.15e-3                      # T4 sees the host, not the plane

CFG_TOML = """
[stations.RX-A]
grid = "EN16"
reference = ["T6"]
cross_check = ["T4"]

[stations.RX-H]
grid = "FN20mv"
reference = ["T6"]
cross_check = ["T4"]
hide_reference = true
"""


def _cfgs(tmp_path):
    p = tmp_path / "stations.toml"
    p.write_text(CFG_TOML)
    return se.load_stations(p)


def clock_error(t):
    """The label plane's error: slow wander of a few hundred microseconds."""
    return 0.3e-3 * math.sin(t / 7000.0) + 0.1e-3


def index_of(t_utc):
    """radiod's input-sample index of a true UTC instant."""
    return (t_utc - EPOCH_UTC) * F_IN


def make_station(station, cfg, nights, seed, wwv_gap_hours=(), hours=range(0, 8),
                 gs_ids=(1, 4), bands=((8927, "HFDL8"), (6559, "HFDL6")),
                 plane_shift=None, ref_follows=True):
    """rows, minutes and broadcasts for one receiving station.

    The anchor carries the plane's error e: it labels index(m) - e*f_in as
    minute m.  The HFDL counter label and the WWV index route both inherit
    e through it; t6_offset_ms reports it.  ``plane_shift(m)`` adds to the
    plane; with ``ref_follows`` the reference sees the shift too."""
    rng = np.random.default_rng(seed)
    shift = plane_shift or (lambda m: 0.0)
    rows, minutes, broadcasts = [], [], []
    for n in range(nights):
        midnight = NIGHT0 + n * DAY_S + 12 * 3600       # 00:00Z after the night's evening
        for h in hours:
            for mi in range(60):
                m = midnight + h * 3600 + mi * 60
                e = clock_error(m) + shift(m)
                e_ref = clock_error(m) + (shift(m) if ref_follows else 0.0)
                minutes.append({
                    "kind": "minute", "utc_minute": int(m),
                    "t6_offset_ms": e_ref * 1e3,
                    "t4_offset_ms": (e_ref + HOST_MINUS_PLANE_S + rng.normal(0, 0.05e-3)) * 1e3,
                    "t2_offset_ms": None, "t5_vs_adopted": None,
                    "t3_vs_adopted": {"shadow_residual_ns": 0.0, "sigma_ns": 1e6,
                                      "vs_tier": "T6"},
                    "t3_shadow": {"t6_anchor_input_index": index_of(m) - e * F_IN,
                                  "t6_anchor_utc_ns": int(m) * 10**9},
                    "offset_signs": SIGNS,
                })
                if h not in wwv_gap_hours:
                    true_arr = m + 0.006 + 0.5e-3 + rng.normal(0, 1.0e-3)
                    broadcasts.append({
                        "kind": "broadcast", "utc_minute": int(m),
                        "channel": "SHARED_10000", "station": "WWV",
                        "residual_ms": 0.0, "sigma_ms": 1.0, "expected_delay_ms": 6.0,
                        "fold_snr_db": 10.0, "arrival_input_index": index_of(true_arr),
                    })
                    # T3's own frame only: must stay out of check 4
                    broadcasts.append({
                        "kind": "broadcast", "utc_minute": int(m),
                        "channel": "SHARED_10000", "station": "WWVH",
                        "residual_ms": 0.0, "sigma_ms": 1.0, "expected_delay_ms": 9.0,
                        "fold_snr_db": 10.0, "arrival_input_index": None,
                    })
                for gs in gs_ids:
                    gsll = se.GROUND_STATIONS[gs][1:]
                    prior = se.path_prior_s(cfg.latlon, gsll)
                    for fi, (freq, band) in enumerate(bands):
                        for k in range(2):
                            nominal = m + 32.0 * k + 0.5 * gs
                            dc = (prior + OFFSET_S + clock_error(nominal) + shift(m)
                                  + rng.normal(0, SCATTER_S))
                            ld = 0.002 + rng.normal(0, 0.001)
                            rows.append({
                                "radiod_id": station, "gs_id": gs,
                                "gs_name": se.GROUND_STATIONS[gs][0],
                                "freq_khz": freq, "band": band,
                                "frame_index": ((n * 24 + h) * 60 + mi) * 2 + k,
                                "frame_offset": 0,
                                "nominal_utc": nominal, "t_level_active": "T6",
                                "utc_sync": True, "found": True, "flags": "",
                                "excluded": "", "snr_db": 25.0, "doppler_hz": 0.1,
                                "delay_counter_s": dc, "delay_s": dc + ld,
                                "label_diff_s": ld,
                                "input_samprate": F_IN, "input_epoch_gps_s": EPOCH_GPS,
                                "arrival_input_index": index_of(nominal + dc),
                            })
    return se.StationData(rows=rows, minutes=minutes, broadcasts=broadcasts)


def dataset(tmp_path, nights=4, **kw):
    cfgs = _cfgs(tmp_path)
    data = {
        "RX-A": make_station("RX-A", cfgs["RX-A"], nights, 1, wwv_gap_hours=(3, 4), **kw),
        "RX-H": make_station("RX-H", cfgs["RX-H"], nights, 2, wwv_gap_hours=(5,), **kw),
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
    for gs, g in c1["per_gs"].items():
        assert abs(g["offset_s"] - OFFSET_S) < 0.05e-3
    assert c1["steady"] is True
    assert c1["within_key_sd_s"] < 0.05e-3


def test_check1_chain_difference_is_no_wander(tmp_path):
    """A steady offset with a 0.6 ms longer receive chain at one site:
    each key holds still, so check 1 must hold."""
    data, cfgs = dataset(tmp_path, nights=3)
    for r in data["RX-H"].rows:
        r["delay_counter_s"] += 0.6e-3
    c1 = se.evaluate(data, cfgs, train_nights=1).check1
    assert c1["steady"] is True
    for gs, g in c1["per_gs"].items():
        assert g["within_key_sd_s"] < 0.05e-3, gs
        assert g["common_sd_s"] < 0.05e-3, gs


def _shift_night(data, night, pred, ds):
    for st, sd in data.items():
        for r in sd.rows:
            if se.night_of(r["nominal_utc"]) == night and pred(st, r):
                r["delay_counter_s"] += ds


def test_check1_splits_offset_wander_from_path(tmp_path):
    # The ground station's offset moves on one night: common to every site.
    data, cfgs = dataset(tmp_path, nights=3)
    _shift_night(data, "20261102", lambda st, r: r["gs_id"] == 4, 0.3e-3)
    c1 = se.evaluate(data, cfgs, train_nights=1).check1
    g = c1["per_gs"][4]
    assert c1["steady"] is False
    assert g["common_sd_s"] > 0.1e-3
    assert g["between_key_sd_s"] < 0.05e-3
    # One site's path changes on one night: it shows between the sites.
    data, cfgs = dataset(tmp_path, nights=3)
    _shift_night(data, "20261102",
                 lambda st, r: r["gs_id"] == 1 and st == "RX-A", 0.3e-3)
    g1 = se.evaluate(data, cfgs, train_nights=1).check1["per_gs"][1]
    assert g1["between_key_sd_s"] > 0.05e-3
    assert g1["between_key_sd_s"] > 3 * g["between_key_sd_s"]


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
            if se.night_of(b["utc_minute"]) in ev.score_nights \
                    and b["arrival_input_index"] is not None:
                b["arrival_input_index"] += 0.004 * F_IN
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
                                                   "vs_tier": "T6"}}
    # reading minus truth, in seconds
    assert se.reference_error_s(row, ["T4"]) == (pytest.approx(2.0e-3), "T4")
    assert se.reference_error_s(row, ["T6"]) == (pytest.approx(-0.5e-3), "T6")
    # T5 minus T6 = +1 ms, so the T6 plane reads 1 ms early.
    assert se.reference_error_s(row, ["T5"]) == (pytest.approx(-1.0e-3), "T5")
    assert se.reference_error_s(row, ["T6", "T5"])[1] == "T6"
    row2 = dict(row, t6_offset_ms=None)
    assert se.reference_error_s(row2, ["T6", "T5"])[1] == "T5"
    assert se.reference_error_s(dict(row2, t5_vs_adopted=None), ["T6", "T5"]) == (None, None)
    # T5 against any bench but T6 measures another plane: no answer.
    t5_vs_t3 = dict(row2, t5_vs_adopted={"shadow_residual_ns": 1.0e6, "vs_tier": "T3"})
    assert se.reference_error_s(t5_vs_t3, ["T5"]) == (None, None)
    # A changed convention refuses rather than scores with the wrong sign.
    bad = dict(SIGNS, t4_offset_ms="source minus host")
    with pytest.raises(se.SignConventionError):
        se.reference_error_s(dict(row, offset_signs=bad), ["T4"])
    # So does a missing one: no row, no key.
    with pytest.raises(se.SignConventionError):
        se.reference_error_s(dict(row, offset_signs=None), ["T6"])
    no_t6 = {k: v for k, v in SIGNS.items() if k != "t6_offset_ms"}
    with pytest.raises(se.SignConventionError):
        se.reference_error_s(dict(row, offset_signs=no_t6), ["T6"])


def test_t4_cannot_score_the_anchor_plane(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[stations.X]\ngrid = "EN16"\nreference = ["T4"]\n')
    with pytest.raises(ValueError, match="cross_check"):
        se.load_stations(p)


def test_plane_invariance(tmp_path):
    """Shift the plane and its reference together: the score holds.  Shift
    the plane alone: the score moves."""
    wander = lambda m: 0.5e-3 * math.sin(m / 3000.0)        # noqa: E731
    base_d, cfgs = dataset(tmp_path, nights=3)
    both_d, _ = dataset(tmp_path, nights=3, plane_shift=wander, ref_follows=True)
    alone_d, _ = dataset(tmp_path, nights=3, plane_shift=wander, ref_follows=False)
    base = se.evaluate(base_d, cfgs, train_nights=1).scores
    both = se.evaluate(both_d, cfgs, train_nights=1).scores
    alone = se.evaluate(alone_d, cfgs, train_nights=1).scores
    for key in base:
        assert both[key]["n"] == base[key]["n"], key
        assert both[key]["rms_s"] == pytest.approx(base[key]["rms_s"], abs=1e-8), key
        assert both[key]["p95_s"] == pytest.approx(base[key]["p95_s"], abs=1e-8), key
    for st in ("RX-A", "RX-H"):
        assert alone[(st, "wwv+hfdl")]["p95_s"] > base[(st, "wwv+hfdl")]["p95_s"] + 0.2e-3


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
    assert ev.multifreq_duplicates == {}
    # A second row for one (gs, frame, frequency) gets counted, not paired.
    data["RX-A"].rows.append(dict(data["RX-A"].rows[0]))
    assert se.evaluate(data, cfgs, train_nights=1).multifreq_duplicates == {"RX-A": 1}
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
    # like for like: the minutes both arms reconstruct
    assert wwv["common"]["n"] == both["common"]["n"] == 2 * 6 * 60
    assert both["common"]["p95_s"] < wwv["common"]["p95_s"]
    # and the minutes only HFDL fills, apart
    assert both["hfdl_only"]["n"] == 2 * 2 * 60
    assert both["hfdl_only"]["p95_s"] < 0.2e-3
    # the WWVH rows (residual_ms only) never entered
    assert ev.exclusions["RX-A"]["wwv_residual_route_excluded"] == 3 * 6 * 60
    # the hidden station still scores
    assert ev.scores[("RX-H", "wwv+hfdl")]["n"] > 0


def test_check4_hour_needs_coverage(tmp_path):
    """One minute does not make an hour usable."""
    data, cfgs = dataset(tmp_path, nights=3)
    data["RX-A"].broadcasts = [
        b for b in data["RX-A"].broadcasts
        if not (se._hour(b["utc_minute"]) == 6 and (b["utc_minute"] // 60) % 60 >= 10)]
    wwv = se.evaluate(data, cfgs, train_nights=1).scores[("RX-A", "wwv")]
    assert wwv["empty_hours"] == 2 * 3           # hours 3, 4 and the thin 6
    assert wwv["by_hour"][6]["n"] == 2 * 10


def test_per_station_night_window(tmp_path):
    data, cfgs = dataset(tmp_path, nights=3)
    cfgs["RX-A"].night_hours = (2, 5)
    ev = se.evaluate(data, cfgs, train_nights=1)
    assert ev.scores[("RX-A", "wwv")]["hours"] == 2 * 3
    assert ev.scores[("RX-H", "wwv")]["hours"] == 2 * 8      # the 00-12Z default
    wrap = se._in_hours
    assert wrap(23, (22, 4)) and wrap(1, (22, 4)) and not wrap(12, (22, 4))


def test_cross_check_reports_t4_beside_the_plane(tmp_path):
    data, cfgs = dataset(tmp_path, nights=2)
    cc = se.evaluate(data, cfgs, train_nights=1).cross_checks["RX-A"]["T4"]
    assert cc["n_both"] == 2 * 8 * 60
    assert abs(cc["minus_plane_median_s"] - HOST_MINUS_PLANE_S) < 0.01e-3


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
    ((m1, k1, y1, s1),) = out
    assert k1 == ("wwv", "WWV_20000", "WWV") and y1 == pytest.approx(late, abs=1e-7)
    assert counts["wwv_index_route"] == 1 and counts["wwv_residual_route_excluded"] == 1


def test_anchor_from_another_radiod_run_drops_out():
    f_in = 64.8e6
    m = 1_793_000_040
    epoch = 1_792_990_000.0
    good = {"utc_minute": m, "t3_shadow": {"t6_anchor_input_index": (m - epoch) * f_in,
                                           "t6_anchor_utc_ns": m * 10**9}}
    other = {"utc_minute": m + 60, "t3_shadow": {
        "t6_anchor_input_index": (m + 60 - epoch + 7200) * f_in,   # a run 2 h earlier
        "t6_anchor_utc_ns": (m + 60) * 10**9}}
    from collections import defaultdict
    counts = defaultdict(int)
    out = se._anchor_series(se.StationData(minutes=[good, other]), f_in, [epoch], counts)
    assert list(out) == [m]
    assert counts["anchor_other_run"] == 1


def test_one_key_counts_once_per_minute(tmp_path):
    """Rows of one key in one minute average first: correlated copies of one
    key must not outvote the other keys."""
    data, cfgs = dataset(tmp_path, nights=3)
    ev = se.evaluate(data, cfgs, train_nights=1)
    rows = data["RX-A"].rows
    copies = [dict(r) for r in rows if r["gs_id"] == 4 and r["band"] == "HFDL8"]
    rows.extend(copies * 3)
    ev2 = se.evaluate(data, cfgs, train_nights=1)
    key = ("RX-A", "wwv+hfdl")
    assert ev2.recon[key].keys() == ev.recon[key].keys()
    for m in ev.recon[key]:
        assert ev2.recon[key][m][0] == pytest.approx(ev.recon[key][m][0], abs=1e-12)
