"""Squitter rows to sigmondb rows, held to the back-fill by a shared parity fixture.

wsprdaemon-server/tests/fixtures/sigmondb/hfdl/ holds the same three files; its back-fill
test re-reads parity-arrivals.jsonl, so the back-fill and this mapping move only together.
"""
import csv
import json
from pathlib import Path

from hfdl_recorder.timing.sigmondb_rows import (
    ARRIVAL_COLUMNS, CAPTURE_COLUMNS, arrival_for, captures_for, utc_iso)

FIX = Path(__file__).parent / "fixtures" / "sigmondb"
_BOOL = ("found", "utc_sync")
_INT = ("gs_id", "freq_khz", "frame_index", "frame_offset", "analyzer_version")


def _csv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _jsonl(p):
    with open(p) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _typed(r):
    """A CSV row with the types batch._row_for gives it."""
    out = {}
    for k, v in r.items():
        if v == "":
            out[k] = None
        elif k in _BOOL:
            out[k] = v == "True"
        elif k in _INT:
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


T6ROW = _typed(_csv(FIX / "t6-group.csv")[0])


def test_t6row_is_what_the_tests_assume():
    assert T6ROW["found"] and T6ROW["t_level_active"] == "T6" and T6ROW["gs_id"] == 4
    assert abs(T6ROW["delay_counter_s"] - 0.0835) < 1e-4


def test_live_arrivals_equal_the_backfill():
    exp = _jsonl(FIX / "parity-arrivals.jsonl")
    got = [a for rid, f in (("AC0G/B4", "t6-group.csv"), ("UACNJ", "t5-group.csv"))
           for r in _csv(FIX / f) if (a := arrival_for(_typed(r), rid))]
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        e.pop("received_at"); assert set(g) == set(e)
        assert {k: g[k] for k in g if k != "engineering"} == {k: e[k] for k in e if k != "engineering"}
        assert e["engineering"].items() <= g["engineering"].items()


def test_columns_are_ddl_order_without_received_at():
    e = _jsonl(FIX / "parity-arrivals.jsonl")[0]
    assert list(ARRIVAL_COLUMNS) == [k for k in e if k != "received_at"]
    assert list(arrival_for(T6ROW, "AC0G/B4")) == list(ARRIVAL_COLUMNS)
    (c,) = captures_for([T6ROW], "AC0G/B4")
    assert list(c) == list(CAPTURE_COLUMNS) and "received_at" not in CAPTURE_COLUMNS


def test_label_only_arrival_travels_null_with_basis_label():
    a = arrival_for(dict(T6ROW, delay_counter_s=None), "AC0G/B4")
    assert a["engineering"]["delay_basis"] == "label" and a["origin"] == "" and a["u_epoch_ns"] is None
    assert a["delay_ns"] == round(T6ROW["delay_s"] * 1e9)


def test_t6_counter_arrival_is_assumed_230ns():
    a = arrival_for(T6ROW, "AC0G/B4")
    assert (a["u_epoch_ns"], a["k"], a["a_level_provenance"], a["origin"]) == (230.0, 2.0, "assumed", "t6_anchor")
    assert (a["counter_epoch_id"], a["tx_id"], a["clean_tag"]) == ("ep-1477095220", "hfdl:4", "squitter")
    assert a["schema_version"] == 1 and a["detector_version"] == "hfdl-analyzer-2"
    assert a["time"] == utc_iso(T6ROW["nominal_utc"])


def test_t6_without_counter_map_and_t5_are_null():
    assert arrival_for(dict(T6ROW, counter_map=None), "X")["origin"] == ""
    assert arrival_for(dict(T6ROW, counter_map="not json"), "X")["origin"] == ""
    assert arrival_for(dict(T6ROW, t_level_active="T5"), "X")["u_epoch_ns"] is None


def test_unfound_excluded_or_delayless_rows_give_no_arrival():
    assert arrival_for(dict(T6ROW, found=False), "X") is None
    assert arrival_for(dict(T6ROW, excluded="implausible_delay"), "X") is None
    assert arrival_for(dict(T6ROW, delay_counter_s=None, delay_s=None), "X") is None


def test_engineering_values_are_strings():
    a = arrival_for(dict(T6ROW, flags=None), "X")
    e = a["engineering"]
    assert all(isinstance(v, str) for v in e.values())
    assert (e["frame_index"], e["frame_offset"], e["flags"]) == ("1800", "1", "")
    assert set(e) == {"frame_index", "frame_offset", "flags", "second_peak_ratio",
                      "second_peak_delay_s", "peak_width_s", "lobe_distortion",
                      "t_level_active", "delay_basis"}


def test_utc_iso_format():
    assert utc_iso(0.0) == "1970-01-01T00:00:00.000000Z"
    assert utc_iso(1.5) == "1970-01-01T00:00:01.500000Z"


def test_two_ground_stations_one_frequency_one_capture():        # Review Focus 4
    rows = [dict(T6ROW, gs_id=4), dict(T6ROW, gs_id=1, nominal_utc=T6ROW["nominal_utc"] + 32, found=False)]
    (c,) = captures_for(rows, "AC0G/B4")
    assert c["expected"] == {"hfdl:4": 1.0, "hfdl:1": 0.0} and (c["n_expected"], c["n_found"]) == (2, 1)
    assert c["duration_s"] == 64.0


def test_capture_fields_from_the_fixture():
    rows = [_typed(r) for r in _csv(FIX / "t6-group.csv")]
    (c,) = captures_for(rows, "AC0G/B4")
    t = [r["nominal_utc"] for r in rows]
    assert c["time"] == utc_iso(min(t)) and c["duration_s"] == max(t) - min(t) + 32.0
    assert (c["n_expected"], c["n_found"], c["expected"]) == (5, 2, {"hfdl:4": 0.4})
    assert (c["centre_hz"], c["rate_hz_per_s"], c["ring_ref"]) == (11387e3, 0.0, "")
    assert (c["origin"], c["counter_epoch_id"], c["u_epoch_ns"]) == ("t6_anchor", "ep-1477095220", 230.0)
    assert c["engineering"] == {"source": "squitter-batch", "n_rows": "5", "n_anchored": "5",
                                "n_rows:hfdl:4": "5", "n_found:hfdl:4": "2"}
    (c5,) = captures_for([_typed(r) for r in _csv(FIX / "t5-group.csv")], "UACNJ")
    assert c5["origin"] == "" and c5["engineering"]["n_anchored"] == "0"


def test_two_frequencies_make_two_captures():
    caps = captures_for([T6ROW, dict(T6ROW, freq_khz=13276)], "X")
    assert sorted(c["centre_hz"] for c in caps) == [11387e3, 13276e3]


def test_error_rows_without_nominal_count_nowhere():
    assert captures_for([{"found": False, "freq_khz": 11387, "analyzer_version": 2}], "X") == []
