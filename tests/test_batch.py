"""Batch squitter analysis: window files in, ``squitter_timing`` rows out."""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from hfdl_recorder.timing import batch
from hfdl_recorder.timing import waveform as w

FS = 6000.0
F_IN = 64_800_000.0
GPS_TO_UTC = 315_964_800 - 18           # GPS seconds -> Unix UTC seconds
EPOCH_GPS = 1_477_000_000.25            # radiod run start, GPS scale
EPOCH_UTC = EPOCH_GPS + GPS_TO_UTC
NOMINAL = 1_793_000_000.0
REG_DAY = datetime.fromtimestamp(NOMINAL, timezone.utc).strftime("%Y%m%d")


class FakeWriter:
    def __init__(self):
        self.inserts = []
        self.closed = False
        self.is_noop = False

    def insert(self, rows):
        self.inserts.extend(rows)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class Factory:
    def __init__(self):
        self.writer = FakeWriter()
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return self.writer


def _window(root, delay, *, frame=0, nominal=NOMINAL, idx_offset=0.0,
            status_late_s=0.0, utc_sync=True, epoch_gps=EPOCH_GPS,
            with_index=True, x=None, name=None):
    """Write one window file; returns its path and the counter label."""
    t0_counter = nominal - 0.03
    idx0 = round((t0_counter - (epoch_gps + GPS_TO_UTC)) * F_IN) + idx_offset
    t0 = (epoch_gps + GPS_TO_UTC) + idx0 / F_IN + status_late_s
    if x is None:
        x = w.synth_squitter(FS, delay, snr_db=20, rng=np.random.default_rng(frame + 7))
    meta = {
        "t0_utc": t0, "nominal_utc": nominal, "fs": FS, "freq_khz": 8912,
        "gs_id": 7, "gs_name": "Test", "frame_index": frame, "frame_offset": 2,
        "band": "8", "anchor_source": "status", "offset_ns": 0,
        "t_level_active": "T3", "utc_sync": utc_sync,
        "input_index0": float(idx0) if with_index else None,
        "input_samprate": F_IN if with_index else None,
        "counter_map": None,
        "input_epoch_gps_s": epoch_gps if with_index else None,
        "anchor_id": 1, "anchor_rtp": 5, "anchor_age_s": 3.0,
        "n_gap_samples": 0, "decimation": {"id": "fft-band-select-v1"},
    }
    day = "20261102"
    d = root / "radiod0" / day
    d.mkdir(parents=True, exist_ok=True)
    p = d / (name or f"7-8912-{frame}.npz")
    np.savez(p, x=np.asarray(x, dtype=np.complex64), meta=np.str_(json.dumps(meta)))
    return p, idx0


def _registration(regdir, idx0, *, ahead_s=-60.0, epoch_gps=EPOCH_GPS,
                  epoch_id="ep-1477000000", day=REG_DAY):
    """A minute row whose anchor sits ahead_s from the window start."""
    regdir.mkdir(parents=True, exist_ok=True)
    a_idx = float(round(idx0 + ahead_s * F_IN))
    utc = (epoch_gps + GPS_TO_UTC) + a_idx / F_IN
    ns = int(round(utc * 1e9))
    row = {"t3": {"counter_epoch_id": epoch_id},
           "t3_shadow": {"t6_anchor_input_index": a_idx, "t6_anchor_utc_ns": ns}}
    with open(regdir / f"minutes-{day}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def _run(tmp_path, **kw):
    fac = Factory()
    counts = batch.run_batch(tmp_path / "win", registration_dir=tmp_path / "reg",
                             writer_factory=fac, **kw)
    return fac, counts


def test_rows_from_synthetic_windows(tmp_path):
    delays = (0.0050, 0.01234, 0.0201)
    for i, d in enumerate(delays):
        _window(tmp_path / "win", d, frame=i)
    fac, counts = _run(tmp_path)
    rows = sorted(fac.writer.inserts, key=lambda r: r["frame_index"])
    assert len(rows) == 3 and counts["analysed"] == 3
    for r, d in zip(rows, delays):
        assert r["found"] and abs(r["delay_s"] - d) < 20e-6
        assert r["radiod_id"] == "radiod0" and r["gs_id"] == 7
        assert r["delay_counter_s"] is None and r["label_diff_s"] is None
        assert r["excluded"] == ""
    assert fac.writer.closed


def test_counter_delay_from_registration(tmp_path):
    _, idx0 = _window(tmp_path / "win", 0.0123)
    _registration(tmp_path / "reg", idx0)
    fac, _ = _run(tmp_path)
    r = fac.writer.inserts[0]
    assert r["delay_counter_s"] is not None
    assert abs(r["delay_counter_s"] - r["delay_s"]) < 1e-6
    exp_idx = idx0 + (r["arrival_utc"] - r["t0_utc"]) * F_IN
    assert abs(r["arrival_input_index"] - exp_idx) < 1.0


@pytest.mark.parametrize("kw", [
    dict(epoch_gps=EPOCH_GPS + 7200.0, epoch_id="ep-1477007200"),   # other run
    dict(ahead_s=-300.0),                                          # too old
    dict(ahead_s=+300.0),                                          # too far ahead
])
def test_registration_not_usable_gives_null(tmp_path, kw):
    _, idx0 = _window(tmp_path / "win", 0.0123)
    _registration(tmp_path / "reg", idx0, **kw)
    fac, _ = _run(tmp_path)
    r = fac.writer.inserts[0]
    assert r["delay_counter_s"] is None and r["label_diff_s"] is None
    assert r["found"]


def test_nearest_registration_wins(tmp_path):
    _, idx0 = _window(tmp_path / "win", 0.0123)
    _registration(tmp_path / "reg", idx0, ahead_s=-100.0)
    _registration(tmp_path / "reg", idx0, ahead_s=-20.0)
    fac, _ = _run(tmp_path)
    assert abs(fac.writer.inserts[0]["delay_counter_s"]
               - fac.writer.inserts[0]["delay_s"]) < 1e-6


def test_label_diff_recorded(tmp_path):
    _, idx0 = _window(tmp_path / "win", 0.0123, status_late_s=0.0023)
    _registration(tmp_path / "reg", idx0)
    fac, _ = _run(tmp_path)
    r = fac.writer.inserts[0]
    assert abs(r["label_diff_s"] - 0.0023) < 2e-6
    # the counter label is the better clock: its delay sits 2.3 ms earlier
    assert abs((r["delay_s"] - r["delay_counter_s"]) - 0.0023) < 2e-6


def test_utc_sync_false_marked(tmp_path):
    _window(tmp_path / "win", 0.0123, utc_sync=False)
    fac, _ = _run(tmp_path)
    assert fac.writer.inserts[0]["excluded"] == "utc_sync_false"


def test_implausible_delay_marked(tmp_path):
    p, _ = _window(tmp_path / "win", 0.0123, frame=1)
    with np.load(p) as z:
        x, meta = z["x"], json.loads(str(z["meta"]))
    meta["nominal_utc"] = meta["nominal_utc"] - 0.2     # burst now 0.2 s after nominal
    np.savez(p, x=x, meta=np.str_(json.dumps(meta)))
    fac, _ = _run(tmp_path)
    r = [r for r in fac.writer.inserts if r["frame_index"] == 1][0]
    assert r["found"] and r["delay_s"] > 0.1
    assert r["excluded"] == "implausible_delay"


def test_both_exclusions_join(tmp_path):
    p, _ = _window(tmp_path / "win", 0.0123, utc_sync=False)
    with np.load(p) as z:
        x, meta = z["x"], json.loads(str(z["meta"]))
    meta["nominal_utc"] -= 0.2
    np.savez(p, x=x, meta=np.str_(json.dumps(meta)))
    fac, _ = _run(tmp_path)
    assert fac.writer.inserts[0]["excluded"] == "utc_sync_false,implausible_delay"


def test_second_run_skips_done(tmp_path):
    _window(tmp_path / "win", 0.0123)
    fac, _ = _run(tmp_path)
    assert len(fac.writer.inserts) == 1
    assert list((tmp_path / "win").rglob("*.done"))
    fac2, counts = _run(tmp_path)
    assert fac2.writer.inserts == [] and counts["analysed"] == 0
    assert counts["skipped_done"] == 1


def test_writer_is_local(tmp_path):
    _window(tmp_path / "win", 0.0123)
    fac, _ = _run(tmp_path)
    assert fac.calls and fac.calls[0]["local"] is True
    assert fac.calls[0]["table"] == "squitter_timing" and fac.calls[0]["mode"] == "hfdl"


def test_bad_window_gets_batch_error_row_and_done(tmp_path):
    _window(tmp_path / "win", 0.0123, frame=0)
    bad = tmp_path / "win" / "radiod0" / "20261102" / "7-8912-9.npz"
    bad.write_bytes(b"not an npz")
    fac, counts = _run(tmp_path)
    by = {r["frame_index"]: r for r in fac.writer.inserts}
    assert by[0]["found"]
    bad_rows = [r for r in fac.writer.inserts if r["reason"].startswith("batch_error")]
    assert len(bad_rows) == 1 and bad_rows[0]["found"] is False
    assert bad.with_name(bad.name + ".done").exists()
    assert counts["errors"] == 1


def test_tmp_files_skipped_and_since_filter(tmp_path):
    p, _ = _window(tmp_path / "win", 0.0123)
    (p.parent / "7-8912-5.npz.tmp").write_bytes(b"partial")
    fac, _ = _run(tmp_path, since=NOMINAL + 10)
    assert fac.writer.inserts == []
    fac, _ = _run(tmp_path, since=NOMINAL - 10)
    assert len(fac.writer.inserts) == 1


def test_malformed_registration_lines_tolerated(tmp_path):
    _, idx0 = _window(tmp_path / "win", 0.0123)
    reg = tmp_path / "reg"
    reg.mkdir()
    (reg / f"minutes-{REG_DAY}.jsonl").write_text('{broken\n\n{"t3_shadow": null}\n[1]\n')
    _registration(reg, idx0)
    fac, _ = _run(tmp_path)
    assert fac.writer.inserts[0]["delay_counter_s"] is not None


def test_missing_registration_dir_ok(tmp_path):
    _window(tmp_path / "win", 0.0123)
    fac = Factory()
    batch.run_batch(tmp_path / "win", registration_dir=tmp_path / "nope",
                    writer_factory=fac)
    assert fac.writer.inserts[0]["delay_counter_s"] is None


def test_max_windows_caps_a_run(tmp_path):
    for i in range(3):
        _window(tmp_path / "win", 0.0123, frame=i)
    fac, counts = _run(tmp_path, max_windows=2)
    assert len(fac.writer.inserts) == 2 and counts["deferred"] == 1
    fac2, _ = _run(tmp_path, max_windows=2)
    assert len(fac2.writer.inserts) == 1


def test_csv_output(tmp_path):
    _window(tmp_path / "win", 0.0123)
    out = tmp_path / "o.csv"
    _run(tmp_path, csv_path=out)
    lines = out.read_text().splitlines()
    assert len(lines) == 2 and "delay_s" in lines[0].split(",")
