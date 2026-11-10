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
    """Buffers until flush, like the real sink writer."""

    def __init__(self, noop=False, fail_flush=False, fail_insert=False):
        self.inserts = []           # durable rows
        self._buf = []
        self.closed = False
        self.is_noop = noop
        self.fail_flush = fail_flush
        self.fail_insert = fail_insert

    @property
    def buffered(self):
        return len(self._buf)

    def insert(self, rows):
        if self.fail_insert:
            raise RuntimeError("BufferFull")
        if not self.is_noop:
            self._buf.extend(rows)

    def flush(self):
        if self.fail_flush or self.is_noop:
            return
        self.inserts.extend(self._buf)
        self._buf = []

    def close(self):
        self.closed = True


class Factory:
    """One FakeWriter per table; ``kw`` configures the squitter_timing one.
    Pre-set ``writers[<table>]`` to configure another."""

    def __init__(self, **kw):
        self.writer = FakeWriter(**kw)
        self.writers = {"squitter_timing": self.writer}
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return self.writers.setdefault(kw["table"], FakeWriter())


def _window(root, delay, *, frame=0, nominal=NOMINAL, idx_offset=0.0,
            status_late_s=0.0, utc_sync=True, epoch_gps=EPOCH_GPS,
            with_index=True, x=None, name=None, freq_khz=8912, gs_id=7):
    """Write one window file; returns its path and the counter label."""
    t0_counter = nominal - 0.03
    idx0 = round((t0_counter - (epoch_gps + GPS_TO_UTC)) * F_IN) + idx_offset
    t0 = (epoch_gps + GPS_TO_UTC) + idx0 / F_IN + status_late_s
    if x is None:
        x = w.synth_squitter(FS, delay, snr_db=20, rng=np.random.default_rng(frame + 7))
    meta = {
        "t0_utc": t0, "nominal_utc": nominal, "fs": FS, "freq_khz": freq_khz,
        "gs_id": gs_id, "gs_name": "Test", "frame_index": frame, "frame_offset": 2,
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
    p = d / (name or f"{gs_id}-{freq_khz}-{frame}.npz")
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


def _run(tmp_path, fac=None, **kw):
    fac = fac or Factory()
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
    exp_idx = idx0 + r["arrival_offset_s"] * F_IN
    assert abs(r["arrival_input_index"] - exp_idx) < 1e-3
    # the epoch-difference route agrees only to its ~0.24 us grain
    assert abs(r["arrival_input_index"]
               - (idx0 + (r["arrival_utc"] - r["t0_utc"]) * F_IN)) < 20.0


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


def test_vanished_windows_count_apart_and_log_once(tmp_path, monkeypatch, caplog):
    """The keeper prunes a window between listing and reading: expected life."""
    for i in range(5):
        _window(tmp_path / "win", 0.0123, frame=i)
    real = batch._row_for

    def pruned(path, regs):
        if path.name.split("-")[2].startswith(("1.", "3.")):   # frames 1 and 3
            path.unlink()
        return real(path, regs)

    monkeypatch.setattr(batch, "_row_for", pruned)
    with caplog.at_level("INFO", logger="hfdl_recorder.timing.batch"):
        fac, counts = _run(tmp_path)
    assert counts["vanished"] == 2 and counts["errors"] == 0
    assert counts["analysed"] == 3 and len(fac.writer.inserts) == 3
    done = [d.name for d in (tmp_path / "win").glob("*/*/*.done")]
    assert len(done) == 3
    mentions = [r for r in caplog.records if "vanished before" in r.getMessage() or "FileNotFound" in r.getMessage()]
    assert len(mentions) == 1 and mentions[0].levelname == "INFO"
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_corrupt_window_still_an_error_with_warning(tmp_path, caplog):
    _window(tmp_path / "win", 0.0123, frame=0)
    bad = tmp_path / "win" / "radiod0" / "20261102" / "7-8912-9.npz"
    bad.write_bytes(b"not an npz")
    with caplog.at_level("INFO", logger="hfdl_recorder.timing.batch"):
        _, counts = _run(tmp_path)
    assert counts["errors"] == 1 and counts["vanished"] == 0
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


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


def _dones(tmp_path):
    return list((tmp_path / "win").rglob("*.done"))


def test_failed_flush_leaves_no_markers_and_reruns(tmp_path):
    for i in range(3):
        _window(tmp_path / "win", 0.0123, frame=i)
    fac, counts = _run(tmp_path, Factory(fail_flush=True))
    assert fac.writer.inserts == [] and _dones(tmp_path) == []
    assert counts["stopped"]
    fac2, _ = _run(tmp_path)
    assert len(fac2.writer.inserts) == 3 and len(_dones(tmp_path)) == 3


def test_insert_failure_stops_without_marking(tmp_path):
    _window(tmp_path / "win", 0.0123)
    fac, counts = _run(tmp_path, Factory(fail_insert=True))
    assert _dones(tmp_path) == [] and counts["stopped"]


def test_noop_writer_marks_nothing(tmp_path):
    _window(tmp_path / "win", 0.0123)
    fac, counts = _run(tmp_path, Factory(noop=True))
    assert _dones(tmp_path) == [] and counts["analysed"] == 0 and counts["stopped"]


def test_periodic_commit_marks_only_durable_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "FLUSH_EVERY", 2)
    for i in range(3):
        _window(tmp_path / "win", 0.0123, frame=i)
    fac = Factory()
    real = fac.writer.flush
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) >= 2:          # the final commit fails
            return
        real()
    fac.writer.flush = flaky
    _run(tmp_path, fac)
    assert len(fac.writer.inserts) == 2 and len(_dones(tmp_path)) == 2


def test_since_skips_before_analysis(tmp_path, monkeypatch):
    _window(tmp_path / "win", 0.0123)
    monkeypatch.setattr(batch, "time_squitter",
                        lambda *a, **k: pytest.fail("analysed a pre-since window"))
    fac, counts = _run(tmp_path, since=NOMINAL + 10)
    assert counts["skipped_since"] == 1 and fac.writer.inserts == []


def test_registration_skew_up_to_2s_still_matches(tmp_path):
    _, idx0 = _window(tmp_path / "win", 0.0123)
    _registration(tmp_path / "reg", idx0, epoch_gps=EPOCH_GPS + 0.8)
    fac, _ = _run(tmp_path)
    assert fac.writer.inserts[0]["delay_counter_s"] is not None


# ---------------------------------------------------------------- final review

def test_no_sink_mode_runs_without_sigmond(tmp_path, monkeypatch):
    """Off the station: no sigmond installed, rows land in the files."""
    import sys
    monkeypatch.setitem(sys.modules, "sigmond", None)
    monkeypatch.setitem(sys.modules, "sigmond.hamsci_sink", None)
    for i in range(2):
        _window(tmp_path / "win", 0.0123, frame=i)
    js, cs = tmp_path / "out" / "squitter.jsonl", tmp_path / "out" / "squitter.csv"
    js.parent.mkdir()
    counts = batch.run_batch(tmp_path / "win", registration_dir=tmp_path / "reg",
                             sink=False, jsonl_path=js, csv_path=cs)
    assert counts["analysed"] == 2 and not counts["stopped"]
    rows = [json.loads(l) for l in js.read_text().splitlines()]
    assert len(rows) == 2 and all(r["found"] for r in rows)
    assert len(cs.read_text().splitlines()) == 3
    assert len(_dones(tmp_path)) == 2
    # a rerun adds nothing and keeps what the jsonl holds
    counts = batch.run_batch(tmp_path / "win", registration_dir=tmp_path / "reg",
                             sink=False, jsonl_path=js)
    assert counts["analysed"] == 0 and len(js.read_text().splitlines()) == 2


def test_no_sink_needs_an_output_file(tmp_path):
    _window(tmp_path / "win", 0.0123)
    with pytest.raises(ValueError):
        batch.run_batch(tmp_path / "win", registration_dir=tmp_path / "reg", sink=False)


def test_no_sink_unwritable_output_marks_nothing(tmp_path):
    _window(tmp_path / "win", 0.0123)
    counts = batch.run_batch(tmp_path / "win", registration_dir=tmp_path / "reg",
                             sink=False, jsonl_path=tmp_path / "nodir" / "x.jsonl")
    assert counts["stopped"] and _dones(tmp_path) == []


def test_rows_carry_analyzer_version_and_schema(tmp_path):
    from hfdl_recorder.timing.analyzer import ANALYZER_VERSION
    assert ANALYZER_VERSION >= 2
    _window(tmp_path / "win", 0.0123)
    fac, _ = _run(tmp_path)
    assert fac.writer.inserts[0]["analyzer_version"] == ANALYZER_VERSION
    assert fac.calls[0]["schema_version"] == batch.SCHEMA_VERSION >= 2


def test_done_from_an_older_analyser_reanalyses(tmp_path, monkeypatch):
    from hfdl_recorder.timing.analyzer import ANALYZER_VERSION
    p, _ = _window(tmp_path / "win", 0.0123)
    done = p.with_name(p.name + ".done")
    done.touch()                         # b72dc39 wrote empty markers: version 1
    fac, counts = _run(tmp_path)
    assert counts["analysed"] == 1 and len(fac.writer.inserts) == 1
    assert int(done.read_text()) == ANALYZER_VERSION
    fac, counts = _run(tmp_path)
    assert counts["analysed"] == 0 and counts["skipped_done"] == 1
    done.write_text(str(ANALYZER_VERSION + 1))     # a newer analyser's marker
    fac, counts = _run(tmp_path)
    assert counts["analysed"] == 0
    monkeypatch.setattr(batch, "ANALYZER_VERSION", ANALYZER_VERSION + 2)
    fac, counts = _run(tmp_path)
    assert counts["analysed"] == 1


def test_arrival_input_index_from_in_window_offset(tmp_path):
    """No subtraction of two epochs: the index comes from the in-window
    offset, exact to well under a sample at 64.8 MS/s."""
    from hfdl_recorder.timing.analyzer import time_squitter
    p, idx0 = _window(tmp_path / "win", 0.0123)
    fac, _ = _run(tmp_path)
    r = fac.writer.inserts[0]
    with np.load(p) as z:
        meta = json.loads(str(z["meta"]))
        arr = time_squitter(z["x"], FS, meta["t0_utc"], meta["nominal_utc"])
    assert abs(arr.arrival_utc - (meta["t0_utc"] + arr.arrival_offset_s)) < 1e-6
    assert r["arrival_offset_s"] == pytest.approx(arr.arrival_offset_s, abs=0)
    assert r["arrival_input_index"] - idx0 == pytest.approx(
        arr.arrival_offset_s * F_IN, abs=1e-3)


def test_window_waits_for_its_registration(tmp_path):
    """hf-timestd writes the minute row after the window: a window analysed
    first must stay unmarked and retry, but not forever."""
    _, idx0 = _window(tmp_path / "win", 0.0123)
    _registration(tmp_path / "reg", idx0, ahead_s=-300.0)    # newest row: too old
    fac, counts = _run(tmp_path, now=NOMINAL + 60)
    assert fac.writer.inserts == [] and _dones(tmp_path) == []
    assert counts["awaiting_registration"] == 1 and not counts["stopped"]
    _registration(tmp_path / "reg", idx0, ahead_s=+30.0)     # it arrives
    fac, counts = _run(tmp_path, now=NOMINAL + 300)
    assert fac.writer.inserts[0]["delay_counter_s"] is not None
    assert len(_dones(tmp_path)) == 1


def test_registration_absent_for_hours_gives_up(tmp_path):
    _window(tmp_path / "win", 0.0123)
    fac, counts = _run(tmp_path, now=NOMINAL + batch.REG_WAIT_S + 1)
    assert fac.writer.inserts[0]["delay_counter_s"] is None
    assert len(_dones(tmp_path)) == 1


def test_registration_past_the_window_but_unusable_marks_done(tmp_path):
    _, idx0 = _window(tmp_path / "win", 0.0123)
    _registration(tmp_path / "reg", idx0, ahead_s=+300.0)    # newer, too far
    fac, counts = _run(tmp_path, now=NOMINAL + 400)
    assert fac.writer.inserts[0]["delay_counter_s"] is None
    assert len(_dones(tmp_path)) == 1


def test_cli_exits_nonzero_when_the_run_stopped(tmp_path, monkeypatch):
    import sys
    from hfdl_recorder import cli
    from hfdl_recorder.timing import batch as b
    seen = {}

    def fake(window_dir, **kw):
        seen["dir"] = window_dir
        return {"stopped": True}
    monkeypatch.setattr(b, "run_batch", fake)
    monkeypatch.setattr(sys, "argv", ["hfdl-recorder", "squitter-analyze",
                                      "--dir", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1


def test_cli_dir_defaults_from_config(tmp_path, monkeypatch):
    import sys
    from hfdl_recorder import cli
    from hfdl_recorder.timing import batch as b
    cfg = tmp_path / "c.toml"
    cfg.write_text('[timing_squitters]\nwindow_dir = "%s"\n\n[[radiod]]\nid = "r"\n'
                   'status = "r.local"\n' % (tmp_path / "w"))
    seen = {}

    def fake(window_dir, **kw):
        seen["dir"] = window_dir
        return {"stopped": False}
    monkeypatch.setattr(b, "run_batch", fake)
    monkeypatch.setattr(sys, "argv", ["hfdl-recorder", "squitter-analyze",
                                      "--config", str(cfg)])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0
    assert str(seen["dir"]) == str(tmp_path / "w")


# ------------------------------------------------------- sigmondb rows (Task 7)

def _as_t6(p):
    """Rewrite a window's meta as a T6 window with a counter map."""
    with np.load(p) as z:
        x, meta = z["x"], json.loads(str(z["meta"]))
    meta["t_level_active"] = "T6"
    meta["counter_map"] = {"input_epoch_gps_s": EPOCH_GPS}
    np.savez(p, x=x, meta=np.str_(json.dumps(meta)))


def _run_calls(tmp_path):
    _window(tmp_path / "win", 0.0123)
    fac, _ = _run(tmp_path)
    return fac.calls


def test_found_t6_window_writes_an_assumed_arrival(tmp_path):
    fac = Factory()
    p, idx0 = _window(tmp_path / "win", 0.012)
    _as_t6(p)
    _registration(tmp_path / "reg", idx0)
    _, counts = _run(tmp_path, fac, reporter_id="AC0G/B4")
    (a,) = fac.writers["arrival"].inserts
    assert a["reporter_id"] == "AC0G/B4" and a["engineering"]["delay_basis"] == "counter"
    assert a["tx_id"] == "hfdl:7" and a["a_level_provenance"] == "assumed"
    assert a["counter_epoch_id"] == "ep-1477000000"
    assert counts["arrivals"] == 1 and counts["captures"] == 1
    (c,) = fac.writers["capture"].inserts
    assert c["reporter_id"] == "AC0G/B4" and c["n_expected"] == 1 and c["n_found"] == 1


def test_no_reporter_id_gives_an_empty_one(tmp_path):
    fac = Factory()
    _window(tmp_path / "win", 0.012)
    _run(tmp_path, fac)
    assert fac.writers["arrival"].inserts[0]["reporter_id"] == ""
    assert fac.writers["capture"].inserts[0]["reporter_id"] == ""


def test_sigmondb_writers_are_not_local_and_name_the_producer(tmp_path):
    calls = _run_calls(tmp_path)
    kw = {c["table"]: c for c in calls}
    assert calls[0]["table"] == "squitter_timing" and calls[0]["local"] is True
    for t in ("arrival", "capture"):
        assert (kw[t]["mode"], kw[t]["schema_version"], kw[t]["producer"]) == (
            "sigmondb", 1, "hfdl-recorder")
        assert not kw[t].get("local")


def test_one_capture_per_frequency_per_run(tmp_path):
    _window(tmp_path / "win", 0.0123, frame=0, freq_khz=11387)
    _window(tmp_path / "win", 0.0123, frame=0, freq_khz=13276, gs_id=3)
    fac1, _ = _run(tmp_path)
    _window(tmp_path / "win", 0.0123, frame=1, freq_khz=11387, nominal=NOMINAL + 64)
    fac2, _ = _run(tmp_path)
    caps = fac1.writers["capture"].inserts + fac2.writers["capture"].inserts
    assert len(fac1.writers["capture"].inserts) == 2 and len(caps) == 3
    assert len({(c["centre_hz"], c["time"]) for c in caps}) == 3
    assert all(c["n_expected"] == 1 for c in caps)


def test_failed_arrival_flush_marks_nothing(tmp_path):
    for i in range(2):
        _window(tmp_path / "win", 0.0123, frame=i)
    fac = Factory()
    fac.writers["arrival"] = FakeWriter(fail_flush=True)
    _, counts = _run(tmp_path, fac)
    assert _dones(tmp_path) == [] and counts["stopped"]
    assert fac.writers["capture"].inserts == []


def test_stopped_run_writes_captures_for_committed_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "FLUSH_EVERY", 1)
    for i in range(4):
        _window(tmp_path / "win", 0.0123, frame=i)
    fac = Factory()
    real, n = fac.writer.insert, []

    def third_fails(rows):
        n.append(1)
        if len(n) == 3:
            raise RuntimeError("BufferFull")
        real(rows)
    fac.writer.insert = third_fails
    _, counts = _run(tmp_path, fac)
    assert counts["stopped"] and len(_dones(tmp_path)) == 2
    assert len(fac.writers["arrival"].inserts) == 2
    (c,) = fac.writers["capture"].inserts
    assert c["n_expected"] == 2 and counts["captures"] == 1


def test_no_sink_mode_writes_no_sigmondb(tmp_path):
    _window(tmp_path / "win", 0.0123)
    fac = Factory()
    counts = batch.run_batch(tmp_path / "win", registration_dir=tmp_path / "reg",
                             sink=False, jsonl_path=tmp_path / "o.jsonl",
                             writer_factory=fac)
    assert not any(c.get("table") == "arrival" for c in fac.calls)
    assert counts["arrivals"] == 0 and counts["captures"] == 0


def test_a_bad_arrival_costs_only_its_row_and_logs_once(tmp_path, monkeypatch, caplog):
    for i in range(3):
        _window(tmp_path / "win", 0.0123, frame=i)
    real = batch._row_for

    def no_snr(path, regs):
        row, missing = real(path, regs)
        if not path.name.endswith("-2.npz"):
            row["snr_db"] = None          # NaN, cleaned: arrival_for raises
        return row, missing
    monkeypatch.setattr(batch, "_row_for", no_snr)
    fac = Factory()
    with caplog.at_level("INFO", logger="hfdl_recorder.timing.batch"):
        _, counts = _run(tmp_path, fac)
    assert not counts["stopped"] and len(_dones(tmp_path)) == 3
    assert len(fac.writer.inserts) == 3 and len(fac.writers["arrival"].inserts) == 1
    (c,) = fac.writers["capture"].inserts
    assert c["n_expected"] == 3 and c["n_found"] == 1
    assert len([r for r in caplog.records if "no arrival" in r.getMessage()]) == 1


def test_cli_passes_the_instance_reporter_id(tmp_path, monkeypatch):
    import sys
    from hfdl_recorder import cli
    from hfdl_recorder.timing import batch as b
    cfg = tmp_path / "c.toml"
    cfg.write_text('[instance]\nreporter_id = "AC0G/B4"\n\n[timing_squitters]\n'
                   'window_dir = "%s"\n' % (tmp_path / "w"))
    seen = {}

    def fake(window_dir, **kw):
        seen.update(kw)
        return {"stopped": False}
    monkeypatch.setattr(b, "run_batch", fake)
    monkeypatch.setattr(sys, "argv", ["hfdl-recorder", "squitter-analyze",
                                      "--config", str(cfg)])
    with pytest.raises(SystemExit):
        cli.main()
    assert seen["reporter_id"] == "AC0G/B4"
