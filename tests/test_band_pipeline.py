"""Tests for BandPipeline.build_argv (golden-string).

We only validate the argv that BandPipeline would pass to dumphfdl.
The supervisor loop, subprocess lifecycle, and ka9q callback path are
tested separately (and integration-tested with a live radiod).
"""

from __future__ import annotations

from hfdl_recorder.bands import HFDL_BANDS
from hfdl_recorder.core.band_pipeline import BandPipeline, _f32_iq_to_cf32

import numpy as np


def _make_config(**overrides) -> dict:
    cfg = {
        "station": {"station_id": "TEST-1"},
        "paths": {
            "dumphfdl":  "/opt/git/sigmond/hfdl-recorder/bin/dumphfdl",
            "spool_dir": "/var/lib/hfdl-recorder",
            "log_dir":   "/var/log/hfdl-recorder",
            "systable":  "/var/lib/hfdl-recorder/systable.conf",
        },
        "sinks": {"local_json": True, "airframes_io": False},
    }
    for key, val in overrides.items():
        cfg[key] = val
    return cfg


def test_build_argv_hfdl21_default():
    pipeline = BandPipeline(
        band=HFDL_BANDS["HFDL21"],
        radiod_id="rx888",
        config=_make_config(),
    )
    argv = pipeline.build_argv()

    assert argv[0] == "/opt/git/sigmond/hfdl-recorder/bin/dumphfdl"
    assert "--iq-file" in argv and "-" in argv
    # Sample format / rate.
    assert argv[argv.index("--sample-format") + 1] == "cf32"
    assert argv[argv.index("--sample-rate") + 1] == "80000"
    # Center freq in kHz; whole-kHz band → no decimal.
    assert argv[argv.index("--centerfreq") + 1] == "21964"
    # Station id passed through.
    assert argv[argv.index("--station-id") + 1] == "TEST-1"
    # Systable read+save uses same path.
    sys_idx = argv.index("--system-table")
    assert argv[sys_idx + 1] == "/var/lib/hfdl-recorder/systable.conf"
    save_idx = argv.index("--system-table-save")
    assert argv[save_idx + 1] == "/var/lib/hfdl-recorder/systable.conf"
    # JSON sink.
    assert "--output" in argv
    assert any(
        "decoded:json:file:path=/var/lib/hfdl-recorder/rx888/HFDL21.json" in a
        for a in argv
    )
    # Channel kHz are positional at the end (last 9 entries for HFDL21).
    expected = [str(int(k)) for k in HFDL_BANDS["HFDL21"].channels_khz]
    assert argv[-len(expected):] == expected


def test_build_argv_hfdl10_half_khz_center():
    pipeline = BandPipeline(
        band=HFDL_BANDS["HFDL10"],
        radiod_id="rx888",
        config=_make_config(),
    )
    argv = pipeline.build_argv()
    # HFDL10 center is 10061.5 kHz — the only half-kHz center in our table
    # along with HFDL8 (8902.5).
    assert argv[argv.index("--centerfreq") + 1] == "10061.5"


def test_build_argv_airframes_appends_second_output():
    cfg = _make_config()
    cfg["sinks"]["airframes_io"] = True
    pipeline = BandPipeline(
        band=HFDL_BANDS["HFDL13"],
        radiod_id="rx888",
        config=cfg,
    )
    argv = pipeline.build_argv()
    output_specs = [argv[i + 1] for i, a in enumerate(argv) if a == "--output"]
    assert any("decoded:json:file:path=" in s for s in output_specs)
    assert any("feed.airframes.io" in s for s in output_specs)


def test_build_argv_no_station_id_omitted():
    cfg = _make_config()
    cfg["station"]["station_id"] = ""
    pipeline = BandPipeline(
        band=HFDL_BANDS["HFDL5"],
        radiod_id="rx888",
        config=cfg,
    )
    argv = pipeline.build_argv()
    assert "--station-id" not in argv


def test_f32_iq_to_cf32_complex64():
    # Two IQ samples: (0.5, -0.25), (1.0, -1.0).
    samples = np.array([0.5 - 0.25j, 1.0 - 1.0j], dtype=np.complex64)
    blob = _f32_iq_to_cf32(samples)
    # 4 float32 values × 4 bytes = 16 bytes.
    assert len(blob) == 16
    out = np.frombuffer(blob, dtype=np.float32)
    assert out[0] == np.float32(0.5)
    assert out[1] == np.float32(-0.25)
    assert out[2] == np.float32(1.0)
    assert out[3] == np.float32(-1.0)


def test_f32_iq_to_cf32_passes_through_large_values():
    # CF32 has no clipping — values outside [-1, 1] pass through.
    samples = np.array([2.0 + 0.0j, -2.0 + 0.0j], dtype=np.complex64)
    out = np.frombuffer(_f32_iq_to_cf32(samples), dtype=np.float32)
    assert out[0] == np.float32(2.0)
    assert out[2] == np.float32(-2.0)


def test_f32_iq_to_cf32_zeros_nan_and_inf():
    samples = np.array(
        [complex(np.nan, 1.0), complex(1.0, np.inf), complex(-np.inf, 0.5)],
        dtype=np.complex64,
    )
    out = np.frombuffer(_f32_iq_to_cf32(samples), dtype=np.float32)
    # NaN/Inf zeroed; finite values preserved.
    assert out[0] == np.float32(0.0)   # was NaN
    assert out[1] == np.float32(1.0)   # finite
    assert out[2] == np.float32(1.0)   # finite
    assert out[3] == np.float32(0.0)   # was +inf
    assert out[4] == np.float32(0.0)   # was -inf
    assert out[5] == np.float32(0.5)   # finite


# -- window keeper beside the dumphfdl write (Task 5) --

import io
from types import SimpleNamespace


class _FakeProc:
    def __init__(self):
        self.stdin = io.BytesIO()

    def poll(self):
        return None


class _RecordingKeeper:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def feed(self, samples, rtp_start, gap_samples=0):
        self.calls.append((np.array(samples, copy=True), rtp_start))
        self.gaps = getattr(self, "gaps", []) + [gap_samples]
        if self.fail:
            raise RuntimeError("keeper broke")


def _drive(keeper) -> bytes:
    pipeline = BandPipeline(band=HFDL_BANDS["HFDL13"], radiod_id="rx888",
                            config=_make_config())
    pipeline._first_sample_logged = True
    pipeline._proc = _FakeProc()
    if keeper is not None:
        pipeline.set_keeper(keeper)
    rng = np.random.default_rng(1)
    rtp = 4_294_967_000          # wraps during the run
    for _ in range(50):
        x = (rng.standard_normal(2000) + 1j * rng.standard_normal(2000)).astype(np.complex64)
        x[7] = complex(np.nan, 1.0)
        pipeline._on_samples(x, SimpleNamespace(delivered_rtp_start=rtp))
        rtp = (rtp + 2000) & 0xFFFFFFFF
    return pipeline._proc.stdin.getvalue()


def test_dumphfdl_bytes_unchanged_with_keeper(tmp_path):
    from hfdl_recorder.timing.window_keeper import WindowKeeper

    off = _drive(None)
    rec = _RecordingKeeper()
    assert _drive(rec) == off
    assert _drive(_RecordingKeeper(fail=True)) == off
    real = WindowKeeper(band=HFDL_BANDS["HFDL13"], fs=100_000, out_dir=tmp_path,
                        ground_stations=["Riverhead, New York"],
                        anchor_fn=lambda rtp: None, counter_map_fn=lambda: None)
    real.start()
    try:
        assert _drive(real) == off
    finally:
        real.stop()
    # The keeper saw every block, with its RTP, after dumphfdl's copy.
    assert len(rec.calls) == 50
    assert rec.calls[0][1] == 4_294_967_000
    assert rec.calls[1][1] == (4_294_967_000 + 2000) & 0xFFFFFFFF


def test_keeper_skips_block_without_rtp():
    rec = _RecordingKeeper()
    pipeline = BandPipeline(band=HFDL_BANDS["HFDL13"], radiod_id="rx888",
                            config=_make_config())
    pipeline._first_sample_logged = True
    pipeline._proc = _FakeProc()
    pipeline.set_keeper(rec)
    x = np.zeros(10, dtype=np.complex64)
    pipeline._on_samples(x, SimpleNamespace(delivered_rtp_start=None))
    pipeline._on_samples(x, None)
    assert rec.calls == []
    assert len(pipeline._proc.stdin.getvalue()) == 2 * 10 * 8


class _FakeMulti:
    def __init__(self, ssrc):
        self._slots = {ssrc: object()}

    def add_channel(self, **kw):
        return SimpleNamespace(ssrc=next(iter(self._slots)))


def _attached(ssrc=0x10):
    pipeline = BandPipeline(band=HFDL_BANDS["HFDL13"], radiod_id="rx888",
                            config=_make_config())
    pipeline._first_sample_logged = True
    multi = _FakeMulti(ssrc)
    pipeline.attach(multi)
    return pipeline, multi


def test_may_poll_only_the_live_current_ssrc():
    pipeline, multi = _attached(0x10)
    assert not pipeline.may_poll(0x10)                 # no sample yet
    pipeline._on_samples(np.zeros(4, np.complex64), None)
    assert pipeline.may_poll(0x10)
    assert not pipeline.may_poll(0x11)
    # The stream drops: no polls while it is down.
    pipeline._on_stream_dropped("silence")
    assert not pipeline.may_poll(0x10)
    # MultiStream re-keys the slot to a new SSRC on restore.
    multi._slots = {0x20: object()}
    pipeline._on_stream_restored(SimpleNamespace(ssrc=0x20))
    pipeline._on_samples(np.zeros(4, np.complex64), None)
    assert pipeline.current_ssrc == 0x20
    assert not pipeline.may_poll(0x10)
    assert pipeline.may_poll(0x20)
    # A slot that left MultiStream is never polled.
    multi._slots = {}
    assert not pipeline.may_poll(0x20)


def test_may_poll_needs_a_sample_in_the_last_half_second(monkeypatch):
    import hfdl_recorder.core.band_pipeline as bp
    now = [1000.0]
    monkeypatch.setattr(bp.time, "monotonic", lambda: now[0])
    pipeline, _ = _attached(0x10)
    pipeline._on_samples(np.zeros(4, np.complex64), None)
    now[0] += 0.4
    assert pipeline.may_poll(0x10)
    now[0] += 0.2
    assert not pipeline.may_poll(0x10)


def test_keeper_errors_logged_once_per_streak(caplog):
    import logging
    caplog.set_level(logging.ERROR, logger="hfdl_recorder.core.band_pipeline")
    rec = _RecordingKeeper(fail=True)
    pipeline = BandPipeline(band=HFDL_BANDS["HFDL13"], radiod_id="rx888",
                            config=_make_config())
    pipeline._first_sample_logged = True
    pipeline.set_keeper(rec)
    q = SimpleNamespace(delivered_rtp_start=1, batch_gaps=[])
    x = np.zeros(4, np.complex64)
    for fail in (True, True, False, True, True):
        rec.fail = fail
        pipeline._on_samples(x, q)
    assert sum("window keeper failed" in r.getMessage() for r in caplog.records) == 2


def test_gap_samples_reach_keeper():
    rec = _RecordingKeeper()
    pipeline = BandPipeline(band=HFDL_BANDS["HFDL13"], radiod_id="rx888",
                            config=_make_config())
    pipeline._first_sample_logged = True
    pipeline.set_keeper(rec)
    gaps = [SimpleNamespace(duration_samples=100), SimpleNamespace(duration_samples=20)]
    pipeline._on_samples(np.zeros(4, np.complex64),
                         SimpleNamespace(delivered_rtp_start=1, batch_gaps=gaps))
    pipeline._on_samples(np.zeros(4, np.complex64),
                         SimpleNamespace(delivered_rtp_start=5, batch_gaps=[]))
    assert rec.gaps == [120, 0]


def test_may_poll_fails_closed_without_slots():
    class _NoSlots:
        def add_channel(self, **kw):
            return SimpleNamespace(ssrc=0x10)

    pipeline = BandPipeline(band=HFDL_BANDS["HFDL13"], radiod_id="rx888",
                            config=_make_config())
    pipeline._first_sample_logged = True
    pipeline.attach(_NoSlots())
    pipeline._on_samples(np.zeros(4, np.complex64), None)
    assert not pipeline.may_poll(0x10)


def test_channel_frequency_follows_channel_info():
    pipeline, multi = _attached(0x10)
    assert pipeline.channel_freq_hz == HFDL_BANDS["HFDL13"].center_hz
    multi._slots = {0x20: object()}
    pipeline._on_stream_restored(SimpleNamespace(ssrc=0x20, frequency=13_310_002.0))
    assert pipeline.channel_freq_hz == 13_310_002.0
    pipeline._on_stream_restored(SimpleNamespace(ssrc=0x20, frequency=0.0))
    assert pipeline.channel_freq_hz == 13_310_002.0     # 0 means unknown
