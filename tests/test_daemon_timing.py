"""The daemon builds one window keeper per band only when asked (Task 5)."""

from __future__ import annotations

from types import SimpleNamespace

from hfdl_recorder.bands import HFDL_BANDS
from hfdl_recorder.core.band_pipeline import BandPipeline
from hfdl_recorder.core.daemon import HfdlRecorder


def _config(tmp_path, enabled):
    return {
        "station": {"station_id": "T"},
        "paths": {"spool_dir": str(tmp_path), "log_dir": str(tmp_path)},
        "sinks": {},
        "processing": {"radiod_lifetime_frames": 0},
        "timing_squitters": {
            "enabled": enabled,
            "ground_stations": ["Riverhead, New York"],
            "window_dir": str(tmp_path / "squitters"),
            "retention_days": 5, "min_free_gb": 0,
        },
    }


class _Control:
    def __init__(self):
        self.ssrcs = []
        self.listened = []

    def poll_status(self, ssrc, timeout):
        self.ssrcs.append(ssrc)
        return SimpleNamespace(ssrc=ssrc, gps_time=None, rtp_timesnap=None)

    def listen_status(self, callback, duration=None, ssrcs=None):
        self.listened.append(frozenset(ssrcs))


class _FakeMulti:
    def __init__(self, ssrc):
        self._slots = {ssrc: object()}

    def add_channel(self, **kw):
        return SimpleNamespace(ssrc=next(iter(self._slots)))


def _built(tmp_path, ssrc):
    rec = HfdlRecorder(_config(tmp_path, True), {"status": "x.local"})
    rec._control = _Control()
    pipeline = BandPipeline(band=HFDL_BANDS["HFDL13"], radiod_id="x.local",
                            config=rec._config)
    pipeline._first_sample_logged = True
    multi = _FakeMulti(ssrc)
    pipeline.attach(multi)
    rec._pipelines.append(pipeline)
    rec._add_keeper(pipeline, ssrc)
    return rec, pipeline, multi


def test_keeper_built_per_band_and_polls_only_its_ssrc(tmp_path):
    import numpy as np
    rec, pipeline, _ = _built(tmp_path, 0xABCD)
    keeper, pl, _counter = rec._keepers[0]
    assert pipeline._keeper is keeper and pl is pipeline
    assert keeper.out_dir == tmp_path / "squitters" / "x.local"
    pipeline._on_samples(np.zeros(4, np.complex64), None)
    # The anchor polls the band's own SSRC and nothing else, and with no
    # status pair it gives no anchor (no host-clock fallback).
    assert keeper._anchor_fn(123) is None
    assert set(rec._control.ssrcs) == {0xABCD}
    assert rec._control.listened[0] == {0xABCD}


def test_rekeyed_slot_gets_no_polls_to_old_ssrc(tmp_path):
    import numpy as np
    rec, pipeline, multi = _built(tmp_path, 0x10)
    keeper = rec._keepers[0][0]
    multi._slots = {0x20: object()}
    pipeline._on_stream_restored(SimpleNamespace(ssrc=0x20))
    pipeline._on_samples(np.zeros(4, np.complex64), None)
    keeper._anchor_fn(1)
    rec._refresh_counter_maps()
    assert rec._control.ssrcs and set(rec._control.ssrcs) == {0x20}
    assert all(s == {0x20} for s in rec._control.listened)


def test_no_polls_while_stream_is_down(tmp_path):
    import numpy as np
    rec, pipeline, _ = _built(tmp_path, 0x10)
    keeper = rec._keepers[0][0]
    pipeline._on_samples(np.zeros(4, np.complex64), None)
    pipeline._on_stream_dropped("silence")
    keeper._anchor_fn(1)
    rec._refresh_counter_maps()
    assert rec._control.ssrcs == []         # passive listening only


def test_no_keeper_when_disabled(tmp_path):
    rec = HfdlRecorder(_config(tmp_path, False), {"status": "x.local"})
    assert rec._timing_cfg.get("enabled") is False
    rec._start_timing()
    assert rec._keepers == [] and rec._timing_thread is None
