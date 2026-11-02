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

    def poll_status(self, ssrc, timeout):
        self.ssrcs.append(ssrc)
        return SimpleNamespace(gps_time=None, rtp_timesnap=None)


def test_keeper_built_per_band_and_polls_only_its_ssrc(tmp_path):
    rec = HfdlRecorder(_config(tmp_path, True), {"status": "x.local"})
    rec._control = _Control()
    pipeline = BandPipeline(band=HFDL_BANDS["HFDL13"], radiod_id="x.local",
                            config=rec._config)
    rec._pipelines.append(pipeline)
    rec._add_keeper(pipeline, 0xABCD)
    keeper, ssrc, _counter = rec._keepers[0]
    assert pipeline._keeper is keeper and ssrc == 0xABCD
    assert keeper.out_dir == tmp_path / "squitters" / "x.local"
    # The anchor polls the band's own SSRC and nothing else, and with no
    # status pair it gives no anchor (no host-clock fallback).
    assert keeper._anchor_fn(123) is None
    assert set(rec._control.ssrcs) == {0xABCD}


def test_no_keeper_when_disabled(tmp_path):
    rec = HfdlRecorder(_config(tmp_path, False), {"status": "x.local"})
    assert rec._timing_cfg.get("enabled") is False
    rec._start_timing()
    assert rec._keepers == [] and rec._timing_thread is None
