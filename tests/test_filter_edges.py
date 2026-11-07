"""The band filter must survive a radiod restart.

MultiStream re-creates a dropped channel from what ``add_channel`` stored.
Edges set only by a separate ``set_filter`` call never reach that store, so
after a radiod restart the channel falls back to the "iq" preset's ±5 kHz
and dumphfdl hears one 10 kHz slice of the band (UACNJ, 11-04 → 11-07).
"""
from unittest import mock

from hfdl_recorder.bands import HFDL_BANDS
from hfdl_recorder.core.band_pipeline import BandPipeline
from hfdl_recorder.core.radiod import band_filter_edges


def test_edges_span_the_band_less_the_guard():
    lo, hi = band_filter_edges(160_000)
    assert (lo, hi) == (-78_500, 78_500)       # matches K3LR's live HFDL8


def test_attach_hands_the_edges_to_add_channel():
    band = HFDL_BANDS["HFDL8"]
    pipeline = BandPipeline(band=band, radiod_id="test", config={"paths": {}})
    multi = mock.MagicMock()
    multi.add_channel.return_value = mock.Mock(ssrc=7, frequency=band.center_hz)
    pipeline.attach(multi)
    kw = multi.add_channel.call_args.kwargs
    assert (kw["low_edge"], kw["high_edge"]) == band_filter_edges(band.samprate_hz)
