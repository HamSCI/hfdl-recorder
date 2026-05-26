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
