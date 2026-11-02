"""Tests for label-check: labelled_lag_seconds, the TOML writer, CLI shape."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from hfdl_recorder.timing.label_check import (
    build_repeat_record,
    labelled_lag_seconds,
    write_corrections,
)

REF_FS = 12000
PROBE_FS = 100000
DURATION = 2.0          # seconds; periodic, so both rates sample it exactly
BAND_HZ = 5000.0        # signal occupies +-5 kHz, inside the 12 kS/s Nyquist


def _spectrum(seed: int) -> dict:
    """Random complex line spectrum on 1/DURATION grid, |f| < BAND_HZ."""
    rng = np.random.default_rng(seed)
    k = np.arange(-int(BAND_HZ * DURATION), int(BAND_HZ * DURATION))
    coef = rng.normal(size=k.size) + 1j * rng.normal(size=k.size)
    return {"k": k, "coef": coef}


def _sample(spec: dict, fs: int, delay_s: float = 0.0) -> np.ndarray:
    """Exact samples of x(t - delay_s) at fs; x is periodic over DURATION."""
    n = int(fs * DURATION)
    full = np.zeros(n, dtype=complex)
    f = spec["k"] / DURATION
    full[spec["k"] % n] = spec["coef"] * np.exp(-2j * np.pi * f * delay_s)
    return np.fft.ifft(full) * n / np.sqrt(spec["k"].size)


def test_zero_lag_when_labels_agree():
    spec = _spectrum(1)
    ref = _sample(spec, REF_FS)
    probe = _sample(spec, PROBE_FS)
    lag, corr = labelled_lag_seconds(ref, REF_FS, 1000.0, probe, PROBE_FS, 1000.0)
    assert abs(lag) < 2e-6
    assert corr > 0.9


def test_reports_known_label_offset():
    spec = _spectrum(2)
    ref = _sample(spec, REF_FS)
    probe = _sample(spec, PROBE_FS)
    lag, corr = labelled_lag_seconds(
        ref, REF_FS, 1000.0, probe, PROBE_FS, 1000.0 + 3.7e-3)
    assert lag == pytest.approx(3.7e-3, abs=2e-6)
    assert corr > 0.9


def test_fractional_sample_offset():
    spec = _spectrum(3)
    ref = _sample(spec, REF_FS)
    # Probe content runs 0.3 ref-sample behind its label: label runs late.
    probe = _sample(spec, PROBE_FS, delay_s=0.3 / REF_FS)
    lag, _ = labelled_lag_seconds(ref, REF_FS, 1000.0, probe, PROBE_FS, 1000.0)
    assert lag == pytest.approx(0.3 / REF_FS, abs=2e-6)


def test_uncorrelated_streams_report_low_corr():
    ref = _sample(_spectrum(4), REF_FS)
    probe = _sample(_spectrum(5), PROBE_FS)
    _, corr = labelled_lag_seconds(ref, REF_FS, 1000.0, probe, PROBE_FS, 1000.0)
    assert corr < 0.2


def test_whole_second_label_offset_with_partial_overlap():
    spec = _spectrum(6)
    ref = _sample(spec, REF_FS)
    probe = _sample(spec, PROBE_FS)
    lag, corr = labelled_lag_seconds(
        ref, REF_FS, 1000.0, probe, PROBE_FS, 1000.0 + 0.25)
    assert lag == pytest.approx(0.25, abs=2e-6)
    assert corr > 0.9


def test_repeat_record_shape():
    rec = build_repeat_record(
        band="HFDL13", probe_fs=80000, ref_fs=12000, lag_s=1.5e-4,
        corr=0.97, anchor_source="rtp_to_utc+authority", t_level_active="T3")
    assert list(rec) == ["band", "probe_fs", "ref_fs", "lag_us", "corr",
                         "anchor_source", "t_level_active"]
    assert rec["lag_us"] == pytest.approx(150.0)
    json.dumps(rec)


def test_write_corrections_median_per_band(tmp_path):
    out = tmp_path / "sub" / "label_corrections.toml"
    recs = [
        {"band": "HFDL13", "lag_us": 100.0},
        {"band": "HFDL13", "lag_us": 104.0},
        {"band": "HFDL13", "lag_us": 500.0},
        {"band": "HFDL8", "lag_us": -20.0},
    ]
    write_corrections(out, recs)
    tomllib = pytest.importorskip("tomllib") if sys.version_info >= (3, 11) \
        else pytest.importorskip("tomli")
    data = tomllib.loads(out.read_text())
    assert data["HFDL13"]["lag_us"] == 104.0
    assert data["HFDL13"]["spread_us"] == 400.0
    assert data["HFDL13"]["n"] == 3
    assert data["HFDL8"]["lag_us"] == -20.0


def test_write_corrections_merges_other_bands(tmp_path):
    out = tmp_path / "c.toml"
    write_corrections(out, [{"band": "HFDL8", "lag_us": 1.0}])
    write_corrections(out, [{"band": "HFDL13", "lag_us": 2.0}])
    text = out.read_text()
    assert "[HFDL8]" in text and "[HFDL13]" in text


def test_cli_parses_label_check():
    from hfdl_recorder.cli import build_parser
    args = build_parser().parse_args([
        "label-check", "--radiod-id", "rx", "--ref-freq-hz", "10000000",
        "--band", "HFDL13", "--seconds", "30", "--repeats", "5", "--json"])
    assert args.command == "label-check"
    assert args.ref_freq_hz == 10_000_000.0
    assert args.band == "HFDL13"
    assert args.seconds == 30.0 and args.repeats == 5 and args.json
    assert str(args.out) == "/var/lib/hfdl-recorder/label_corrections.toml"
