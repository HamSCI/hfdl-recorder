"""Tests for config loading and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from hfdl_recorder.bands import HFDL_BANDS
from hfdl_recorder.config import (
    get_enabled_band_names,
    get_enabled_bands,
    load_config,
    resolve_radiod_block,
    resolve_radiod_status,
)

FIXTURE = Path(__file__).parent / "fixtures" / "test-config.toml"


def test_load_config_applies_defaults():
    cfg = load_config(FIXTURE)
    assert cfg["paths"]["dumphfdl"] == "/usr/bin/true"
    assert cfg["paths"]["spool_dir"].endswith("spool")
    # Defaults from DEFAULTS dict.
    assert "log_dir" in cfg["paths"]
    assert "sinks" in cfg
    assert cfg["sinks"]["local_json"] is True
    assert cfg["sinks"]["airframes_io"] is False
    assert cfg["sinks"]["airframes_station_id"] == ""


def test_load_config_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.toml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_resolve_radiod_block_single():
    cfg = load_config(FIXTURE)
    block = resolve_radiod_block(cfg, None)
    assert block["status"] == "test-rx888-status.local"


def test_resolve_radiod_block_unknown_id():
    cfg = load_config(FIXTURE)
    with pytest.raises(ValueError):
        resolve_radiod_block(cfg, "nope")


def test_get_enabled_band_names():
    cfg = load_config(FIXTURE)
    block = resolve_radiod_block(cfg, None)
    assert get_enabled_band_names(block) == ["HFDL21", "HFDL13", "HFDL5"]


def test_get_enabled_bands_resolves():
    cfg = load_config(FIXTURE)
    block = resolve_radiod_block(cfg, None)
    bands = get_enabled_bands(block)
    assert [b.name for b in bands] == ["HFDL21", "HFDL13", "HFDL5"]
    assert bands[0].center_hz == HFDL_BANDS["HFDL21"].center_hz


def test_get_enabled_bands_unknown_raises():
    cfg = load_config(FIXTURE)
    block = resolve_radiod_block(cfg, None)
    block["bands"] = {"enabled": ["HFDL21", "NOPE"]}
    with pytest.raises(ValueError):
        get_enabled_bands(block)


def test_resolve_radiod_status_from_field():
    cfg = load_config(FIXTURE)
    block = resolve_radiod_block(cfg, None)
    assert resolve_radiod_status(block) == "test-rx888-status.local"


def test_resolve_radiod_status_missing_raises():
    """Phase 6: a block without `status` is rejected, not back-filled."""
    with pytest.raises(ValueError):
        resolve_radiod_status({})


def test_timing_squitters_off_by_default():
    cfg = load_config(FIXTURE)
    ts = cfg["timing_squitters"]
    assert ts["enabled"] is False
    assert ts["ground_stations"] == [
        "Riverhead, New York", "San Francisco, California", "Molokai, Hawaii"]
    assert ts["window_dir"] == "/var/lib/hfdl-recorder/squitters"
    assert ts["retention_days"] == 5
    assert ts["min_free_gb"] == 5


def test_timing_squitters_template_matches_defaults():
    from hfdl_recorder.config import tomllib
    tpl = Path(__file__).parent.parent / "config" / "hfdl-recorder-config.toml.template"
    with open(tpl, "rb") as f:
        raw = tomllib.load(f)
    ts = raw["timing_squitters"]
    assert ts["enabled"] is False
    assert ts["ground_stations"] == [
        "Riverhead, New York", "San Francisco, California", "Molokai, Hawaii"]


def test_timing_squitters_rejects_bad_values(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(FIXTURE.read_text() + "\n[timing_squitters]\nretention_days = -1\n")
    with pytest.raises(ValueError):
        load_config(p)
