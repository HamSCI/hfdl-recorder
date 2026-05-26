"""Tests for contract v0.7 inventory/validate JSON builders."""

from __future__ import annotations

from pathlib import Path

from hfdl_recorder.config import load_config
from hfdl_recorder.contract import (
    CONTRACT_VERSION,
    build_inventory,
    build_validate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "test-config.toml"


def test_contract_version_is_0_7():
    assert CONTRACT_VERSION == "0.7"


def test_inventory_required_top_level_keys():
    cfg = load_config(FIXTURE)
    inv = build_inventory(cfg, FIXTURE)
    for key in (
        "client", "version", "contract_version", "config_path",
        "log_paths", "log_level", "instances", "deps", "issues",
    ):
        assert key in inv, f"missing top-level inventory key: {key}"
    assert inv["client"] == "hfdl-recorder"
    assert inv["contract_version"] == "0.7"


def test_inventory_instance_shape():
    cfg = load_config(FIXTURE)
    inv = build_inventory(cfg, FIXTURE)
    assert len(inv["instances"]) == 1
    inst = inv["instances"][0]
    # RADIOD-IDENTIFICATION.md §3.1 (Phase 6): instance and radiod_id
    # are both the canonical mDNS multicast status name.
    assert inst["instance"] == "test-rx888-status.local"
    assert inst["radiod_id"] == "test-rx888-status.local"
    assert inst["modes"] == ["hfdl"]
    assert inst["bands"] == ["HFDL21", "HFDL13", "HFDL5"]
    assert inst["ka9q_channels"] == 3
    # frequencies sorted ascending (band centers).
    assert inst["frequencies_hz"] == sorted(inst["frequencies_hz"])
    assert inst["data_destination"] is None  # contract §7
    assert inst["uses_timing_calibration"] is False


def test_inventory_timing_authority_applied_v0_7():
    """CONTRACT v0.7 §3/§18 — runtime-state field for the §18
    subscription. hfdl-recorder runs in RTP-default mode (HFDL
    frame decoding is ms-tolerant), so the field is present and
    explicitly None — distinguishes contract-aware-in-default-mode
    from a pre-v0.7 client."""
    cfg = load_config(FIXTURE)
    inv = build_inventory(cfg, FIXTURE)
    inst = inv["instances"][0]
    assert "timing_authority_applied" in inst
    assert inst["timing_authority_applied"] is None


def test_inventory_data_sinks_v0_6():
    """CONTRACT v0.6 §17.3: every instance has a data_sinks array.

    File sinks are always declared. ClickHouse support has been
    removed suite-wide; only file sinks remain.
    """
    cfg = load_config(FIXTURE)
    inv = build_inventory(cfg, FIXTURE)
    inst = inv["instances"][0]
    assert "data_sinks" in inst
    kinds = {s["kind"] for s in inst["data_sinks"]}
    assert "file" in kinds
    assert "clickhouse" not in kinds
    for sink in inst["data_sinks"]:
        for required in ("kind", "target", "retention_days", "mb_per_day"):
            assert required in sink, f"sink missing {required}"


def test_validate_passes_with_fixture():
    cfg = load_config(FIXTURE)
    payload = build_validate(cfg, FIXTURE)
    fails = [i for i in payload["issues"] if i["severity"] == "fail"]
    assert payload["ok"] is True, f"unexpected fails: {fails}"


def test_validate_fails_when_dumphfdl_missing(tmp_path):
    cfg = load_config(FIXTURE)
    cfg["paths"]["dumphfdl"] = str(tmp_path / "not-a-binary")
    payload = build_validate(cfg)
    assert payload["ok"] is False
    fails = [i for i in payload["issues"] if i["severity"] == "fail"]
    assert any("dumphfdl not found" in i["message"] for i in fails)


def test_validate_flags_unknown_band(tmp_path):
    cfg = load_config(FIXTURE)
    cfg["radiod"][0]["bands"]["enabled"] = ["HFDL21", "BOGUS"]
    payload = build_validate(cfg)
    assert payload["ok"] is False
    assert any(
        i["severity"] == "fail" and "BOGUS" in i["message"]
        for i in payload["issues"]
    )


def test_validate_flags_duplicate_band():
    cfg = load_config(FIXTURE)
    cfg["radiod"][0]["bands"]["enabled"] = ["HFDL21", "HFDL21"]
    payload = build_validate(cfg)
    assert payload["ok"] is False
    assert any(
        i["severity"] == "fail" and "twice" in i["message"]
        for i in payload["issues"]
    )


def test_validate_fails_airframes_without_station_id():
    cfg = load_config(FIXTURE)
    cfg["station"]["station_id"] = ""
    cfg["sinks"]["airframes_io"] = True
    payload = build_validate(cfg)
    assert payload["ok"] is False
    assert any(
        i["severity"] == "fail" and "station_id" in i["message"]
        for i in payload["issues"]
    )
