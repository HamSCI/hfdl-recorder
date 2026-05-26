"""Client-contract v0.6 inventory and validate JSON builders."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from hfdl_recorder.bands import HFDL_BANDS
from hfdl_recorder.config import (
    get_enabled_band_names,
    get_enabled_bands,
    resolve_radiod_status,
)
from hfdl_recorder.version import GIT_INFO

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "0.7"


def build_inventory(config: dict, config_path: Path) -> dict:
    """Build the inventory --json payload per contract v0.6."""
    paths = config.get("paths", {})
    sinks = config.get("sinks", {})
    log_dir = paths.get("log_dir", "/var/log/hfdl-recorder")
    spool_dir = paths.get("spool_dir", "/var/lib/hfdl-recorder")

    try:
        version = pkg_version("hfdl-recorder")
    except Exception:
        version = "0.1.0"

    radiod_blocks = config.get("radiod", [])
    if isinstance(radiod_blocks, dict):
        radiod_blocks = [radiod_blocks]

    instances = []
    all_log_paths: dict[str, Any] = {}

    for block in radiod_blocks:
        # Phase 6 cutover (RADIOD-IDENTIFICATION.md §3.1): the mDNS
        # multicast status name IS the identifier.
        status_dns = resolve_radiod_status(block)
        radiod_id = status_dns
        try:
            bands = get_enabled_bands(block)
        except ValueError:
            bands = []

        chain_delay_env = (
            f"RADIOD_{radiod_id.upper().replace('-', '_').replace('.', '_')}_CHAIN_DELAY_NS"
        )
        chain_delay_raw = os.environ.get(chain_delay_env)
        chain_delay = int(chain_delay_raw) if chain_delay_raw else None

        # Per-band frequency list reported as the band centers; the
        # in-band channel frequencies are dumphfdl-internal and not part
        # of the sigmond contract surface.
        frequencies_hz = sorted(b.center_hz for b in bands)

        # CONTRACT v0.6 §17 — output sinks per instance.  Local JSON files
        # are always declared (canonical artifact dumphfdl writes); the
        # ChTailer additionally stages parsed `hfdl.spots` rows into
        # sigmond's local SQLite sink, which `hs-uploader` drains upstream.
        data_sinks: list[dict[str, Any]] = [
            {
                "kind":           "file",
                "target":         f"{spool_dir}/{radiod_id}",
                "schema_ref":     None,
                "retention_days": 0,
                "mb_per_day":     _estimate_json_mb_per_day(bands),
            },
            {
                "kind":           "file",
                "target":         log_dir,
                "schema_ref":     None,
                "retention_days": 365,
                "mb_per_day":     5,
            },
        ]

        # RADIOD-IDENTIFICATION.md §3.2 — inventory radiod_id is the
        # mDNS control/status multicast name (the only functional
        # identifier).
        inventory_radiod_id = status_dns
        instance = {
            "instance": radiod_id,
            "radiod_id": inventory_radiod_id,
            "host": "localhost",
            "radiod_status_dns": status_dns,
            # Per CLIENT-CONTRACT.md §7: ka9q-python owns the multicast
            # destination. We don't pre-resolve at inventory time; the
            # daemon reports the resolved address in its own status.
            "data_destination": None,
            "ka9q_channels": len(bands),
            "frequencies_hz": frequencies_hz,
            "modes": ["hfdl"] if bands else [],
            "bands": [b.name for b in bands],
            "data_sinks": data_sinks,
            "uses_timing_calibration": False,
            "provides_timing_calibration": False,
            "chain_delay_ns_applied": chain_delay,
            # CONTRACT v0.7 §18 — runtime-state field for the §18
            # subscription. hfdl-recorder runs in RTP-default mode
            # (HFDL frame decoding is ms-tolerant; no hard-deadline
            # scheduling against UTC). Reported as null to satisfy
            # the v0.7 inventory shape; sigmond's adapter
            # distinguishes contract-aware-in-default-mode from a
            # pre-v0.7 client by the field's explicit presence.
            "timing_authority_applied": None,
        }
        instances.append(instance)

        # The process log goes to the systemd journal
        # (StandardOutput=journal) — see it via `smd log hfdl-recorder`.
        # log_paths lists only file-based logs (for `smd log --files`).
        instance_logs: dict[str, Any] = {}
        per_band_logs = {
            b.name: f"{log_dir}/{radiod_id}-{b.name}.log" for b in bands
        }
        if per_band_logs:
            instance_logs["bands"] = per_band_logs
        json_sinks = {
            b.name: f"{spool_dir}/{radiod_id}/{b.name}.json" for b in bands
        }
        if json_sinks and sinks.get("local_json", True):
            instance_logs["json"] = json_sinks
        all_log_paths[radiod_id] = instance_logs

    effective_level = logging.getLogger().getEffectiveLevel()
    log_level_name = logging.getLevelName(effective_level)

    payload: dict[str, Any] = {
        "client": "hfdl-recorder",
        "version": version,
        "contract_version": CONTRACT_VERSION,
        "config_path": str(config_path),
    }

    if GIT_INFO:
        payload["git"] = GIT_INFO

    if all_log_paths:
        payload["log_paths"] = all_log_paths

    payload["log_level"] = log_level_name
    payload["instances"] = instances
    payload["deps"] = {
        "binary": [
            {"name": "dumphfdl", "note": "HFDL waveform decoder + libacars; built by scripts/build-dumphfdl.sh"},
        ],
        "git": [
            {"name": "ka9q-radio", "note": "provides radiod IQ multicast; HFDL fragment in config/fragments/hfdl.conf"},
        ],
        "pypi": [
            {"name": "ka9q-python", "version": ">=3.8.0"},
        ],
    }
    payload["issues"] = _collect_issues(config, paths)

    return payload


def build_validate(config: dict, config_path: Path | None = None) -> dict:
    """Build the validate --json payload per contract v0.4 §12.3."""
    paths = config.get("paths", {})
    issues = _collect_issues(config, paths)
    payload: dict[str, Any] = {
        "ok": not any(i["severity"] == "fail" for i in issues),
    }
    if config_path is not None:
        payload["config_path"] = str(config_path)
    payload["issues"] = issues
    return payload


def _collect_issues(config: dict, paths: dict) -> list[dict]:
    """Run validation checks and return issues list."""
    issues: list[dict] = []

    station = config.get("station", {})
    if not station.get("station_id"):
        issues.append({
            "severity": "warn",
            "instance": "all",
            "message": (
                "station.station_id is empty — required by dumphfdl "
                "--station-id and any aggregator feed"
            ),
        })

    sinks = config.get("sinks", {})
    if sinks.get("airframes_io") and not station.get("station_id"):
        issues.append({
            "severity": "fail",
            "instance": "all",
            "message": "sinks.airframes_io requires station.station_id",
        })

    dumphfdl = paths.get("dumphfdl", "/opt/git/sigmond/hfdl-recorder/bin/dumphfdl")
    if not (shutil.which(dumphfdl) or Path(dumphfdl).is_file()):
        issues.append({
            "severity": "fail",
            "instance": "all",
            "message": (
                f"dumphfdl not found at {dumphfdl} — "
                f"run scripts/build-dumphfdl.sh"
            ),
        })
    else:
        version_str = _probe_dumphfdl_version(dumphfdl)
        if version_str:
            issues.append({
                "severity": "info",
                "instance": "all",
                "message": f"dumphfdl reports: {version_str}",
            })

    radiod_blocks = config.get("radiod", [])
    if isinstance(radiod_blocks, dict):
        radiod_blocks = [radiod_blocks]
    if not radiod_blocks:
        issues.append({
            "severity": "fail",
            "instance": "all",
            "message": "no [[radiod]] blocks configured",
        })

    for block in radiod_blocks:
        rid = block.get("status", "<unnamed>")
        if not block.get("status"):
            issues.append({
                "severity": "fail",
                "instance": rid,
                "message": (
                    "[[radiod]] block has no `status` field "
                    "(mDNS multicast name)"
                ),
            })

        names = get_enabled_band_names(block)
        if not names:
            issues.append({
                "severity": "warn",
                "instance": rid,
                "message": "no HFDL bands enabled in [radiod.bands]",
            })

        # Unknown band names — fail loudly so a typo doesn't become silence.
        for name in names:
            if name not in HFDL_BANDS:
                issues.append({
                    "severity": "fail",
                    "instance": rid,
                    "message": (
                        f"unknown band {name!r}; valid: "
                        f"{', '.join(sorted(HFDL_BANDS))}"
                    ),
                })

        # Duplicate band — silently dropped by ka9q-python's per-channel
        # subscription; treat as the HFDL-equivalent of the SSRC collision
        # check in psk-recorder (contract v0.4 §12.2).
        seen: set[str] = set()
        for name in names:
            if name in seen:
                issues.append({
                    "severity": "fail",
                    "instance": rid,
                    "message": (
                        f"band {name!r} listed twice in [radiod.bands].enabled"
                    ),
                })
            seen.add(name)

        # Squitter-only band: useful but low-yield without companion bands.
        if "HFDL15" in names and len(names) == 1:
            issues.append({
                "severity": "info",
                "instance": rid,
                "message": (
                    "HFDL15 (15025 kHz) is squitter-only; consider enabling "
                    "additional bands for traffic decodes"
                ),
            })

    return issues


def _estimate_json_mb_per_day(bands) -> int:
    """Rough decoded-JSON volume estimate.

    HFDL traffic is bursty; ~20 kB/day per band is a reasonable order-of-
    magnitude for an active station. Used by sigmond's disk-budget summary.
    """
    return max(1, len(list(bands)))


def _probe_dumphfdl_version(path: str) -> str:
    """Run ``dumphfdl --version`` and return the first line, or empty string."""
    try:
        proc = subprocess.run(
            [path, "--version"],
            capture_output=True, text=True, timeout=3,
        )
        out = (proc.stdout or proc.stderr).strip().splitlines()
        return out[0] if out else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""
