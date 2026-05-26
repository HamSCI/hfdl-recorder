"""TOML config loader and defaults for hfdl-recorder."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from hfdl_recorder.bands import HFDL_BANDS, HfdlBand


DEFAULT_CONFIG_PATH = Path("/etc/hfdl-recorder/hfdl-recorder-config.toml")
PER_INSTANCE_CONFIG_DIR = Path("/etc/hfdl-recorder")


def resolve_config_path(
    instance: Optional[str] = None,
    explicit_path: Optional[Path] = None,
) -> Path:
    """Resolve config path per sigmond MULTI-INSTANCE-ARCHITECTURE.md §4.

    Precedence: explicit_path > $HFDL_RECORDER_CONFIG > per-instance
    /etc/hfdl-recorder/<instance>.toml (when given and exists) > legacy
    /etc/hfdl-recorder/hfdl-recorder-config.toml (with DeprecationWarning
    when --instance was given but per-instance file is missing).
    """
    if explicit_path is not None:
        return Path(explicit_path)
    env_override = os.environ.get("HFDL_RECORDER_CONFIG")
    if env_override:
        return Path(env_override)
    if instance:
        per_instance = PER_INSTANCE_CONFIG_DIR / f"{instance}.toml"
        if per_instance.exists():
            return per_instance
        warnings.warn(
            f"per-instance config {per_instance} not found; falling "
            f"back to legacy shared config {DEFAULT_CONFIG_PATH}. "
            f"Migrate this host with `sudo smd instance migrate` "
            f"(MULTI-INSTANCE-ARCHITECTURE.md §6).",
            DeprecationWarning,
            stacklevel=2,
        )
    return DEFAULT_CONFIG_PATH


def extract_reporter_id(config_or_path) -> Optional[str]:
    """Read reporter_id from a per-instance config's [instance] block.

    Accepts a parsed TOML dict or a Path.  Returns None when no
    [instance] block.  Callers should NOT fall back to args.instance
    (the systemd %i, typically a radiod id, not a reporter id);
    row construction falls back to radiod_id instead.
    """
    if isinstance(config_or_path, dict):
        raw = config_or_path
    else:
        path = Path(config_or_path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            return None
    inst = raw.get("instance")
    if not isinstance(inst, dict):
        return None
    rid = inst.get("reporter_id")
    if not isinstance(rid, str) or not rid:
        return None
    return rid

DEFAULTS: dict[str, Any] = {
    "paths": {
        "dumphfdl":  "/opt/git/sigmond/hfdl-recorder/bin/dumphfdl",
        "spool_dir": "/var/lib/hfdl-recorder",
        "log_dir":   "/var/log/hfdl-recorder",
        "systable":  "/var/lib/hfdl-recorder/systable.conf",
    },
    "sinks": {
        "local_json":   True,
        "airframes_io": False,
    },
    "processing": {
        # radiod LIFETIME tag (ka9q-python ≥3.13.0, ka9q-radio ≥0f8b622).
        # Channels self-destruct after this many radiod main-loop frames
        # (~50 Hz at the default 20 ms blocktime, so 6000 ≈ 2 min).  The
        # daemon refreshes lifetime every (frames / 4) seconds while
        # running, so a crashed/killed daemon leaves no residual channels
        # on radiod within ~2 min.  0 = infinite (no LIFETIME tag, no
        # keep-alive — radiod owns the channel for its full template
        # default).
        "radiod_lifetime_frames": 6000,
    },
}

# Encoding integer matches ka9q-python's Encoding enum (s16be = 2). The
# radiod HFDL fragment ships every band as s16be, so we hard-code it here.
DEFAULT_PRESET = "iq"
DEFAULT_ENCODING = "s16be"


def load_config(path: Path | None = None) -> dict:
    """Load and merge config with defaults."""
    config_path = path or Path(
        os.environ.get("HFDL_RECORDER_CONFIG", str(DEFAULT_CONFIG_PATH))
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    raw.setdefault("paths", {})
    for key, val in DEFAULTS["paths"].items():
        raw["paths"].setdefault(key, val)

    raw.setdefault("sinks", {})
    for key, val in DEFAULTS["sinks"].items():
        raw["sinks"].setdefault(key, val)

    raw.setdefault("processing", {})
    for key, val in DEFAULTS["processing"].items():
        raw["processing"].setdefault(key, val)

    lifetime = raw["processing"]["radiod_lifetime_frames"]
    if not isinstance(lifetime, int) or lifetime < 0:
        raise ValueError(
            f"processing.radiod_lifetime_frames must be a non-negative int "
            f"(frames; ~50 Hz at default blocktime); got {lifetime!r}"
        )

    return raw


# Default HFDL band selection used for synthesized [[radiod]] blocks
# (matches the curated set in the config template).
_DEFAULT_HFDL_BANDS = [
    "HFDL21", "HFDL13", "HFDL11", "HFDL10", "HFDL8", "HFDL6", "HFDL5",
]

# Where ka9q-radio keeps its per-instance conf files.  When hfdl-recorder
# can't find a [[radiod]] block in its own config, this is the canonical
# source of truth for the radiod's status DNS — same approach
# wsprdaemon-client uses (see lib/wdlib/v4_parser.py).
_RADIOD_CONF_DIR = Path("/etc/radio")


def _read_status_from_radiod_conf(conf_path: Path) -> str | None:
    """Return the `status =` value from a radiod conf, or None on miss."""
    try:
        for raw in conf_path.read_text().splitlines():
            line = raw.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() == "status":
                # Strip trailing inline comments, quotes, and whitespace
                val = val.split("#", 1)[0].strip().strip('"').strip("'")
                return val or None
    except OSError:
        pass
    return None


def _synthesize_radiod_block_from_conf(radiod_id: str) -> dict | None:
    """Build a minimal [[radiod]] block from /etc/radio/radiod@<id>.conf.

    Returns None if the conf doesn't exist or doesn't name a status DNS.
    The synthesized block uses the curated default band list — operators
    who want a different set should declare an explicit [[radiod]] block.
    """
    conf = _RADIOD_CONF_DIR / f"radiod@{radiod_id}.conf"
    if not conf.exists():
        return None
    status = _read_status_from_radiod_conf(conf)
    if not status:
        return None
    return {
        "status":   status,
        "bands":    {"enabled": list(_DEFAULT_HFDL_BANDS)},
        "_source":  f"synthesized from {conf}",
    }


def resolve_radiod_block(config: dict, radiod_id: str | None) -> dict:
    """Find the [[radiod]] block matching radiod_id.

    Resolution order (Phase 6 cutover —
    RADIOD-IDENTIFICATION.md §3.1):
      1. Match an explicit [[radiod]] block via the canonical
         ``status`` field (the mDNS multicast name).
      2. If no match (or no blocks at all) and
         /etc/radio/radiod@<id>.conf exists locally, synthesize a
         block from the conf's ``status =`` line.
      3. If radiod_id is None and exactly one /etc/radio/radiod@*.conf
         exists, autodetect that one.
    """
    radiod_blocks = config.get("radiod", [])
    if isinstance(radiod_blocks, dict):
        radiod_blocks = [radiod_blocks]

    # Explicit match first
    if radiod_id is not None:
        for block in radiod_blocks:
            if block.get("status") == radiod_id:
                return block
        # Fall through to filesystem
        synth = _synthesize_radiod_block_from_conf(radiod_id)
        if synth is not None:
            return synth
        available = [b.get("status", "<unnamed>") for b in radiod_blocks]
        raise ValueError(
            f"No [[radiod]] block with status={radiod_id!r} in "
            f"config and no /etc/radio/radiod@{radiod_id}.conf on disk. "
            f"Config blocks: {available or ['(none)']}.  "
            "If you see legacy `id`/`radiod_status` fields in the "
            "config, run `sudo smd radiod migrate --yes`."
        )

    # radiod_id is None → either pick the one config block, or autodetect
    # from the filesystem when the config has no blocks.
    if len(radiod_blocks) == 1:
        return radiod_blocks[0]
    if not radiod_blocks and _RADIOD_CONF_DIR.is_dir():
        confs = sorted(_RADIOD_CONF_DIR.glob("radiod@*.conf"))
        if len(confs) == 1:
            only_id = confs[0].stem.split("@", 1)[1]
            synth = _synthesize_radiod_block_from_conf(only_id)
            if synth is not None:
                return synth
    if not radiod_blocks:
        raise ValueError(
            "Config has no [[radiod]] blocks and "
            f"no unambiguous local radiod conf in {_RADIOD_CONF_DIR}"
        )
    raise ValueError(
        f"--radiod-id required: config has {len(radiod_blocks)} "
        f"[[radiod]] blocks"
    )


def get_enabled_band_names(radiod_block: dict) -> list[str]:
    """Return the list of band names enabled for this radiod, in config order."""
    bands_block = radiod_block.get("bands", {})
    return list(bands_block.get("enabled", []))


def get_enabled_bands(radiod_block: dict) -> list[HfdlBand]:
    """Resolve enabled band names against the static HFDL_BANDS table.

    Unknown names raise ValueError so misconfigurations fail loud rather
    than silently dropping a band the operator thought was enabled.
    """
    resolved: list[HfdlBand] = []
    for name in get_enabled_band_names(radiod_block):
        band = HFDL_BANDS.get(name)
        if band is None:
            raise ValueError(
                f"Unknown HFDL band {name!r}. "
                f"Valid: {', '.join(sorted(HFDL_BANDS))}"
            )
        resolved.append(band)
    return resolved


def resolve_radiod_status(radiod_block: dict) -> str:
    """Resolve the radiod mDNS control/status multicast name.

    Reads the canonical ``status`` field per
    RADIOD-IDENTIFICATION.md §3.1.  Phase 6 cutover (this release)
    removed the legacy paths (``RADIOD_<ID>_STATUS`` env override
    and ``radiod_status`` field) — operators with legacy configs
    must run ``sudo smd radiod migrate --yes``.
    """
    status = radiod_block.get("status")
    if not status:
        raise ValueError(
            "[[radiod]] block has no `status` field.  Run "
            "`sudo smd radiod migrate --yes` if this config still "
            "uses the legacy `radiod_status` field."
        )
    return status
