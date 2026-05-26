"""Interactive `config init` and `config edit` for hfdl-recorder.

Implements CONTRACT-v0.5 §14: sigmond invokes these via
`smd config init|edit hfdl-recorder [<instance>]`, passing
`STATION_CALL`, `STATION_GRID`, `SIGMOND_INSTANCE`, and
`SIGMOND_RADIOD_STATUS` as advisory defaults.

hfdl-recorder's `[station].station_id` doubles as the
airframes.io-registered station name; the convention is
`<CALLSIGN>-<n>` per radiod source.  Operators can override.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from .config import DEFAULT_CONFIG_PATH


def _find_template() -> Optional[Path]:
    candidates = [
        Path(__file__).resolve().parent.parent.parent
            / "config" / "hfdl-recorder-config.toml.template",
        Path("/opt/git/sigmond/hfdl-recorder/config/hfdl-recorder-config.toml.template"),
        Path("/usr/local/share/hfdl-recorder/hfdl-recorder-config.toml.template"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def cmd_config_init(args) -> int:
    target = _resolve_target(args)
    if target.exists() and not getattr(args, "reconfig", False):
        _err(f"{target} already exists.  Pass --reconfig to overwrite, or "
             f"run `hfdl-recorder config edit` instead.")
        return 1

    template = _find_template()
    if template is None:
        _err("hfdl-recorder template not found; reinstall the package")
        return 1

    body = template.read_text()
    values = _collect_init_values(args)
    body = _apply_init_substitutions(body, values)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    _ok(f"wrote {target}")
    _info(f"station_id: {values['station_id']}    grid: {values['grid']}")
    _info(f"radiod:     id={values['radiod_id']}  status={values['radiod_status']}")
    _info("")
    _info("Next steps:")
    _info(f"  1. Review [radiod.bands].enabled in {target}")
    _info(f"  2. If feeding airframes.io, register {values['station_id']!r} "
          f"at https://app.airframes.io")
    _info(f"  3. Validate: hfdl-recorder validate --json")
    _info(f"  4. Start:    sudo systemctl enable --now "
          f"hfdl-recorder@{values['radiod_id']}.service")
    return 0


def cmd_config_edit(args) -> int:
    target = _resolve_target(args)
    if not target.exists():
        _err(f"{target} does not exist.  Run `hfdl-recorder config init` first.")
        return 1

    try:
        with open(target, "rb") as f:
            current = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        _err(f"failed to read {target}: {e}")
        return 1

    cur_station = (current.get("station") or {}).get("station_id", "")
    cur_grid    = (current.get("station") or {}).get("grid_square", "")
    blocks = _radiod_blocks(current)
    block, block_index = _select_radiod_block(blocks, args)
    if block is None:
        return 1
    cur_id     = block.get("id", "")
    cur_status = block.get("radiod_status", "")

    if getattr(args, "non_interactive", False):
        _info(f"station.station_id    = {cur_station}")
        _info(f"station.grid_square   = {cur_grid}")
        _info(f"radiod[{block_index}].id            = {cur_id}")
        _info(f"radiod[{block_index}].radiod_status = {cur_status}")
        return 0

    new_station = _prompt("station_id (airframes.io registration)",
                          cur_station or _default_station_id())
    new_grid    = _prompt("Grid square",
                          cur_grid or os.environ.get("STATION_GRID", ""))
    new_id      = _prompt("Radiod id",
                          cur_id or os.environ.get("SIGMOND_INSTANCE", ""))
    new_status  = _prompt("Radiod status DNS",
                          cur_status or
                          os.environ.get("SIGMOND_RADIOD_STATUS", ""))

    body = target.read_text()
    body = _replace_station_field(body, "station_id",  new_station)
    body = _replace_station_field(body, "grid_square", new_grid)
    body = _replace_radiod_field(body, block_index, "id",            new_id)
    body = _replace_radiod_field(body, block_index, "radiod_status", new_status)

    if body == target.read_text():
        _info("no changes")
        return 0

    target.write_text(body)
    _ok(f"updated {target}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_target(args) -> Path:
    return Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)


def _default_station_id() -> str:
    """Compose <CALLSIGN>-<n> per CONTRACT-v0.5 §14.6.

    The suffix `n` is `SIGMOND_RADIOD_INDEX` when set by the dispatcher
    (the 1-based declaration order of the radiod this instance is
    bound to, including the single-radiod case where it's just `1`).
    When the env var is absent or unparseable — e.g. a standalone
    invocation outside sigmond — we default to `1`.

    Airframes.io requires the suffix even in the single-radiod case,
    so unlike most clients we don't drop it when COUNT == 1.

    Returns "" when no callsign is known.
    """
    call = os.environ.get("STATION_CALL", "").strip()
    if not call:
        return ""
    try:
        index = int(os.environ.get("SIGMOND_RADIOD_INDEX", "1") or "1")
    except ValueError:
        index = 1
    return f"{call}-{index}"


def _discover_radiods(timeout: float = 5.0) -> list[dict]:
    """Return discovered radiods or [] on failure (lazy ka9q-python
    import, avahi missing, etc.).  Per RADIOD-IDENTIFICATION.md §4."""
    try:
        from ka9q.discovery import discover_radiod_services
        return discover_radiod_services(timeout=timeout) or []
    except Exception:
        return []


def _pick_radiod_status_from_discovery(
    discovered: list[dict], env_status: str, instance_hint: str,
) -> str:
    """Interactive discovery flow (zero / one / multi cases)."""
    if not discovered:
        print("\033[33m⚠\033[0m  No radiod instances broadcasting on the "
              "local network.")
        _info("Install + start radiod before continuing:")
        _info("  sudo smd install ka9q-radio")
        _info("Continuing with manual entry — the daemon will refuse to "
              "start if the multicast name is unreachable.")
        default = env_status or (
            f"{instance_hint}-status.local" if instance_hint else "")
        return _prompt("Radiod status DNS (manual entry)", default,
                       required=True)

    if len(discovered) == 1:
        only = discovered[0]
        _info(f"One radiod discovered: {only['hostname']!r} "
              f"(advertised: {only['name']!r})")
        confirm = _prompt(
            f"Use {only['hostname']!r}? [Y/n]", "Y").strip().lower()
        if confirm in ("", "y", "yes"):
            return only["hostname"]
        return _prompt("Radiod status DNS (manual entry)",
                       env_status or only["hostname"], required=True)

    _info("Multiple radiods discovered on the LAN:")
    for i, svc in enumerate(discovered, 1):
        _info(f"  [{i}] {svc['hostname']:<32} (advertised: {svc['name']!r})")
    while True:
        choice = _prompt(
            f"Pick a radiod [1-{len(discovered)}]", "1").strip()
        try:
            idx = int(choice) - 1
        except ValueError:
            print("\033[33m⚠\033[0m  Enter a number from the menu.")
            continue
        if 0 <= idx < len(discovered):
            return discovered[idx]["hostname"]
        print(f"\033[33m⚠\033[0m  Out of range; pick 1-{len(discovered)}.")


def _collect_init_values(args) -> dict:
    grid = os.environ.get("STATION_GRID", "")
    instance = os.environ.get("SIGMOND_INSTANCE", "")
    status = os.environ.get("SIGMOND_RADIOD_STATUS", "")
    default_station = _default_station_id() or "YOURCALL-1"

    if getattr(args, "non_interactive", False):
        # Env wins; else single-radiod auto-pick; else placeholder.
        if status:
            radiod_status = status
        else:
            discovered = _discover_radiods()
            if len(discovered) == 1:
                radiod_status = discovered[0]["hostname"]
            else:
                radiod_status = (
                    f"{instance}-status.local"
                    if instance else "my-rx888-status.local"
                )
        return {
            "station_id":    default_station,
            "grid":          grid or "AA00aa",
            "radiod_id":     instance or "my-rx888",
            "radiod_status": radiod_status,
        }

    station_id  = _prompt("station_id (airframes.io registration)",
                          default_station, required=True)
    grid_square = _prompt("Grid square", grid, required=True)

    # RADIOD-IDENTIFICATION.md §4 — discovery-driven radiod selection.
    discovered = _discover_radiods()
    radiod_status = _pick_radiod_status_from_discovery(
        discovered, status, instance)

    # Legacy local label — Phase 6 cutover removes this prompt.
    radiod_id_default = instance or _derive_label_from_status(radiod_status)
    radiod_id   = _prompt("Radiod id (local label — legacy, will be retired)",
                          radiod_id_default, required=True)
    return {
        "station_id":    station_id,
        "grid":          grid_square,
        "radiod_id":     radiod_id,
        "radiod_status": radiod_status,
    }


def _derive_label_from_status(status: str) -> str:
    """Strip mDNS suffixes for a default local label."""
    base = (status or "").strip()
    for suffix in ("-status.local", ".local"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base or "default"


def _apply_init_substitutions(body: str, values: dict) -> str:
    body = _replace_station_field(body, "station_id",  values["station_id"])
    body = _replace_station_field(body, "grid_square", values["grid"])
    # RADIOD-IDENTIFICATION.md §3.1 — canonical field is `status`
    # (multicast mDNS name).  The legacy `id` + `radiod_status` lines
    # are commented out in the template post-Phase 3.
    body = _replace_radiod_field(body, 0, "status", values["radiod_status"])
    return body


def _radiod_blocks(config: dict) -> list[dict]:
    blocks = config.get("radiod", [])
    if isinstance(blocks, dict):
        blocks = [blocks]
    return list(blocks)


def _select_radiod_block(blocks: list[dict], args) -> tuple:
    if not blocks:
        _err("config has no [[radiod]] blocks")
        return None, -1

    target_id = os.environ.get("SIGMOND_INSTANCE", "") or \
                getattr(args, "radiod_id", None)

    if target_id:
        for i, b in enumerate(blocks):
            if b.get("id") == target_id:
                return b, i
        _err(f"no [[radiod]] block with id={target_id!r}; "
             f"available: {', '.join(b.get('id', '?') for b in blocks)}")
        return None, -1

    if len(blocks) == 1:
        return blocks[0], 0

    if getattr(args, "non_interactive", False):
        _err("multiple [[radiod]] blocks; specify with --radiod-id or "
             "SIGMOND_INSTANCE")
        return None, -1

    print("\nMultiple [[radiod]] blocks present.  Pick one:")
    for i, b in enumerate(blocks, start=1):
        print(f"  {i}) id={b.get('id', '?')}  status={b.get('radiod_status', '?')}")
    while True:
        choice = input(f"Select [1-{len(blocks)}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(blocks):
                return blocks[idx], idx
        except ValueError:
            pass
        print("  invalid choice")


# ---------------------------------------------------------------------------
# Field substitution
# ---------------------------------------------------------------------------

def _replace_station_field(body: str, key: str, value: str) -> str:
    pat = re.compile(
        r'^(\s*' + re.escape(key) + r'\s*=\s*)"[^"]*"(.*)$', re.MULTILINE
    )
    in_station = False
    out_lines: list[str] = []
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            in_station = (stripped == "[station]")
        if in_station:
            line = pat.sub(rf'\g<1>"{value}"\g<2>', line)
        out_lines.append(line)
    return ''.join(out_lines)


def _replace_radiod_field(body: str, index: int, key: str, value: str) -> str:
    pat = re.compile(
        r'^(\s*' + re.escape(key) + r'\s*=\s*)"[^"]*"(.*)$', re.MULTILINE
    )
    out_lines: list[str] = []
    radiod_count = -1
    in_target = False
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "[[radiod]]":
            radiod_count += 1
            in_target = (radiod_count == index)
        elif (stripped.startswith('[[') and stripped.endswith(']]')
              and stripped != "[[radiod]]"):
            in_target = False
        elif (stripped.startswith('[') and not stripped.startswith('[[')
              and not stripped.startswith('[radiod.')):
            in_target = False
        if in_target:
            line = pat.sub(rf'\g<1>"{value}"\g<2>', line)
        out_lines.append(line)
    return ''.join(out_lines)


# ---------------------------------------------------------------------------
# Prompts and UI
# ---------------------------------------------------------------------------

def _prompt(label: str, default: str, *, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(f"  {label}{suffix}: ").strip()
        except EOFError:
            raw = ""
        result = raw or default
        if result or not required:
            return result
        print("  This field is required.")


def _ok(msg: str) -> None:
    print(f"\033[32m✓\033[0m {msg}")


def _info(msg: str) -> None:
    print(f"  {msg}")


def _err(msg: str) -> None:
    print(f"\033[31m✗\033[0m {msg}", file=sys.stderr)
