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
    _info(f"radiod:     status={values['radiod_status']}")
    _info("")
    _info("Next steps:")
    _info(f"  1. Review [radiod.bands].enabled in {target}")
    _info(f"  2. If feeding airframes.io, register {values['station_id']!r} "
          f"at https://app.airframes.io")
    _info(f"  3. Validate: hfdl-recorder validate --json")
    _info(f"  4. Start:    sudo systemctl enable --now "
          f"hfdl-recorder@{values['radiod_status']}.service")
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
    cur_status = block.get("status", "")

    if getattr(args, "non_interactive", False):
        _info(f"station.station_id    = {cur_station}")
        _info(f"station.grid_square   = {cur_grid}")
        _info(f"radiod[{block_index}].status = {cur_status}")
        return 0

    new_station = _prompt("station_id (airframes.io registration)",
                          cur_station or _default_station_id())
    new_grid    = _prompt("Grid square",
                          cur_grid or os.environ.get("STATION_GRID", ""))
    new_status  = _prompt("Radiod status DNS",
                          cur_status or
                          os.environ.get("SIGMOND_RADIOD_STATUS", ""))

    body = target.read_text()
    body = _replace_station_field(body, "station_id",  new_station)
    body = _replace_station_field(body, "grid_square", new_grid)
    body = _replace_radiod_field(body, block_index, "status", new_status)

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
            "radiod_status": radiod_status,
        }

    station_id  = _prompt("station_id (airframes.io registration)",
                          default_station, required=True)
    grid_square = _prompt("Grid square", grid, required=True)

    # RADIOD-IDENTIFICATION.md §4 — discovery-driven radiod selection.
    discovered = _discover_radiods()
    radiod_status = _pick_radiod_status_from_discovery(
        discovered, status, instance)

    return {
        "station_id":    station_id,
        "grid":          grid_square,
        "radiod_status": radiod_status,
    }


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
            if b.get("status") == target_id:
                return b, i
        _err(f"no [[radiod]] block with status={target_id!r}; "
             f"available: {', '.join(b.get('status', '?') for b in blocks)}")
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


# ---------------------------------------------------------------------------
# CLIENT-CONTRACT §14 — JSON config-roundtrip surface.
#
# `hfdl-recorder config show --json [--defaults]`   reads the TOML
#   file on disk and emits it as JSON on stdout.  With `--defaults`,
#   DEFAULTS is deep-merged into the file's content so every default
#   key is present (the wizard uses this to populate every form
#   field on a freshly-installed host).
#
# `hfdl-recorder config apply --json -`   reads a JSON dict from
#   stdin, deep-merges it into the existing TOML file, and atomically
#   rewrites the file.  Only sections in _APPLY_ALLOWED_SECTIONS are
#   accepted; payload type-checking is structural only.  Comments
#   and source ordering are NOT preserved — the file is rewritten
#   from the merged dict.
#
# Pattern lifted from wspr-recorder commit ad8f637 and psk-recorder's
# original implementation.  Difference vs wspr-recorder: `--defaults`
# is functional here (hfdl-recorder has a canonical DEFAULTS dict in
# config.py covering [paths], [sinks], [processing]).
# ---------------------------------------------------------------------------

import copy
import json
import tempfile

from .config import DEFAULTS, DEFAULT_CONFIG_PATH


# Sections allowed in the apply payload.  Matches hfdl-recorder's
# actual schema: [station], [paths], [sinks], [processing], plus
# `[[radiod]]` array-of-tables and the `[instance]` block prepended by
# `smd instance migrate`.  Anything outside this set is rejected to
# protect the file from typos and future schema additions reaching
# disk without explicit review.
_APPLY_ALLOWED_SECTIONS = {
    "instance", "station", "paths", "sinks",
    "processing", "radiod",
}


def cmd_config_show(args) -> int:
    """Emit the on-disk TOML (or DEFAULTS-merged) as JSON on stdout."""
    config_path = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
    if getattr(args, "defaults", False):
        # DEFAULTS as the base, then the file's content overlays it.
        # Sigmond's wizard uses this on first-run / freshly-installed
        # hosts where the file may not yet have every key but the form
        # should still render with sensible placeholders.
        merged = copy.deepcopy(DEFAULTS)
        if config_path.is_file():
            try:
                with open(config_path, "rb") as f:
                    file_data = tomllib.load(f)
                merged = _deep_merge(merged, file_data)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                print(f"config show: cannot read {config_path}: {exc}",
                      file=sys.stderr)
                return 2
        out = merged
    else:
        if not config_path.is_file():
            out = {}
        else:
            try:
                with open(config_path, "rb") as f:
                    out = tomllib.load(f)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                print(f"config show: cannot read {config_path}: {exc}",
                      file=sys.stderr)
                return 2
    json.dump(out, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0


def cmd_config_apply(args) -> int:
    """Read a JSON dict on stdin, validate, atomically write the TOML.

    Section whitelist + structural type checks (each section must be
    a table, except `radiod` which is a list of tables).  No per-key
    type enforcement — sigmond's wizard owns input typing on its end.
    """
    config_path = Path(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"config apply: stdin is not valid JSON: {exc}",
              file=sys.stderr)
        return 2

    if not isinstance(payload, dict):
        print(f"config apply: top-level JSON must be an object, "
              f"got {type(payload).__name__}", file=sys.stderr)
        return 2

    unknown = set(payload.keys()) - _APPLY_ALLOWED_SECTIONS
    if unknown:
        print(f"config apply: section(s) not writable via apply: "
              f"{sorted(unknown)} "
              f"(allowed: {sorted(_APPLY_ALLOWED_SECTIONS)})",
              file=sys.stderr)
        return 2

    for section, fields in payload.items():
        if section == "radiod":
            # Array-of-tables: structural check only.  The wizard pilot
            # doesn't yet edit `[[radiod]]` for hfdl-recorder (that's
            # the multi-band per-radiod-instance shape — a follow-up).
            if not isinstance(fields, list):
                print(f"config apply: [[radiod]] must be a list, "
                      f"got {type(fields).__name__}", file=sys.stderr)
                return 2
            continue
        if not isinstance(fields, dict):
            print(f"config apply: [{section}] must be a table, "
                  f"got {type(fields).__name__}", file=sys.stderr)
            return 2

    # Deep-merge with existing file.
    if config_path.is_file():
        with open(config_path, "rb") as f:
            existing = tomllib.load(f)
    else:
        existing = {}
    merged = _deep_merge(existing, payload)

    text = _serialize_toml(merged)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(config_path.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    try:
        tmp.chmod(0o644)
    except PermissionError:
        pass
    tmp.replace(config_path)
    print(f"wrote {config_path}")
    return 0


# ---------------------------------------------------------------------------
# Helpers (deep_merge + minimal TOML serializer).  Identical to the
# wspr-recorder commit ad8f637 versions — could be lifted to a shared
# sigmond library in a future cleanup, but inlined here per-client for
# install-script self-containedness.
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict where overlay's keys win over base's.

    Nested dicts merge recursively; lists and scalars overwrite.
    """
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = repr(v)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise TypeError(f"unsupported TOML scalar type: {type(v).__name__}")


def _toml_inline_array(arr: list) -> str:
    parts = []
    for x in arr:
        if isinstance(x, (str, bool, int, float)):
            parts.append(_toml_scalar(x))
        else:
            parts.append(json.dumps(x))
    return "[" + ", ".join(parts) + "]"


def _serialize_toml(d: dict, parent: str = "") -> str:
    """Serialize ``d`` to a deterministic TOML string.

    Handles scalars, nested dicts (rendered as ``[section.child]``),
    and arrays-of-tables (rendered as ``[[section]]``).  Arrays of
    scalars render inline.  Does NOT preserve comments or original
    ordering — keys are sorted within each section for determinism.
    """
    lines: list[str] = []
    scalars: list[tuple[str, object]] = []
    nested: list[tuple[str, dict]] = []
    array_of_tables: list[tuple[str, list]] = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            nested.append((k, v))
        elif (isinstance(v, list) and v
              and all(isinstance(item, dict) for item in v)):
            array_of_tables.append((k, v))
        else:
            scalars.append((k, v))
    if scalars:
        if parent:
            lines.append(f"[{parent}]")
        for k, v in scalars:
            if isinstance(v, list):
                lines.append(f"{k} = {_toml_inline_array(v)}")
            else:
                lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
    for k, sub in nested:
        header = f"{parent}.{k}" if parent else k
        lines.append(_serialize_toml(sub, parent=header))
    for k, blocks in array_of_tables:
        header = f"{parent}.{k}" if parent else k
        for block in blocks:
            lines.append(f"[[{header}]]")
            for bk in sorted(block.keys()):
                bv = block[bk]
                if isinstance(bv, dict):
                    lines.append(_serialize_toml({bk: bv}, parent=header))
                elif isinstance(bv, list):
                    lines.append(f"{bk} = {_toml_inline_array(bv)}")
                else:
                    lines.append(f"{bk} = {_toml_scalar(bv)}")
            lines.append("")
    return "\n".join(lines)
