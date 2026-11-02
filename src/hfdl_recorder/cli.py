"""hfdl-recorder CLI entry point.

Subcommands:
    inventory   — contract v0.4 JSON inventory
    validate    — contract v0.4 config validation
    version     — version + git block
    daemon      — long-running supervisor of per-band dumphfdl subprocesses
    status      — health check
    label-check — compare HFDL UTC labels with hf-timestd's
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path


def _resolve_log_level() -> int:
    """Resolve log level per contract v0.4 §11 precedence.

    1. --log-level CLI flag (handled by caller, not here)
    2. HFDL_RECORDER_LOG_LEVEL env var
    3. CLIENT_LOG_LEVEL env var
    4. Default: INFO
    """
    for env_key in ("HFDL_RECORDER_LOG_LEVEL", "CLIENT_LOG_LEVEL"):
        val = os.environ.get(env_key, "").upper().strip()
        if val and hasattr(logging, val):
            return getattr(logging, val)
    return logging.INFO


def _install_sighup_handler() -> None:
    """Re-read log level from env on SIGHUP (contract v0.4 §11)."""
    def _on_sighup(signum, frame):
        level = _resolve_log_level()
        logging.getLogger().setLevel(level)
        logging.getLogger(__name__).info(
            "SIGHUP: log level set to %s", logging.getLevelName(level)
        )
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _on_sighup)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hfdl-recorder",
        description="HFDL recorder for ka9q-radio (one dumphfdl per band)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    def _add_common(sub):
        sub.add_argument(
            "--config", type=Path, default=None,
            help="Path to hfdl-recorder-config.toml",
        )
        sub.add_argument(
            "--log-level", default=None,
            help="Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        )

    sub_inv = subparsers.add_parser("inventory", help="Contract v0.4 inventory")
    sub_inv.add_argument("--json", action="store_true", default=True)
    _add_common(sub_inv)

    sub_val = subparsers.add_parser("validate", help="Contract v0.4 validation")
    sub_val.add_argument("--json", action="store_true", default=True)
    _add_common(sub_val)

    sub_ver = subparsers.add_parser("version", help="Version info")
    sub_ver.add_argument("--json", action="store_true", default=True)
    _add_common(sub_ver)

    sub_daemon = subparsers.add_parser("daemon", help="Run recorder daemon")
    sub_daemon.add_argument(
        "--instance", default=None,
        help="Reporter-ID instance (loads /etc/hfdl-recorder/<instance>.toml "
             "when present; falls back to shared config otherwise). "
             "See sigmond's MULTI-INSTANCE-ARCHITECTURE.md §6.",
    )
    sub_daemon.add_argument(
        "--radiod-id", default=None,
        help="ID of the [[radiod]] block to use",
    )
    _add_common(sub_daemon)

    sub_status = subparsers.add_parser("status", help="Health check")
    _add_common(sub_status)

    # Configuration interview (CONTRACT-v0.5 §14).
    sub_cfg = subparsers.add_parser(
        "config",
        help="initialize or edit hfdl-recorder configuration",
    )
    cfg_sub = sub_cfg.add_subparsers(dest="config_command")

    sub_init = cfg_sub.add_parser(
        "init", help="write a fresh hfdl-recorder-config.toml from template")
    sub_init.add_argument("--reconfig", action="store_true",
                          help="overwrite existing config")
    sub_init.add_argument("--non-interactive", action="store_true",
                          help="use env-var defaults, do not prompt")
    _add_common(sub_init)

    sub_edit = cfg_sub.add_parser(
        "edit", help="review and update an existing config")
    sub_edit.add_argument("--non-interactive", action="store_true",
                          help="show current values, do not prompt")
    sub_edit.add_argument("--radiod-id", default=None,
                          help="focus edits on a specific [[radiod]] block")
    _add_common(sub_edit)

    # CLIENT-CONTRACT §14 JSON-roundtrip surface.  Sigmond's in-TUI
    # Textual config wizard requires `show --json` + `apply --json -`;
    # without them sigmond falls back to whiptail.  See configurator.
    sub_show = cfg_sub.add_parser(
        "show", help="emit current config (TOML→JSON) on stdout")
    sub_show.add_argument("--json", action="store_true", default=True)
    sub_show.add_argument("--defaults", action="store_true",
                          help="merge DEFAULTS into the output")
    _add_common(sub_show)

    sub_apply = cfg_sub.add_parser(
        "apply", help="apply a JSON payload (from stdin) to the config")
    sub_apply.add_argument("--json", action="store_true", default=True)
    sub_apply.add_argument("input", nargs="?", default="-",
                           help="JSON payload path or `-` for stdin (default)")
    _add_common(sub_apply)

    sub_lc = subparsers.add_parser(
        "label-check",
        help="compare an HFDL channel's UTC label with hf-timestd's",
    )
    sub_lc.add_argument("--radiod-id", default=None,
                        help="ID of the [[radiod]] block to use")
    sub_lc.add_argument("--ref-freq-hz", type=float, required=True,
                        help="frequency of the hf-timestd reference channel")
    sub_lc.add_argument("--ref-ssrc", type=int, default=None,
                        help="pick the reference channel by SSRC if several "
                             "iq channels share the frequency")
    sub_lc.add_argument("--band", required=True,
                        help="HFDL band whose sample rate the probe uses")
    sub_lc.add_argument("--seconds", type=float, default=30.0)
    sub_lc.add_argument("--repeats", type=int, default=5)
    sub_lc.add_argument("--json", action="store_true",
                        help="print one JSON object per repeat")
    sub_lc.add_argument("--out", type=Path,
                        default=Path("/var/lib/hfdl-recorder/label_check_results.toml"),
                        help="write the per-band median lag here, for the "
                             "operator; nothing reads this file (the batch "
                             "applies no correction)")
    _add_common(sub_lc)

    sub_sq = subparsers.add_parser(
        "squitter-analyze",
        help="time saved squitter windows into the local squitter_timing table",
    )
    sub_sq.add_argument("--dir", type=Path, default=None,
                        help="window directory (default: the config's "
                             "[timing_squitters] window_dir)")
    sub_sq.add_argument("--instance", default=None,
                        help="resolve the config as the daemon does "
                             "(/etc/hfdl-recorder/<instance>.toml, else the "
                             "shared config)")
    sub_sq.add_argument("--radiod-id", default=None,
                        help="analyse this radiod's windows only")
    sub_sq.add_argument("--since", default=None,
                        help="skip windows nominally before this UTC time "
                             "(ISO 8601, or Unix seconds)")
    sub_sq.add_argument("--csv", type=Path, default=None,
                        help="also write this run's rows to this CSV file")
    sub_sq.add_argument("--jsonl", type=Path, default=None,
                        help="also append the rows to this JSON-lines file")
    sub_sq.add_argument("--no-sink", action="store_true",
                        help="store rows only in --csv/--jsonl, never in "
                             "sigmond's sink (off-station use; sigmond need "
                             "not be installed)")
    sub_sq.add_argument("--registration-dir", type=Path,
                        default=Path("/var/lib/timestd/t3_shadow"),
                        help="hf-timestd T3 shadow minute rows (read only)")
    sub_sq.add_argument("--max", type=int, default=2000, dest="max_windows",
                        help="analyse at most this many windows per run")
    _add_common(sub_sq)

    return parser


def _parse_since(text):
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()


def main():
    _contract_quiet = any(
        arg in ("inventory", "validate", "version")
        for arg in sys.argv[1:3]
    )

    root = logging.getLogger()
    if _contract_quiet:
        root.setLevel(logging.WARNING)
    else:
        root.setLevel(_resolve_log_level())

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        )
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            if _contract_quiet:
                handler.setLevel(logging.WARNING)

    if not _contract_quiet:
        logging.info("hfdl-recorder starting")

    parser = build_parser()

    args = parser.parse_args()

    if args.log_level and not _contract_quiet:
        level_name = args.log_level.upper()
        if hasattr(logging, level_name):
            root.setLevel(getattr(logging, level_name))

    if args.command == "inventory":
        _handle_inventory(args)
    elif args.command == "validate":
        _handle_validate(args)
    elif args.command == "version":
        _handle_version(args)
    elif args.command == "daemon":
        _handle_daemon(args)
    elif args.command == "status":
        _handle_status(args)
    elif args.command == "config":
        _handle_config(args)
    elif args.command == "label-check":
        from hfdl_recorder.timing.label_check import run_label_check
        sys.exit(run_label_check(args))
    elif args.command == "squitter-analyze":
        sys.exit(_handle_squitter_analyze(args, parser))
    else:
        parser.print_help()
        sys.exit(1)


def _squitter_dir(args) -> Path:
    """--dir, else [timing_squitters] window_dir from the config the daemon
    reads, else the built-in default when no config exists (off-station)."""
    if args.dir is not None:
        return args.dir
    from hfdl_recorder.config import DEFAULTS, load_config, resolve_config_path
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        path = resolve_config_path(instance=args.instance, explicit_path=args.config)
    try:
        return Path(load_config(path)["timing_squitters"]["window_dir"])
    except FileNotFoundError:
        default = DEFAULTS["timing_squitters"]["window_dir"]
        logging.warning("no config at %s; window dir defaults to %s", path, default)
        return Path(default)


def _handle_squitter_analyze(args, parser) -> int:
    from hfdl_recorder.timing import batch
    if args.no_sink and not (args.csv or args.jsonl):
        parser.error("--no-sink needs --csv or --jsonl")
    counts = batch.run_batch(
        _squitter_dir(args), registration_dir=args.registration_dir,
        since=_parse_since(args.since), radiod_id=args.radiod_id,
        max_windows=args.max_windows, csv_path=args.csv, jsonl_path=args.jsonl,
        sink=not args.no_sink)
    return 1 if counts.get("stopped") else 0


def _handle_config(args):
    from hfdl_recorder import configurator

    sub = getattr(args, "config_command", None)
    if sub == "init":
        sys.exit(configurator.cmd_config_init(args))
    if sub == "edit":
        sys.exit(configurator.cmd_config_edit(args))
    if sub == "show":
        sys.exit(configurator.cmd_config_show(args))
    if sub == "apply":
        sys.exit(configurator.cmd_config_apply(args))
    print("usage: hfdl-recorder config {init|edit|show|apply}")
    sys.exit(2)


def _handle_inventory(args):
    from hfdl_recorder.config import DEFAULT_CONFIG_PATH, load_config
    from hfdl_recorder.contract import build_inventory

    config_path = args.config or Path(
        os.environ.get("HFDL_RECORDER_CONFIG", str(DEFAULT_CONFIG_PATH))
    )
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        payload = {
            "client": "hfdl-recorder",
            "version": "0.1.0",
            "contract_version": "0.8",
            "config_path": str(config_path),
            "instances": [],
            "issues": [
                {
                    "severity": "fail",
                    "instance": "all",
                    "message": f"config not found: {config_path}",
                }
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    payload = build_inventory(config, config_path)
    print(json.dumps(payload, indent=2))


def _handle_validate(args):
    from hfdl_recorder.config import DEFAULT_CONFIG_PATH, load_config
    from hfdl_recorder.contract import build_validate

    config_path = args.config or Path(
        os.environ.get("HFDL_RECORDER_CONFIG", str(DEFAULT_CONFIG_PATH))
    )
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        payload = {
            "ok": False,
            "config_path": str(config_path),
            "issues": [
                {
                    "severity": "fail",
                    "instance": "all",
                    "message": f"config not found: {config_path}",
                }
            ],
        }
        print(json.dumps(payload, indent=2))
        sys.exit(1)
        return

    payload = build_validate(config, config_path)
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        sys.exit(1)


def _handle_version(args):
    from hfdl_recorder import __version__
    from hfdl_recorder.version import GIT_INFO

    payload = {
        "client": "hfdl-recorder",
        "version": __version__,
    }
    if GIT_INFO:
        payload["git"] = GIT_INFO
    print(json.dumps(payload, indent=2))


def _handle_daemon(args):
    _install_sighup_handler()
    logger = logging.getLogger("hfdl_recorder.daemon")

    from hfdl_recorder.config import (
        extract_reporter_id, load_config, resolve_config_path,
        resolve_radiod_block,
    )
    from hfdl_recorder.core.daemon import HfdlRecorder

    # Phase-5 cutover (sigmond MULTI-INSTANCE-ARCHITECTURE.md §4)
    instance = getattr(args, "instance", None)
    config_path = resolve_config_path(
        instance=instance, explicit_path=args.config,
    )
    config = load_config(config_path)
    try:
        radiod_block = resolve_radiod_block(config, args.radiod_id)
    except ValueError:
        # Post-`smd instance migrate` soft-cutover: the systemd
        # template still passes --radiod-id %i (= reporter ID),
        # which doesn't match the [[radiod]] block's id (= mDNS
        # source label) in the per-instance config — and the
        # /etc/radio/radiod@<reporter-id>.conf synth path doesn't
        # exist either.  When --instance was given the per-instance
        # config defines this instance's source unambiguously; fall
        # through to None (single-block resolution).  Warn so the
        # mismatch stays visible.  Legacy multi-radiod shared
        # configs (no --instance) keep strict id matching.
        if args.radiod_id is None or instance is None:
            raise
        logger.warning(
            "--radiod-id=%r did not resolve in %s (--instance=%r); "
            "falling through to single-block per-instance path.  "
            "Drop --radiod-id from the systemd template once all "
            "reporters are migrated.",
            args.radiod_id, config_path, instance,
        )
        radiod_block = resolve_radiod_block(config, None)
    reporter_id = extract_reporter_id(config_path)

    logger.info(
        "Starting hfdl-recorder daemon for radiod %s "
        "(config=%s, reporter_id=%s)",
        radiod_block.get("status", "<unconfigured>"), config_path,
        reporter_id or "<derived from radiod_id at row time>",
    )

    recorder = HfdlRecorder(config, radiod_block, reporter_id=reporter_id)
    recorder.run()


def _handle_status(args):
    print("hfdl-recorder: not running (Phase 1 not yet implemented)")
    sys.exit(2)


if __name__ == "__main__":
    main()
