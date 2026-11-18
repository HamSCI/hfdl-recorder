"""One instance per receiver, keyed by reporter
(tasks/transporter-reporters/2026-11-18-design.md).

On a transporter (AC0G-B2 for B1A and B1B) hfdl-recorder runs one instance
per receiver, hfdl-recorder@AC0G-B1A, from /etc/hfdl-recorder/AC0G-B1A.toml:
[instance] reporter_id = "AC0G/B1A" and that receiver's one [[radiod]]
block.  Its squitter timer (hfdl-recorder-squitters@AC0G-B1A) analyses that
radiod's windows and names the rows AC0G/B1A.  A station still on the legacy
shared config keeps today's behaviour.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from types import SimpleNamespace
from unittest import mock

import pytest

from hfdl_recorder import cli, config as cfgmod, configurator
from hfdl_recorder.core import daemon as daemon_mod
from hfdl_recorder.timing import batch


def _block(status):
    return (f'[[radiod]]\nstatus = "{status}"\n'
            f'[radiod.bands]\nenabled = ["HFDL13", "HFDL8"]\n\n')


STATION = '[station]\nstation_id = "AC0G-1"\ngrid_square = "EM38ww"\n\n'


@pytest.fixture
def etc(tmp_path, monkeypatch):
    d = tmp_path / "etc"
    d.mkdir()
    radio = tmp_path / "radio"
    radio.mkdir()
    monkeypatch.setattr(cfgmod, "PER_INSTANCE_CONFIG_DIR", d)
    monkeypatch.setattr(cfgmod, "DEFAULT_CONFIG_PATH", d / "hfdl-recorder-config.toml")
    monkeypatch.setattr(configurator, "DEFAULT_CONFIG_PATH", d / "hfdl-recorder-config.toml")
    monkeypatch.setattr(cfgmod, "_RADIOD_CONF_DIR", radio)
    monkeypatch.delenv("HFDL_RECORDER_CONFIG", raising=False)
    return d


def _instance_file(etc, key, rid, status):
    p = etc / f"{key}.toml"
    p.write_text(f'[instance]\nreporter_id = "{rid}"\n\n' + STATION + _block(status))
    return p


def _shared_file(etc):
    p = etc / "hfdl-recorder-config.toml"
    p.write_text(STATION + _block("b1a-status.local") + _block("b1b-status.local"))
    return p


# --- config resolution --------------------------------------------------------

def test_reporter_instance_resolves_its_own_file(etc):
    p = _instance_file(etc, "AC0G-B1A", "AC0G/B1A", "b1a-status.local")
    assert cfgmod.resolve_config_path(instance="AC0G-B1A") == p
    assert cfgmod.resolve_config_path(instance="AC0G\\x3dB1A") == p


def test_per_instance_names_skip_the_shared_file(etc):
    _shared_file(etc)
    _instance_file(etc, "AC0G-B1A", "AC0G/B1A", "b1a-status.local")
    _instance_file(etc, "AC0G-B1B", "AC0G/B1B", "b1b-status.local")
    assert cfgmod.per_instance_names_from_dir(str(etc)) == ["AC0G-B1A", "AC0G-B1B"]


# --- daemon -------------------------------------------------------------------

def _run_daemon(argv, monkeypatch):
    seen = {}

    class FakeRecorder:
        def __init__(self, config, radiod_block, reporter_id=None):
            seen.update(block=radiod_block, reporter_id=reporter_id)

        def run(self):
            pass

    monkeypatch.setattr(daemon_mod, "HfdlRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "_install_sighup_handler", lambda: None)
    monkeypatch.setattr(sys, "argv", ["hfdl-recorder", *argv])
    cli.main()
    return seen


def test_daemon_for_a_reporter_instance(etc, tmp_path, monkeypatch):
    _shared_file(etc)
    _instance_file(etc, "AC0G-B1A", "AC0G/B1A", "b1a-status.local")
    # A radiod conf named like the key must not replace the file's block.
    (tmp_path / "radio" / "radiod@AC0G-B1A.conf").write_text(
        "[global]\nstatus = other-status.local\n")
    seen = _run_daemon(["daemon", "--instance", "AC0G-B1A",
                        "--radiod-id", "AC0G-B1A"], monkeypatch)
    assert seen["reporter_id"] == "AC0G/B1A"
    assert seen["block"]["status"] == "b1a-status.local"
    assert seen["block"]["bands"]["enabled"] == ["HFDL13", "HFDL8"]


def test_daemon_legacy_radiod_instance_unchanged(etc, monkeypatch):
    _shared_file(etc)
    with pytest.warns(DeprecationWarning):
        seen = _run_daemon(["daemon", "--instance", "b1b-status.local",
                            "--radiod-id", "b1b-status.local"], monkeypatch)
    assert seen["reporter_id"] is None
    assert seen["block"]["status"] == "b1b-status.local"


# --- squitter timer -----------------------------------------------------------

def _run_squitters(argv, monkeypatch):
    seen = {}

    def fake(window_dir, **kw):
        seen.update(kw, window_dir=window_dir)
        return {"stopped": False}

    monkeypatch.setattr(batch, "run_batch", fake)
    monkeypatch.setattr(sys, "argv", ["hfdl-recorder", "squitter-analyze", *argv])
    with pytest.raises(SystemExit):
        cli.main()
    return seen


def test_squitter_timer_follows_the_reporter_instance(etc, monkeypatch):
    _instance_file(etc, "AC0G-B1A", "AC0G/B1A", "b1a-status.local")
    seen = _run_squitters(["--instance", "AC0G-B1A", "--radiod-id", "AC0G-B1A"],
                          monkeypatch)
    assert seen["reporter_id"] == "AC0G/B1A"
    # The window keeper files windows under the radiod's status name.
    assert seen["radiod_id"] == "b1a-status.local"


def test_squitter_timer_legacy_unchanged(etc, monkeypatch):
    _shared_file(etc)
    seen = _run_squitters(["--instance", "b1b-status.local",
                           "--radiod-id", "b1b-status.local"], monkeypatch)
    assert seen["reporter_id"] is None
    assert seen["radiod_id"] == "b1b-status.local"


# --- inventory / validate -----------------------------------------------------

def _contract(argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["hfdl-recorder", *argv])
    try:
        cli.main()
    except SystemExit:
        pass
    return json.loads(capsys.readouterr().out)


def test_inventory_for_one_instance(etc, monkeypatch, capsys):
    _instance_file(etc, "AC0G-B1A", "AC0G/B1A", "b1a-status.local")
    inv = _contract(["inventory", "--json", "--instance", "AC0G-B1A"], monkeypatch, capsys)
    assert [(i["instance"], i["reporter_id"], i["radiod_id"]) for i in inv["instances"]] == [
        ("AC0G-B1A", "AC0G/B1A", "b1a-status.local")]


def test_migrated_station_inventories_every_receiver(etc, monkeypatch, capsys):
    _instance_file(etc, "AC0G-B1A", "AC0G/B1A", "b1a-status.local")
    _instance_file(etc, "AC0G-B1B", "AC0G/B1B", "b1b-status.local")
    inv = _contract(["inventory", "--json"], monkeypatch, capsys)
    assert {(i["instance"], i["radiod_id"]) for i in inv["instances"]} == {
        ("AC0G-B1A", "b1a-status.local"), ("AC0G-B1B", "b1b-status.local")}
    assert len(inv["config_paths"]) == 2


def test_legacy_inventory_reads_the_shared_config(etc, monkeypatch, capsys):
    shared = _shared_file(etc)
    _instance_file(etc, "AC0G-B1A", "AC0G/B1A", "b1a-status.local")
    inv = _contract(["inventory", "--json"], monkeypatch, capsys)
    assert inv["config_path"] == str(shared)
    assert [i["instance"] for i in inv["instances"]] == [
        "b1a-status.local", "b1b-status.local"]
    assert {i["reporter_id"] for i in inv["instances"]} == {None}


# --- config init for one receiver ---------------------------------------------

SIGMOND_ENV = {"SIGMOND_RADIOD_COUNT": "2", "STATION_CALL": "AC0G",
               "STATION_GRID": "EM38ww", "STATION_REPORTER_ID": "AC0G/B2",
               "RADIOD_B1A_STATUS": "b1a-status.local",
               "RADIOD_B1B_STATUS": "b1b-status.local"}


def _init(tmp_path, env, **kw):
    args = SimpleNamespace(non_interactive=True, reconfig=False, config=None,
                           radiod_id=None, instance=None, reporter_id=None,
                           radiod_status=None)
    for k, v in kw.items():
        setattr(args, k, v)
    base = {k: v for k, v in os.environ.items()
            if not k.startswith(("SIGMOND_", "STATION_", "RADIOD_"))}
    with mock.patch.dict(os.environ, {**base, **env}, clear=True), \
         mock.patch.object(configurator, "_discover_radiods", return_value=[]), \
         mock.patch.object(configurator, "COORDINATION_ENV", tmp_path / "absent.env"):
        return configurator.cmd_config_init(args)


def test_init_writes_the_instance_block_and_only_its_radiod(etc, tmp_path):
    rc = _init(tmp_path, {**SIGMOND_ENV, "SIGMOND_INSTANCE": "AC0G-B1A",
                          "SIGMOND_RADIOD_STATUS": "b1a-status.local"},
               reporter_id="AC0G/B1A")
    assert rc == 0
    cfg = tomllib.loads((etc / "AC0G-B1A.toml").read_text())
    assert cfg["instance"]["reporter_id"] == "AC0G/B1A"
    blocks = cfg["radiod"] if isinstance(cfg["radiod"], list) else [cfg["radiod"]]
    assert [b["status"] for b in blocks] == ["b1a-status.local"]
    assert not (etc / "hfdl-recorder-config.toml").exists()


def test_init_takes_station_reporter_when_its_key_is_the_instance(etc, tmp_path):
    rc = _init(tmp_path, {**SIGMOND_ENV, "STATION_REPORTER_ID": "AC0G/B4"},
               instance="AC0G-B4", radiod_status="b4-status.local")
    assert rc == 0
    cfg = tomllib.loads((etc / "AC0G-B4.toml").read_text())
    assert cfg["instance"]["reporter_id"] == "AC0G/B4"


def test_init_refuses_the_transporters_reporter_for_a_receiver(etc, tmp_path):
    assert _init(tmp_path, SIGMOND_ENV, instance="AC0G-B1A",
                 radiod_status="b1a-status.local") == 1
    assert not (etc / "AC0G-B1A.toml").exists()


def test_init_refuses_without_the_receivers_radiod(etc, tmp_path):
    assert _init(tmp_path, SIGMOND_ENV, instance="AC0G-B1A",
                 reporter_id="AC0G/B1A") == 1


def test_legacy_init_without_an_instance_is_unchanged(etc, tmp_path):
    rc = _init(tmp_path, SIGMOND_ENV)
    assert rc == 0
    cfg = tomllib.loads((etc / "hfdl-recorder-config.toml").read_text())
    assert "instance" not in cfg
    blocks = cfg["radiod"] if isinstance(cfg["radiod"], list) else [cfg["radiod"]]
    assert [b["status"] for b in blocks] == ["b1a-status.local", "b1b-status.local"]


def test_cli_parses_the_instance_arguments(monkeypatch):
    seen = {}
    monkeypatch.setattr(configurator, "cmd_config_init",
                        lambda a: seen.update(vars(a)) or 0)
    monkeypatch.setattr(sys, "argv", ["hfdl-recorder", "config", "init",
                                      "--non-interactive", "--instance", "AC0G-B1A",
                                      "--reporter-id", "AC0G/B1A",
                                      "--radiod-status", "b1a-status.local"])
    with pytest.raises(SystemExit):
        cli.main()
    assert (seen["instance"], seen["reporter_id"], seen["radiod_status"]) == (
        "AC0G-B1A", "AC0G/B1A", "b1a-status.local")
