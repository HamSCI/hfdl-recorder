"""Instance naming for install.sh: one hfdl-recorder@<status> per block."""

from __future__ import annotations

import subprocess
import sys

from hfdl_recorder.config import radiod_instance_names


def test_status_names_instance():
    cfg = {"radiod": [{"status": "N8UR-status.local"}]}
    assert radiod_instance_names(cfg) == ["N8UR-status.local"]


def test_status_wins_over_legacy_id():
    cfg = {"radiod": [{"status": "bee1-status.local", "id": "old"}]}
    assert radiod_instance_names(cfg) == ["bee1-status.local"]


def test_legacy_id_only():
    assert radiod_instance_names({"radiod": [{"id": "bee1"}]}) == ["bee1"]


def test_placeholder_and_empty_skipped():
    cfg = {"radiod": [{"status": "<configure-via-config-init>"},
                      {"status": ""}, {"status": "  "}, {}]}
    assert radiod_instance_names(cfg) == []


def test_no_blocks():
    assert radiod_instance_names({}) == []


def test_single_dict_block():
    assert radiod_instance_names({"radiod": {"status": "a-status.local"}}) == [
        "a-status.local"]


def test_never_default_and_dedup():
    cfg = {"radiod": [{"status": "a.local"}, {"status": "a.local"}, {}]}
    names = radiod_instance_names(cfg)
    assert names == ["a.local"]
    assert "default" not in names


def test_cli_prints_names_from_file(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[radiod]]\nstatus = "x-status.local"\n'
                 '[[radiod]]\nstatus = "<configure-via-config-init>"\n')
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; from hfdl_recorder.config import "
         "radiod_instance_names_from_file as f; "
         "print('\\n'.join(f(sys.argv[1])))", str(p)],
        capture_output=True, text=True, check=True).stdout
    assert out.split() == ["x-status.local"]
