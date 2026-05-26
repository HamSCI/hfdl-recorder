"""Tests for `hfdl-recorder config init|edit` (CONTRACT-v0.5 §14)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hfdl_recorder import configurator


def _ns(**kwargs):
    base = dict(non_interactive=True, reconfig=False,
                config=None, radiod_id=None)
    base.update(kwargs)
    return SimpleNamespace(**base)


def _clear_env(*names):
    for n in names:
        os.environ.pop(n, None)


class StationIdSuffixTests(unittest.TestCase):
    """CONTRACT-v0.5 §14.6: hfdl-recorder station_id derives from
    STATION_CALL and SIGMOND_RADIOD_INDEX (1-based, set by the
    dispatcher).  No instance-name parsing."""

    def test_suffix_from_radiod_index(self):
        with mock.patch.dict(os.environ, {
            'STATION_CALL':         'AC0G',
            'SIGMOND_RADIOD_INDEX': '3',
        }, clear=False):
            self.assertEqual(configurator._default_station_id(), 'AC0G-3')

    def test_suffix_defaults_to_1_when_index_unset(self):
        with mock.patch.dict(os.environ, {'STATION_CALL': 'AC0G'},
                             clear=False):
            _clear_env('SIGMOND_RADIOD_INDEX')
            self.assertEqual(configurator._default_station_id(), 'AC0G-1')

    def test_instance_name_is_not_parsed(self):
        # bee1-rx888 has digits but no SIGMOND_RADIOD_INDEX → safe '1'.
        with mock.patch.dict(os.environ, {
            'STATION_CALL':     'AC0G',
            'SIGMOND_INSTANCE': 'bee1-rx888',
        }, clear=False):
            _clear_env('SIGMOND_RADIOD_INDEX')
            self.assertEqual(configurator._default_station_id(), 'AC0G-1')

    def test_empty_when_no_call(self):
        _clear_env('STATION_CALL', 'SIGMOND_RADIOD_INDEX')
        self.assertEqual(configurator._default_station_id(), '')


class FieldSubstitutionTests(unittest.TestCase):
    def test_station_field_replace_isolated(self):
        body = (
            '[station]\n'
            'station_id  = "OLD"\n'
            'grid_square = "AA00aa"\n'
            '\n'
            '[paths]\n'
            'station_id  = "NOT_TOUCHED"\n'
        )
        out = configurator._replace_station_field(body, 'station_id', 'AC0G-1')
        self.assertIn('station_id  = "AC0G-1"', out)
        self.assertIn('station_id  = "NOT_TOUCHED"', out)

    def test_radiod_field_replace_first_block(self):
        body = (
            '[[radiod]]\n'
            'id            = "old1"\n'
            'radiod_status = "old1.local"\n'
            '\n'
            '[radiod.bands]\n'
            'enabled = ["HFDL13"]\n'
            '\n'
            '[[radiod]]\n'
            'id            = "old2"\n'
        )
        out = configurator._replace_radiod_field(body, 0, 'id', 'NEW1')
        self.assertIn('id            = "NEW1"', out)
        self.assertIn('id            = "old2"', out)


class InitCommandTests(unittest.TestCase):
    def test_writes_template_using_radiod_index(self):
        # The dispatcher sets SIGMOND_RADIOD_INDEX to disambiguate radiods.
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'cfg.toml'
            args = _ns(config=target, non_interactive=True)
            with mock.patch.dict(os.environ, {
                'STATION_CALL':           'AC0G',
                'STATION_GRID':           'EM38',
                'SIGMOND_INSTANCE':       'bee1-rx888',
                'SIGMOND_RADIOD_STATUS':  'bee1-status.local',
                'SIGMOND_RADIOD_INDEX':   '3',
            }, clear=False):
                rc = configurator.cmd_config_init(args)

            self.assertEqual(rc, 0)
            text = target.read_text()
            self.assertIn('station_id  = "AC0G-3"', text)
            self.assertIn('grid_square = "EM38"', text)
            # RADIOD-IDENTIFICATION.md §3.1: canonical field is `status`.
            self.assertIn('status = "bee1-status.local"', text)

    def test_station_id_defaults_to_dash_1_when_no_index(self):
        # Standalone invocation (no sigmond) → INDEX unset → '-1'.
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'cfg.toml'
            args = _ns(config=target, non_interactive=True)
            with mock.patch.dict(os.environ, {
                'STATION_CALL':           'AC0G',
                'STATION_GRID':           'EM38',
                'SIGMOND_INSTANCE':       'bee1-rx888',
                'SIGMOND_RADIOD_STATUS':  'bee1-status.local',
            }, clear=False):
                _clear_env('SIGMOND_RADIOD_INDEX')
                rc = configurator.cmd_config_init(args)

            self.assertEqual(rc, 0)
            text = target.read_text()
            self.assertIn('station_id  = "AC0G-1"', text)
            # RADIOD-IDENTIFICATION.md §3.1: canonical field is `status`.
            self.assertIn('status = "bee1-status.local"', text)

    def test_refuses_overwrite_without_reconfig(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'cfg.toml'
            target.write_text('[station]\nstation_id = "EXISTING"\n')
            args = _ns(config=target, non_interactive=True)
            rc = configurator.cmd_config_init(args)
            self.assertEqual(rc, 1)
            self.assertIn('EXISTING', target.read_text())

    def test_safe_defaults_when_env_unset(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'cfg.toml'
            args = _ns(config=target, non_interactive=True)
            _clear_env('STATION_CALL', 'STATION_GRID',
                       'SIGMOND_INSTANCE', 'SIGMOND_RADIOD_STATUS')
            rc = configurator.cmd_config_init(args)
            self.assertEqual(rc, 0)
            text = target.read_text()
            self.assertIn('"YOURCALL-1"', text)
            self.assertIn('"my-rx888"', text)


class EditCommandTests(unittest.TestCase):
    def _initial_config(self) -> str:
        return (
            '[station]\n'
            'station_id  = "OLD-1"\n'
            'grid_square = "AA00aa"\n'
            '\n'
            '[[radiod]]\n'
            'id            = "old"\n'
            'radiod_status = "old.local"\n'
            '\n'
            '[radiod.bands]\n'
            'enabled = ["HFDL13"]\n'
        )

    def test_non_interactive_displays_only(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'cfg.toml'
            initial = self._initial_config()
            target.write_text(initial)
            args = _ns(config=target, non_interactive=True)
            rc = configurator.cmd_config_edit(args)
            self.assertEqual(rc, 0)
            self.assertEqual(target.read_text(), initial)

    def test_errors_when_target_absent(self):
        with tempfile.TemporaryDirectory() as d:
            args = _ns(config=Path(d) / 'absent.toml',
                       non_interactive=True)
            self.assertEqual(configurator.cmd_config_edit(args), 1)


if __name__ == '__main__':
    unittest.main()
