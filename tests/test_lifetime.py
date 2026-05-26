"""Tests for radiod channel-lifetime keep-alive (ka9q-python ≥3.13.0).

hfdl-recorder opts into ka9q-python / radiod's LIFETIME tag so a
crashed or killed daemon can't leave its per-band channels lingering
on radiod beyond ~`radiod_lifetime_frames / 50` seconds (≈2 min at
the default).

Surfaces under test:
  * config: ``[processing] radiod_lifetime_frames`` defaults to 6000,
    validates non-negative int, sentinel 0 = "no LIFETIME tag".
  * BandPipeline.attach: forwards ``lifetime`` to MultiStream.add_channel
    and returns the SSRC for the daemon to register.
  * keep-alive thread: refreshes every (multi, ssrc) entry at
    (frames/50/4) cadence; tolerates per-call failures.

The full provisioning path (live ka9q + radiod) is exercised by
integration smoke-tests, not here.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = str(REPO_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hfdl_recorder.config import DEFAULTS, load_config


class ConfigDefaultsTests(unittest.TestCase):

    def test_default_is_6000_frames(self):
        self.assertEqual(
            DEFAULTS["processing"]["radiod_lifetime_frames"], 6000,
        )

    def _write_config(self, body: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".toml", delete=False,
        )
        tmp.write(body)
        tmp.flush()
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink)
        return path

    def _minimal_radiod(self) -> str:
        return (
            '[[radiod]]\nstatus = "host"\n'
            '[radiod.bands]\nenabled = ["HFDL21"]\n'
        )

    def test_missing_section_falls_back_to_default(self):
        path = self._write_config(self._minimal_radiod())
        cfg = load_config(path)
        self.assertEqual(
            cfg["processing"]["radiod_lifetime_frames"], 6000,
        )

    def test_explicit_value_honored(self):
        path = self._write_config(
            '[processing]\nradiod_lifetime_frames = 3000\n'
            + self._minimal_radiod()
        )
        cfg = load_config(path)
        self.assertEqual(
            cfg["processing"]["radiod_lifetime_frames"], 3000,
        )

    def test_zero_means_no_lifetime_tag(self):
        path = self._write_config(
            '[processing]\nradiod_lifetime_frames = 0\n'
            + self._minimal_radiod()
        )
        cfg = load_config(path)
        self.assertEqual(cfg["processing"]["radiod_lifetime_frames"], 0)

    def test_negative_rejected(self):
        path = self._write_config(
            '[processing]\nradiod_lifetime_frames = -5\n'
            + self._minimal_radiod()
        )
        with self.assertRaisesRegex(ValueError, "radiod_lifetime_frames"):
            load_config(path)

    def test_non_int_rejected(self):
        path = self._write_config(
            '[processing]\nradiod_lifetime_frames = "many"\n'
            + self._minimal_radiod()
        )
        with self.assertRaisesRegex(ValueError, "radiod_lifetime_frames"):
            load_config(path)


class BandPipelineLifetimeTests(unittest.TestCase):
    """BandPipeline.attach forwards lifetime= and returns ssrc."""

    def _make_pipeline(self):
        from hfdl_recorder.core.band_pipeline import BandPipeline
        from hfdl_recorder.bands import HFDL_BANDS
        return BandPipeline(
            band=HFDL_BANDS["HFDL21"],
            radiod_id="test",
            config={"paths": {}},
        )

    def test_attach_forwards_lifetime(self):
        pipeline = self._make_pipeline()
        multi = mock.MagicMock()
        multi.add_channel.return_value = mock.Mock(ssrc=42)

        ssrc = pipeline.attach(multi, lifetime=6000)

        self.assertEqual(ssrc, 42)
        kwargs = multi.add_channel.call_args.kwargs
        self.assertEqual(kwargs["lifetime"], 6000)

    def test_attach_lifetime_none_default(self):
        pipeline = self._make_pipeline()
        multi = mock.MagicMock()
        multi.add_channel.return_value = mock.Mock(ssrc=99)

        pipeline.attach(multi)

        kwargs = multi.add_channel.call_args.kwargs
        self.assertIsNone(kwargs["lifetime"])


class _DaemonForKeepAliveTests:
    """Build an HfdlRecorder with a stub config + manually populated
    private state.  Bypasses _provision (which imports ka9q at runtime).
    """

    @staticmethod
    def make(lifetime_frames: int):
        from hfdl_recorder.core.daemon import HfdlRecorder
        cfg = {
            "paths": {}, "station": {}, "sinks": {},
            "processing": {"radiod_lifetime_frames": lifetime_frames},
        }
        radiod = {"status": "host"}
        return HfdlRecorder(cfg, radiod)


class KeepAliveLoopTests(unittest.TestCase):

    def test_no_thread_started_when_no_entries(self):
        rec = _DaemonForKeepAliveTests.make(6000)
        rec._start_lifetime_keepalive()
        self.assertIsNone(rec._lifetime_thread)

    def test_thread_refreshes_all_entries(self):
        rec = _DaemonForKeepAliveTests.make(200)
        m1, m2 = mock.MagicMock(), mock.MagicMock()
        rec._lifetime_entries = [(m1, 100), (m1, 101), (m2, 200)]
        rec._running = True

        thread = threading.Thread(
            target=rec._lifetime_loop, args=(0.05,), daemon=True,
        )
        thread.start()
        time.sleep(0.18)
        rec._running = False
        thread.join(timeout=1.0)

        m1.set_channel_lifetime.assert_any_call(100, 200)
        m1.set_channel_lifetime.assert_any_call(101, 200)
        m2.set_channel_lifetime.assert_any_call(200, 200)

    def test_failure_does_not_crash_loop(self):
        rec = _DaemonForKeepAliveTests.make(200)
        m_bad = mock.MagicMock()
        m_bad.set_channel_lifetime.side_effect = RuntimeError("radiod down")
        m_good = mock.MagicMock()
        rec._lifetime_entries = [(m_bad, 100), (m_good, 200)]
        rec._running = True

        thread = threading.Thread(
            target=rec._lifetime_loop, args=(0.05,), daemon=True,
        )
        thread.start()
        time.sleep(0.12)
        rec._running = False
        thread.join(timeout=1.0)

        m_good.set_channel_lifetime.assert_any_call(200, 200)

    def test_zero_frames_read_through(self):
        rec = _DaemonForKeepAliveTests.make(0)
        self.assertEqual(rec._radiod_lifetime_frames, 0)


if __name__ == "__main__":
    unittest.main()
