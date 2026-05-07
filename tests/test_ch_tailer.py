"""Tests for hfdl_recorder.core.ch_tailer (CONTRACT v0.6 §17 wiring)."""
from __future__ import annotations

import json
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hfdl_recorder.core.ch_tailer import (
    ChTailer,
    parse_hfdl_frame,
    _icao_to_int,
)


# ── Realistic frame fixtures (libacars JSON shape) ─────────────────────────

# Squitter (ground station beacon) — minimal but plausible.
SPDU_FRAME = {
    "hfdl": {
        "app":         {"name": "dumphfdl", "ver": "1.4.1"},
        "station":     "AC0G-EM38ww",
        "t":           {"sec": 1746628496, "usec": 123_000},
        "freq":        21934000,
        "bit_rate":    1800,
        "sig_level":   -75.5,
        "noise_level": -110.2,
        "freq_skew":   0.5,
        "slot":        "A",
        "spdu": {
            "src":    {"type": "ground", "id": 11, "name": "Krasnoyarsk"},
            "gs":     {"id": 11, "name": "Krasnoyarsk"},
            "version": 0,
        },
    }
}

# Aircraft → ground (uplink direction tag).  ACARS body present.
LPDU_ACARS_FRAME = {
    "hfdl": {
        "app":         {"name": "dumphfdl", "ver": "1.4.1"},
        "station":     "AC0G-EM38ww",
        "t":           {"sec": 1746628500, "usec": 0},
        "freq":        13321000,
        "bit_rate":    1800,
        "sig_level":   -82.0,
        "noise_level": -113.0,
        "freq_skew":   -0.3,
        "slot":        "B",
        "lpdu": {
            "direction": "downlink",
            "src":       {"type": "ground", "id": 4, "name": "Reykjavik"},
            "dst":       {"type": "aircraft", "icao": "ABC123"},
            "ac_info":   {"icao": "0xABC123", "tail": "N12345",
                          "operator": "Acme Air", "type": "B738"},
            "acars": {
                "label":    "DM",
                "flight_id": "ACA855",
                "msg_text": "POS:KORD/...",
            },
            "hfnpdu": {
                "position": {"lat": 41.97, "lon": -87.90, "alt": 35000},
            },
        },
    }
}

NON_HFDL_LINE  = '{"some_other": "object"}'
GARBAGE_LINE   = "this is not json"
EMPTY_LINE     = ""

LINE_SPDU       = json.dumps(SPDU_FRAME)
LINE_LPDU_ACARS = json.dumps(LPDU_ACARS_FRAME)


class TestIcaoConversion(unittest.TestCase):

    def test_int_passthrough(self):
        self.assertEqual(_icao_to_int(0xABC123), 0xABC123)

    def test_hex_string_with_prefix(self):
        self.assertEqual(_icao_to_int("0xABC123"), 0xABC123)

    def test_hex_string_no_prefix(self):
        self.assertEqual(_icao_to_int("abc123"), 0xABC123)

    def test_decimal_string(self):
        self.assertEqual(_icao_to_int("11258179"), 11258179)

    def test_none_returns_none(self):
        self.assertIsNone(_icao_to_int(None))
        self.assertIsNone(_icao_to_int(""))

    def test_invalid_returns_none(self):
        self.assertIsNone(_icao_to_int("not-hex"))

    def test_out_of_range_returns_none(self):
        self.assertIsNone(_icao_to_int(1 << 25))


class TestParseHfdlFrame(unittest.TestCase):

    def test_spdu_frame_round_trips_metadata(self):
        row = parse_hfdl_frame(LINE_SPDU, band_name="HFDL21")
        self.assertIsNotNone(row)
        self.assertEqual(row["band_name"], "HFDL21")
        self.assertEqual(row["station_id"], "AC0G-EM38ww")
        self.assertEqual(row["frequency"], 21_934_000)
        self.assertAlmostEqual(row["frequency_mhz"], 21.934, places=3)
        self.assertEqual(row["bit_rate"], 1800)
        self.assertAlmostEqual(row["sig_level"], -75.5, places=2)
        self.assertEqual(row["slot"], "A")
        self.assertEqual(row["direction"], "downlink")     # spdu == ground
        self.assertEqual(row["ground_station"], "Krasnoyarsk")
        # 1746628496 → 2025-05-07 14:34:56 UTC
        self.assertEqual(row["time"],
                         datetime(2025, 5, 7, 14, 34, 56, 123_000))
        self.assertEqual(row["raw_json"], LINE_SPDU)
        # ACARS / position fields absent for a squitter.
        self.assertEqual(row["acars_label"], "")
        self.assertIsNone(row["position_lat"])

    def test_lpdu_acars_frame_extracts_aircraft_fields(self):
        row = parse_hfdl_frame(LINE_LPDU_ACARS, band_name="HFDL13")
        self.assertIsNotNone(row)
        self.assertEqual(row["band_name"], "HFDL13")
        self.assertEqual(row["direction"], "downlink")
        self.assertEqual(row["ground_station"], "Reykjavik")
        self.assertEqual(row["icao_addr"], 0xABC123)
        self.assertEqual(row["aircraft_reg"], "N12345")
        self.assertEqual(row["acars_label"], "DM")
        self.assertEqual(row["flight"], "ACA855")
        self.assertEqual(row["acars_message"], "POS:KORD/...")
        # Position extracted.
        self.assertAlmostEqual(row["position_lat"], 41.97, places=2)
        self.assertAlmostEqual(row["position_lon"], -87.90, places=2)
        self.assertEqual(row["position_alt_ft"], 35000)

    def test_non_hfdl_object_rejected(self):
        self.assertIsNone(parse_hfdl_frame(NON_HFDL_LINE, band_name="HFDL21"))

    def test_garbage_line_rejected(self):
        self.assertIsNone(parse_hfdl_frame(GARBAGE_LINE, band_name="HFDL21"))

    def test_empty_line_rejected(self):
        self.assertIsNone(parse_hfdl_frame(EMPTY_LINE, band_name="HFDL21"))

    def test_missing_timestamp_rejected(self):
        broken = json.dumps({"hfdl": {"freq": 1000, "t": {"sec": 0}}})
        self.assertIsNone(parse_hfdl_frame(broken, band_name="HFDL21"))


# ── Tailer with fake writer ─────────────────────────────────────────────────

class FakeWriter:
    def __init__(self, noop=False):
        self._noop = noop
        self.health = "noop" if noop else "ok"
        self.inserts: list = []
        self.flushed = 0
        self.closed = False

    @property
    def is_noop(self):
        return self._noop

    def insert(self, rows):
        self.inserts.extend(rows)

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed = True


class TestChTailer(unittest.TestCase):

    def _make_tailer(self, json_path: Path, *, noop=False):
        fake = FakeWriter(noop=noop)
        tailer = ChTailer(
            json_path=json_path,
            band_name="HFDL21",
            radiod_id="test-rx888",
            host_call="AC0G",
            host_grid="EM38ww",
            processing_version="0.1.0+abc",
            writer_factory=lambda batch_rows: fake,
        )
        return tailer, fake

    def test_skips_history_at_startup(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "HFDL21.json"
            jp.write_text(LINE_SPDU + "\n")            # pre-existing
            tailer, fake = self._make_tailer(jp)
            tailer.start()
            try:
                time.sleep(1.5)
            finally:
                tailer.stop(timeout=2.0)
            self.assertEqual(fake.inserts, [])

    def test_consumes_appended_frames(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "HFDL21.json"
            jp.write_text("")
            tailer, fake = self._make_tailer(jp)
            tailer.start()
            try:
                with open(jp, "a") as f:
                    f.write(LINE_SPDU + "\n")
                    f.write(LINE_LPDU_ACARS + "\n")
                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline and len(fake.inserts) < 2:
                    time.sleep(0.1)
            finally:
                tailer.stop(timeout=2.0)
            self.assertEqual(len(fake.inserts), 2)
            for row in fake.inserts:
                self.assertEqual(row["host_call"], "AC0G")
                self.assertEqual(row["host_grid"], "EM38ww")
                self.assertEqual(row["radiod_id"], "test-rx888")
                self.assertEqual(row["processing_version"], "0.1.0+abc")

    def test_handles_partial_line_split_across_reads(self):
        """A line written half-now / half-later must still parse once whole."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "HFDL21.json"
            jp.write_text("")
            tailer, fake = self._make_tailer(jp)
            tailer.start()
            try:
                # Write the line in two chunks across two poll cycles.
                half = len(LINE_SPDU) // 2
                with open(jp, "a") as f:
                    f.write(LINE_SPDU[:half])
                time.sleep(1.5)
                self.assertEqual(fake.inserts, [],
                                 "partial line should not parse yet")
                with open(jp, "a") as f:
                    f.write(LINE_SPDU[half:] + "\n")
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and len(fake.inserts) < 1:
                    time.sleep(0.1)
            finally:
                tailer.stop(timeout=2.0)
            self.assertEqual(len(fake.inserts), 1)
            self.assertEqual(fake.inserts[0]["station_id"], "AC0G-EM38ww")

    def test_skips_unparseable_lines_silently(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "HFDL21.json"
            jp.write_text("")
            tailer, fake = self._make_tailer(jp)
            tailer.start()
            try:
                with open(jp, "a") as f:
                    f.write(GARBAGE_LINE + "\n")
                    f.write(NON_HFDL_LINE + "\n")
                    f.write(LINE_SPDU + "\n")
                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline and len(fake.inserts) < 1:
                    time.sleep(0.1)
            finally:
                tailer.stop(timeout=2.0)
            self.assertEqual(len(fake.inserts), 1,
                             "garbage and non-HFDL lines should be silently skipped")


if __name__ == "__main__":
    unittest.main()
