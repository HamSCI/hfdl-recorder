"""Daily UTC rotation of dumphfdl's per-band JSON spool files."""

from __future__ import annotations

import gzip
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from hfdl_recorder.core.ch_tailer import ChTailer
from hfdl_recorder.core.spool_rotation import SpoolRotator

from tests.test_ch_tailer import LINE_SPDU

NOW = datetime(2026, 11, 13, 0, 5, tzinfo=timezone.utc)
YESTERDAY = datetime(2026, 11, 12, 23, 59, tzinfo=timezone.utc)


def _spool(tmp_path, band="HFDL21", text="a\nb\n", when=YESTERDAY):
    d = tmp_path / "rid"
    d.mkdir(exist_ok=True)
    p = d / f"{band}.json"
    p.write_text(text)
    os.utime(p, (when.timestamp(), when.timestamp()))
    return p


def _rot(tmp_path, keep=3, now=NOW, bands=("HFDL21",)):
    return SpoolRotator(tmp_path / "rid", list(bands), keep_days=keep,
                        now_fn=lambda: now)


def test_rotates_yesterdays_file(tmp_path):
    p = _spool(tmp_path)
    _rot(tmp_path).rotate_once()
    gz = tmp_path / "rid" / "HFDL21.json.20261112.gz"
    assert gzip.decompress(gz.read_bytes()) == b"a\nb\n"
    assert p.stat().st_size == 0
    with open(p, "a") as f:                      # dumphfdl appends (O_APPEND)
        f.write("c\n")
    assert p.read_text() == "c\n"


def test_empty_file_untouched(tmp_path):
    p = _spool(tmp_path, text="")
    _rot(tmp_path).rotate_once()
    assert list(p.parent.glob("*.gz")) == []


def test_already_rotated_today_is_noop(tmp_path):
    p = _spool(tmp_path)
    r = _rot(tmp_path)
    r.rotate_once()
    with open(p, "a") as f:
        f.write("today\n")
    r.rotate_once()
    assert p.read_text() == "today\n"
    assert len(list(p.parent.glob("*.gz"))) == 1


def test_todays_content_is_not_rotated(tmp_path):
    p = _spool(tmp_path, when=NOW)
    _rot(tmp_path).rotate_once()
    assert p.read_text() == "a\nb\n"


def test_keeps_newest_n_per_band(tmp_path):
    p = _spool(tmp_path)
    for day in ("20261105", "20261106", "20261107", "20261108"):
        (p.parent / f"HFDL21.json.{day}.gz").write_bytes(b"x")
    (p.parent / "HFDL5.json.20261101.gz").write_bytes(b"x")   # other band
    _rot(tmp_path, keep=3).rotate_once()
    left = sorted(f.name for f in p.parent.glob("HFDL21.json.*.gz"))
    assert left == ["HFDL21.json.20261107.gz", "HFDL21.json.20261108.gz",
                    "HFDL21.json.20261112.gz"]
    assert (p.parent / "HFDL5.json.20261101.gz").exists()


def test_error_logs_and_continues(tmp_path, caplog, monkeypatch):
    _spool(tmp_path, "HFDL21")
    p2 = _spool(tmp_path, "HFDL13")
    real = gzip.GzipFile

    def boom(*a, **kw):
        if "HFDL21" in str(kw.get("filename", "")) or \
                "HFDL21" in str(getattr(kw.get("fileobj"), "name", "")):
            raise OSError("disk full")
        return real(*a, **kw)

    monkeypatch.setattr(gzip, "GzipFile", boom)
    with caplog.at_level(logging.WARNING):
        _rot(tmp_path, bands=("HFDL21", "HFDL13")).rotate_once()
    assert any("HFDL21" in r.message and r.levelno == logging.WARNING
               for r in caplog.records)
    assert p2.stat().st_size == 0                # next band still rotated
    assert (tmp_path / "rid" / "HFDL21.json").read_text() == "a\nb\n"


def test_missing_file_is_fine(tmp_path):
    (tmp_path / "rid").mkdir()
    _rot(tmp_path).rotate_once()


def test_tailer_follows_a_truncation_without_rereading(tmp_path):
    p = _spool(tmp_path, text="", when=NOW)
    got = []
    t = ChTailer(json_path=p, band_name="HFDL21", radiod_id="rid",
                 on_squitter=got.append, stage_spots=False)
    t.start()
    t.stop()                                     # drive polls by hand
    with open(p, "a") as f:
        f.write(LINE_SPDU + "\n" + LINE_SPDU + "\n")
    t._poll_once()
    assert len(got) == 2
    os.utime(p, (YESTERDAY.timestamp(), YESTERDAY.timestamp()))
    _rot(tmp_path).rotate_once()
    t._poll_once()                               # sees the shrink
    assert len(got) == 2                         # nothing re-read
    with open(p, "a") as f:
        f.write(LINE_SPDU + "\n")
    t._poll_once()
    assert len(got) == 3                         # new frame not skipped


def _line(when):
    return json.dumps({"hfdl": {"t": {"sec": int(when.timestamp()),
                                      "usec": 0}}}) + "\n"


def test_first_line_date_beats_todays_mtime(tmp_path):
    # Busy band: mtime is already today, content starts yesterday.
    p = _spool(tmp_path, text=_line(YESTERDAY) + _line(NOW), when=NOW)
    _rot(tmp_path).rotate_once()
    gz = tmp_path / "rid" / "HFDL21.json.20261112.gz"
    assert b"\n" in gzip.decompress(gz.read_bytes())
    assert p.stat().st_size == 0


def test_file_written_across_midnight_rotates_once(tmp_path):
    clock = [datetime(2026, 11, 12, 20, 0, tzinfo=timezone.utc)]
    p = _spool(tmp_path, text=_line(clock[0]), when=clock[0])
    r = SpoolRotator(tmp_path / "rid", ["HFDL21"], keep_days=3,
                     now_fn=lambda: clock[0])
    r.rotate_once()                       # day 1: nothing to rotate
    assert not list(p.parent.glob("*.gz"))
    for hh in (23, 24, 25, 26):           # a frame every hour, checks too
        clock[0] = datetime(2026, 11, 12, 0, 0, tzinfo=timezone.utc) \
            .replace(day=12 + hh // 24, hour=hh % 24, minute=1)
        with open(p, "a") as f:
            f.write(_line(clock[0]))
        os.utime(p, (clock[0].timestamp(),) * 2)
        r.rotate_once()
    assert [f.name for f in p.parent.glob("*.gz")] == \
        ["HFDL21.json.20261112.gz"]


def test_truncate_failure_removes_archive_and_backs_off(tmp_path, caplog,
                                                        monkeypatch):
    p = _spool(tmp_path)
    calls = []

    def bad(path, n):
        calls.append(path)
        raise PermissionError("nope")

    monkeypatch.setattr(os, "truncate", bad)
    r = _rot(tmp_path)
    with caplog.at_level(logging.WARNING):
        r.rotate_once()
        r.rotate_once()
        r.rotate_once()
    assert not list(p.parent.glob("*.gz*"))       # no archive left behind
    assert len(calls) == 1                         # one attempt per day
    assert sum(r_.levelno == logging.WARNING for r_ in caplog.records) == 1
    assert p.read_text() == "a\nb\n"


def test_prune_orders_collisions_by_date_then_number(tmp_path):
    d = tmp_path / "rid"
    d.mkdir()
    names = ["20261111", "20261112", "20261112.1", "20261112.2",
             "20261112.10", "20261110"]
    for n in names:
        (d / f"HFDL21.json.{n}.gz").write_bytes(b"x")
    _rot(tmp_path, keep=3).rotate_once()
    assert sorted(f.name for f in d.glob("*.gz")) == [
        "HFDL21.json.20261112.1.gz", "HFDL21.json.20261112.10.gz",
        "HFDL21.json.20261112.2.gz"]


def test_leftover_tmp_cleaned_at_startup(tmp_path):
    d = tmp_path / "rid"
    d.mkdir()
    (d / "HFDL21.json.20261111.gz.tmp").write_bytes(b"partial")
    (d / "HFDL5.json.20261111.gz.tmp").write_bytes(b"other band")
    _rot(tmp_path)
    assert not (d / "HFDL21.json.20261111.gz.tmp").exists()
    assert (d / "HFDL5.json.20261111.gz.tmp").exists()
