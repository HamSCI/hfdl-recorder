"""Daily UTC rotation of dumphfdl's per-band JSON spool files.

dumphfdl holds ``<BAND>.json`` open with O_APPEND, so the live file is
never renamed.  A busy band writes every few seconds, so the file's
mtime says nothing about the day it covers.  The rotator tracks, per
band, the UTC date the live content starts.  It seeds that date at
first sight from the timestamp of the file's first line (dumphfdl
``hfdl.t.sec``), falling back to the file's mtime.  The state lives in
memory; a restart re-seeds it from the file.

When the current UTC date passes the start date and the file holds
data, the rotator gzips the file to ``<BAND>.json.<YYYYMMDD>.gz`` (the
start date; a few minutes of the new day may ride along), truncates the
live file in place, and sets the start date to today.  The newest
``keep_days`` rotated files per band stay.

Every error logs a WARNING and never stops the daemon or dumphfdl.  A
band that cannot truncate retries at most once per UTC day.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_KEEP_DAYS = 3
_GZIP_LEVEL = 6


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _first_line_date(path: Path) -> Optional[date]:
    """UTC date of the first frame in the spool file, or None."""
    try:
        with open(path, "rb") as fh:
            line = fh.readline(1 << 20)
        sec = json.loads(line)["hfdl"]["t"]["sec"]
        return datetime.fromtimestamp(int(sec), timezone.utc).date()
    except Exception:
        return None


class SpoolRotator:
    def __init__(self, spool_dir: Path, bands: Iterable[str], *,
                 keep_days: int = DEFAULT_KEEP_DAYS,
                 now_fn: Callable[[], datetime] = _utcnow) -> None:
        self.spool_dir = Path(spool_dir)
        self.bands = list(bands)
        self.keep_days = max(1, int(keep_days))
        self._now = now_fn
        self._start: dict[str, date] = {}     # band -> date content starts
        self._blocked: dict[str, date] = {}   # band -> day a rotation failed
        self._clean_tmp()

    def _clean_tmp(self) -> None:
        for band in self.bands:
            try:
                for tmp in self.spool_dir.glob(f"{band}.json.*.gz.tmp"):
                    tmp.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("spool rotation: tmp cleanup band=%s: %s",
                               band, e)

    def rotate_once(self) -> None:
        for band in self.bands:
            try:
                self._rotate_band(band)
            except Exception as e:
                logger.warning("spool rotation band=%s failed: %s", band, e)
            try:
                self._prune_band(band)
            except Exception as e:
                logger.warning("spool pruning band=%s failed: %s", band, e)

    def _rotate_band(self, band: str) -> None:
        today = self._now().date()
        if self._blocked.get(band) == today:
            return                            # failed today; retry tomorrow
        live = self.spool_dir / f"{band}.json"
        try:
            st = live.stat()
        except FileNotFoundError:
            return
        if st.st_size == 0:
            self._start[band] = today         # content will start today
            return
        if band not in self._start:
            self._start[band] = (
                _first_line_date(live)
                or datetime.fromtimestamp(st.st_mtime, timezone.utc).date())
        covers = self._start[band]
        if covers >= today:
            return
        self._blocked[band] = today           # cleared below on success
        stamp = covers.strftime("%Y%m%d")
        dest = self.spool_dir / f"{band}.json.{stamp}.gz"
        n = 0
        while dest.exists():                  # never overwrite earlier data
            n += 1
            dest = self.spool_dir / f"{band}.json.{stamp}.{n}.gz"
        tmp = dest.with_name(dest.name + ".tmp")
        copied = 0
        try:
            with open(live, "rb") as src, open(tmp, "wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb",
                                   compresslevel=_GZIP_LEVEL) as out:
                    # Copy to the current end, then re-check: lines
                    # dumphfdl appends during the copy join the archive.
                    while True:
                        size = os.fstat(src.fileno()).st_size
                        if size <= copied:
                            break
                        shutil.copyfileobj(_Limited(src, size - copied), out)
                        copied = size
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(tmp, dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        try:
            os.truncate(live, 0)
        except BaseException:
            dest.unlink(missing_ok=True)      # data stays in the live file
            raise
        del self._blocked[band]
        self._start[band] = today
        logger.info("spool rotation band=%s: %d bytes -> %s",
                    band, copied, dest.name)

    def _prune_band(self, band: str) -> None:
        pat = re.compile(rf"{re.escape(band)}\.json\.(\d{{8}})(?:\.(\d+))?\.gz")
        keyed = []
        for p in self.spool_dir.iterdir():
            m = pat.fullmatch(p.name)
            if m:
                keyed.append(((m.group(1), int(m.group(2) or 0)), p))
        keyed.sort(key=lambda kp: kp[0])
        for _key, p in keyed[:-self.keep_days]:
            try:
                p.unlink()
            except OSError as e:
                logger.warning("spool pruning: cannot delete %s: %s",
                               p.name, e)


class _Limited:
    """Read at most ``n`` bytes from ``fh`` (for copyfileobj)."""

    def __init__(self, fh, n: int) -> None:
        self._fh, self._left = fh, n

    def read(self, size: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        size = self._left if size < 0 else min(size, self._left)
        data = self._fh.read(size)
        self._left -= len(data)
        return data
