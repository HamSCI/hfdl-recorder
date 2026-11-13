"""Daily UTC rotation of dumphfdl's per-band JSON spool files.

dumphfdl holds ``<BAND>.json`` open with O_APPEND, so the live file is
never renamed.  A file whose last write falls on an earlier UTC day is
copied into ``<BAND>.json.<YYYYMMDD>.gz`` (the UTC date it covers) and
then truncated in place; dumphfdl's next write lands at offset 0.  The
newest ``keep_days`` rotated files per band stay; older ones go.

Every error logs a WARNING and never stops the daemon or dumphfdl.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

DEFAULT_KEEP_DAYS = 3


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SpoolRotator:
    def __init__(self, spool_dir: Path, bands: Iterable[str], *,
                 keep_days: int = DEFAULT_KEEP_DAYS,
                 now_fn: Callable[[], datetime] = _utcnow) -> None:
        self.spool_dir = Path(spool_dir)
        self.bands = list(bands)
        self.keep_days = max(1, int(keep_days))
        self._now = now_fn

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
        live = self.spool_dir / f"{band}.json"
        try:
            st = live.stat()
        except FileNotFoundError:
            return
        if st.st_size == 0:
            return
        covers = datetime.fromtimestamp(st.st_mtime, timezone.utc).date()
        if covers >= self._now().date():
            return
        stamp = covers.strftime("%Y%m%d")
        dest = self.spool_dir / f"{band}.json.{stamp}.gz"
        n = 0
        while dest.exists():                     # never overwrite earlier data
            n += 1
            dest = self.spool_dir / f"{band}.json.{stamp}.{n}.gz"
        tmp = dest.with_name(dest.name + ".tmp")
        copied = 0
        try:
            with open(live, "rb") as src, gzip.open(tmp, "wb") as out:
                # Copy to the current end, then re-check: lines dumphfdl
                # appends during the copy join the archive, not the void.
                while True:
                    size = os.fstat(src.fileno()).st_size
                    if size <= copied:
                        break
                    shutil.copyfileobj(_Limited(src, size - copied), out)
                    copied = size
            os.replace(tmp, dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        os.truncate(live, 0)
        logger.info("spool rotation band=%s: %d bytes -> %s",
                    band, copied, dest.name)

    def _prune_band(self, band: str) -> None:
        rotated = sorted(p.name for p in
                         self.spool_dir.glob(f"{band}.json.*.gz"))
        for name in rotated[:-self.keep_days]:
            try:
                (self.spool_dir / name).unlink()
            except OSError as e:
                logger.warning("spool pruning: cannot delete %s: %s", name, e)


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
