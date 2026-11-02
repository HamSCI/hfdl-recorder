"""Window keeper: save labelled IQ windows around expected squitters.

One keeper serves one band.  ``BandPipeline._on_samples`` hands it every
block, with the block's RTP timestamp, right after the CF32 write to
dumphfdl.  ``ChTailer`` hands it every decoded SPDU.  From the SPDUs it learns
which (ground station, frequency, frame offset k) currently transmit, among
the chosen ground stations and inside its band.  For each slot it expects, it
saves one window: 30 ms before the nominal slot start to the slot's end,
narrowed to one HFDL channel at 6000 S/s.

What runs where:

- ``feed`` (the MultiStream thread) copies the block into a ring held in RTP
  space and books windows.  It does no DSP, takes no lock, reads no clock,
  and never waits: when the writer's queue is full it drops the window and
  counts it.
- The writer thread asks ``anchor_fn`` for the status-pair anchor, reads each
  finished window back out of the ring, mixes the channel's dial frequency
  to 0 Hz, resamples to 6000 S/s, and writes the npz atomically.  It also
  prunes old day folders.

Two labels name each window's first sample (spec, "Changed after the label
gate"): the counter label (``input_index0``, from the channel's counter map)
and the status-pair label (``t0_utc``, from ``anchor_fn``).  The anchor
places the windows; the 30 ms lead absorbs its ordinary error.  ``t0_utc`` is
saved exactly as the anchor gives it.

The host clock never labels anything.  ``time.monotonic`` paces the prune and
nothing else.  Window ages, frequency silence and the frame index all come
from the sample count and the anchor.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from hfdl_recorder.bands import HfdlBand
from hfdl_recorder.timing.channel_status import ANCHOR_POLLS, LISTEN_S, POLL_TIMEOUT_S
from hfdl_recorder.timing.frame_grid import FRAME_S, FRAMES_PER_DAY, SLOT_S, next_slot_starts

logger = logging.getLogger(__name__)

OUT_FS = 6000                 # saved rate; the analyser expects it
LEAD_S = 0.030                # window opens this long before the nominal start
WINDOW_OUT = round((LEAD_S + SLOT_S) * OUT_FS)   # 14949 samples, 2.49 s
MARGIN_S = 0.050              # extra input read each side, cut off after resampling
FLAT_HZ = 2800.0              # passband flat to here, both halves
EDGE_HZ = 3000.0              # raised-cosine taper reaches 0 at the output Nyquist
FILTER_GUARD_HZ = 1500        # the daemon's radiod filter guard (daemon._provision)
SILENT_FRAMES = 3             # stop windowing a frequency unheard this long
PLAN_LEAD_S = 2.0             # book a window this far ahead of its start
PLAN_BACK_S = 1.5             # ... or this far behind, while the ring holds it
PLAN_EVERY_S = 0.25           # how often feed() books
# The writer thread takes anchors and saves windows in turn, so a finished
# window can wait behind one anchor.  The worst anchor (passive listen, then
# every poll timing out) is LISTEN_S + ANCHOR_POLLS * POLL_TIMEOUT_S = 3.5 s;
# WRITER_SAVE_S covers the save ahead of it.  The ring holds one extended
# window plus RING_SLACK_S of input: 5 s of complex64 is 11 MB on the widest
# band (HFDL5, 277.2 kS/s), 3.2 MB at 80 kS/s.
ANCHOR_WORST_S = LISTEN_S + ANCHOR_POLLS * POLL_TIMEOUT_S
WRITER_SAVE_S = 1.5
RING_SLACK_S = ANCHOR_WORST_S + WRITER_SAVE_S   # 5.0 s
ANCHOR_RETRY_S = 10.0         # samples to wait after a failed anchor
COUNTER_LABEL_TOL_S = 5.0     # counter vs status-pair label: beyond this, another run
PRUNE_EVERY_S = 3600.0
REANCHOR_S = 600.0            # take a fresh least-late anchor this often
DISK_RESUME_GB = 1.0          # resume saving at min_free_gb + this (hysteresis)
GAP_HISTORY = 64              # blocks with zero-fill remembered for the meta
WRITER_NICE = 10              # added to the process's niceness, writer thread only

# GPS epoch minus Unix epoch, and GPS minus UTC.  Only the 5 s run check
# uses them, never a label.
_GPS_UNIX_S = 315_964_800
_GPS_LEAP_S = 18

DECIMATION_ID = "fft-band-select-v1"


def _rtp_diff(a: int, b: int) -> int:
    d = (a - b) & 0xFFFFFFFF
    return d - 0x100000000 if d >= 0x80000000 else d


def _smooth(n: int) -> int:
    """Smallest 2-3-5-smooth integer >= n (a fast FFT length)."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


@dataclass(frozen=True)
class _Target:
    gs_id: int
    gs_name: str
    freq_khz: int
    k: int
    heard_abs: int            # ring sample count when last heard
    utc_sync: Optional[bool]


@dataclass(frozen=True)
class _Anchor:
    gen: int
    id: int                   # counts every anchor this keeper took
    rtp: int                  # the anchored RTP
    abs_: int                 # ring sample count of the anchored RTP
    utc: float
    source: str
    offset_ns: Optional[int]
    t_level_active: Optional[str]


@dataclass
class _Job:
    gen: int
    key: tuple
    target: _Target
    nominal_utc: float
    frame_index: int
    s0_abs: int               # input sample of the first saved sample
    ext_abs: int              # first input sample read
    end_abs: int              # one past the last input sample read
    rtp0: int
    t0_utc: float
    anchor: _Anchor


class WindowKeeper:
    """Keeps labelled windows around expected squitters for one band."""

    def __init__(
        self,
        band: HfdlBand,
        fs: int,
        out_dir: Path,
        ground_stations: list[str],
        anchor_fn: Callable[[int], object],
        counter_map_fn: Callable[[], object],
        max_queue: int = 8,
        *,
        min_free_gb: float = 5.0,
        retention_days: int = 5,
        free_bytes_fn: Optional[Callable[[Path], int]] = None,
        tier_fn: Optional[Callable[[], Optional[str]]] = None,
    ):
        self._band = band
        self._fs = int(fs)
        self._out_dir = Path(out_dir)
        self._chosen = frozenset(ground_stations)
        self._anchor_fn = anchor_fn
        self._counter_map_fn = counter_map_fn
        self._min_free = float(min_free_gb) * 1024**3
        self._retention_days = int(retention_days)
        self._free_bytes = free_bytes_fn or (lambda p: shutil.disk_usage(p).free)
        self._tier_fn = tier_fn

        # Resampling plan: fs -> 6000 by up/down = L/M.  The extended read
        # holds q*M input samples, q*L output; margins are whole multiples
        # of L output samples so output sample j sits exactly at input
        # sample s0 + j*M/L.
        g = math.gcd(self._fs, OUT_FS)
        self._L, self._M = OUT_FS // g, self._fs // g
        m_q = math.ceil(MARGIN_S * OUT_FS / self._L)
        self._margin_out = m_q * self._L
        self._margin_in = m_q * self._M
        q = _smooth(math.ceil((WINDOW_OUT + 2 * self._margin_out) / self._L))
        self._n_out_ext = q * self._L
        self._n_in_ext = q * self._M
        self._taper = self._make_taper()

        self._ring_len = self._n_in_ext + int(RING_SLACK_S * self._fs)
        self._ring = np.zeros(self._ring_len, dtype=np.complex64)
        self._max_block = 0

        # feed() state.  Only the MultiStream thread writes these.
        self._head_abs = 0                 # samples ever fed
        self._expect_rtp: Optional[int] = None
        self._gen = 0
        self._gen_abs = 0                  # ring count where this generation began
        self._gen_rtp = 0
        self._next_plan_abs = 0
        self._next_anchor_abs = 0
        self._anchor_pending = False       # an anchor request sits with the writer
        self._anchor_seq = 0
        # Blocks that carried zero-fill: ((abs_start, n, gap_samples), ...),
        # replaced whole so the writer reads it without a lock.
        self._gaps: tuple = ()
        self._planned: list[_Job] = []
        self._booked: dict[tuple, int] = {}

        # Shared, replaced whole (never mutated) so feed() reads without a lock.
        self._anchor: Optional[_Anchor] = None
        self._targets: dict[tuple, _Target] = {}
        self._targets_lock = threading.Lock()   # serialises writers only
        self._anchored_evt = threading.Event()
        self.last_feed_mono = 0.0          # liveness for the daemon's poller

        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._disk_low = False
        self._last_prune: Optional[float] = None
        self._current_day: Optional[str] = None   # from the latest window's nominal UTC
        self._frame_bad = False
        self._anchor_bad = False           # in a streak of failed anchors

        self._stats = {
            "saved": 0, "dropped_queue": 0, "dropped_disk": 0,
            "dropped_overrun": 0, "reanchors": 0, "anchors": 0,
            "anchor_failures": 0, "no_counter_map": 0,
            "counter_map_refused": 0, "errors": 0,
            "skipped_late": 0, "frame_mismatch": 0,
        }

    @property
    def out_dir(self) -> Path:
        return self._out_dir

    # -- lifecycle --

    def start(self) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"hfdl-{self._band.name}-windows",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None and timeout > 0:
            self._thread.join(timeout=timeout)

    def flush(self) -> None:
        """Wait until the writer has handled every queued job (tests)."""
        self._q.join()

    def wait_anchored(self, timeout: float) -> bool:
        return self._anchored_evt.wait(timeout)

    def stats(self) -> dict:
        out = dict(self._stats)
        out["targets"] = len(self._live_targets())
        out["anchored"] = self._anchor is not None and self._anchor.gen == self._gen
        return out

    def stats_line(self) -> str:
        st = self.stats()
        return (f"{self._band.name} squitter windows: " + " ".join(
            f"{k}={v}" for k, v in st.items()))

    # -- squitters (ChTailer thread) --

    def note_squitter(self, spdu_frame: dict) -> None:
        """Learn (gs, freq, k) from one decoded SPDU frame.

        The source ground station transmitted on hfdl.freq at frame_offset k.
        gs_status lists each ground station's active frequencies: a chosen
        station whose list drops a frequency stops being windowed there.
        dumphfdl's t.sec (host clock) is never read.
        """
        try:
            self._note_squitter(spdu_frame)
        except Exception:  # noqa: BLE001 - a bad frame must not reach the tailer
            logger.debug("%s: unusable squitter frame", self._band.name,
                         exc_info=True)

    def _note_squitter(self, frame: dict) -> None:
        hfdl = frame.get("hfdl", frame)
        spdu = hfdl.get("spdu")
        if not isinstance(spdu, dict):
            return
        heard = self._head_abs
        announced: dict[int, tuple[frozenset, Optional[bool]]] = {}
        for entry in spdu.get("gs_status") or []:
            gs = entry.get("gs") or {}
            if gs.get("name") not in self._chosen:
                continue
            freqs = frozenset(int(round(float(f["freq"])))
                              for f in entry.get("freqs") or []
                              if isinstance(f, dict) and f.get("freq") is not None)
            if not freqs:
                continue                  # no resolvable list: no information
            announced[int(gs["id"])] = (freqs, entry.get("utc_sync"))

        self._check_frame_index(spdu.get("frame_index"), heard)

        src = spdu.get("src") or {}
        new: Optional[_Target] = None
        if src.get("name") in self._chosen:
            gs_id = int(src["id"])
            freq_khz = int(round(int(hfdl["freq"]) / 1000))
            k = int(spdu["frame_offset"])
            if 0 <= k < 13 and self._in_band(freq_khz):
                sync = announced.get(gs_id, (None, None))[1]
                new = _Target(gs_id, src["name"], freq_khz, k, heard, sync)

        with self._targets_lock:
            targets = dict(self._targets)
            for key, t in list(targets.items()):
                ann = announced.get(t.gs_id)
                if ann is None:
                    continue
                if t.freq_khz not in ann[0]:
                    del targets[key]
                elif ann[1] is not None and ann[1] != t.utc_sync:
                    targets[key] = _Target(t.gs_id, t.gs_name, t.freq_khz,
                                           t.k, t.heard_abs, ann[1])
            if new is not None:
                if new.utc_sync is None and (new.gs_id, new.freq_khz) in targets:
                    old = targets[(new.gs_id, new.freq_khz)]
                    new = _Target(new.gs_id, new.gs_name, new.freq_khz, new.k,
                                  new.heard_abs, old.utc_sync)
                targets[(new.gs_id, new.freq_khz)] = new
            self._targets = targets

    def _check_frame_index(self, frame_index, heard_abs: int) -> None:
        """The squitter names its frame; the anchor names the frame now.

        A decode lands within the frame it was sent in or the next one, so
        any other difference means the anchor (or the decode) is wrong by a
        frame or more.  Counted always; logged once per streak.
        """
        a = self._anchor
        if frame_index is None or a is None or a.gen != self._gen:
            return
        head_utc = a.utc + (heard_abs - a.abs_) / self._fs
        ours = math.floor(head_utc / FRAME_S) % FRAMES_PER_DAY
        lag = (ours - int(frame_index)) % FRAMES_PER_DAY
        if lag <= 1:
            if self._frame_bad:
                self._frame_bad = False
                logger.info("%s: squitter frame_index agrees with the anchor again",
                            self._band.name)
            return
        self._stats["frame_mismatch"] += 1
        if not self._frame_bad:
            self._frame_bad = True
            logger.warning(
                "%s: squitter frame_index %d, but the anchor puts this frame at "
                "%d; the anchor or the decode is off by a frame or more "
                "(further mismatches counted, not logged)",
                self._band.name, int(frame_index), ours)

    def _in_band(self, freq_khz: int) -> bool:
        f_mix = freq_khz * 1000 - self._band.center_hz
        return abs(f_mix) + EDGE_HZ <= self._fs / 2 - FILTER_GUARD_HZ

    def _live_targets(self) -> list[_Target]:
        limit = SILENT_FRAMES * FRAME_S * self._fs
        head = self._head_abs
        return [t for t in self._targets.values() if head - t.heard_abs <= limit]

    # -- samples (MultiStream thread) --

    def feed(self, samples: np.ndarray, rtp_start: int, gap_samples: int = 0) -> None:
        """Copy one block into the ring and book finished windows.  Never blocks.

        ``gap_samples``: zero-fill MultiStream put in this block (its
        quality.batch_gaps); windows that overlap the block record it.
        """
        x = np.asarray(samples)
        if x.dtype != np.complex64:
            if np.iscomplexobj(x):
                x = x.astype(np.complex64)
            else:
                x = np.ascontiguousarray(x, dtype=np.float32).view(np.complex64)
        n = len(x)
        if n == 0:
            return
        self.last_feed_mono = time.monotonic()
        rtp_start = int(rtp_start) & 0xFFFFFFFF
        if self._expect_rtp is None or rtp_start != self._expect_rtp:
            self._new_generation(rtp_start, first=self._expect_rtp is None)

        # Ring write.  head_abs moves only after the data lands, so a reader
        # that checks head after copying knows whether it was overwritten.
        if n > self._max_block:
            self._max_block = n
        src = x if n <= self._ring_len else x[-self._ring_len:]
        start = (self._head_abs + n - len(src)) % self._ring_len
        first = min(len(src), self._ring_len - start)
        self._ring[start:start + first] = src[:first]
        if first < len(src):
            self._ring[:len(src) - first] = src[first:]
        if gap_samples:
            self._gaps = self._gaps[-(GAP_HISTORY - 1):] + (
                (self._head_abs, n, int(gap_samples)),)
        self._head_abs += n
        self._expect_rtp = (rtp_start + n) & 0xFFFFFFFF

        anchor = self._anchor
        if anchor is None or anchor.gen != self._gen:
            self._ask_anchor(rtp_start, self._head_abs - n)
            return
        if self._head_abs - anchor.abs_ >= REANCHOR_S * self._fs:
            self._ask_anchor(rtp_start, self._head_abs - n)   # a periodic refresh
        if self._head_abs >= self._next_plan_abs:
            self._next_plan_abs = self._head_abs + int(PLAN_EVERY_S * self._fs)
            self._plan(anchor)
        self._dispatch()

    def _new_generation(self, rtp_start: int, first: bool) -> None:
        """A discontinuity: the old anchor and every unfinished window go."""
        if not first:
            self._stats["reanchors"] += 1
            logger.warning(
                "%s: RTP discontinuity (expected %s, got %d); re-anchoring",
                self._band.name, self._expect_rtp, rtp_start)
        self._gen += 1
        self._gen_abs = self._head_abs
        self._gen_rtp = rtp_start
        self._planned = []
        self._booked = {}
        self._anchored_evt.clear()
        self._anchor_pending = False
        self._next_anchor_abs = self._head_abs
        self._next_plan_abs = self._head_abs

    def _ask_anchor(self, rtp: int, abs_: int) -> None:
        if self._anchor_pending or self._head_abs < self._next_anchor_abs:
            return
        try:
            self._q.put_nowait(("anchor", self._gen, abs_, rtp))
            self._anchor_pending = True
        except queue.Full:
            pass                          # try again on a later block

    def _plan(self, a: _Anchor) -> None:
        fs = self._fs
        head_utc = a.utc + (self._head_abs - a.abs_) / fs
        oldest = max(self._gen_abs,
                     self._head_abs - self._ring_len + self._max_block)
        for t in self._live_targets():
            for nominal in next_slot_starts(t.k, head_utc - PLAN_BACK_S,
                                            head_utc + PLAN_LEAD_S):
                frame_abs = round((nominal - t.k * SLOT_S) / FRAME_S)
                key = (t.gs_id, t.freq_khz, frame_abs)
                if key in self._booked:
                    continue
                s0 = a.abs_ + math.floor((nominal - LEAD_S - a.utc) * fs)
                ext = s0 - self._margin_in
                if ext < oldest:
                    self._booked[key] = s0          # too late; never retry it
                    self._stats["skipped_late"] += 1
                    continue
                self._booked[key] = s0
                self._planned.append(_Job(
                    gen=self._gen, key=key, target=t, nominal_utc=nominal,
                    frame_index=frame_abs % FRAMES_PER_DAY,
                    s0_abs=s0, ext_abs=ext, end_abs=ext + self._n_in_ext,
                    rtp0=(self._gen_rtp + s0 - self._gen_abs) & 0xFFFFFFFF,
                    t0_utc=a.utc + (s0 - a.abs_) / fs, anchor=a))
        self._planned.sort(key=lambda j: j.end_abs)
        stale = self._head_abs - 2 * int(FRAME_S * fs)
        if len(self._booked) > 256:
            self._booked = {k: v for k, v in self._booked.items() if v > stale}

    def _dispatch(self) -> None:
        while self._planned and self._planned[0].end_abs <= self._head_abs:
            job = self._planned.pop(0)
            try:
                self._q.put_nowait(("window", job))
            except queue.Full:
                self._stats["dropped_queue"] += 1

    # -- writer thread --

    def _run(self) -> None:
        # Linux nice is per thread: the writer yields to radiod, dumphfdl and
        # the sample path.  Raising one's own niceness needs no privilege.
        try:
            tid = threading.get_native_id()
            os.setpriority(os.PRIO_PROCESS, tid,
                           min(19, os.getpriority(os.PRIO_PROCESS, tid) + WRITER_NICE))
        except (AttributeError, OSError):
            pass
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                if item is None:
                    return
                if item[0] == "anchor":
                    self._do_anchor(*item[1:])
                else:
                    self._save_window(item[1])
                    self._maybe_prune()
            except Exception:  # noqa: BLE001 - the writer must outlive one bad window
                self._stats["errors"] += 1
                logger.exception("%s: squitter window writer failed", self._band.name)
            finally:
                self._q.task_done()

    def _do_anchor(self, gen: int, abs_: int, rtp: int) -> None:
        try:
            self._take_anchor(gen, abs_, rtp)
        finally:
            self._anchor_pending = False

    def _take_anchor(self, gen: int, abs_: int, rtp: int) -> None:
        try:
            a = self._anchor_fn(rtp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: anchor failed: %s", self._band.name, exc)
            a = None
        if a is None or not getattr(a, "rtp_referenced", True):
            # No host-clock labels: wait and ask again (a refresh keeps the
            # anchor it has meanwhile).
            self._stats["anchor_failures"] += 1
            self._next_anchor_abs = abs_ + int(ANCHOR_RETRY_S * self._fs)
            if not self._anchor_bad:
                self._anchor_bad = True
                logger.warning(
                    "%s: no anchor (no valid status reply with a status pair); "
                    "retrying every %.0f s of samples, silently until one succeeds",
                    self._band.name, ANCHOR_RETRY_S)
            return
        if self._anchor_bad:
            self._anchor_bad = False
            logger.info("%s: anchor recovered", self._band.name)
        if gen != self._gen:
            return                          # another discontinuity since asking
        old = self._anchor
        self._anchor_seq += 1
        snap = getattr(a, "snapshot", None)
        self._anchor = _Anchor(
            gen=gen, id=self._anchor_seq, rtp=int(rtp) & 0xFFFFFFFF, abs_=abs_,
            utc=float(a.utc), source=str(a.source),
            offset_ns=getattr(a, "offset_ns", None),
            t_level_active=getattr(snap, "t_level_active", None))
        self._stats["anchors"] += 1
        self._anchored_evt.set()
        if old is not None and old.gen == gen:
            # The step between the extrapolated old anchor and the new one:
            # ADC rate error over the interval plus the two anchors' lateness.
            step_ms = (a.utc - (old.utc + (abs_ - old.abs_) / self._fs)) * 1e3
            logger.info("%s: squitter windows re-anchored rtp=%d utc=%.6f (%s), "
                        "step %+.3f ms after %.0f s", self._band.name, rtp, a.utc,
                        a.source, step_ms, (abs_ - old.abs_) / self._fs)
        else:
            logger.info("%s: squitter windows anchored rtp=%d utc=%.6f (%s)",
                        self._band.name, rtp, a.utc, a.source)

    def _read_ring(self, a: int, n: int) -> Optional[np.ndarray]:
        idx = a % self._ring_len
        first = min(n, self._ring_len - idx)
        out = np.empty(n, dtype=np.complex64)
        out[:first] = self._ring[idx:idx + first]
        if first < n:
            out[first:] = self._ring[:n - first]
        # Valid only if feed() has not since overwritten sample a.
        if a < self._head_abs + self._max_block - self._ring_len:
            return None
        return out

    def _save_window(self, job: _Job) -> None:
        if job.gen != self._gen:
            return                          # a discontinuity came after booking
        self._current_day = datetime.fromtimestamp(
            job.nominal_utc, tz=timezone.utc).strftime("%Y%m%d")
        try:
            free = self._free_bytes(self._out_dir)
        except OSError:
            free = 0
        # Hysteresis: stop below min_free_gb, resume only DISK_RESUME_GB above.
        limit = self._min_free + (DISK_RESUME_GB * 1024**3 if self._disk_low else 0)
        if free < limit:
            self._stats["dropped_disk"] += 1
            if not self._disk_low:
                self._disk_low = True
                logger.warning(
                    "%s: only %.1f GB free under %s (min %.1f GB); saving no "
                    "squitter windows until %.1f GB are free",
                    self._band.name, free / 1024**3, self._out_dir,
                    self._min_free / 1024**3,
                    self._min_free / 1024**3 + DISK_RESUME_GB)
                self._maybe_prune(force=True)
            return
        if self._disk_low:
            self._disk_low = False
            logger.info("%s: disk space back (%.1f GB free); squitter windows resume",
                        self._band.name, free / 1024**3)

        raw = self._read_ring(job.ext_abs, self._n_in_ext)
        if raw is None or job.gen != self._gen:
            self._stats["dropped_overrun"] += 1
            return
        n_bad = int(np.count_nonzero(~np.isfinite(raw.view(np.float32))))
        x = self._narrow(raw, job.target.freq_khz)

        meta = self._meta(job, n_bad)
        meta["n_gap_samples"] = sum(
            g for start, n, g in self._gaps
            if start < job.end_abs and start + n > job.ext_abs)
        day = datetime.fromtimestamp(job.nominal_utc, tz=timezone.utc).strftime("%Y%m%d")
        folder = self._out_dir / day
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{job.target.gs_id}-{job.target.freq_khz}-{job.frame_index}.npz"
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as fh:
                np.savez(fh, x=x, meta=np.str_(json.dumps(meta)))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        self._stats["saved"] += 1

    def _make_taper(self) -> np.ndarray:
        f = np.fft.fftfreq(self._n_out_ext, 1.0 / OUT_FS)
        a = np.abs(f)
        h = np.where(a <= FLAT_HZ, 1.0, 0.0)
        edge = (a > FLAT_HZ) & (a < EDGE_HZ)
        h[edge] = 0.5 * (1 + np.cos(np.pi * (a[edge] - FLAT_HZ) / (EDGE_HZ - FLAT_HZ)))
        return h

    def _narrow(self, raw: np.ndarray, freq_khz: int) -> np.ndarray:
        """Mix the dial to 0 Hz and resample to 6000 S/s by band selection.

        Keeps -3..+3 kHz, flat to +-2.8 kHz, raised-cosine to zero at 3 kHz.
        The negative half stays: the analyser reads noise there.  Output
        sample j sits at input sample s0 + j*fs/6000 exactly.
        """
        x = np.nan_to_num(raw.astype(np.complex128), nan=0.0, posinf=0.0, neginf=0.0)
        f_mix = freq_khz * 1000 - self._band.center_hz
        if f_mix:
            n = np.arange(len(x))
            x *= np.exp(-2j * np.pi * (f_mix / self._fs) * n)
        spec = np.fft.fft(x)
        no = self._n_out_ext
        half = no // 2
        sel = np.empty(no, dtype=np.complex128)
        sel[:half] = spec[:half]
        sel[half:] = spec[len(spec) - (no - half):]
        sel *= self._taper
        y = np.fft.ifft(sel) * (no / len(x))
        return y[self._margin_out:self._margin_out + WINDOW_OUT].astype(np.complex64)

    def _meta(self, job: _Job, n_bad: int) -> dict:
        t, a = job.target, job.anchor
        tier = a.t_level_active
        if self._tier_fn is not None:
            try:
                tier = self._tier_fn()
            except Exception:  # noqa: BLE001
                pass
        meta = {
            "t0_utc": job.t0_utc,
            "nominal_utc": job.nominal_utc,
            "fs": OUT_FS,
            "freq_khz": t.freq_khz,
            "gs_id": t.gs_id,
            "gs_name": t.gs_name,
            "frame_index": job.frame_index,
            "frame_offset": t.k,
            "band": self._band.name,
            "band_fs": self._fs,
            "band_center_hz": self._band.center_hz,
            "anchor_source": a.source,
            "anchor_id": a.id,
            "anchor_rtp": a.rtp,
            "anchor_age_s": (job.s0_abs - a.abs_) / self._fs,
            "offset_ns": a.offset_ns,
            "t_level_active": tier,
            "utc_sync": t.utc_sync,
            "rtp0": job.rtp0,
            "lead_s": LEAD_S,
            "n_nonfinite": n_bad,
            "decimation": {
                "id": DECIMATION_ID, "in_fs": self._fs, "out_fs": OUT_FS,
                "up": self._L, "down": self._M,
                "flat_hz": FLAT_HZ, "edge_hz": EDGE_HZ,
                "mix_hz": t.freq_khz * 1000 - self._band.center_hz,
            },
            "input_index0": None,
            "input_samprate": None,
            "counter_map": None,
            "input_epoch_gps_s": None,
        }
        cmap = None
        try:
            cmap = self._counter_map_fn()
        except Exception:  # noqa: BLE001
            logger.debug("%s: counter map read failed", self._band.name, exc_info=True)
        if cmap is None:
            self._stats["no_counter_map"] += 1
            return meta
        idx = cmap.input_index(job.rtp0)
        # A map from another radiod run gives an index hours off.  This
        # check runs at 5 s grain and labels nothing.
        implied = cmap.input_epoch_gps_s + idx / cmap.input_samprate + _GPS_UNIX_S - _GPS_LEAP_S
        if abs(implied - job.t0_utc) > COUNTER_LABEL_TOL_S:
            self._stats["counter_map_refused"] += 1
            return meta
        meta.update(input_index0=float(idx), input_samprate=cmap.input_samprate,
                    counter_map=cmap.to_dict(),
                    input_epoch_gps_s=cmap.input_epoch_gps_s)
        return meta

    def _maybe_prune(self, force: bool = False) -> None:
        """Remove day folders older than retention_days before the current
        day (the latest window's nominal UTC, saved or not)."""
        now = time.monotonic()
        if not force and self._last_prune is not None \
                and now - self._last_prune < PRUNE_EVERY_S:
            return
        if self._current_day is None:
            return
        self._last_prune = now
        newest = datetime.strptime(self._current_day, "%Y%m%d")
        cutoff = (newest - timedelta(days=self._retention_days)).strftime("%Y%m%d")
        try:
            entries = list(self._out_dir.iterdir())
        except OSError:
            return
        for p in entries:
            name = p.name
            if len(name) != 8 or not name.isdigit() or not p.is_dir():
                continue
            if name < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                logger.info("%s: pruned squitter windows %s", self._band.name, name)
