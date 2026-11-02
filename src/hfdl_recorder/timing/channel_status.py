"""One HFDL band channel's own status: its counter map and its anchor.

The window keeper labels each window twice (spec, "Changed after the label
gate").  This module supplies both labels from the band channel's own status
replies, off the sample path:

- ``ChannelCounterMap`` learns the channel's CounterMap (RTP to radiod's
  input-sample index) from (INPUT_SAMPLES, RTP_TIMESNAP) pairs, as
  hf-timestd's ``counter_map_probe.py`` does: five polls at the start, then
  one a tick, majority three of five, rolled forward to the newest RTP,
  dropped when stale or replaced on a new radiod run.  hf-timestd is not a
  dependency, so this is the small equivalent hfdl-recorder needs.
- ``least_late_anchor`` takes several status replies and keeps the one that
  puts a given RTP earliest in UTC.  radiod reads GPS_TIME when it builds a
  status packet but takes RTP_TIMESNAP from the last block, so every reply
  runs late by its own amount; the earliest runs least late.

Channel hygiene: on radiod, a command addressed to an unknown SSRC mints a
channel with infinite lifetime.  Callers pass only the band's own live SSRC,
the one ``BandPipeline.attach`` returned and whose samples are arriving.

The monotonic clock only measures how long the map has gone without a reply.
The host clock labels nothing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

from hamsci_dsp.counter_map import CounterMap, CounterPoll

logger = logging.getLogger(__name__)

POLLS_AT_START = 5
HISTORY = 5                  # three of five is a strict majority; never widen
MIN_AGREE = 3
RUN_MISS_BLOCKS = 1000       # a restart resets INPUT_SAMPLES far past this
ANCHOR_POLLS = 5


def counter_poll_from_reply(reply, output_samprate: int) -> Optional[CounterPoll]:
    """One status reply -> one poll, or None (a reply missing a field, or
    from another output rate, is not a poll)."""
    fe = getattr(reply, "frontend", None)
    gps = getattr(reply, "gps_time", None)
    rtp = getattr(reply, "rtp_timesnap", None)
    n_in = getattr(fe, "input_samples", None)
    f_in = getattr(fe, "input_samprate", None)
    ilen = getattr(reply, "filter_blocksize", None)
    if not gps or rtp is None or n_in is None or not f_in or not ilen:
        return None
    if getattr(reply, "output_samprate", None) != output_samprate:
        return None
    return CounterPoll(rtp_timesnap=int(rtp), input_samples=int(n_in),
                       filter_blocksize=int(ilen), input_samprate=int(f_in),
                       output_samprate=int(output_samprate), gps_time_ns=int(gps))


class ChannelCounterMap:
    """The counter map of one channel, kept from its own polls.  Thread-safe."""

    def __init__(self, name: str, output_samprate: int,
                 mono_fn: Callable[[], float] = time.monotonic):
        self._name = name
        self._rate = int(output_samprate)
        self._mono = mono_fn
        # Half the signed-32 reach: a held map never aliases.
        self._stale_s = 0.5 * (2**31 / self._rate)
        self._lock = threading.Lock()
        self._polls: deque = deque(maxlen=HISTORY)
        self._held: Optional[CounterMap] = None
        self._last_rtp: Optional[int] = None
        self._last_fresh: Optional[float] = None
        self.steps = 0

    @property
    def polls_wanted(self) -> int:
        with self._lock:
            return POLLS_AT_START if len(self._polls) < MIN_AGREE else 1

    def add_reply(self, reply) -> None:
        poll = counter_poll_from_reply(reply, self._rate)
        if poll is not None:
            self.add(poll)

    def add(self, poll: CounterPoll) -> None:
        with self._lock:
            if self._is_new_run(poll):
                logger.info("counter map for %s: new radiod run, map replaced",
                            self._name)
                self._polls.clear()
                self._held = None
            self._polls.append(poll)
            self._last_rtp = poll.rtp_timesnap
            self._last_fresh = self._mono()
            new = CounterMap.from_polls(list(self._polls), min_agree=MIN_AGREE)
            if new is not None:
                self._adopt(new)

    def current(self) -> Optional[CounterMap]:
        """The held map rolled to the newest RTP, or None (none yet, or stale)."""
        with self._lock:
            if self._held is None or self._last_fresh is None:
                return None
            if self._mono() - self._last_fresh > self._stale_s:
                logger.error("counter map for %s dropped: no fresh reply for "
                             "over %.1f h, a held map could alias",
                             self._name, self._stale_s / 3600)
                self._held = None
                self._polls.clear()
                return None
            self._held = self._held.rolled_forward(self._last_rtp)
            return self._held

    def _is_new_run(self, poll: CounterPoll) -> bool:
        newest = self._polls[-1] if self._polls else None
        if newest is not None and \
                poll.input_samples < newest.input_samples - newest.filter_blocksize:
            return True
        ref = self._held
        if ref is not None:
            if (poll.filter_blocksize, poll.input_samprate) != (ref.ilen, ref.input_samprate):
                return True
            return abs(ref.input_index(poll.rtp_timesnap) - poll.input_samples) > \
                RUN_MISS_BLOCKS * ref.ilen
        if newest is None:
            return False
        ilen = newest.filter_blocksize
        if (poll.filter_blocksize, poll.input_samprate) != (ilen, newest.input_samprate):
            return True
        olen = newest.output_samprate * ilen // newest.input_samprate
        if olen <= 0:
            return True
        d = (poll.rtp_timesnap - newest.rtp_timesnap + 2**31) % 2**32 - 2**31
        expect = newest.input_samples + d * ilen / olen
        return abs(expect - poll.input_samples) > RUN_MISS_BLOCKS * ilen

    def _adopt(self, new: CounterMap) -> None:
        held = self._held
        if held is None:
            self._held = new
            return
        rolled = held.rolled_forward(new.rtp_ref)
        blocks = (new.input_ref - rolled.input_index(new.rtp_ref)) / new.ilen
        # Same run: keep the least-late epoch, so lateness never creeps in.
        if abs(blocks) <= RUN_MISS_BLOCKS:
            new = CounterMap(rtp_ref=new.rtp_ref, input_ref=new.input_ref,
                             ilen=new.ilen, olen=new.olen,
                             input_samprate=new.input_samprate,
                             input_epoch_gps_s=min(held.input_epoch_gps_s,
                                                   new.input_epoch_gps_s),
                             n_agree=new.n_agree)
            if abs(blocks) >= 1e-6:
                self.steps += 1
                logger.error("COUNTER MAP STEP %s: intercept moved %.3f "
                             "block(s) within one radiod run (step %d)",
                             self._name, blocks, self.steps)
        self._held = new


class _StatusPair:
    """The (GPS_TIME, RTP_TIMESNAP) pair of one reply, shaped for rtp_to_utc."""

    def __init__(self, gps_time: int, rtp_timesnap: int, sample_rate: int):
        self._pair = (gps_time, rtp_timesnap)
        self.sample_rate = sample_rate

    def get_anchor(self):
        return self._pair


def least_late_anchor(control, ssrc: int, rtp: int, sample_rate: int, *,
                      authority_reader=None, counter: Optional[ChannelCounterMap] = None,
                      n: int = ANCHOR_POLLS, timeout: float = 1.0,
                      rtp_to_utc=None):
    """Anchor ``rtp`` through the least late of ``n`` status replies.

    Returns the AnchorUTC that puts ``rtp`` earliest, or None when no reply
    carried a status pair (never a host-clock fallback).  Each reply also
    feeds ``counter``, so a re-anchor after a radiod restart relearns the
    counter map at once.
    """
    from hamsci_dsp.timing import acquire_anchor_utc
    if rtp_to_utc is None:
        from ka9q import rtp_to_utc

    snapshot = None
    if authority_reader is not None:
        try:
            snapshot = authority_reader.read()
        except Exception as exc:  # noqa: BLE001
            logger.warning("authority read failed at anchor: %s", exc)
    best = None
    for _ in range(n):
        try:
            reply = control.poll_status(ssrc, timeout)
        except Exception as exc:  # noqa: BLE001
            logger.debug("anchor poll of ssrc %d failed: %s", ssrc, exc)
            continue
        if counter is not None:
            counter.add_reply(reply)
        gps, snap_rtp = getattr(reply, "gps_time", None), getattr(reply, "rtp_timesnap", None)
        if not gps or snap_rtp is None:
            continue
        a = acquire_anchor_utc(
            first_rtp=rtp, channel_info=_StatusPair(int(gps), int(snap_rtp), sample_rate),
            rtp_to_utc=rtp_to_utc, snapshot=snapshot, sample_rate=sample_rate)
        if not a.rtp_referenced:
            continue
        if best is None or a.utc < best.utc:
            best = a
    return best
