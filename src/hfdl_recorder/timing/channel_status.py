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
channel with infinite lifetime.  So ``gather_status`` listens first: a passive
``listen_status`` for the one SSRC hears radiod's status broadcasts and any
replies to other commands, and can mint nothing.  Only when that falls short
does it poll, and only while ``gate(ssrc)`` holds.  The daemon's gate
(``BandPipeline.may_poll``) admits the slot's CURRENT SSRC (MultiStream may
re-key a slot on restore), while the stream is up and a sample arrived in the
last 0.5 s.  Every reply must match the band's SSRC, output rate and centre
frequency, or it is ignored.

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
LISTEN_S = 1.0               # passive listen before any poll
POLL_TIMEOUT_S = 0.5
FREQ_TOL_HZ = 1.0


def reply_matches(reply, ssrc: int, output_samprate: int, freq_hz: float) -> bool:
    """True only for a status reply from this band's own channel."""
    if getattr(reply, "ssrc", None) != ssrc:
        return False
    if getattr(reply, "output_samprate", None) != output_samprate:
        return False
    f = getattr(reply, "frequency", None)
    return f is not None and abs(float(f) - float(freq_hz)) <= FREQ_TOL_HZ


class _RejectLog:
    """Warn once per streak, per SSRC, when replies fail ``reply_matches``.

    If radiod reports another rate or frequency than the band expects, every
    reply gets rejected and no anchor or counter map ever forms; without this
    the failure would stay silent.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._bad: set = set()

    def check(self, reply, ssrc: int, rate: int, freq_hz: float) -> bool:
        ok = reply_matches(reply, ssrc, rate, freq_hz)
        with self._lock:
            if ok:
                if ssrc in self._bad:
                    self._bad.discard(ssrc)
                    logger.info("status replies for ssrc %d match the band again", ssrc)
                return True
            first = ssrc not in self._bad
            self._bad.add(ssrc)
        if first:
            f = getattr(reply, "frequency", None)
            logger.warning(
                "status reply rejected: ssrc %s, output rate %s, frequency %s Hz; "
                "expected ssrc %d, rate %d, frequency %.0f Hz (+-%.0f).  No anchor "
                "or counter map forms from such replies (further rejections for "
                "this ssrc not logged until one matches)",
                getattr(reply, "ssrc", None), getattr(reply, "output_samprate", None),
                "None" if f is None else f"{float(f):.0f}",
                ssrc, rate, freq_hz, FREQ_TOL_HZ)
        return False


_REJECTS = _RejectLog()


class _ConvertFailLog:
    """rtp_to_utc raising on a status pair: logged once per streak per SSRC."""

    def __init__(self):
        self._lock = threading.Lock()
        self._bad: set = set()

    def failed(self, ssrc: int, exc: Exception) -> None:
        with self._lock:
            first = ssrc not in self._bad
            self._bad.add(ssrc)
        if first:
            logger.warning("rtp_to_utc failed on ssrc %d's status pair: %s; no "
                           "anchor from it (further failures not logged until "
                           "one converts)", ssrc, exc)

    def ok(self, ssrc: int) -> None:
        with self._lock:
            if ssrc not in self._bad:
                return
            self._bad.discard(ssrc)
        logger.info("rtp_to_utc converts ssrc %d's status pairs again", ssrc)


_CONVERT_FAILS = _ConvertFailLog()


def gather_status(control, ssrc: int, n: int, *, rate: int, freq_hz: float,
                  gate: Callable[[int], bool], listen_s: float = LISTEN_S,
                  poll_timeout: float = POLL_TIMEOUT_S) -> list:
    """Up to ``n`` validated status replies for ``ssrc``.

    Listens passively first.  Then polls, at most ``n`` times, each only
    while ``gate(ssrc)`` holds; a closed gate stops the polling at once.
    """
    got: list = []

    def heard(st):
        if len(got) < n and _REJECTS.check(st, ssrc, rate, freq_hz):
            got.append(st)

    listen = getattr(control, "listen_status", None)
    if listen is not None and listen_s > 0:
        try:
            listen(heard, duration=listen_s, ssrcs={ssrc})
        except Exception as exc:  # noqa: BLE001
            logger.debug("status listen for ssrc %d failed: %s", ssrc, exc)
    for _ in range(n):
        if len(got) >= n or not gate(ssrc):
            break
        try:
            reply = control.poll_status(ssrc, min(poll_timeout, POLL_TIMEOUT_S))
        except Exception as exc:  # noqa: BLE001
            logger.debug("status poll of ssrc %d failed: %s", ssrc, exc)
            continue
        if _REJECTS.check(reply, ssrc, rate, freq_hz):
            got.append(reply)
    return got


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

    def __init__(self, name: str, output_samprate: int, freq_hz: float,
                 mono_fn: Callable[[], float] = time.monotonic):
        self._name = name
        self._rate = int(output_samprate)
        self._freq = float(freq_hz)
        self._ssrc: Optional[int] = None
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

    def bind(self, ssrc: int, freq_hz: Optional[float] = None) -> None:
        """Follow this SSRC (and the frequency its channel reports).  A new
        SSRC (a re-keyed slot) starts the map over."""
        with self._lock:
            if freq_hz:
                self._freq = float(freq_hz)
            if ssrc == self._ssrc:
                return
            self._ssrc = ssrc
            self._polls.clear()
            self._held = None
            self._last_rtp = None
            self._last_fresh = None

    @property
    def ssrc(self) -> Optional[int]:
        return self._ssrc

    def add_reply(self, reply) -> None:
        """Learn from one reply, if it comes from the bound channel."""
        if self._ssrc is None or not _REJECTS.check(reply, self._ssrc, self._rate, self._freq):
            return
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
    """The (GPS_TIME, RTP_TIMESNAP) pair of one reply, shaped like the
    ChannelInfo that ka9q.rtp_to_utc reads: get_anchor(), sample_rate and
    chain_delay_correction_ns (ChannelInfo defaults it to None)."""

    def __init__(self, gps_time: int, rtp_timesnap: int, sample_rate: int,
                 chain_delay_correction_ns: Optional[int] = None):
        self._pair = (gps_time, rtp_timesnap)
        self.sample_rate = sample_rate
        self.chain_delay_correction_ns = chain_delay_correction_ns

    def get_anchor(self):
        return self._pair


def least_late_anchor(control, ssrc: int, rtp: int, sample_rate: int, *,
                      freq_hz: float, gate: Callable[[int], bool],
                      authority_reader=None, counter: Optional[ChannelCounterMap] = None,
                      n: int = ANCHOR_POLLS, listen_s: float = LISTEN_S,
                      poll_timeout: float = POLL_TIMEOUT_S, rtp_to_utc=None,
                      chain_delay_correction_ns: Optional[int] = None):
    """Anchor ``rtp`` through the least late of up to ``n`` status replies.

    Returns the AnchorUTC that puts ``rtp`` earliest, or None when no valid
    reply carried a status pair (never a host-clock fallback).  Each reply
    also feeds ``counter``, so a re-anchor relearns the counter map at once.

    ``chain_delay_correction_ns`` comes from the band's ChannelInfo (None
    unless something set it), as rtp_to_utc would see it there.  Each pair is
    tried through rtp_to_utc here first: a pair it cannot convert is skipped
    and logged once per streak, so acquire_anchor_utc's own per-call warning
    never floods the journal.
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
    replies = gather_status(control, ssrc, n, rate=sample_rate, freq_hz=freq_hz,
                            gate=gate, listen_s=listen_s, poll_timeout=poll_timeout)
    best = None
    for reply in replies:
        if counter is not None and counter.ssrc == ssrc:
            counter.add_reply(reply)
        gps, snap_rtp = getattr(reply, "gps_time", None), getattr(reply, "rtp_timesnap", None)
        if not gps or snap_rtp is None:
            continue
        pair = _StatusPair(int(gps), int(snap_rtp), sample_rate,
                           chain_delay_correction_ns)
        try:
            ok = rtp_to_utc(int(rtp) & 0xFFFFFFFF, pair) is not None
        except Exception as exc:  # noqa: BLE001
            _CONVERT_FAILS.failed(ssrc, exc)
            continue
        if not ok:
            continue
        _CONVERT_FAILS.ok(ssrc)
        a = acquire_anchor_utc(
            first_rtp=rtp, channel_info=pair,
            rtp_to_utc=rtp_to_utc, snapshot=snapshot, sample_rate=sample_rate)
        if not a.rtp_referenced:
            continue
        if best is None or a.utc < best.utc:
            best = a
    return best
