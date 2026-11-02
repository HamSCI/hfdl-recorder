"""Tests for the band channel's counter map and least-late anchor (Task 5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hfdl_recorder.timing.channel_status import (
    ChannelCounterMap,
    least_late_anchor,
)

F_IN = 64_800_000
F_OUT = 100_000
ILEN = 1_296_000            # 20 ms of input
OLEN = 2_000                # 20 ms of output
GPS_UNIX = 315_964_800 - 18  # Unix = GPS + this


def reply(block, *, r0=5_000, i0=10**12, epoch_gps=1.4e9, late_s=0.0,
          rate=F_OUT):
    n_in = i0 + block * ILEN
    gps_s = epoch_gps + n_in / F_IN + late_s
    return SimpleNamespace(
        gps_time=int(round(gps_s * 1e9)),
        rtp_timesnap=(r0 + block * OLEN) & 0xFFFFFFFF,
        filter_blocksize=ILEN, output_samprate=rate,
        frontend=SimpleNamespace(input_samples=n_in, input_samprate=F_IN))


def test_map_from_polls_and_roll_forward():
    cm = ChannelCounterMap("HFDL13", F_OUT)
    assert cm.polls_wanted == 5
    for b in (0, 3, 9, 12, 20):
        cm.add_reply(reply(b, late_s=0.001 * b))
    m = cm.current()
    assert m is not None and m.n_agree == 5
    assert m.input_index(5_000 + 7 * OLEN) == 10**12 + 7 * ILEN
    assert cm.polls_wanted == 1
    # Least-late epoch kept.
    assert m.input_epoch_gps_s == pytest.approx(1.4e9, abs=1e-6)


def test_other_rate_reply_is_not_a_poll():
    cm = ChannelCounterMap("HFDL13", F_OUT)
    for b in range(5):
        cm.add_reply(reply(b, rate=12_000))
    assert cm.current() is None


def test_new_run_replaces_map():
    cm = ChannelCounterMap("HFDL13", F_OUT)
    for b in range(5):
        cm.add_reply(reply(b))
    # radiod restarted: INPUT_SAMPLES and RTP start over.
    for b in range(3):
        cm.add_reply(reply(b, r0=77, i0=10**6, epoch_gps=1.4e9 + 3600))
    m = cm.current()
    assert m is not None
    assert m.input_index(77) == 10**6
    assert m.input_epoch_gps_s == pytest.approx(1.4e9 + 3600, abs=1e-6)


def test_stale_map_dropped():
    now = [0.0]
    cm = ChannelCounterMap("HFDL13", F_OUT, mono_fn=lambda: now[0])
    for b in range(5):
        cm.add_reply(reply(b))
    assert cm.current() is not None
    now[0] = 0.5 * 2**31 / F_OUT + 1
    assert cm.current() is None


class _Control:
    def __init__(self, replies):
        self.replies = list(replies)
        self.ssrcs = []

    def poll_status(self, ssrc, timeout):
        self.ssrcs.append(ssrc)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _rtp_to_utc(rtp, ch, wallclock_hint_sec=None):
    gps, snap = ch.get_anchor()
    d = ((rtp - snap + 2**31) % 2**32) - 2**31
    return gps / 1e9 + GPS_UNIX + d / ch.sample_rate


def test_least_late_anchor_keeps_earliest():
    lates = [0.004, 0.001, 0.250, TimeoutError("x"), 0.003]
    replies = [r if isinstance(r, Exception) else reply(10 + i, late_s=r)
               for i, r in enumerate(lates)]
    ctl = _Control(replies)
    cm = ChannelCounterMap("HFDL13", F_OUT)
    rtp = 5_000 + 30 * OLEN
    a = least_late_anchor(ctl, 0x1234, rtp, F_OUT, counter=cm,
                          rtp_to_utc=_rtp_to_utc)
    true_utc = 1.4e9 + (10**12 + 30 * ILEN) / F_IN + GPS_UNIX
    assert a.utc == pytest.approx(true_utc + 0.001, abs=1e-6)
    assert a.rtp_referenced
    assert ctl.ssrcs == [0x1234] * 5        # only the band's own SSRC
    assert cm.current() is not None         # the replies taught the map too


def test_least_late_anchor_never_falls_back_to_host_clock():
    ctl = _Control([SimpleNamespace(gps_time=None, rtp_timesnap=None)] * 5)
    assert least_late_anchor(ctl, 1, 0, F_OUT, rtp_to_utc=_rtp_to_utc) is None
