"""Tests for the band channel's counter map and least-late anchor (Task 5)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from hfdl_recorder.timing.channel_status import (
    ChannelCounterMap,
    gather_status,
    least_late_anchor,
    reply_matches,
)

F_IN = 64_800_000
F_OUT = 100_000
ILEN = 1_296_000            # 20 ms of input
OLEN = 2_000                # 20 ms of output
GPS_UNIX = 315_964_800 - 18  # Unix = GPS + this
SSRC = 0x1234
FREQ = 13_310_000.0


def reply(block, *, r0=5_000, i0=10**12, epoch_gps=1.4e9, late_s=0.0,
          rate=F_OUT, ssrc=SSRC, freq=FREQ):
    n_in = i0 + block * ILEN
    gps_s = epoch_gps + n_in / F_IN + late_s
    return SimpleNamespace(
        ssrc=ssrc, frequency=freq,
        gps_time=int(round(gps_s * 1e9)),
        rtp_timesnap=(r0 + block * OLEN) & 0xFFFFFFFF,
        filter_blocksize=ILEN, output_samprate=rate,
        frontend=SimpleNamespace(input_samples=n_in, input_samprate=F_IN))


@pytest.fixture(autouse=True)
def _fresh_reject_log(monkeypatch):
    import hfdl_recorder.timing.channel_status as cs
    monkeypatch.setattr(cs, "_REJECTS", cs._RejectLog())
    monkeypatch.setattr(cs, "_CONVERT_FAILS", cs._ConvertFailLog())


def _cm(**kw):
    cm = ChannelCounterMap("HFDL13", F_OUT, FREQ, **kw)
    cm.bind(SSRC)
    return cm


def test_map_from_polls_and_roll_forward():
    cm = _cm()
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
    cm = _cm()
    for b in range(5):
        cm.add_reply(reply(b, rate=12_000))
    assert cm.current() is None


def test_new_run_replaces_map():
    cm = _cm()
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
    cm = _cm(mono_fn=lambda: now[0])
    for b in range(5):
        cm.add_reply(reply(b))
    assert cm.current() is not None
    now[0] = 0.5 * 2**31 / F_OUT + 1
    assert cm.current() is None


class _Control:
    """Polls answer from ``replies``; a passive listen hears ``broadcast``."""

    def __init__(self, replies=(), broadcast=()):
        self.replies = list(replies)
        self.broadcast = list(broadcast)
        self.polled = []
        self.listened = []

    def poll_status(self, ssrc, timeout):
        self.polled.append((ssrc, timeout))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def listen_status(self, callback, duration=None, ssrcs=None):
        self.listened.append((frozenset(ssrcs), duration))
        for r in self.broadcast:
            if r.ssrc in ssrcs:
                callback(r)


def _rtp_to_utc(rtp, ch, wallclock_hint_sec=None):
    gps, snap = ch.get_anchor()
    d = ((rtp - snap + 2**31) % 2**32) - 2**31
    return gps / 1e9 + GPS_UNIX + d / ch.sample_rate


def _anchor(ctl, rtp, gate=lambda ssrc: True, counter=None, ssrc=SSRC):
    return least_late_anchor(ctl, ssrc, rtp, F_OUT, freq_hz=FREQ, gate=gate,
                             counter=counter, rtp_to_utc=_rtp_to_utc)


def test_least_late_anchor_keeps_earliest():
    lates = [0.004, 0.001, 0.250, TimeoutError("x"), 0.003]
    replies = [r if isinstance(r, Exception) else reply(10 + i, late_s=r)
               for i, r in enumerate(lates)]
    ctl = _Control(replies=replies)
    cm = _cm()
    rtp = 5_000 + 30 * OLEN
    a = _anchor(ctl, rtp, counter=cm)
    true_utc = 1.4e9 + (10**12 + 30 * ILEN) / F_IN + GPS_UNIX
    assert a.utc == pytest.approx(true_utc + 0.001, abs=1e-6)
    assert a.rtp_referenced
    assert {s for s, _ in ctl.polled} == {SSRC}  # only the band's own SSRC
    assert all(t <= 0.5 for _, t in ctl.polled)
    assert cm.current() is not None         # the replies taught the map too


def test_passive_listen_preferred_over_polls():
    ctl = _Control(broadcast=[reply(b, late_s=0.002 * b) for b in range(1, 6)])
    a = _anchor(ctl, 5_000)
    assert a is not None
    assert ctl.polled == []                 # heard enough without commanding
    assert ctl.listened and ctl.listened[0][0] == {SSRC}


def test_closed_gate_sends_no_command():
    ctl = _Control(replies=[reply(1)] * 5)
    assert _anchor(ctl, 5_000, gate=lambda ssrc: False) is None
    assert ctl.polled == []
    got = gather_status(ctl, SSRC, 3, rate=F_OUT, freq_hz=FREQ,
                        gate=lambda ssrc: False)
    assert got == [] and ctl.polled == []


def test_wrong_rate_or_frequency_or_ssrc_ignored():
    bad = [reply(1, rate=12_000), reply(2, freq=FREQ + 5000.0),
           reply(3, ssrc=SSRC + 1), reply(4, freq=None)]
    for r in bad:
        assert not reply_matches(r, SSRC, F_OUT, FREQ)
    assert reply_matches(reply(5), SSRC, F_OUT, FREQ)
    # The anchor ignores them all...
    ctl = _Control(replies=list(bad) + [reply(9, late_s=0.0)],
                   broadcast=[reply(7, freq=FREQ + 5000.0, late_s=-1.0)])
    a = _anchor(ctl, 5_000)
    true_utc = 1.4e9 + 10**12 / F_IN + GPS_UNIX
    assert a.utc == pytest.approx(true_utc, abs=1e-6)
    # ...and so does the counter map.
    cm = _cm()
    for r in bad * 2:
        cm.add_reply(r)
    assert cm.current() is None


def test_rebind_to_new_ssrc_forgets_map():
    cm = _cm()
    for b in range(5):
        cm.add_reply(reply(b))
    assert cm.current() is not None
    cm.bind(SSRC + 7)
    assert cm.current() is None
    cm.add_reply(reply(6))                  # the old SSRC: ignored
    assert cm.polls_wanted == 5


def test_least_late_anchor_never_falls_back_to_host_clock():
    ctl = _Control(replies=[SimpleNamespace(ssrc=SSRC, frequency=FREQ,
                                            output_samprate=F_OUT,
                                            gps_time=None, rtp_timesnap=None)] * 5)
    assert _anchor(ctl, 0) is None


def test_rejected_replies_logged_once_per_streak(caplog):
    caplog.set_level(logging.WARNING, logger="hfdl_recorder.timing.channel_status")
    bad = reply(1, freq=FREQ + 3.0)

    def gather(r):
        return gather_status(_Control(replies=[r] * 3), SSRC, 3, rate=F_OUT,
                             freq_hz=FREQ, gate=lambda s: True, listen_s=0)

    assert gather(bad) == [] and gather(bad) == []
    warns = [r.getMessage() for r in caplog.records]
    assert len(warns) == 1
    assert "13310003" in warns[0] and "13310000" in warns[0]
    assert str(F_OUT) in warns[0]
    assert len(gather(reply(2))) == 3          # a good reply ends the streak
    gather(bad)
    assert len(caplog.records) == 2
    # The counter map shares the same once-per-streak warning.
    cm = _cm()
    cm.add_reply(reply(5))                     # ends the open streak
    cm.add_reply(reply(3, rate=12_000))
    cm.add_reply(reply(4, rate=12_000))
    assert len(caplog.records) == 3


def _near_now_replies(lates, chain=None):
    """Replies whose GPS time sits near the host clock, so the real
    rtp_to_utc's wrap hint picks the same epoch."""
    import time as _t
    epoch = _t.time() - GPS_UNIX - 10**12 / F_IN
    return epoch, [reply(b, epoch_gps=epoch, late_s=late)
                   for b, late in enumerate(lates)]


def test_least_late_anchor_through_real_rtp_to_utc():
    """No fake: ka9q.rtp_to_utc reads every attribute a ChannelInfo has."""
    epoch, reps = _near_now_replies([0.003, 0.001, 0.002])
    ctl = _Control(replies=reps)
    a = least_late_anchor(ctl, SSRC, 5_000, F_OUT, freq_hz=FREQ,
                          gate=lambda s: True, n=3, listen_s=0)
    assert a is not None and a.rtp_referenced
    true_utc = epoch + 10**12 / F_IN + GPS_UNIX
    assert a.utc == pytest.approx(true_utc + 0.001, abs=2e-6)
    # A chain-delay correction from the band's ChannelInfo is applied, as
    # rtp_to_utc applies it for a ChannelInfo.
    epoch, reps = _near_now_replies([0.001])
    b = least_late_anchor(_Control(replies=reps), SSRC, 5_000, F_OUT,
                          freq_hz=FREQ, gate=lambda s: True, n=1, listen_s=0,
                          chain_delay_correction_ns=250_000)
    assert b.utc == pytest.approx(epoch + 10**12 / F_IN + GPS_UNIX + 0.001
                                  - 250e-6, abs=2e-6)


def test_conversion_failure_logged_once_per_streak(caplog):
    caplog.set_level(logging.WARNING)

    def broken(rtp, ch, wallclock_hint_sec=None):
        raise AttributeError("no such attribute")

    for _ in range(3):
        a = least_late_anchor(_Control(replies=[reply(b) for b in range(5)]),
                              SSRC, 5_000, F_OUT, freq_hz=FREQ,
                              gate=lambda s: True, listen_s=0, rtp_to_utc=broken)
        assert a is None
    # Fifteen failed pairs, one line, none from hamsci_dsp's per-call path.
    assert len(caplog.records) == 1
    assert caplog.records[0].name == "hfdl_recorder.timing.channel_status"
