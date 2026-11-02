"""Tests for the squitter window keeper (Task 5).

A synthetic stream runs through ``WindowKeeper.feed`` in 20 ms blocks, each
with its RTP timestamp.  ``anchor_fn`` is a fake that knows the stream's true
RTP-to-UTC map, so every test can check the saved labels against the truth.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pytest

from hamsci_dsp.counter_map import CounterMap
from hamsci_dsp.timing import AnchorUTC

from hfdl_recorder.bands import HfdlBand
from hfdl_recorder.timing.frame_grid import SLOT_S
from hfdl_recorder.timing.window_keeper import (
    LEAD_S,
    OUT_FS,
    PLAN_LEAD_S,
    WindowKeeper,
)

FS = 24_000
# 13.310 MHz centre, 24 kS/s: 13312 kHz sits at +2 kHz, 13315 at +5 kHz,
# 13303 at -7 kHz.  All three fit inside the band with the filter guard.
BAND = HfdlBand("TESTB", 13_310_000, FS, (13303, 13312, 13315))
FRAME0 = 1_792_000_000.0           # a frame boundary: 56,000,000 x 32 s
CHOSEN = ["Riverhead, New York", "San Francisco, California"]
WINDOW_OUT = round((LEAD_S + SLOT_S) * OUT_FS)


def _rtp_diff(a: int, b: int) -> int:
    d = (a - b) & 0xFFFFFFFF
    return d - 0x100000000 if d >= 0x80000000 else d


class Stream:
    """Synthetic IQ stream with a known RTP-to-UTC map.

    A segment is one contiguous RTP run; ``jump`` starts a new one (a radiod
    restart).  ``signal`` maps time since FRAME0 to complex samples (relative,
    since float64 phase at 1.8e9 s jitters by milliradians).
    """

    def __init__(self, utc0: float, rtp0: int, signal=None, block_s=0.02):
        self.utc = utc0
        self.rtp = rtp0 & 0xFFFFFFFF
        self.block = int(round(block_s * FS))
        self.signal = signal
        self.segments = [(self.rtp, self.utc)]

    def jump(self, rtp_new: int, utc_gap_s: float = 0.0):
        self.rtp = rtp_new & 0xFFFFFFFF
        self.utc += utc_gap_s
        self.segments.append((self.rtp, self.utc))

    def run(self, keeper, seconds: float, on_block=None, pace=True):
        n_blocks = int(round(seconds * FS / self.block))
        for _ in range(n_blocks):
            if on_block is not None:
                on_block(self.utc)
            t = (self.utc - FRAME0) + np.arange(self.block) / FS
            if self.signal is None:
                x = np.zeros(self.block, dtype=np.complex64)
            else:
                x = self.signal(t).astype(np.complex64)
            keeper.feed(x, self.rtp)
            if pace:
                # Real time runs far slower than this loop.  Give the writer
                # the time it would have had: to take the anchor, and to
                # empty its queue before the ring wraps.
                keeper.wait_anchored(2.0)
                keeper.flush()
            self.rtp = (self.rtp + self.block) & 0xFFFFFFFF
            self.utc += self.block / FS

    def true_utc(self, rtp: int) -> float:
        """UTC of an RTP value in the newest segment that holds it."""
        for seg_rtp, seg_utc in reversed(self.segments):
            d = _rtp_diff(rtp, seg_rtp)
            if d >= 0:
                return seg_utc + d / FS
        raise KeyError(rtp)

    def anchor_fn(self, rtp: int) -> AnchorUTC:
        return AnchorUTC(utc=self.true_utc(rtp), source="rtp_to_utc+authority",
                         offset_seconds=0.0, offset_ns=0, snapshot=None,
                         rtp_referenced=True)


def spdu(gs_id, name, freq_khz, k, frame_index=0, status=None, utc_sync=True):
    """A dumphfdl SPDU frame, shaped like the ND capture."""
    if status is None:
        status = [{"gs": {"id": gs_id, "name": name}, "utc_sync": utc_sync,
                   "freqs": [{"id": 1, "freq": float(freq_khz)}]}]
    return {"hfdl": {
        "t": {"sec": 1, "usec": 0},
        "freq": int(freq_khz * 1000),
        "spdu": {"src": {"type": "Ground station", "id": gs_id, "name": name},
                 "frame_index": frame_index, "frame_offset": k,
                 "gs_status": status}}}


def _keeper(tmp_path, stream, **kw):
    kw.setdefault("counter_map_fn", lambda: None)
    kw.setdefault("min_free_gb", 0.0)      # the dev box's own disk is no test
    k = WindowKeeper(band=BAND, fs=FS, out_dir=tmp_path,
                     ground_stations=CHOSEN, anchor_fn=stream.anchor_fn, **kw)
    k.start()
    return k


def _windows(tmp_path):
    out = []
    for p in sorted(tmp_path.glob("*/*.npz")):
        with np.load(p) as z:
            out.append((p, z["x"], json.loads(str(z["meta"]))))
    return out


def _settle(keeper):
    keeper.flush()


def test_window_starts_30ms_before_nominal(tmp_path):
    s = Stream(FRAME0 + 1.0, rtp0=123_456)
    k = _keeper(tmp_path, s)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        s.run(k, 40.0)
        _settle(k)
    finally:
        k.stop()
    wins = _windows(tmp_path)
    assert len(wins) == 1, repr(sorted(k.stats().items()))
    path, x, meta = wins[0]
    nominal = FRAME0 + 5 * SLOT_S
    assert meta["nominal_utc"] == pytest.approx(nominal, abs=1e-6)
    assert abs(meta["t0_utc"] - (nominal - 0.030)) <= 1.0 / OUT_FS
    assert meta["t0_utc"] <= nominal - 0.030 + 1e-9   # never less lead
    assert len(x) == WINDOW_OUT
    assert x.dtype == np.complex64
    assert meta["fs"] == OUT_FS
    assert meta["freq_khz"] == 13312
    assert meta["gs_id"] == 4 and meta["gs_name"] == "Riverhead, New York"
    assert meta["frame_offset"] == 5
    assert meta["frame_index"] == (int(FRAME0) // 32) % 2700
    assert meta["band"] == "TESTB"
    assert meta["utc_sync"] is True
    assert meta["anchor_source"] == "rtp_to_utc+authority"
    assert meta["input_index0"] is None            # no counter map
    assert path.name == f"4-13312-{meta['frame_index']}.npz"
    assert path.parent.name == "20261014"
    assert not list(tmp_path.glob("*/*.tmp"))


def test_t0_label_matches_signal_onset(tmp_path):
    """A tone that switches on at the nominal slot start appears 30 ms into
    the window, on the +1440 Hz subcarrier, so label and content agree."""
    nominal = FRAME0 + 5 * SLOT_S
    f_tone = (13312 * 1000 + 1440) - BAND.center_hz

    def sig(t):
        on = t >= nominal - FRAME0
        return on * np.exp(2j * np.pi * f_tone * t)

    s = Stream(FRAME0 + 1.0, rtp0=7, signal=sig)
    k = _keeper(tmp_path, s)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        s.run(k, 16.0)
        _settle(k)
    finally:
        k.stop()
    (_, x, meta), = _windows(tmp_path)
    mag = np.abs(x)
    onset = int(np.argmax(mag > 0.5))
    expect = (nominal - meta["t0_utc"]) * OUT_FS
    assert abs(onset - expect) <= 2
    # Spectrum after the onset: one peak at +1440 Hz.
    seg = x[400:400 + 6000]
    spec = np.abs(np.fft.fft(seg))
    freqs = np.fft.fftfreq(len(seg), 1.0 / OUT_FS)
    assert freqs[int(np.argmax(spec))] == pytest.approx(1440.0, abs=2.0)
    # Gain stays flat: amplitude 1 in, about 1 out.
    assert np.median(mag[400:-400]) == pytest.approx(1.0, abs=0.02)


def test_negative_half_kept_flat(tmp_path):
    """The analyser reads noise from the negative half: a tone at -2.5 kHz
    from the dial must pass at full amplitude."""
    f_tone = (13312 * 1000 - 2500) - BAND.center_hz
    s = Stream(FRAME0 + 1.0, rtp0=7,
               signal=lambda t: np.exp(2j * np.pi * f_tone * t))
    k = _keeper(tmp_path, s)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        s.run(k, 16.0)
        _settle(k)
    finally:
        k.stop()
    (_, x, _), = _windows(tmp_path)
    assert np.median(np.abs(x[400:-400])) == pytest.approx(1.0, abs=0.02)


def test_neighbour_channel_suppressed(tmp_path):
    """A burst on the next HFDL channel (+3 kHz dial, +4440 Hz) stays out."""
    f_tone = (13315 * 1000 + 1440) - BAND.center_hz
    s = Stream(FRAME0 + 1.0, rtp0=7,
               signal=lambda t: np.exp(2j * np.pi * f_tone * t))
    k = _keeper(tmp_path, s)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        s.run(k, 16.0)
        _settle(k)
    finally:
        k.stop()
    (_, x, _), = _windows(tmp_path)
    assert np.max(np.abs(x[400:-400])) < 1e-3


def test_counter_label(tmp_path):
    s = Stream(FRAME0 + 1.0, rtp0=1000)
    # Input rate 4x the output rate; input index 10**9 at rtp 1000.
    cmap = CounterMap(rtp_ref=1000, input_ref=10**9, ilen=4 * 480, olen=480,
                      input_samprate=4 * FS,
                      input_epoch_gps_s=(FRAME0 + 1.0 - 315964800 + 18)
                      - 10**9 / (4 * FS),
                      n_agree=5)
    k = _keeper(tmp_path, s, counter_map_fn=lambda: cmap)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        s.run(k, 16.0)
        _settle(k)
    finally:
        k.stop()
    (_, _, meta), = _windows(tmp_path)
    rtp0 = meta["rtp0"]
    assert meta["input_index0"] == pytest.approx(cmap.input_index(rtp0))
    assert meta["input_samprate"] == 4 * FS
    assert meta["counter_map"] == cmap.to_dict()
    assert meta["input_epoch_gps_s"] == cmap.input_epoch_gps_s
    # The counter label and the status-pair label name the same sample.
    off = (meta["input_index0"] - 10**9) / (4 * FS)
    assert FRAME0 + 1.0 + off == pytest.approx(meta["t0_utc"], abs=1e-6)


def test_counter_map_from_another_run_refused(tmp_path):
    """A map whose implied UTC sits hours from the anchor belongs to another
    radiod run: the window saves with input_index0 = null."""
    s = Stream(FRAME0 + 1.0, rtp0=1000)
    cmap = CounterMap(rtp_ref=1000, input_ref=10**9, ilen=1920, olen=480,
                      input_samprate=4 * FS,
                      input_epoch_gps_s=(FRAME0 - 7200 - 315964800 + 18)
                      - 10**9 / (4 * FS),
                      n_agree=5)
    k = _keeper(tmp_path, s, counter_map_fn=lambda: cmap)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        s.run(k, 16.0)
        _settle(k)
    finally:
        k.stop()
    (_, _, meta), = _windows(tmp_path)
    assert meta["input_index0"] is None
    assert k.stats()["counter_map_refused"] == 1


def test_only_live_chosen_stations(tmp_path):
    s = Stream(FRAME0 + 1.0, rtp0=5)
    k = _keeper(tmp_path, s)
    try:
        k.note_squitter(spdu(9, "Agana, Guam", 13312, 5))
        # Chosen, but its frequency lies outside this band.
        k.note_squitter(spdu(4, "Riverhead, New York", 11387, 6))
        s.run(k, 40.0)
        _settle(k)
    finally:
        k.stop()
    assert _windows(tmp_path) == []
    assert k.stats()["saved"] == 0


def test_follows_frequency_change(tmp_path):
    """Riverhead moves from 13312 (k=5) to 13315 (k=8).  13312 windows stop
    once it has gone 3 frames unheard; 13315 windows start."""
    s = Stream(FRAME0 + 1.0, rtp0=99)
    k = _keeper(tmp_path, s)
    both = [{"gs": {"id": 4, "name": "Riverhead, New York"}, "utc_sync": True,
             "freqs": [{"id": 1, "freq": 13312.0}, {"id": 2, "freq": 13315.0}]}]
    seen: set[int] = set()
    last_old = {"utc": None}
    switch_at = FRAME0 + 3 * 32

    def on_block(utc):
        # Both squitters of frame f decode 3 s after the k=8 slot ends.
        f = int((utc - FRAME0 - 3.0 - 9 * SLOT_S) // 32)
        if f < 0 or f in seen:
            return
        seen.add(f)
        if FRAME0 + f * 32 < switch_at:
            k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5,
                                 status=both))
            last_old["utc"] = utc
        k.note_squitter(spdu(4, "Riverhead, New York", 13315, 8, status=both))

    try:
        s.run(k, 9 * 32, on_block=on_block)
        _settle(k)
    finally:
        k.stop()
    wins = [m for _, _, m in _windows(tmp_path)]
    old = [m["nominal_utc"] for m in wins if m["freq_khz"] == 13312]
    new = [m["nominal_utc"] for m in wins if m["freq_khz"] == 13315]
    assert old and new
    silent_from = last_old["utc"]
    # No 13312 window more than 3 frames (+ the planning lead) after the
    # last time it was heard; and some within that span (no early cut-off).
    assert max(old) <= silent_from + 3 * 32 + PLAN_LEAD_S
    assert max(old) > switch_at
    assert max(new) > max(old) + 32


def test_announcement_drops_frequency(tmp_path):
    """A squitter whose gs_status no longer lists 13312 stops it at once."""
    s = Stream(FRAME0 + 1.0, rtp0=99)
    k = _keeper(tmp_path, s)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        s.run(k, 32.0)                     # one 13312 window (frame 0)
        only_new = [{"gs": {"id": 4, "name": "Riverhead, New York"},
                     "utc_sync": True, "freqs": [{"id": 2, "freq": 11387.0}]}]
        # Another GS's squitter carries Riverhead's new frequency list.
        k.note_squitter(spdu(1, "San Francisco, California", 13303, 2,
                             status=only_new))
        s.run(k, 64.0)
        _settle(k)
    finally:
        k.stop()
    wins = [m for _, _, m in _windows(tmp_path)]
    assert [m["freq_khz"] for m in wins if m["gs_id"] == 4] == [13312]
    # San Francisco's own channel is now live.
    assert any(m["gs_id"] == 1 and m["freq_khz"] == 13303 for m in wins)


def test_reanchors_on_rtp_jump(tmp_path):
    s = Stream(FRAME0 + 1.0, rtp0=1_000_000)
    k = _keeper(tmp_path, s)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        # Stop the first run 1 s into frame 1's window, then restart radiod:
        # its RTP counter starts over, and the stream resumes 2 s later.
        s.run(k, 32 + 5 * SLOT_S + 1.0 - 1.0)
        s.jump(rtp_new=17, utc_gap_s=2.0)
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        s.run(k, 64.0)
        _settle(k)
    finally:
        k.stop()
    st = k.stats()
    assert st["reanchors"] == 1
    wins = [m for _, _, m in _windows(tmp_path)]
    nominals = sorted(m["nominal_utc"] for m in wins)
    f = FRAME0 + 5 * SLOT_S
    # Frame 0 before the jump; frame 1 spans it and is not saved; frames 2
    # and 3 after it, labelled through the new anchor.
    assert nominals == pytest.approx([f, f + 64, f + 96], abs=1e-6)
    jump_utc = s.segments[1][1]
    for m in wins:
        start = m["t0_utc"]
        end = start + WINDOW_OUT / OUT_FS
        assert end < jump_utc - 2.0 or start >= jump_utc
        assert abs(m["t0_utc"] - (m["nominal_utc"] - 0.030)) <= 1.0 / OUT_FS


def test_rtp_wrap_is_not_a_jump(tmp_path):
    """The 32-bit RTP counter wraps mid-window; the stream stays continuous,
    so the window saves whole and nothing re-anchors."""
    wrap_at = FRAME0 + 5 * SLOT_S + 1.0
    rtp0 = (2**32 - int(round((wrap_at - (FRAME0 + 1.0)) * FS))) & 0xFFFFFFFF
    s = Stream(FRAME0 + 1.0, rtp0=rtp0)
    k = _keeper(tmp_path, s)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        s.run(k, 40.0)
        _settle(k)
    finally:
        k.stop()
    assert k.stats()["reanchors"] == 0
    (_, _, meta), = _windows(tmp_path)
    assert abs(meta["t0_utc"] - (meta["nominal_utc"] - 0.030)) <= 1.0 / OUT_FS


def test_feed_never_blocks(tmp_path):
    s = Stream(FRAME0 + 1.0, rtp0=5, block_s=1.0)
    k = _keeper(tmp_path, s)
    try:
        for gs, name in ((4, "Riverhead, New York"),
                         (1, "San Francisco, California")):
            for freq, kk in ((13312, gs), (13315, gs + 4), (13303, gs + 8)):
                k.note_squitter(spdu(gs, name, freq, kk, status=[]))
        s.run(k, 1.0)
        assert k.wait_anchored(2.0)
        k._save_window = lambda job: time.sleep(1.0)
        t0 = time.perf_counter()
        s.run(k, 100.0, pace=False)
        dt = time.perf_counter() - t0
        st = k.stats()
    finally:
        k.stop(timeout=0.0)
    assert dt < 0.1, f"100 feeds took {dt:.3f} s"
    # Six live channels, three frames each: far more windows than the
    # queue holds, so most drop and are counted.
    assert st["dropped_queue"] >= 6 * 3 - 8 - 1


def test_disk_low_stops_saving(tmp_path, caplog):
    s = Stream(FRAME0 + 1.0, rtp0=5)
    k = _keeper(tmp_path, s, min_free_gb=5.0,
                free_bytes_fn=lambda path: 1 * 1024**3)
    caplog.set_level(logging.WARNING, logger="hfdl_recorder.timing.window_keeper")
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        k.note_squitter(spdu(4, "Riverhead, New York", 13315, 9, status=[]))
        s.run(k, 70.0)
        _settle(k)
    finally:
        k.stop()
    assert _windows(tmp_path) == []
    st = k.stats()
    assert st["saved"] == 0
    assert st["dropped_disk"] >= 3
    warnings = [r for r in caplog.records if "free" in r.getMessage()]
    assert len(warnings) == 1


def test_prunes_old_day_folders(tmp_path):
    s = Stream(FRAME0 + 1.0, rtp0=5)
    for name in ("20261001", "20261008", "20261009", "20261013", "notadate"):
        (tmp_path / name).mkdir()
    k = _keeper(tmp_path, s, retention_days=5)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        s.run(k, 16.0)                     # saves into 20261014, then prunes
        _settle(k)
    finally:
        k.stop()
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["20261009", "20261013", "20261014", "notadate"]


def test_feed_cost_is_small(tmp_path):
    """feed() copies into the ring and books windows; it does no DSP."""
    s = Stream(FRAME0 + 1.0, rtp0=5)
    k = _keeper(tmp_path, s)
    try:
        k.note_squitter(spdu(4, "Riverhead, New York", 13312, 5))
        x = np.zeros(480, dtype=np.complex64)
        rtp = 5
        n = 3000
        t0 = time.perf_counter()
        for _ in range(n):
            k.feed(x, rtp)
            rtp += 480
        per_call = (time.perf_counter() - t0) / n
    finally:
        k.stop()
    assert per_call < 200e-6, f"{per_call * 1e6:.1f} us per feed"
