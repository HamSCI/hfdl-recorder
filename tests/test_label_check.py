"""Tests for label-check: labelled_lag_seconds, the TOML writer, CLI shape."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from hfdl_recorder.timing.label_check import (
    build_repeat_record,
    labelled_lag_seconds,
    write_corrections,
)

REF_FS = 12000
PROBE_FS = 100000
DURATION = 2.0          # seconds; periodic, so both rates sample it exactly
BAND_HZ = 5000.0        # signal occupies +-5 kHz, inside the 12 kS/s Nyquist


def _spectrum(seed: int) -> dict:
    """Random complex line spectrum on 1/DURATION grid, |f| < BAND_HZ."""
    rng = np.random.default_rng(seed)
    k = np.arange(-int(BAND_HZ * DURATION), int(BAND_HZ * DURATION))
    coef = rng.normal(size=k.size) + 1j * rng.normal(size=k.size)
    return {"k": k, "coef": coef}


def _sample(spec: dict, fs: int, delay_s: float = 0.0) -> np.ndarray:
    """Exact samples of x(t - delay_s) at fs; x is periodic over DURATION."""
    n = int(fs * DURATION)
    full = np.zeros(n, dtype=complex)
    f = spec["k"] / DURATION
    full[spec["k"] % n] = spec["coef"] * np.exp(-2j * np.pi * f * delay_s)
    return np.fft.ifft(full) * n / np.sqrt(spec["k"].size)


def test_zero_lag_when_labels_agree():
    spec = _spectrum(1)
    ref = _sample(spec, REF_FS)
    probe = _sample(spec, PROBE_FS)
    lag, corr = labelled_lag_seconds(ref, REF_FS, 1000.0, probe, PROBE_FS, 1000.0)
    assert abs(lag) < 2e-6
    assert corr > 0.9


def test_reports_known_label_offset():
    spec = _spectrum(2)
    ref = _sample(spec, REF_FS)
    probe = _sample(spec, PROBE_FS)
    lag, corr = labelled_lag_seconds(
        ref, REF_FS, 1000.0, probe, PROBE_FS, 1000.0 + 3.7e-3)
    assert lag == pytest.approx(3.7e-3, abs=2e-6)
    assert corr > 0.9


def test_fractional_sample_offset():
    spec = _spectrum(3)
    ref = _sample(spec, REF_FS)
    # Probe content runs 0.3 ref-sample behind its label: label runs late.
    probe = _sample(spec, PROBE_FS, delay_s=0.3 / REF_FS)
    lag, _ = labelled_lag_seconds(ref, REF_FS, 1000.0, probe, PROBE_FS, 1000.0)
    assert lag == pytest.approx(0.3 / REF_FS, abs=2e-6)


def test_uncorrelated_streams_report_low_corr():
    ref = _sample(_spectrum(4), REF_FS)
    probe = _sample(_spectrum(5), PROBE_FS)
    _, corr = labelled_lag_seconds(ref, REF_FS, 1000.0, probe, PROBE_FS, 1000.0)
    assert corr < 0.2


def test_whole_second_label_offset_with_partial_overlap():
    spec = _spectrum(6)
    ref = _sample(spec, REF_FS)
    probe = _sample(spec, PROBE_FS)
    lag, corr = labelled_lag_seconds(
        ref, REF_FS, 1000.0, probe, PROBE_FS, 1000.0 + 0.25)
    assert lag == pytest.approx(0.25, abs=2e-6)
    assert corr > 0.9


def test_repeat_record_shape():
    rec = build_repeat_record(
        band="HFDL13", probe_fs=80000, ref_fs=12000, lag_s=1.5e-4,
        corr=0.97, anchor_source="rtp_to_utc+authority",
        ref_source="rtp_to_utc", ref_encoding=4,
        t_level_active="T3")
    assert list(rec) == ["band", "probe_fs", "ref_fs", "lag_us", "corr",
                         "anchor_source", "ref_source", "ref_encoding",
                         "t_level_active"]
    assert rec["lag_us"] == pytest.approx(150.0)
    json.dumps(rec)


def test_write_corrections_median_per_band(tmp_path):
    out = tmp_path / "sub" / "label_corrections.toml"
    recs = [
        {"band": "HFDL13", "lag_us": 100.0},
        {"band": "HFDL13", "lag_us": 104.0},
        {"band": "HFDL13", "lag_us": 500.0},
        {"band": "HFDL8", "lag_us": -20.0},
    ]
    write_corrections(out, recs)
    tomllib = pytest.importorskip("tomllib") if sys.version_info >= (3, 11) \
        else pytest.importorskip("tomli")
    data = tomllib.loads(out.read_text())
    assert data["HFDL13"]["lag_us"] == 104.0
    assert data["HFDL13"]["spread_us"] == 400.0
    assert data["HFDL13"]["n"] == 3
    assert data["HFDL8"]["lag_us"] == -20.0


def test_write_corrections_merges_other_bands(tmp_path):
    out = tmp_path / "c.toml"
    write_corrections(out, [{"band": "HFDL8", "lag_us": 1.0}])
    write_corrections(out, [{"band": "HFDL13", "lag_us": 2.0}])
    text = out.read_text()
    assert "[HFDL8]" in text and "[HFDL13]" in text


def test_cli_parses_label_check():
    from hfdl_recorder.cli import build_parser
    args = build_parser().parse_args([
        "label-check", "--radiod-id", "rx", "--ref-freq-hz", "10000000",
        "--band", "HFDL13", "--seconds", "30", "--repeats", "5", "--json"])
    assert args.command == "label-check"
    assert args.ref_freq_hz == 10_000_000.0
    assert args.band == "HFDL13"
    assert args.seconds == 30.0 and args.repeats == 5 and args.json
    assert str(args.out) == "/var/lib/hfdl-recorder/label_corrections.toml"
# ---- _run_repeat against a fake radiod ----
#
# The fake models the radiod behaviour that matters for channel hygiene
# (ka9q-radio radio_status.c, radio.c):
#   * Any command addressed to one SSRC (poll, tune, LIFETIME, ...) goes
#     through lookup_or_create_chan: an unknown SSRC is MINTED, at 0 Hz, with
#     the template's lifetime 0 = infinite.
#   * Every command resets the countdown to lifestart; a LIFETIME tag sets
#     both lifestart and the countdown.
#   * The countdown runs only when > 0; at 0 the channel is destroyed.
#   * A poll of SSRC 0xffffffff (discovery) lists the running channels and
#     mints nothing.

import threading
from types import SimpleNamespace

from hfdl_recorder.bands import HFDL_BANDS
from hfdl_recorder.timing import label_check as lc

REF_SSRC = 111
PROBE_SSRC = 222
FREQ = 10e6
DISCOVER_FRAMES = 150          # a ~3 s discovery listen at 50 frames/s


class _FakeRadiod:
    """Fake radiod plus the RadiodControl surface label-check uses."""

    status_address = "fake-radiod.local"

    def __init__(self, stale_probe=False, honours_lifetime=True,
                 create_fails=False):
        self.chans = {REF_SSRC: {"freq": FREQ, "life": 0, "start": 0}}
        if stale_probe:
            self.chans[PROBE_SSRC] = {"freq": FREQ, "life": 0, "start": 0}
        self.honours_lifetime = honours_lifetime
        self.create_fails = create_fails
        self.log = []          # (ssrc, op, lifetime) per SSRC-addressed command
        self.minted = []
        self.lock = threading.Lock()

    # -- radiod --
    def _command(self, ssrc, op, lifetime=None, freq=None):
        assert ssrc not in (0, 0xFFFFFFFF)
        with self.lock:
            ch = self.chans.get(ssrc)
            if ch is None:
                ch = {"freq": 0.0, "life": 0, "start": 0}
                self.chans[ssrc] = ch
                self.minted.append(ssrc)
            ch["life"] = ch["start"]
            if lifetime is not None and self.honours_lifetime:
                ch["start"] = ch["life"] = lifetime
            if freq is not None:
                ch["freq"] = freq
            self.log.append((ssrc, op, lifetime))
            return ch

    def tick(self, frames):
        with self.lock:
            for ssrc, ch in list(self.chans.items()):
                if ch["life"] > 0:
                    ch["life"] -= frames
                    if ch["life"] <= 0:
                        del self.chans[ssrc]

    def discover(self):
        self.tick(DISCOVER_FRAMES)
        with self.lock:
            return {s: SimpleNamespace(ssrc=s, frequency=c["freq"])
                    for s, c in self.chans.items()}

    def cmds(self, ssrc):
        return [(op, life) for s, op, life in self.log if s == ssrc]

    # -- RadiodControl surface --
    def poll_channel(self, ssrc, expected_freq=None, timeout=1.0,
                     allow_idle=False, **kw):
        ch = self._command(ssrc, "poll")
        if ch["freq"] or allow_idle:
            return SimpleNamespace(ssrc=ssrc, frequency=ch["freq"])
        return None

    def set_channel_lifetime(self, ssrc, lifetime, encoding=None):
        self._command(ssrc, "lifetime", lifetime=lifetime)

    def remove_channel(self, ssrc):
        self._command(ssrc, "remove", freq=0.0)

    def ensure_channel(self, frequency_hz, preset, sample_rate, encoding=0,
                       lifetime=None, **kw):
        # As ka9q's ensure_channel: pre-poll, create, verify poll.
        ssrc = PROBE_SSRC
        self.poll_channel(ssrc, expected_freq=frequency_hz)
        if self.create_fails:
            raise TimeoutError("fake: create never acknowledged")
        self._command(ssrc, "create", lifetime=lifetime, freq=frequency_hz)
        self.poll_channel(ssrc, expected_freq=frequency_hz)
        return SimpleNamespace(ssrc=ssrc, frequency=frequency_hz,
                               sample_rate=sample_rate, encoding=encoding,
                               multicast_address="239.9.9.9")


class _FakeStream:
    """Reference stream: delivers a full capture synchronously on start()."""
    def __init__(self, channel, on_samples):
        self.on_samples = on_samples

    def start(self):
        n = int(self.on_samples.n_target)
        self.on_samples(np.ones(n, dtype=np.complex64),
                        SimpleNamespace(delivered_rtp_start=1))

    def stop(self):
        pass


def _fake_multi(created, on_start=None, stop_raises=False):
    class FakeMulti:
        """Mirrors ka9q.MultiStream's control traffic: add_channel goes
        through ensure_channel, set_channel_lifetime is a no-op once the
        slot is gone, stop() sends nothing."""

        def __init__(self, control):
            self.control = control
            self.slots = {}

        def add_channel(self, **kw):
            created.append(kw)
            info = self.control.ensure_channel(**kw)
            self.slots[info.ssrc] = kw
            self.cb = kw["on_samples"]
            return info

        def prune_frequency(self, frequency_hz, keep_ssrc=None):
            removed = list(self.slots)
            self.slots.clear()
            return removed

        def start(self):
            if on_start is not None:
                on_start()
            self.cb(np.ones(self.cb.n_target, dtype=np.complex64),
                    SimpleNamespace(delivered_rtp_start=1))

        def set_channel_lifetime(self, ssrc, n):
            if ssrc in self.slots:
                self.control.set_channel_lifetime(ssrc, n)

        def stop(self):
            if stop_raises:
                raise RuntimeError("fake: stop blew up")
    return FakeMulti


def _anchor(source, rtp=True):
    return SimpleNamespace(utc=1000.0, source=source, rtp_referenced=rtp,
                           snapshot=SimpleNamespace(t_level_active="T3"))


def _rtp(*a):
    return _anchor("rtp_to_utc")


def _run(radiod, monkeypatch, label_fn=_rtp, created=None, probe_ssrc=None,
         **multi_kw):
    created = [] if created is None else created
    probe_ssrc = PROBE_SSRC if probe_ssrc is None else probe_ssrc
    monkeypatch.setattr(lc, "_probe_ssrc", lambda *a: probe_ssrc)
    monkeypatch.setattr(lc, "_discover_ssrcs",
                        lambda control: set(radiod.discover()), raising=False)
    monkeypatch.setattr(lc, "REAP_TIMEOUT_SEC", 0.0)
    monkeypatch.setattr(lc, "labelled_lag_seconds", lambda *a: (1e-4, 0.9))
    monkeypatch.setattr(lc.time, "sleep", lambda s: radiod.tick(50))
    ref = SimpleNamespace(ssrc=REF_SSRC, frequency=FREQ, sample_rate=12000,
                          multicast_address="239.1.1.1", encoding=4)
    return lc._run_repeat(
        0, control=radiod, ref=ref, band=HFDL_BANDS["HFDL13"], seconds=0.01,
        reader=None, label_fn=label_fn, ref_stream_cls=_FakeStream,
        multi_cls=_fake_multi(created, **multi_kw))


def test_clean_repeat_leaves_no_channel_behind(monkeypatch):
    radiod = _FakeRadiod()
    rec = _run(radiod, monkeypatch)
    assert rec is not None and rec["lag_us"] == pytest.approx(100.0)
    radiod.tick(10_000)                       # let any finite lifetime expire
    assert set(radiod.chans) == {REF_SSRC}


def test_repeats_back_to_back_leave_no_channel_behind(monkeypatch):
    radiod = _FakeRadiod()
    for _ in range(3):
        assert _run(radiod, monkeypatch) is not None
    radiod.tick(10_000)
    assert set(radiod.chans) == {REF_SSRC}


def test_reference_ssrc_never_addressed(monkeypatch):
    radiod = _FakeRadiod()
    _run(radiod, monkeypatch)
    assert radiod.cmds(REF_SSRC) == []


def test_probe_created_with_finite_lifetime(monkeypatch):
    created = []
    _run(_FakeRadiod(), monkeypatch, created=created)
    assert created[0]["lifetime"] == lc.PROBE_LIFETIME_FRAMES > 0


def test_reap_is_the_last_command_to_the_probe(monkeypatch):
    radiod = _FakeRadiod()
    _run(radiod, monkeypatch)
    cmds = radiod.cmds(PROBE_SSRC)
    assert cmds[-1] == ("lifetime", 1)
    assert cmds.count(("lifetime", 1)) == 1


def test_nothing_addressed_to_probe_before_its_create(monkeypatch):
    # Only ensure_channel's own pre-poll may precede the create; label-check
    # itself must not poll the probe SSRC (a poll mints it).
    radiod = _FakeRadiod()
    _run(radiod, monkeypatch)
    cmds = radiod.cmds(PROBE_SSRC)
    assert cmds[:2] == [("poll", None), ("create", lc.PROBE_LIFETIME_FRAMES)]


def test_stale_probe_found_by_discovery_and_reaped(monkeypatch):
    radiod = _FakeRadiod(stale_probe=True)    # infinite-lifetime orphan
    created = []
    rec = _run(radiod, monkeypatch, created=created)
    assert rec is not None and len(created) == 1
    cmds = radiod.cmds(PROBE_SSRC)
    assert cmds[0] == ("lifetime", 1)         # reaped, never polled first
    radiod.tick(10_000)
    assert set(radiod.chans) == {REF_SSRC}


def test_keepalive_cannot_send_after_reap(monkeypatch):
    monkeypatch.setattr(lc, "KEEPALIVE_SEC", 0.001)
    radiod = _FakeRadiod()
    entered = threading.Event()
    real = radiod.set_channel_lifetime

    def slow_lifetime(ssrc, lifetime, encoding=None):
        if lifetime == lc.PROBE_LIFETIME_FRAMES:
            entered.set()
            threading.Event().wait(0.3)       # a refresh already in flight
        real(ssrc, lifetime, encoding)
    radiod.set_channel_lifetime = slow_lifetime
    _run(radiod, monkeypatch, on_start=lambda: entered.wait(5))
    threading.Event().wait(0.6)               # let a stray refresh land
    cmds = radiod.cmds(PROBE_SSRC)
    assert cmds[-1] == ("lifetime", 1), cmds


def test_stop_raising_still_reaps(monkeypatch):
    radiod = _FakeRadiod()
    with pytest.raises(RuntimeError):
        _run(radiod, monkeypatch, stop_raises=True)
    assert radiod.cmds(PROBE_SSRC)[-1] == ("lifetime", 1)
    radiod.tick(10_000)
    assert set(radiod.chans) == {REF_SSRC}


def test_failed_create_reaps_the_minted_ssrc(monkeypatch):
    radiod = _FakeRadiod(create_fails=True)
    with pytest.raises(TimeoutError):
        _run(radiod, monkeypatch)
    radiod.tick(10_000)
    assert set(radiod.chans) == {REF_SSRC}


def test_probe_that_will_not_die_fails_the_repeat(monkeypatch, capsys):
    radiod = _FakeRadiod(honours_lifetime=False)   # pre-LIFETIME radiod
    rec = _run(radiod, monkeypatch)
    assert rec is None
    assert "still present" in capsys.readouterr().err


def test_stale_probe_that_will_not_die_fails_before_creation(monkeypatch):
    radiod = _FakeRadiod(stale_probe=True, honours_lifetime=False)
    created = []
    assert _run(radiod, monkeypatch, created=created) is None
    assert created == []


def test_wallclock_anchor_drops_the_repeat(monkeypatch, capsys):
    for bad_ref, bad_probe in (("wallclock_fallback", "rtp_to_utc"),
                               ("rtp_to_utc", "authority_on_wallclock")):
        anchors = iter([_anchor(bad_ref, rtp=bad_ref.startswith("rtp")),
                        _anchor(bad_probe, rtp=bad_probe.startswith("rtp"))])
        radiod = _FakeRadiod()
        rec = _run(radiod, monkeypatch, label_fn=lambda *a: next(anchors))
        assert rec is None
        assert "DROPPED" in capsys.readouterr().err
        radiod.tick(10_000)
        assert set(radiod.chans) == {REF_SSRC}


def test_equal_ssrc_refuses_removal(monkeypatch):
    radiod = _FakeRadiod()
    with pytest.raises(SystemExit):
        _run(radiod, monkeypatch, probe_ssrc=REF_SSRC)
    assert radiod.log == []
    with pytest.raises(SystemExit):
        lc._reap_probe(radiod, REF_SSRC, REF_SSRC)
    assert radiod.log == []


def test_discover_ssrcs_uses_native_listen_only(monkeypatch):
    # discover_channels falls back to the `control` utility on an empty
    # listen; label-check must use the native 0xffffffff listen alone.
    import ka9q.discovery as kd
    seen = []
    monkeypatch.setattr(kd, "discover_channels_native",
                        lambda addr, listen_duration=2.0, interface=None:
                        seen.append(addr) or {7: None, 8: None})
    monkeypatch.setattr(kd, "discover_channels_via_control",
                        lambda *a, **k: pytest.fail("fallback used"))
    monkeypatch.setattr(kd, "discover_channels",
                        lambda *a, **k: pytest.fail("fallback-capable path"))
    assert lc._discover_ssrcs(SimpleNamespace(status_address="r.local")) \
        == {7, 8}
    assert seen == ["r.local"]
    monkeypatch.setattr(kd, "discover_channels_native",
                        lambda *a, **k: {})
    assert lc._discover_ssrcs(SimpleNamespace(status_address="r.local")) \
        is None


def test_reference_encoding_guard(monkeypatch):
    from ka9q import Encoding
    with pytest.raises(SystemExit):
        lc._check_ref_encoding(0, 111)
    with pytest.raises(SystemExit):
        lc._check_ref_encoding(Encoding.OPUS, 111)
    assert lc._check_ref_encoding(Encoding.F32LE, 111) == 4
    assert lc._check_ref_encoding(Encoding.S16BE, 111) == 2


def test_find_reference_refuses_unknown_encoding(monkeypatch):
    import ka9q
    mk = lambda enc: SimpleNamespace(ssrc=1, frequency=10e6, preset="iq",
                                     sample_rate=12000, encoding=enc,
                                     multicast_address="239.1.1.1")
    monkeypatch.setattr(ka9q, "discover_channels", lambda s: {1: mk(0)})
    with pytest.raises(SystemExit):
        lc._find_reference("radiod.local", 10e6, None)
    monkeypatch.setattr(ka9q, "discover_channels", lambda s: {1: mk(4)})
    assert lc._find_reference("radiod.local", 10e6, None).encoding == 4
