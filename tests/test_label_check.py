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


# ---- _run_repeat with fakes ----

from types import SimpleNamespace

from hfdl_recorder.bands import HFDL_BANDS
from hfdl_recorder.timing import label_check as lc


class _FakeControl:
    """Fake RadiodControl.  `present` = SSRCs radiod still answers for."""

    def __init__(self, present=(), reap_works=True):
        self.calls = []
        self.present = set(present)
        self.reap_works = reap_works

    def remove_channel(self, ssrc):
        self.calls.append(("remove_channel", ssrc))

    def set_channel_lifetime(self, ssrc, n):
        self.calls.append(("set_channel_lifetime", ssrc, n))
        if n == 1 and self.reap_works:
            self.present.discard(ssrc)

    def poll_channel(self, ssrc, **kw):
        return object() if ssrc in self.present else None

    def __getattr__(self, name):          # any other call is recorded loudly
        def rec(*a, **kw):
            self.calls.append((name, a, kw))
        return rec


class _FakeStream:
    """Delivers a full capture synchronously on start()."""
    def __init__(self, channel, on_samples):
        self.on_samples = on_samples
        self.fs = channel.sample_rate

    def start(self):
        n = int(self.on_samples.n_target)
        self.on_samples(np.ones(n, dtype=np.complex64),
                        SimpleNamespace(delivered_rtp_start=1))

    def stop(self):
        pass


def _fake_multi(control, probe_ssrc, created):
    class FakeMulti:
        def __init__(self, control):
            self.control = control

        def add_channel(self, **kw):
            self.cb = kw["on_samples"]
            created.append(kw)
            self.control.present.add(probe_ssrc)
            return SimpleNamespace(ssrc=probe_ssrc)

        def start(self):
            self.cb(np.ones(self.cb.n_target, dtype=np.complex64),
                    SimpleNamespace(delivered_rtp_start=1))

        def set_channel_lifetime(self, ssrc, n):
            self.control.calls.append(("keepalive", ssrc, n))

        def stop(self):
            pass
    return FakeMulti


def _anchor(source, rtp=True):
    return SimpleNamespace(utc=1000.0, source=source, rtp_referenced=rtp,
                           snapshot=SimpleNamespace(t_level_active="T3"))


def _run(control, probe_ssrc, label_fn, monkeypatch, created=None):
    created = [] if created is None else created
    monkeypatch.setattr(lc, "_probe_ssrc", lambda *a: probe_ssrc)
    monkeypatch.setattr(lc, "REAP_TIMEOUT_SEC", 0.0)
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    ref = SimpleNamespace(ssrc=111, frequency=10e6, sample_rate=12000,
                          multicast_address="239.1.1.1", encoding=4)

    def fake_stream(channel, on_samples):
        return _FakeStream(channel, on_samples)
    return lc._run_repeat(
        0, control=control, ref=ref, band=HFDL_BANDS["HFDL13"], seconds=0.01,
        reader=None, label_fn=label_fn, ref_stream_cls=fake_stream,
        multi_cls=_fake_multi(control, probe_ssrc, created))


def test_reference_path_never_creates_modifies_or_removes(monkeypatch):
    control = _FakeControl()
    monkeypatch.setattr(lc, "labelled_lag_seconds", lambda *a: (1e-4, 0.9))
    rec = _run(control, 222, lambda *a: _anchor("rtp_to_utc"), monkeypatch)
    assert rec["ref_source"] == "rtp_to_utc"
    assert rec["lag_us"] == pytest.approx(100.0)
    # Only the probe SSRC is ever touched; reference SSRC 111 never.
    assert control.calls == [("set_channel_lifetime", 222, 1)]


def test_probe_created_with_finite_lifetime(monkeypatch):
    created = []
    monkeypatch.setattr(lc, "labelled_lag_seconds", lambda *a: (1e-4, 0.9))
    _run(_FakeControl(), 222, lambda *a: _anchor("rtp_to_utc"), monkeypatch,
         created)
    assert created[0]["lifetime"] == lc.PROBE_LIFETIME_FRAMES > 0


def test_removal_sends_lifetime_one(monkeypatch):
    control = _FakeControl(present={222})
    assert lc._reap_probe(control, 222, 111) is True
    assert ("set_channel_lifetime", 222, 1) in control.calls


def test_probe_still_present_fails_the_repeat(monkeypatch, capsys):
    control = _FakeControl(reap_works=False)
    monkeypatch.setattr(lc, "labelled_lag_seconds", lambda *a: (1e-4, 0.9))
    rec = _run(control, 222, lambda *a: _anchor("rtp_to_utc"), monkeypatch)
    assert rec is None
    assert "still present" in capsys.readouterr().err


def test_preexisting_probe_is_reaped_before_creation(monkeypatch):
    control = _FakeControl(present={222})
    created = []
    monkeypatch.setattr(lc, "labelled_lag_seconds", lambda *a: (1e-4, 0.9))
    rec = _run(control, 222, lambda *a: _anchor("rtp_to_utc"), monkeypatch,
               created)
    assert rec is not None
    assert control.calls[0] == ("set_channel_lifetime", 222, 1)
    assert len(created) == 1


def test_stale_probe_that_will_not_die_fails_before_creation(monkeypatch):
    control = _FakeControl(present={222}, reap_works=False)
    created = []
    rec = _run(control, 222, lambda *a: _anchor("rtp_to_utc"), monkeypatch,
               created)
    assert rec is None and created == []


def test_wallclock_anchor_drops_the_repeat(monkeypatch, capsys):
    monkeypatch.setattr(lc, "labelled_lag_seconds", lambda *a: (1e-4, 0.9))
    for bad_ref, bad_probe in (("wallclock_fallback", "rtp_to_utc"),
                               ("rtp_to_utc", "authority_on_wallclock")):
        anchors = iter([_anchor(bad_ref, rtp=bad_ref.startswith("rtp")),
                        _anchor(bad_probe, rtp=bad_probe.startswith("rtp"))])
        rec = _run(_FakeControl(), 222, lambda *a: next(anchors), monkeypatch)
        assert rec is None
        assert "DROPPED" in capsys.readouterr().err


def test_equal_ssrc_refuses_removal(monkeypatch):
    control = _FakeControl()
    with pytest.raises(SystemExit):
        _run(control, 111, lambda *a: _anchor("rtp_to_utc"), monkeypatch)
    assert not control.calls
    with pytest.raises(SystemExit):
        lc._reap_probe(control, 111, 111)
    assert not control.calls


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
