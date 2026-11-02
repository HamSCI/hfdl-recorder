import numpy as np
import pytest

from hfdl_recorder.timing import analyzer as an
from hfdl_recorder.timing import waveform as w

FS = 6000.0
NOMINAL = 1_793_000_000.0           # any UTC; the window starts 30 ms before it
T0 = NOMINAL - 0.03


def _time(x, **kw):
    return an.time_squitter(x, FS, T0, NOMINAL, **kw)


def _frac_delay(x, frac):
    """Delay x by frac samples (band-limited), padding so nothing wraps."""
    n = len(x)
    pad = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.fftfreq(pad)
    y = np.fft.ifft(np.fft.fft(x, pad) * np.exp(-2j * np.pi * f * frac))
    return y[:n]


def _delay_errors(delays, seed=1):
    errs = []
    for i, d in enumerate(delays):
        x = w.synth_squitter(FS, d, snr_db=20, rng=np.random.default_rng(seed + i))
        a = _time(x)
        assert a.found, a.reason
        errs.append(a.delay_s - d)
    return np.array(errs)


DELAYS = (0.0050, 0.01234, 0.0201)


def test_whole_and_fractional_delays():
    errs = _delay_errors(DELAYS)
    assert np.all(np.abs(errs) < 20e-6), errs * 1e6


def test_shift_moves_answer_exactly():
    # Mutation of the input: 37 whole samples plus 0.4 sample must move the
    # answer by exactly 37.4 samples.
    x = w.synth_squitter(FS, 0.0123, snr_db=20, rng=np.random.default_rng(7))
    base = _time(x)
    moved = _time(_frac_delay(np.concatenate([np.zeros(37, complex), x]), 0.4))
    assert base.found and moved.found
    assert abs((moved.delay_s - base.delay_s) - 37.4 / FS) < 2e-6


def test_label_lag_subtracts():
    x = w.synth_squitter(FS, 0.0123, snr_db=20, rng=np.random.default_rng(3))
    a = _time(x)
    b = _time(x, label_lag_s=0.004)
    assert b.delay_s == pytest.approx(a.delay_s - 0.004, abs=1e-9)
    # An epoch double resolves ~0.24 us; delay_s is formed without it.
    assert b.arrival_utc == pytest.approx(a.arrival_utc - 0.004, abs=1e-6)
    assert a.arrival_utc == pytest.approx(NOMINAL + a.delay_s, abs=1e-6)


@pytest.mark.parametrize("dop", [-3.0, 3.0])
def test_doppler_tolerated_and_measured(dop):
    x = w.synth_squitter(FS, 0.01234, snr_db=20, doppler_hz=dop,
                         rng=np.random.default_rng(11))
    a = _time(x)
    assert a.found, a.reason
    assert abs(a.delay_s - 0.01234) < 30e-6
    assert abs(a.doppler_hz - dop) < 0.3


def test_two_path():
    x = w.synth_squitter(FS, 0.0101, snr_db=20, paths=[(0.0, 1.0), (0.0025, 0.6)],
                         rng=np.random.default_rng(5))
    a = _time(x)
    assert a.found, a.reason
    assert abs(a.delay_s - 0.0101) < 30e-6
    assert a.second_peak_delay_s == pytest.approx(0.0025, abs=50e-6)
    assert 0.4 < a.second_peak_ratio < 0.8


def test_m1_shift_reported():
    x = w.synth_squitter(FS, 0.0101, snr_db=20, m1_shift=5,
                         rng=np.random.default_rng(9))
    assert _time(x).m1_shift == 5


def test_no_burst_reports_not_found():
    rng = np.random.default_rng(2)
    n = round((0.03 + 32 / 13) * FS)
    for _ in range(5):
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        a = _time(x)
        assert not a.found
        assert a.reason == "no_peak"


def test_two_bursts_flagged():
    x = (w.synth_squitter(FS, 0.005, snr_db=30, rng=np.random.default_rng(4))
         + w.synth_squitter(FS, 0.405, snr_db=30, rng=np.random.default_rng(6)))
    a = _time(x)
    assert not a.found
    assert a.reason == "ambiguous"


def test_broken_template_fails(monkeypatch):
    # Mutation of the analyzer: a template whose A pair sits one symbol late
    # (as a wrong PREKEY_LEN would put it) must break the 20 us tolerance.
    # It moves the answer about -410 us.  The brief's mutation, one A symbol
    # in ten flipped, moved it only -16 us: inside the tolerance, so too weak
    # to prove the test can fail.
    def shifted():
        a = w.a_symbols()
        return np.concatenate([np.ones(w.PREKEY_LEN + 1), a, a])[:len(a) * 2 + w.PREKEY_LEN]

    monkeypatch.setattr(an, "timing_template", shifted)
    errs = _delay_errors(DELAYS)
    assert np.any(np.abs(errs) >= 20e-6), errs * 1e6

