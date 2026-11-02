import numpy as np

from hfdl_recorder.timing import waveform as w

# Oracle: dumphfdl 1.7 src/hfdl.c A_octets, first 127 bits, MSB first.
_A_OCTETS = [
    0b01011011, 0b10111100, 0b01110100, 0b01010111, 0b00000011, 0b11011001,
    0b10001001, 0b00111001, 0b11110010, 0b00001000, 0b11010101, 0b00110110,
    0b10010100, 0b00101100, 0b00110010, 0b11111110,
]


def _oracle_bits():
    bits = [(o >> (7 - i)) & 1 for o in _A_OCTETS for i in range(8)]
    return np.array(bits[:127])


def test_a_sequence_matches_dumphfdl_octets():
    a = w.a_symbols()
    assert len(a) == w.A_LEN
    assert np.array_equal((a < 0).astype(int), _oracle_bits())


def test_m1_shifts_are_rotations():
    base = np.roll(w.m1_symbols(0), w.M_SHIFTS[0])   # undo shift 0
    seqs = [w.m1_symbols(k) for k in range(8)]
    for k, s in enumerate(seqs):
        assert len(s) == w.M1_LEN
        assert np.array_equal(s, np.roll(base, -w.M_SHIFTS[k]))
    assert len({tuple(s) for s in seqs}) == 8


def test_a_autocorrelation_is_sharp():
    a = w.a_symbols()
    ac = np.array([np.dot(a, np.roll(a, k)) for k in range(len(a))])
    assert ac[0] == len(a)
    assert np.max(np.abs(ac[1:])) < 0.15 * ac[0]


def test_template_layout():
    t = w.timing_template()
    assert len(t) == w.PREKEY_LEN + 2 * w.A_LEN
    assert np.all(t[:w.PREKEY_LEN] == 1)
    assert np.array_equal(t[w.PREKEY_LEN:w.PREKEY_LEN + w.A_LEN], w.a_symbols())


def test_synth_burst_lands_where_asked():
    fs = 6000
    x = w.synth_squitter(fs, 0.0123, snr_db=40, rng=np.random.default_rng(1))
    assert len(x) == round((0.03 + 32 / 13) * fs)
    env = np.abs(x)
    onset = np.argmax(env > 0.5 * np.median(env[len(env) // 2:]))
    assert abs(onset / fs - 0.0423) < 1 / fs


def test_synth_onset_tight_at_high_rate():
    # 18 kS/s: a half-symbol convention error would move onset by ~5 samples.
    fs = 18000
    x = w.synth_squitter(fs, 0.0123, snr_db=40, rng=np.random.default_rng(1))
    env = np.abs(x)
    onset = np.argmax(env > 0.5 * np.median(env[len(env) // 2:]))
    assert abs(onset / fs - 0.0423) < 1 / fs


def test_fractional_delay_is_exact():
    fs = 6000
    rng = np.random.default_rng
    a = w.synth_squitter(fs, 0.0, snr_db=200, rng=rng(1))
    b = w.synth_squitter(fs, 1 / fs, snr_db=200, rng=rng(1))
    # one-sample delay: b[n] == a[n-1] away from the edges
    assert np.allclose(b[100:-100], a[99:-101], atol=1e-6)
    # half-sample delay sits between: compare against the same burst
    # modulated directly at the offset
    h = w.synth_squitter(fs, 0.5 / fs, snr_db=200, rng=rng(1))
    assert np.max(np.abs(h[100:-100] - a[100:-100])) > 0.1


def test_paths_and_doppler():
    fs = 6000
    one = w.synth_squitter(fs, 0.01, snr_db=200, rng=np.random.default_rng(2))
    two = w.synth_squitter(fs, 0.01, snr_db=200, rng=np.random.default_rng(2),
                           paths=[(0.0, 1.0), (0.0025, 0.6)])
    assert np.allclose(two[:100], one[:100], atol=1e-6)
    assert not np.allclose(two, one)
    d = w.synth_squitter(fs, 0.01, snr_db=200, doppler_hz=5.0,
                         rng=np.random.default_rng(2))
    n = np.arange(len(d))
    # derotating recovers the undoppled burst
    assert np.allclose(d * np.exp(-2j * np.pi * 5.0 * n / fs)[: len(d)],
                       one, atol=1e-6)
