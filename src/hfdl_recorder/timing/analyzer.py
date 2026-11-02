"""Time one HFDL squitter against its known preamble.

Input: a window of complex baseband IQ at rate fs (the channel's dial
frequency at 0 Hz, the burst on the +1440 Hz subcarrier) whose first sample
carries the UTC label t0_utc.

Output convention.  The analyzer measures A1, the start of the first A
sequence (the 50 % point of its first pulse, per ``waveform``), and
reports it as ``a1_utc``.  ``arrival_utc`` is A1 less 448 symbols
(248.9 ms): the burst start ARINC 635 implies, and the spec's definition,
so delays stay comparable across stations.  Real pre-keys need not be 448
symbols long; Riverhead's tone starts ~230.5 ms before A1.  Nothing in the
timing depends on the pre-key, and ``prekey_lead_s`` records where the
tone actually started.

The method, in order:

1. Coarse search.  Correlate against the A pair (141 ms) under Doppler
   hypotheses 2 Hz apart over +/- 20 Hz; the strongest match locates A1.
2. Doppler.  An FFT of the pre-key tone from A1 - 225 ms to A1 - 5 ms,
   Hann-windowed, zero-padded and interpolated by parabola.  Magnitude
   only, so the tone's phase does not matter.  Without a tone there (more
   than 1.5 Hz from the coarse hypothesis, or too little window), the
   coarse hypothesis stands.
3. Detection, on the A pair's correlation after Doppler removal.  Its peak
   over its median gives ``snr_db``.  A second peak more than 50 ms away at
   0.8 of the first makes the window ambiguous, skipping +/- 3 ms around
   +/- 70.6 ms, where one A of the template meets the other A of the burst
   at 0.5 by construction.  (The pre-key carries 64 % of the full
   template's energy; a full-template correlation is a broad triangle
   whose shoulder at 50 ms reaches 0.74-0.87 on real bursts, which is why
   detection no longer uses it.)
4. M1.  The eight rotations are tried after the A pair; a clear winner
   labels the squitter and extends the fine template.
5. Fine timing.  Correlate against A, A, M1, M2 (A, A when M1 is unclear),
   interpolated 32 times by zero-padding the cross-spectrum (exact for a
   band-limited correlation); a parabola on the magnitude gives A1.
6. Width and multipath, from the A pair's upsampled correlation: main-lobe
   width and the strongest other path within 50 ms, beyond the A pair's
   first sidelobe (0.19 of the peak, out to 2.3 symbols).  A path there at
   0.8 of the main one or more makes the window ambiguous: which mode is
   "the" arrival becomes a coin toss.
7. Quality, never moving or refusing the answer: ``unresolved_multipath``
   when the A pair's main lobe departs from the clean template's by more
   than noise explains; ``prekey_lead_s`` = A1 minus the tone's fitted
   start (NaN when the tone is too weak, or the window starts inside it).

Limits, measured on synth_squitter:

- An echo closer than about 1.3 ms (2.4 kHz of bandwidth cannot resolve
  it) merges with the main lobe and pulls the answer, roughly in
  proportion to its amplitude.  The flag catches the large pulls; see the
  task report for the measured table.
- A burst whose fine template (A1 to M2) does not lie wholly in the window
  comes back ``truncated``, never timed.

Every not-found result carries NaN in the timing and quality fields; only
``snr_db`` survives once measured.  Reasons: ok, no_peak, ambiguous,
truncated, bad_input (non-finite samples), too_short.

Everything correlates on magnitude, so a polarity-inverted preamble times
the same as an upright one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .waveform import (
    _PULSE_SPAN,
    A_LEN,
    M1_LEN,
    M2_LEN,
    M_SHIFTS,
    PREKEY_LEN,
    SUBCARRIER_HZ,
    SYMBOL_RATE,
    _rc_pulse,
    m1_symbols,
    modulate,
    timing_template,
)

# snr_db = peak / median of the A pair's correlation.  Pure noise scores
# 12.8 dB median, 14.5 dB worst (200 trials), so 17 dB keeps a margin.  The two
# weak Riverhead windows dumphfdl could not decode score 15.8 and 16.2.
FOUND_SNR_DB = 17.0
AMBIGUOUS_RATIO = 0.8       # a far peak at this fraction of the first
AMBIGUOUS_SEP_S = 0.050     # ... more than this far away
MULTIPATH_SPAN_S = 0.050    # other paths are looked for within this span
M1_CLEAR_RATIO = 2.0        # best M1 shift must beat the next by this
LOBE_DISTORTION_FLAG = 0.015    # flag floor for A-pair main-lobe distortion
LOBE_NOISE_FACTOR = 1.5         # ... raised to this x the correlation's relative noise
# The pre-key's start is measured only when its plateau (180 ms) carries
# this integrated SNR (linear); the step fit refines the CUSUM's edge only
# when its own per-sample SNR reaches EDGE_MIN_SNR (about 8 dB input).
PLATEAU_MIN_SNR = 100.0
EDGE_MIN_SNR = 6.0

_ROLLOFF = 0.35
_PAD = 8                    # silent symbols each side of a rendered template
_UPSAMPLE = 32              # interpolation factor for fine timing
_DOPPLER_STEP_HZ = 2.0
_DOPPLER_SPAN_HZ = 20.0     # coarse search covers +/- this
_DOPPLER_AGREE_HZ = 1.5     # pre-key FFT must land this close to the coarse hit
_PREKEY_FFT_FROM_S = 0.225  # Doppler FFT runs from A1 - this ...
_PREKEY_FFT_TO_S = 0.005    # ... to A1 - this
_PREKEY_FFT_MIN_S = 0.050   # shortest usable FFT segment
_TONE_SEARCH_HZ = 30.0      # look for the pre-key tone this far from 1440 Hz
_MAIN_SEARCH_S = 0.002      # fine peak sought this close to the coarse hit
_LOBE_EXCLUDE_SYM = 2.3     # the A pair's first sidelobe (0.19) ends here
_SINGLE_A_S = A_LEN / SYMBOL_RATE   # one A against the other: 0.5 by construction
_SINGLE_A_SKIP_S = 0.003    # ... skipped this far either side in the far test
_LEAD_SEARCH_S = 0.320      # pre-key start sought no earlier than A1 - this
_EDGE_PRE_SYM = 6           # edge refinement window, symbols before ...
_EDGE_POST_SYM = 4          # ... and after the coarse edge


@dataclass
class Arrival:
    """One squitter's timing.  Not-found results carry NaN timing fields.

    ``arrival_utc`` is A1 - 448 symbols (the ARINC 635 burst start, the
    spec's convention); ``a1_utc`` is A1 itself, so nothing downstream need
    depend on the 448.  ``delay_s`` = arrival_utc - nominal slot start.
    """
    found: bool
    arrival_utc: float
    delay_s: float
    snr_db: float
    peak_width_s: float
    second_peak_ratio: float
    second_peak_delay_s: float
    doppler_hz: float
    m1_shift: int | None
    reason: str
    lobe_distortion: float          # RMS main-lobe departure from the clean template
    prekey_lead_s: float            # A1 minus the pre-key tone's start; NaN unobservable
    flags: str                      # comma-separated quality flags, "" when clean
    a1_utc: float                   # start of A1 (UTC); arrival_utc + 448 symbols


def _render(symbols: np.ndarray, fs: float) -> np.ndarray:
    """symbols with _PAD silent symbols each side, so the edges are whole pulses."""
    pad = np.zeros(_PAD)
    return modulate(np.concatenate([pad, symbols, pad]), fs)


def _spectrum(x: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, int]:
    """Cross-spectrum X conj(H) at a length where the correlation does not wrap."""
    n = 1 << int(np.ceil(np.log2(len(x) + len(h))))
    return np.fft.fft(x, n) * np.conj(np.fft.fft(h, n)), n


def _correlate(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """c[k] = sum_n x[n + k] conj(h[n]) for k = 0 .. len(x) - 1."""
    spec, _ = _spectrum(x, h)
    return np.fft.ifft(spec)[:len(x)]


def _correlate_fine(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """|c| at lags k / _UPSAMPLE, k = 0 .. len(x) * _UPSAMPLE - 1.

    Zero-padding the cross-spectrum interpolates the correlation exactly
    when the template is band-limited below fs/2, which the shaped
    subcarrier is (225-2655 Hz at 6 kS/s).
    """
    spec, n = _spectrum(x, h)
    up = np.zeros(n * _UPSAMPLE, dtype=complex)
    half = n // 2
    up[:half] = spec[:half]
    up[-half:] = spec[half:]
    return np.abs(np.fft.ifft(up)[:len(x) * _UPSAMPLE]) * _UPSAMPLE


def _parabola(y: np.ndarray, i: int) -> float:
    """Vertex offset from index i of the parabola through y[i-1], y[i], y[i+1]."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2.0 * b + c
    return 0.0 if den == 0 else 0.5 * (a - c) / den


def _doppler_from_prekey(x: np.ndarray, fs: float, lo: int, hi: int) -> float:
    """Carrier offset of the pre-key tone in x[lo:hi]; NaN when too short.

    Phase-blind: an FFT magnitude peak, so the tone's phase against the A
    pair (which varies burst to burst) does not matter.
    """
    lo, hi = max(lo, 0), min(hi, len(x))
    seg = x[lo:hi]
    if len(seg) < int(_PREKEY_FFT_MIN_S * fs):
        return float("nan")
    nfft = 1 << 18
    spec = np.abs(np.fft.fft(seg * np.hanning(len(seg)), nfft))
    freqs = np.fft.fftfreq(nfft, 1.0 / fs)
    band = np.flatnonzero(np.abs(freqs - SUBCARRIER_HZ) <= _TONE_SEARCH_HZ)
    i = band[np.argmax(spec[band])]
    logs = np.log(np.maximum(spec[i - 1:i + 2], 1e-300))
    off = _parabola(logs, 1)
    return float(freqs[i] + off * fs / nfft - SUBCARRIER_HZ)


def _rotate(x: np.ndarray, fs: float, hz: float) -> np.ndarray:
    return x * np.exp(-2j * np.pi * hz * np.arange(len(x)) / fs)


def _lobe_width(mag: np.ndarray, i: int) -> int:
    """Full width at half maximum of the lobe at index i, in index units."""
    half = 0.5 * mag[i]
    lo = i
    while lo > 0 and mag[lo - 1] >= half:
        lo -= 1
    hi = i
    while hi < len(mag) - 1 and mag[hi + 1] >= half:
        hi += 1
    return hi - lo + 1


def _second_path(mag: np.ndarray, i: int, fs: float) -> tuple[float, float]:
    """(ratio, delay_s) of the strongest local maximum near, but off, the main lobe."""
    per_s = fs * _UPSAMPLE
    span = int(MULTIPATH_SPAN_S * per_s)
    excl = int(_LOBE_EXCLUDE_SYM * per_s / SYMBOL_RATE)
    lo, hi = max(i - span, 1), min(i + span, len(mag) - 2)
    seg = mag[lo - 1:hi + 2]
    peaks = np.flatnonzero((seg[1:-1] >= seg[:-2]) & (seg[1:-1] > seg[2:])) + lo
    peaks = peaks[np.abs(peaks - i) > excl]
    if len(peaks) == 0:
        return 0.0, 0.0
    j = peaks[np.argmax(mag[peaks])]
    return float(mag[j] / mag[i]), float((j - i) / per_s)


def _m1_shift(x: np.ndarray, fs: float, a_start: float) -> int | None:
    """Which M1 rotation follows the A pair, or None when none clearly wins."""
    lag = a_start + (2 * A_LEN - _PAD) * fs / SYMBOL_RATE
    scores = []
    for k in range(len(M_SHIFTS)):
        h = _render(m1_symbols(k), fs)
        best = 0.0
        for d in (-1, 0, 1):
            i = int(round(lag)) + d
            if i < 0 or i + len(h) > len(x):
                continue
            best = max(best, abs(np.vdot(h, x[i:i + len(h)])))
        scores.append(best)
    order = np.argsort(scores)[::-1]
    top, nxt = scores[order[0]], scores[order[1]]
    if top <= 0 or top < M1_CLEAR_RATIO * nxt:
        return None
    return int(order[0])


def _lobe_distortion(pair: np.ndarray, jp: int, ref: np.ndarray, jr: int,
                     fs: float) -> float:
    """RMS difference, over +/- 1 symbol, between the measured A-pair main
    lobe and the clean template's, each scaled to its own peak."""
    w = int(round(fs / SYMBOL_RATE * _UPSAMPLE))
    if jp - w < 0 or jp + w >= len(pair) or jr - w < 0 or jr + w >= len(ref):
        return float("nan")
    m = pair[jp - w:jp + w + 1] / pair[jp]
    r = ref[jr - w:jr + w + 1] / ref[jr]
    return float(np.sqrt(np.mean((m - r) ** 2)))


def _step(t_sym: np.ndarray) -> np.ndarray:
    """A run of +1 symbols switched on at t_sym = 0 (in symbol periods).

    It rises 0 -> 1 and, for any even Nyquist pulse, is antisymmetric about
    its 50 % point at 0; that symmetry, not the exact pulse, sets the edge.
    """
    out = np.zeros_like(t_sym)
    for k in range(_EDGE_PRE_SYM + _EDGE_POST_SYM + 2 * _PULSE_SPAN):
        out += _rc_pulse(t_sym - k - 0.5, _ROLLOFF)
    return out


def _prekey_edge(xd: np.ndarray, fs: float, start: float) -> tuple[float, float]:
    """Refine the pre-key's leading edge near ``start``, in samples.

    Mix the 1440 Hz subcarrier to 0 Hz, then fit a * step(t - e) by least
    squares over a few symbols around ``start`` (a complex and free, e on a
    grid refined by parabola).  Returns (edge, per-sample SNR of the fit);
    NaNs when the fit window leaves the signal.
    """
    sym = fs / SYMBOL_RATE
    lo = int(np.floor(start - (_EDGE_PRE_SYM + 2) * sym))
    hi = int(np.ceil(start + (_EDGE_POST_SYM + 2) * sym))
    if lo < 0 or hi > len(xd):
        return float("nan"), float("nan")
    n = np.arange(lo, hi)
    z = xd[lo:hi] * np.exp(-2j * np.pi * SUBCARRIER_HZ * n / fs)
    grid = start + np.arange(-_EDGE_PRE_SYM * sym, _EDGE_POST_SYM * sym, 0.25)
    score = np.empty(len(grid))
    for i, e in enumerate(grid):
        m = _step((n - e) / sym)
        score[i] = abs(np.dot(z, m)) ** 2 / np.dot(m, m)
    i = int(np.argmax(score))
    e = float(grid[i] + 0.25 * _parabola(score, i))
    m = _step((n - e) / sym)
    a = np.dot(z, m) / np.dot(m, m)
    resid = np.mean(np.abs(z - a * m) ** 2)
    return e, float(abs(a) ** 2 / resid) if resid > 0 else float("inf")


def _noise_sigma(x: np.ndarray, fs: float) -> float:
    """Per-sample noise RMS, from the negative-frequency half of the window.

    The burst sits at +225..+2655 Hz, so below -300 Hz the window holds only
    noise (or another station, which only raises the estimate).
    """
    spec = np.abs(np.fft.fft(x)) ** 2
    f = np.fft.fftfreq(len(x), 1.0 / fs)
    band = (f < -300.0) & (f > -fs / 2 + 100.0)
    if not band.any():
        return float("nan")
    return float(np.sqrt(np.median(spec[band]) / np.log(2) / len(x)))


def _prekey_start(xd: np.ndarray, fs: float, a1: float, first: int) -> float:
    """Where the pre-key tone switches on, in xd samples; NaN if unobservable.

    The tone's phase is taken from its own plateau, 200-20 ms before A1, so
    it need not match the A pair's.  A CUSUM on the projection onto that
    phase (levels 0 and |plateau|) finds the step between ``first`` (the
    window's first sample) or A1 - 320 ms and A1 - 100 ms; the step fit of
    ``_prekey_edge`` then refines it.  Unobservable when the plateau is too
    weak, or the tone already runs at the search's start.
    """
    p_lo, p_hi = int(round(a1 - 0.200 * fs)), int(round(a1 - 0.020 * fs))
    lo = max(first, int(round(a1 - _LEAD_SEARCH_S * fs)))
    hi = int(round(a1 - 0.100 * fs))
    if p_lo < first or hi - lo < 16:
        return float("nan")
    n = np.arange(len(xd))
    z = xd * np.exp(-2j * np.pi * SUBCARRIER_HZ * n / fs)
    c = np.mean(z[p_lo:p_hi])
    resid = np.mean(np.abs(z[p_lo:p_hi] - c) ** 2)
    if not (resid > 0 and abs(c) ** 2 / resid * (p_hi - p_lo) >= PLATEAU_MIN_SNR):
        return float("nan")
    r = np.real(z[lo:hi] * np.conj(c)) / abs(c) - 0.5 * abs(c)
    tail = np.cumsum(r[::-1])[::-1]                 # sum from each e to hi
    k = int(np.argmax(tail))
    if k == 0:
        return float("nan")                         # tone on before the search
    coarse = float(lo + k)
    edge, edge_snr = _prekey_edge(xd, fs, coarse)
    sym = fs / SYMBOL_RATE
    if (edge_snr >= EDGE_MIN_SNR
            and -_EDGE_PRE_SYM + 0.5 < (edge - coarse) / sym < _EDGE_POST_SYM - 0.5):
        return edge
    return coarse


def _not_found(reason: str, snr_db: float = float("nan")) -> Arrival:
    nan = float("nan")
    return Arrival(found=False, arrival_utc=nan, delay_s=nan, snr_db=snr_db,
                   peak_width_s=nan, second_peak_ratio=nan,
                   second_peak_delay_s=nan, doppler_hz=nan, m1_shift=None,
                   reason=reason, lobe_distortion=nan, prekey_lead_s=nan,
                   flags="", a1_utc=nan)


def time_squitter(x: np.ndarray, fs: float, t0_utc: float, nominal_utc: float,
                  label_lag_s: float = 0.0) -> Arrival:
    """Time the squitter in window x; see the module docstring for the method.

    ``label_lag_s`` is how late the window's UTC label runs; it comes off the
    arrival.  ``delay_s`` is arrival minus ``nominal_utc``.  Bad data never
    raises: it comes back found=False with a reason.
    """
    x = np.asarray(x, dtype=complex)
    tpl = np.asarray(timing_template(), dtype=float)
    pair_syms = tpl[PREKEY_LEN:]                    # A, A
    sym = fs / SYMBOL_RATE                          # samples per symbol
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        return _not_found("bad_input")
    if len(x) < len(tpl) * sym:
        return _not_found("too_short")
    if not np.any(x):
        return _not_found("no_peak")

    h_pair = _render(pair_syms, fs)
    longest = len(pair_syms) + M1_LEN + M2_LEN + 2 * _PAD
    # Silence ahead of the window lets every correlation reach negative lags,
    # so a burst near the window's start is still seen (and then judged
    # truncated, not mistimed).
    lead = int(np.ceil(longest * sym))
    xp = np.concatenate([np.zeros(lead, dtype=complex), x])

    # 1. Coarse: A pair under Doppler hypotheses.  The template's sample 0
    # sits _PAD symbols before A1.
    best_mag, best_k, best_hz = -1.0, 0, 0.0
    for hz in np.arange(-_DOPPLER_SPAN_HZ, _DOPPLER_SPAN_HZ + 1e-9, _DOPPLER_STEP_HZ):
        mag = np.abs(_correlate(_rotate(xp, fs, hz), h_pair))
        k = int(np.argmax(mag))
        if mag[k] > best_mag:
            best_mag, best_k, best_hz = float(mag[k]), k, float(hz)
    a1_coarse = best_k + _PAD * sym

    # 2. Doppler from the pre-key tone, phase-blind.
    doppler = _doppler_from_prekey(
        xp, fs, int(round(a1_coarse - _PREKEY_FFT_FROM_S * fs)),
        int(round(a1_coarse - _PREKEY_FFT_TO_S * fs)))
    if not abs(doppler - best_hz) <= _DOPPLER_AGREE_HZ:
        doppler = best_hz                           # no tone where it should be
    xd = _rotate(xp, fs, doppler)

    # 3. Detection on the A pair.  The median runs over the window's own
    # lags, so the silent lead does not dilute it.
    pair_c = np.abs(_correlate(xd, h_pair))
    kp = int(np.argmax(pair_c))
    peak = float(pair_c[kp])
    median = float(np.median(pair_c[lead:]))
    if not (median > 0 and peak > 0):
        return _not_found("no_peak")
    snr_db = float(20 * np.log10(peak / median))
    if snr_db < FOUND_SNR_DB:
        return _not_found("no_peak", snr_db)
    sep = np.abs(np.arange(len(pair_c)) - kp) / fs
    far = (sep > AMBIGUOUS_SEP_S) & ~(np.abs(sep - _SINGLE_A_S) < _SINGLE_A_SKIP_S)
    second_far = float(pair_c[far].max()) if far.any() else 0.0

    # 4. M1, at the A pair's integer peak.
    m1 = _m1_shift(xd, fs, kp + _PAD * sym)
    if m1 is None:
        fine_syms = pair_syms
    else:
        m1_seq = m1_symbols(m1)
        fine_syms = np.concatenate([pair_syms, m1_seq, m1_seq[:M2_LEN]])

    # 5. Fine timing on A, A, M1, M2.  A template cut inside the burst picks
    # up a bias from the pulse tails it shares with the symbols across the
    # cut: before A1 a pre-key of random phase, after M2 data in a real
    # frame, so both average to zero (see the report for the sizes).
    h_fine = _render(fine_syms, fs)
    fine = _correlate_fine(xd, h_fine)
    reach = int(_MAIN_SEARCH_S * fs * _UPSAMPLE)
    lo = max(kp * _UPSAMPLE - reach, 0)
    hi = min(kp * _UPSAMPLE + reach + 1, len(fine))
    j = lo + int(np.argmax(fine[lo:hi]))
    a1_p = (j + _parabola(fine, j)) / _UPSAMPLE + _PAD * sym     # A1, xp samples
    a1 = a1_p - lead                                             # A1, x samples

    # The fine template, A1 to its last symbol, must lie inside the window.
    if a1 < 0 or a1 + len(fine_syms) * sym > len(x):
        return _not_found("truncated", snr_db)

    # 6. Width and other paths from the A pair, upsampled.
    pair = _correlate_fine(xd, h_pair)
    centre = int(round((a1_p - _PAD * sym) * _UPSAMPLE))
    lo = max(centre - _UPSAMPLE * 2, 0)
    jp = lo + int(np.argmax(pair[lo:centre + _UPSAMPLE * 2 + 1]))
    width_s = _lobe_width(pair, jp) / (_UPSAMPLE * fs)
    ratio2, delay2 = _second_path(pair, jp, fs)

    if second_far >= AMBIGUOUS_RATIO * peak or ratio2 >= AMBIGUOUS_RATIO:
        return _not_found("ambiguous", snr_db)

    # 7. Quality; it never moves the answer or refuses it.
    # Lobe distortion: the A pair's main lobe against the clean template's.
    # Noise alone distorts it by up to ~0.8x the correlation's relative noise,
    # so the flag threshold rides above that.
    clean = np.concatenate([np.zeros(len(h_pair), dtype=complex), h_fine])
    ref = _correlate_fine(clean, h_pair)
    jr = int(np.argmax(ref))
    distortion = _lobe_distortion(pair, jp, ref, jr, fs)
    rel_noise = _noise_sigma(x, fs) * float(np.sqrt(np.sum(np.abs(h_pair) ** 2))) / pair[jp]
    flags = []
    if distortion > max(LOBE_DISTORTION_FLAG, LOBE_NOISE_FACTOR * rel_noise):
        flags.append("unresolved_multipath")
    tone_start = _prekey_start(xd, fs, a1_p, lead)
    prekey_lead_s = (a1_p - tone_start) / fs

    # An epoch double resolves only ~0.24 us, so the delay is formed from
    # the small differences first and never by subtracting two epochs.
    start = a1 - PREKEY_LEN * sym                   # the 448-symbol convention
    delay_s = (t0_utc - nominal_utc) + (start / fs - label_lag_s)
    arrival_utc = t0_utc + (start / fs - label_lag_s)
    a1_utc = t0_utc + (a1 / fs - label_lag_s)

    return Arrival(
        found=True,
        arrival_utc=float(arrival_utc),
        delay_s=float(delay_s),
        snr_db=snr_db,
        peak_width_s=float(width_s),
        second_peak_ratio=ratio2,
        second_peak_delay_s=delay2,
        doppler_hz=float(doppler),
        m1_shift=m1,
        reason="ok",
        lobe_distortion=distortion,
        prekey_lead_s=float(prekey_lead_s),
        flags=",".join(flags),
        a1_utc=float(a1_utc),
    )
