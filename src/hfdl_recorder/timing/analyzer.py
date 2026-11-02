"""Time one HFDL squitter against its known preamble.

Input: a window of complex baseband IQ at rate fs (the channel's dial
frequency at 0 Hz, the burst on the +1440 Hz subcarrier) whose first sample
carries the UTC label t0_utc.  Output: when the burst started, in the
convention of ``waveform`` (the 50 % point of the pre-key's leading edge),
and that time minus the slot's nominal start.

The method, in order:

1. Coarse search.  Correlate the window against the A pair (141 ms) under a
   few Doppler hypotheses 2 Hz apart.  The A pair alone tolerates a 1 Hz
   residual, and its strongest match locates the pre-key.
2. Doppler.  The pre-key holds a constant tone at 1440 Hz plus the Doppler
   for 249 ms.  An FFT of 240 ms of it, Hann-windowed, zero-padded and
   interpolated by parabola, measures the Doppler; the window is then
   rotated back by it.
3. Detection.  Correlate against the whole timing template (pre-key, A, A).
   Its peak over its median gives ``snr_db``, and a second peak more than
   50 ms away at 0.8 of the first makes the window ambiguous.
4. M1.  The eight M1 rotations are tried at the symbols after the A pair.
   The winner labels the squitter and extends the fine template.
5. Fine timing.  Correlate against pre-key, A, A, M1, M2 (M1 and M2 only
   when M1 is clear), interpolated 32 times by zero-padding the
   cross-spectrum, which is exact for a band-limited correlation; a
   parabola on the magnitude gives the arrival.  The 249 ms tone gives this
   correlation a broad triangular pedestal, but it varies little across
   the few samples the parabola sees.
6. Width and multipath.  The A pair's correlation has no pedestal, so it
   gives the main-lobe width and the strongest other path within 50 ms,
   beyond the A pair's first sidelobe (0.19 of the peak, out to 2.3 symbols).
   A path there at 0.8 of the main one or more makes the window ambiguous:
   which mode is "the" arrival becomes a coin toss.
7. Quality flags, which never move or refuse the answer:
   ``unresolved_multipath`` when the A pair's main lobe departs from the
   clean template's by more than noise explains, and
   ``prekey_edge_mismatch`` when the pre-key's own leading edge, fitted
   from the tone without the A pair or PREKEY_LEN, sits more than 300 us
   (or 5 sd of its own scatter) from the A-derived start, or cannot be
   found under a strong plateau.

Limits, measured on synth_squitter:

- An echo closer than about 1.3 ms (2.4 kHz of bandwidth cannot resolve
  it) merges with the main lobe and pulls the answer, roughly in
  proportion to its amplitude, by up to ~0.4x the echo spacing at 0.6
  relative amplitude: -116 us for 0.6 at 0.3 ms,
  +113 us for 0.6 at 0.6 ms, +26 us for 0.6 at 1.0 ms, -18 us for 0.6 at
  1.3 ms.  ``unresolved_multipath`` catches these; at high SNR the largest
  unflagged pull seen was -21 us (0.1 at 0.3 ms), and the flag's threshold
  rises with noise.
- A burst whose fine template (pre-key edge to M2) does not lie wholly in
  the window comes back ``truncated``, never timed.
- The coarse search covers +/- 20 Hz of carrier offset.

Every not-found result carries NaN in the timing and quality fields; only
``snr_db`` survives once it has been measured.  Reasons: ok, no_peak,
ambiguous, truncated, bad_input (non-finite samples), too_short.

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

# snr_db = peak / median of the whole-template correlation.  Pure noise
# scores 10.3-14.7 dB (300 trials, median 11.2), so the brief's 10 dB would
# call noise found.  A burst at -16 dB input SNR (per 6 kHz) still scores
# 16-19 dB, and its timing error there runs 47 us rms.
FOUND_SNR_DB = 17.0
AMBIGUOUS_RATIO = 0.8       # a far peak at this fraction of the first
AMBIGUOUS_SEP_S = 0.050     # ... more than this far away
MULTIPATH_SPAN_S = 0.050    # other paths are looked for within this span
M1_CLEAR_RATIO = 2.0        # best M1 shift must beat the next by this
LOBE_DISTORTION_FLAG = 0.015    # flag floor for A-pair main-lobe distortion
LOBE_NOISE_FACTOR = 1.5         # ... raised to this x the correlation's relative noise
PREKEY_EDGE_FLAG_S = 300e-6     # pre-key edge vs A-derived start
# The edge is checked only when the pre-key plateau's per-sample SNR
# (linear) reaches this, about 8 dB input SNR.  The fitted edge scatters
# with sd ~ EDGE_SD_K / sqrt(edge fit SNR): 81 us at 8 dB, 62 at 10, 49 at
# 12 (200 trials each), so the flag fires beyond 300 us or 5 sd, whichever
# is larger.
EDGE_MIN_SNR = 6.0
EDGE_SD_K = 210e-6

_ROLLOFF = 0.35
_PAD = 8                    # silent symbols each side of a rendered template
_UPSAMPLE = 32              # interpolation factor for fine timing
_DOPPLER_STEP_HZ = 2.0
_DOPPLER_SPAN_HZ = 20.0     # coarse search covers +/- this
_PREKEY_GUARD_S = 0.005     # skip this much at each end of the pre-key
_PREKEY_FFT_S = 0.240
_TONE_SEARCH_HZ = 30.0      # look for the pre-key tone this far from 1440 Hz
_MAIN_SEARCH_S = 0.005      # fine peak sought this close to the coarse hit
_LOBE_EXCLUDE_SYM = 2.3     # the A pair's first sidelobe (0.19) ends here
_EDGE_PRE_SYM = 6           # pre-key edge sought this far before the A-derived start
_EDGE_POST_SYM = 4          # ... and this far after it
_EDGE_MISSING_RATIO = 0.25  # edge fit this far below the plateau's SNR: no edge there


@dataclass
class Arrival:
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
    prekey_edge_offset_s: float     # fitted pre-key edge minus A-derived start; NaN unchecked
    flags: str                      # comma-separated quality flags, "" when clean


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


def _doppler_from_prekey(x: np.ndarray, fs: float, start: int) -> float:
    """Carrier offset of the pre-key tone that starts at sample ``start``."""
    lo = max(start + int(round(_PREKEY_GUARD_S * fs)), 0)
    hi = min(lo + int(round(_PREKEY_FFT_S * fs)), len(x))
    seg = x[lo:hi]
    if len(seg) < 16:
        return 0.0
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
    """The pre-key's leading edge, in samples, found from the tone alone.

    Mix the 1440 Hz subcarrier to 0 Hz, then fit a * step(t - e) by least
    squares over a few symbols around ``start`` (a complex and free, e on a
    grid refined by parabola).  This uses neither the A pair nor
    PREKEY_LEN, so it checks the zero the template sets.  Returns (edge,
    per-sample SNR of the fit); NaNs when the fit window leaves the signal.
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


def _prekey_plateau_snr(xd: np.ndarray, fs: float, start: float) -> float:
    """Per-sample SNR of the pre-key tone, 60-200 ms after ``start``."""
    lo = int(round(start + 0.060 * fs))
    hi = int(round(start + 0.200 * fs))
    if lo < 0 or hi > len(xd):
        return float("nan")
    n = np.arange(lo, hi)
    z = xd[lo:hi] * np.exp(-2j * np.pi * SUBCARRIER_HZ * n / fs)
    a = np.mean(z)
    resid = np.mean(np.abs(z - a) ** 2)
    return float(abs(a) ** 2 / resid) if resid > 0 else float("inf")


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


def _not_found(reason: str, snr_db: float = float("nan")) -> Arrival:
    nan = float("nan")
    return Arrival(found=False, arrival_utc=nan, delay_s=nan, snr_db=snr_db,
                   peak_width_s=nan, second_peak_ratio=nan,
                   second_peak_delay_s=nan, doppler_hz=nan, m1_shift=None,
                   reason=reason, lobe_distortion=nan,
                   prekey_edge_offset_s=nan, flags="")


def time_squitter(x: np.ndarray, fs: float, t0_utc: float, nominal_utc: float,
                  label_lag_s: float = 0.0) -> Arrival:
    """Time the squitter in window x; see the module docstring for the method.

    ``label_lag_s`` is how late the window's UTC label runs; it comes off the
    arrival.  ``delay_s`` is arrival minus ``nominal_utc``.  Bad data never
    raises: it comes back found=False with a reason.
    """
    x = np.asarray(x, dtype=complex)
    tpl = np.asarray(timing_template(), dtype=float)
    prekey_len = PREKEY_LEN
    sym = fs / SYMBOL_RATE                          # samples per symbol
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        return _not_found("bad_input")
    if len(x) < len(tpl) * sym:
        return _not_found("too_short")
    if not np.any(x):
        return _not_found("no_peak")

    h_full = _render(tpl, fs)
    h_pair = _render(tpl[prekey_len:], fs)
    longest = len(tpl) + M1_LEN + M2_LEN + 2 * _PAD
    # Silence ahead of the window lets every correlation reach negative lags,
    # so a burst that starts in the window's first few ms, or before it, is
    # still seen (and then judged truncated, not mistimed).
    lead = int(np.ceil(longest * sym))
    xp = np.concatenate([np.zeros(lead, dtype=complex), x])

    # 1. Coarse: A pair under Doppler hypotheses.
    best_mag, best_k = -1.0, 0
    for hz in np.arange(-_DOPPLER_SPAN_HZ, _DOPPLER_SPAN_HZ + 1e-9, _DOPPLER_STEP_HZ):
        mag = np.abs(_correlate(_rotate(xp, fs, hz), h_pair))
        k = int(np.argmax(mag))
        if mag[k] > best_mag:
            best_mag, best_k = float(mag[k]), k
    # The A-pair template's sample 0 sits _PAD symbols before the first A.
    burst_start = int(round(best_k + (_PAD - prekey_len) * sym))

    # 2. Doppler from the pre-key tone.
    doppler = _doppler_from_prekey(xp, fs, burst_start)
    xd = _rotate(xp, fs, doppler)

    # 3. Detection on the whole template.  The median runs over the window's
    # own lags, so the silent lead does not dilute it.
    full = np.abs(_correlate(xd, h_full))
    kf = int(np.argmax(full))
    peak = float(full[kf])
    median = float(np.median(full[lead:]))
    if not (median > 0 and peak > 0):
        return _not_found("no_peak")
    snr_db = float(20 * np.log10(peak / median))
    if snr_db < FOUND_SNR_DB:
        return _not_found("no_peak", snr_db)
    far = np.abs(np.arange(len(full)) - kf) > AMBIGUOUS_SEP_S * fs
    second_far = float(full[far].max()) if far.any() else 0.0

    # 4. M1 first, at the whole template's integer peak, so the fine-timing
    # template can run through it.  It labels the squitter; a wrong or
    # missing M1 only shortens the fine template.
    a_coarse = kf + (_PAD + prekey_len) * sym      # first A symbol, samples
    m1 = _m1_shift(xd, fs, a_coarse)
    if m1 is None:
        fine_syms = tpl
    else:
        m1_seq = m1_symbols(m1)
        fine_syms = np.concatenate([tpl, m1_seq, m1_seq[:M2_LEN]])

    # 5. Fine timing on the whole template, run through M1 and M2 when M1 is
    # known.  A template cut inside the burst biases the peak through the
    # pulse tails it shares with the known symbols across the cut.  Measured
    # at 80 dB on synth_squitter: A pair alone -1.5 us (pre-key before it,
    # M1 after), pre-key+A+A -0.65 us, +M1 +0.9 us, +M1+M2 -0.55 us (the
    # synthetic T blocks follow M2).  In a real frame random data follows
    # M2, so there the edge term averages to zero.  The pedestal stays
    # smooth across the few samples the parabola sees.
    h_fine = _render(fine_syms, fs)
    fine = _correlate_fine(xd, h_fine)
    reach = int(_MAIN_SEARCH_S * fs * _UPSAMPLE)
    lo = max(kf * _UPSAMPLE - reach, 0)
    hi = min(kf * _UPSAMPLE + reach + 1, len(fine))
    j = lo + int(np.argmax(fine[lo:hi]))
    start_p = (j + _parabola(fine, j)) / _UPSAMPLE + _PAD * sym   # in xp samples
    start = start_p - lead                                         # in x samples

    # The whole fine template, pre-key edge to its last symbol, must lie
    # inside the window; a burst cut at either end times wrongly.
    if start < 0 or start + len(fine_syms) * sym > len(x):
        return _not_found("truncated", snr_db)

    # 6. Width and other paths from the A pair, which has no pedestal.
    pair = _correlate_fine(xd, h_pair)
    centre = int(round((start_p + (prekey_len - _PAD) * sym) * _UPSAMPLE))
    lo = max(centre - _UPSAMPLE * 2, 0)
    jp = lo + int(np.argmax(pair[lo:centre + _UPSAMPLE * 2 + 1]))
    width_s = _lobe_width(pair, jp) / (_UPSAMPLE * fs)
    ratio2, delay2 = _second_path(pair, jp, fs)

    if second_far >= AMBIGUOUS_RATIO * peak or ratio2 >= AMBIGUOUS_RATIO:
        return _not_found("ambiguous", snr_db)

    # 7. Quality flags; they never move the answer or refuse it.
    # Lobe distortion: the A pair's main lobe against the clean template's.
    # Noise alone distorts it by up to ~0.8x the correlation's relative noise
    # (measured -13..10 dB), so the flag threshold rides above that.
    ref = _correlate_fine(h_fine, h_pair)
    jr = int(np.argmax(ref))
    distortion = _lobe_distortion(pair, jp, ref, jr, fs)
    rel_noise = _noise_sigma(x, fs) * float(np.sqrt(np.sum(np.abs(h_pair) ** 2))) / pair[jp]
    flags = []
    if not distortion <= max(LOBE_DISTORTION_FLAG, LOBE_NOISE_FACTOR * rel_noise):
        flags.append("unresolved_multipath")
    # Pre-key edge, checked only when the pre-key's plateau has the SNR for
    # it.  A strong plateau with no edge where the A pair puts one (the fit
    # finds no step, or pins to its search limit) is itself a mismatch.
    edge_offset_s = float("nan")
    plateau_snr = _prekey_plateau_snr(xd, fs, start_p)
    if plateau_snr >= EDGE_MIN_SNR:
        edge, edge_snr = _prekey_edge(xd, fs, start_p)
        offset = (edge - start_p) / sym             # symbols
        pinned = not -_EDGE_PRE_SYM + 0.5 < offset < _EDGE_POST_SYM - 0.5
        if edge_snr >= EDGE_MIN_SNR and not pinned:
            edge_offset_s = (edge - start_p) / fs
            limit = max(PREKEY_EDGE_FLAG_S, 5 * EDGE_SD_K / np.sqrt(edge_snr))
            if abs(edge_offset_s) > limit:
                flags.append("prekey_edge_mismatch")
        elif pinned or edge_snr < _EDGE_MISSING_RATIO * plateau_snr:
            flags.append("prekey_edge_mismatch")

    # An epoch double resolves only ~0.24 us, so the delay is formed from
    # the small differences first and never by subtracting two epochs.
    delay_s = (t0_utc - nominal_utc) + (start / fs - label_lag_s)
    arrival_utc = t0_utc + (start / fs - label_lag_s)

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
        prekey_edge_offset_s=float(edge_offset_s),
        flags=",".join(flags),
    )
