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

Everything correlates on magnitude, so a polarity-inverted preamble times
the same as an upright one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .waveform import (
    A_LEN,
    M2_LEN,
    M_SHIFTS,
    PREKEY_LEN,
    SUBCARRIER_HZ,
    SYMBOL_RATE,
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

_PAD = 8                    # silent symbols each side of a rendered template
_UPSAMPLE = 32              # interpolation factor for fine timing
_DOPPLER_STEP_HZ = 2.0
_DOPPLER_SPAN_HZ = 8.0      # coarse search covers +/- this
_PREKEY_GUARD_S = 0.005     # skip this much at each end of the pre-key
_PREKEY_FFT_S = 0.240
_TONE_SEARCH_HZ = 30.0      # look for the pre-key tone this far from 1440 Hz
_MAIN_SEARCH_S = 0.005      # fine peak sought this close to the coarse hit
_LOBE_EXCLUDE_SYM = 2.3     # the A pair's first sidelobe (0.19) ends here


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


def time_squitter(x: np.ndarray, fs: float, t0_utc: float, nominal_utc: float,
                  label_lag_s: float = 0.0) -> Arrival:
    """Time the squitter in window x; see the module docstring for the method.

    ``label_lag_s`` is how late the window's UTC label runs; it comes off the
    arrival.  ``delay_s`` is arrival minus ``nominal_utc``.
    """
    x = np.asarray(x, dtype=complex)
    tpl = np.asarray(timing_template(), dtype=float)
    prekey_len = PREKEY_LEN
    h_full = _render(tpl, fs)
    h_pair = _render(tpl[prekey_len:], fs)
    sym = fs / SYMBOL_RATE                          # samples per symbol

    # 1. Coarse: A pair under Doppler hypotheses.
    best_mag, best_k = -1.0, 0
    for hz in np.arange(-_DOPPLER_SPAN_HZ, _DOPPLER_SPAN_HZ + 1e-9, _DOPPLER_STEP_HZ):
        mag = np.abs(_correlate(_rotate(x, fs, hz), h_pair))
        k = int(np.argmax(mag))
        if mag[k] > best_mag:
            best_mag, best_k = float(mag[k]), k
    # The A-pair template's sample 0 sits _PAD symbols before the first A.
    burst_start = int(round(best_k + (_PAD - prekey_len) * sym))

    # 2. Doppler from the pre-key tone.
    doppler = _doppler_from_prekey(x, fs, burst_start)
    xd = _rotate(x, fs, doppler)

    # 3. Detection on the whole template.
    full = np.abs(_correlate(xd, h_full))
    kf = int(np.argmax(full))
    peak = full[kf]
    median = float(np.median(full))
    snr_db = float(20 * np.log10(peak / median)) if median > 0 else float("inf")
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
    fine = _correlate_fine(xd, _render(fine_syms, fs))
    reach = int(_MAIN_SEARCH_S * fs * _UPSAMPLE)
    lo = max(kf * _UPSAMPLE - reach, 0)
    hi = min(kf * _UPSAMPLE + reach + 1, len(fine))
    j = lo + int(np.argmax(fine[lo:hi]))
    start = (j + _parabola(fine, j)) / _UPSAMPLE + _PAD * sym   # burst start, samples
    # An epoch double resolves only ~0.24 us, so the delay is formed from
    # the small differences first and never by subtracting two epochs.
    delay_s = (t0_utc - nominal_utc) + (start / fs - label_lag_s)
    arrival_utc = t0_utc + (start / fs - label_lag_s)

    # 6. Width and other paths from the A pair, which has no pedestal.
    pair = _correlate_fine(xd, h_pair)
    centre = int(round((start + (prekey_len - _PAD) * sym) * _UPSAMPLE))
    lo = max(centre - _UPSAMPLE * 2, 0)
    jp = lo + int(np.argmax(pair[lo:centre + _UPSAMPLE * 2 + 1]))
    width_s = _lobe_width(pair, jp) / (_UPSAMPLE * fs)
    ratio2, delay2 = _second_path(pair, jp, fs)

    if snr_db < FOUND_SNR_DB:
        found, reason = False, "no_peak"
    elif second_far >= AMBIGUOUS_RATIO * peak:
        found, reason = False, "ambiguous"
    else:
        found, reason = True, "ok"

    if not found:
        m1 = None

    return Arrival(
        found=found,
        arrival_utc=float(arrival_utc),
        delay_s=float(delay_s),
        snr_db=snr_db,
        peak_width_s=float(width_s),
        second_peak_ratio=ratio2,
        second_peak_delay_s=delay2,
        doppler_hz=float(doppler),
        m1_shift=m1,
        reason=reason,
    )
