"""HFDL waveform constants and a synthetic squitter generator.

The preamble sequences are protocol facts (ARINC 635).  The bit lists below
were cross-checked against dumphfdl 1.7 ``src/hfdl.c`` (``A_octets``,
``M1_bits``, ``M_shifts``, ``T``).  The pulse shaping and modulation here
are our own.

All preamble sequences are BPSK.  Bit 0 maps to +1 and bit 1 to -1 (the
liquid-dsp BPSK convention dumphfdl uses).  The pre-key is constant +1.

Time convention: symbol k occupies [k/1800, (k+1)/1800) s and its pulse is
centred at (k+0.5)/1800 s.  A constant run of +1 symbols therefore reaches
half its plateau amplitude at the start of the first symbol.
"""

from __future__ import annotations

import numpy as np

SYMBOL_RATE = 1800
SUBCARRIER_HZ = 1440.0      # the channel's dial frequency sits at 0 Hz
PREKEY_LEN = 448
A_LEN = 127
M1_LEN = 127
M2_LEN = 15
T_LEN = 15
M_SHIFTS = (72, 82, 113, 123, 61, 103, 93, 9)
SLOT_S = 32 / 13            # single-slot duration
_LEAD_S = 0.03              # window starts this long before the nominal burst
_PULSE_SPAN = 8             # raised-cosine pulse kept to +/- this many symbols

# A: 127 bits, MSB first from 16 octets (ARINC 635; dumphfdl A_octets).
_A_OCTETS = (
    0b01011011, 0b10111100, 0b01110100, 0b01010111, 0b00000011, 0b11011001,
    0b10001001, 0b00111001, 0b11110010, 0b00001000, 0b11010101, 0b00110110,
    0b10010100, 0b00101100, 0b00110010, 0b11111110,
)
_A_BITS = [(o >> (7 - i)) & 1 for o in _A_OCTETS for i in range(8)][:A_LEN]

# M1 base sequence (ARINC 635; dumphfdl M1_bits).  M1 for shift index k is
# this list rotated left by M_SHIFTS[k].
_M1_BITS = [
    0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
    1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1,
    0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
    1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
]

# T training block, 15 bits (dumphfdl T = 0x9AF).
_T_BITS = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]


def _bpsk(bits) -> np.ndarray:
    return 1.0 - 2.0 * np.asarray(bits, dtype=float)


def a_symbols() -> np.ndarray:
    return _bpsk(_A_BITS)


def m1_symbols(shift_index: int) -> np.ndarray:
    return np.roll(_bpsk(_M1_BITS), -M_SHIFTS[shift_index])


def timing_template() -> np.ndarray:
    """Pre-key (+1) then A then A: the 702 symbols a squitter always starts with."""
    return np.concatenate([np.ones(PREKEY_LEN), a_symbols(), a_symbols()])


def _rc_pulse(x: np.ndarray, rolloff: float) -> np.ndarray:
    """Raised-cosine pulse at x symbol periods from its centre."""
    den = 1.0 - (2.0 * rolloff * x) ** 2
    sing = np.abs(den) < 1e-9
    safe = np.where(sing, 1.0, den)
    p = np.sinc(x) * np.cos(np.pi * rolloff * x) / safe
    edge = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * rolloff))
    return np.where(sing, edge, p)


def _render(symbols, fs: float, n: int, t0: float, rolloff: float) -> np.ndarray:
    """n samples at fs of the symbol stream starting t0 seconds after sample 0.

    Evaluates the pulse at real-valued times, so t0 and fs need not be
    multiples of the symbol period.  The subcarrier is attached to the
    delayed signal, so a delay moves the carrier phase as a real path does.
    """
    syms = np.asarray(symbols, dtype=float)
    t = np.arange(n) / fs - t0                      # time since burst start
    pos = t * SYMBOL_RATE - 0.5                     # pulse index of each sample
    k0 = np.floor(pos).astype(int)
    out = np.zeros(n)
    for off in range(-_PULSE_SPAN + 1, _PULSE_SPAN + 1):
        k = k0 + off
        ok = (k >= 0) & (k < len(syms))
        kk = np.clip(k, 0, len(syms) - 1)
        out += np.where(ok, syms[kk] * _rc_pulse(pos - k, rolloff), 0.0)
    return out * np.exp(2j * np.pi * SUBCARRIER_HZ * t)


def modulate(symbols: np.ndarray, fs: float, rolloff: float = 0.35) -> np.ndarray:
    """Complex baseband of ``symbols`` (real, one per 1/1800 s) at rate fs."""
    n = int(round(len(symbols) * fs / SYMBOL_RATE))
    return _render(symbols, fs, n, 0.0, rolloff)


def _burst_symbols(m1_shift: int, rng) -> np.ndarray:
    """Pre-key, A, A, M1, M2, nine T blocks, random BPSK data to slot end.

    The nine T blocks sit back to back after M2.  Real frames interleave
    training with data; that detail does not affect the first 702 symbols.
    """
    m1 = m1_symbols(m1_shift)
    parts = [timing_template(), m1, m1[:M2_LEN], np.tile(_bpsk(_T_BITS), 9)]
    head = np.concatenate(parts)
    total = int(round(SLOT_S * SYMBOL_RATE))
    data = rng.choice([-1.0, 1.0], size=max(total - len(head), 0))
    return np.concatenate([head, data])


def synth_squitter(fs: float, delay_s: float, *, snr_db: float,
                   doppler_hz: float = 0.0, paths=(), m1_shift: int = 0,
                   rng) -> np.ndarray:
    """A test window holding one squitter.

    The window lasts 0.03 + 32/13 s.  The burst starts at 0.03 + delay_s.
    ``paths`` is empty (one path, amplitude 1) or a list of
    (extra_delay_s, amplitude) pairs, each relative to delay_s, that replace
    the default path.  Complex white noise is added at ``snr_db`` relative to
    the mean power of the direct burst.  ``doppler_hz`` rotates the burst.
    """
    n = int(round((_LEAD_S + SLOT_S) * fs))
    syms = _burst_symbols(m1_shift, rng)
    rolloff = 0.35
    x = np.zeros(n, dtype=complex)
    for extra, amp in (paths or [(0.0, 1.0)]):
        x += amp * _render(syms, fs, n, _LEAD_S + delay_s + extra, rolloff)
    x *= np.exp(2j * np.pi * doppler_hz * np.arange(n) / fs)
    direct = _render(syms, fs, n, _LEAD_S + delay_s, rolloff)
    live = np.abs(direct) > 0
    power = np.mean(np.abs(direct[live]) ** 2) if live.any() else 1.0
    sigma = np.sqrt(power / 10 ** (snr_db / 10) / 2)
    noise = sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return x + noise
