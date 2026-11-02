"""label-check: does an HFDL channel's UTC label agree with hf-timestd's?

Two channels on one radiod carry the same RF.  Label both streams with
``acquire_anchor_utc`` and cross-correlate: the lag between the labelled
streams equals the label difference plus the filter-delay difference, with
no ionosphere in it.  Deleting and recreating the probe channel between
repeats exposes a per-channel RTP origin (the 2026-10-06 regression).

``labelled_lag_seconds`` is pure numpy and unit-tested.  The radiod-facing
code below it stays thin and runs only against a live radiod.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
import threading
import time
from math import gcd
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("/var/lib/hfdl-recorder/label_corrections.toml")
_INTERP = 8   # cross-correlation interpolation factor before the parabola


def _fft_resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Resample complex ``x`` from fs_in to fs_out in the FFT domain.

    Keeps the time origin: output sample i sits at i / fs_out, as input
    sample i sat at i / fs_in.  Trims the input to a whole number of output
    samples first.  Bins above the lower Nyquist are dropped (band-limit).
    """
    g = gcd(fs_in, fs_out)
    block = fs_in // g
    n = (len(x) // block) * block
    m = n // block * (fs_out // g)
    if n == 0:
        raise ValueError("capture too short to resample")
    spec = np.fft.fft(x[:n])
    out = np.zeros(m, dtype=complex)
    half = min(n, m) // 2
    out[:half] = spec[:half]
    out[m - half:] = spec[n - half:]
    return np.fft.ifft(out) * (m / n)


def labelled_lag_seconds(ref: np.ndarray, ref_fs: int, ref_t0_utc: float,
                         probe: np.ndarray, probe_fs: int,
                         probe_t0_utc: float) -> tuple[float, float]:
    """Lag between two labelled sample streams of the same RF.

    Returns (lag in seconds, peak correlation 0..1).  Positive lag: the
    probe's label runs late, i.e. the probe labels a given piece of signal
    with a later UTC than the reference does.
    """
    ref = np.asarray(ref, dtype=complex)
    probe = np.asarray(probe, dtype=complex)
    if probe_fs != ref_fs:
        probe = _fft_resample(probe, probe_fs, ref_fs)

    # Align by label: drop leading samples of whichever stream starts early.
    skip = int(round((probe_t0_utc - ref_t0_utc) * ref_fs))
    t0_ref, t0_probe = ref_t0_utc, probe_t0_utc
    if skip > 0:
        ref = ref[skip:]
        t0_ref += skip / ref_fs
    elif skip < 0:
        probe = probe[-skip:]
        t0_probe += -skip / ref_fs
    n = min(len(ref), len(probe))
    if n < 16:
        raise ValueError("no usable overlap between the labelled streams")
    ref, probe = ref[:n], probe[:n]

    # c[d] = sum_i ref[i + d] * conj(probe[i]); peak at d = ref index of a
    # signal feature minus its probe index.
    size = 2 * n
    xr = np.fft.fft(ref, size)
    xp = np.fft.fft(probe, size)
    cross = xr * np.conj(xp)
    # Zero-pad the cross-spectrum (split at Nyquist) to interpolate c(d).
    big = np.zeros(size * _INTERP, dtype=complex)
    big[:size // 2] = cross[:size // 2]
    big[-(size // 2):] = cross[size // 2:]
    mag = np.abs(np.fft.ifft(big)) * _INTERP
    k = int(np.argmax(mag))
    y0 = mag[k]
    ym, yp = mag[k - 1], mag[(k + 1) % mag.size]
    denom = ym - 2 * y0 + yp
    frac = 0.5 * (ym - yp) / denom if denom != 0 else 0.0
    pos = k + frac
    if pos > mag.size / 2:
        pos -= mag.size
    d = pos / _INTERP

    # Normalise by the energy of the two streams over the overlap at the
    # peak, so a lag that costs overlap does not read as a weak correlation.
    di = int(round(d))
    r_ov = ref[max(di, 0):n + min(di, 0)]
    p_ov = probe[max(-di, 0):n - max(di, 0)]
    norm = math.sqrt(float(np.sum(np.abs(r_ov) ** 2) * np.sum(np.abs(p_ov) ** 2)))
    peak = float(y0 / norm) if norm > 0 else 0.0
    lag = (t0_probe - t0_ref) - d / ref_fs
    return float(lag), min(max(peak, 0.0), 1.0)


def build_repeat_record(*, band: str, probe_fs: int, ref_fs: int, lag_s: float,
                        corr: float, anchor_source: str,
                        t_level_active) -> dict:
    """One JSON-able result per repeat (the `--json` output shape)."""
    return {
        "band": band,
        "probe_fs": probe_fs,
        "ref_fs": ref_fs,
        "lag_us": lag_s * 1e6,
        "corr": corr,
        "anchor_source": anchor_source,
        "t_level_active": t_level_active,
    }


def write_corrections(path: Path, records: list[dict]) -> None:
    """Write ``[<band>] lag_us = <median>`` (plus spread_us, n) to ``path``.

    Bands already in the file and absent from ``records`` are kept, so
    separate per-band runs accumulate in one file.
    """
    try:
        import tomllib
    except ImportError:  # python < 3.11
        import tomli as tomllib

    path = Path(path)
    table: dict[str, dict] = {}
    if path.exists():
        table = {k: dict(v) for k, v in tomllib.loads(path.read_text()).items()
                 if isinstance(v, dict)}
    by_band: dict[str, list[float]] = {}
    for r in records:
        by_band.setdefault(r["band"], []).append(float(r["lag_us"]))
    for band, lags in by_band.items():
        table[band] = {
            "lag_us": statistics.median(lags),
            "spread_us": max(lags) - min(lags),
            "n": len(lags),
        }
    lines = ["# written by `hfdl-recorder label-check`; lag_us = median label "
             "lag, probe minus hf-timestd reference\n"]
    for band in sorted(table):
        lines.append(f"\n[{band}]\n")
        for key, val in table[band].items():
            lines.append(f"{key} = {val!r}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


# ---- radiod-facing code: thin, exercised only against a live radiod ----

class _PinnedDestination:
    """RadiodControl proxy that pins ensure_channel to an existing group.

    The SSRC hashes the destination, so subscribing to hf-timestd's channel
    needs hf-timestd's multicast address, not our per-client one.
    """

    def __init__(self, control, destination: str):
        self._control = control
        self._destination = destination

    def ensure_channel(self, **kw):
        return self._control.ensure_channel(destination=self._destination, **kw)

    def __getattr__(self, name):
        return getattr(self._control, name)


def _find_reference(status: str, freq_hz: float, ref_ssrc: int | None):
    from ka9q import discover_channels

    chans = discover_channels(status)
    cands = [c for c in chans.values()
             if abs(c.frequency - freq_hz) <= 1.0 and c.preset == "iq"]
    if ref_ssrc is not None:
        cands = [c for c in cands if c.ssrc == ref_ssrc]
    if len(cands) != 1:
        found = [(c.ssrc, c.preset, c.sample_rate, c.multicast_address)
                 for c in cands]
        raise SystemExit(
            f"label-check: expected one iq channel at {freq_hz:.0f} Hz, "
            f"found {len(cands)}: {found}; pass --ref-ssrc to choose")
    return cands[0]


class _Collector:
    """on_samples sink: gathers `n_target` samples, remembers first RTP."""

    def __init__(self, n_target: int):
        self.n_target = n_target
        self.parts: list[np.ndarray] = []
        self.n = 0
        self.first_rtp: int | None = None
        self.done = threading.Event()

    def __call__(self, samples, quality):
        if self.done.is_set():
            return
        if self.first_rtp is None:
            self.first_rtp = quality.delivered_rtp_start
        self.parts.append(np.asarray(samples))
        self.n += len(samples)
        if self.n >= self.n_target:
            self.done.set()

    def samples(self) -> np.ndarray:
        return np.concatenate(self.parts)[: self.n_target]


def _label(collector, info, rate, authority_reader):
    from hamsci_dsp.timing import acquire_anchor_utc
    from ka9q import rtp_to_utc

    a = acquire_anchor_utc(
        first_rtp=collector.first_rtp, channel_info=info,
        rtp_to_utc=rtp_to_utc, authority_reader=authority_reader,
        samples_behind=0, sample_rate=rate)
    return a


def _wait_gone(control, ssrc: int, timeout: float = 90.0) -> None:
    """Wait until radiod no longer answers for a removed SSRC."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        if control.poll_channel(ssrc, timeout=1.0, allow_idle=True) is None:
            return
        time.sleep(2.0)
    logger.warning("probe SSRC %d still answering after %.0f s", ssrc, timeout)


def run_label_check(args) -> int:
    from hamsci_dsp.timing import AuthorityReader
    from ka9q import MultiStream, RadiodControl

    from hfdl_recorder.bands import HFDL_BANDS
    from hfdl_recorder.config import (
        load_config, resolve_config_path, resolve_radiod_block,
        resolve_radiod_status,
    )
    from hfdl_recorder.core.radiod import HFDL_ENCODING, HFDL_PRESET

    band = HFDL_BANDS.get(args.band)
    if band is None:
        print(f"label-check: unknown band {args.band!r}", file=sys.stderr)
        return 2
    config = load_config(resolve_config_path(instance=None,
                                             explicit_path=args.config))
    status = resolve_radiod_status(resolve_radiod_block(config, args.radiod_id))
    ref = _find_reference(status, args.ref_freq_hz, args.ref_ssrc)
    ref_fs = int(ref.sample_rate)
    freq = float(ref.frequency)
    reader = AuthorityReader()
    control = RadiodControl(status, client_id="hfdl-label-check")
    records: list[dict] = []
    try:
        for i in range(args.repeats):
            ref_c = _Collector(int(args.seconds * ref_fs))
            probe_c = _Collector(int(args.seconds * band.samprate_hz))
            ref_ms = MultiStream(control=_PinnedDestination(
                control, ref.multicast_address))
            probe_ms = MultiStream(control=control)
            ref_info = ref_ms.add_channel(
                frequency_hz=freq, preset=ref.preset, sample_rate=ref_fs,
                encoding=ref.encoding, on_samples=ref_c)
            if ref_info.ssrc != ref.ssrc:
                raise SystemExit(
                    f"label-check: subscribed SSRC {ref_info.ssrc} is not "
                    f"hf-timestd's {ref.ssrc}; refusing to measure a "
                    f"channel of our own")
            probe_info = probe_ms.add_channel(
                frequency_hz=freq, preset=HFDL_PRESET,
                sample_rate=band.samprate_hz, encoding=HFDL_ENCODING,
                on_samples=probe_c)
            probe_ms.start()
            ref_ms.start()
            try:
                limit = args.seconds + 30
                ok = (probe_c.done.wait(limit) and ref_c.done.wait(limit))
            finally:
                ref_ms.stop()
                probe_ms.stop()
                control.remove_channel(probe_info.ssrc)
            if not ok:
                print(f"label-check: repeat {i}: capture timed out",
                      file=sys.stderr)
                _wait_gone(control, probe_info.ssrc)
                continue
            a_ref = _label(ref_c, ref_info, ref_fs, reader)
            a_probe = _label(probe_c, probe_info, band.samprate_hz, reader)
            lag, corr = labelled_lag_seconds(
                ref_c.samples(), ref_fs, a_ref.utc,
                probe_c.samples(), band.samprate_hz, a_probe.utc)
            snap = a_probe.snapshot
            rec = build_repeat_record(
                band=band.name, probe_fs=band.samprate_hz, ref_fs=ref_fs,
                lag_s=lag, corr=corr, anchor_source=a_probe.source,
                t_level_active=getattr(snap, "t_level_active", None))
            records.append(rec)
            print(json.dumps(rec) if args.json else
                  f"{band.name} repeat {i}: lag {rec['lag_us']:+.1f} us "
                  f"corr {corr:.3f} ({a_probe.source})", flush=True)
            _wait_gone(control, probe_info.ssrc)
    finally:
        control.close()
    if records:
        write_corrections(args.out, records)
    return 0 if len(records) == args.repeats else 1
