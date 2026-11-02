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
                        ref_source: str, ref_encoding: int, t_level_active) -> dict:
    """One JSON-able result per repeat (the `--json` output shape)."""
    return {
        "band": band,
        "probe_fs": probe_fs,
        "ref_fs": ref_fs,
        "lag_us": lag_s * 1e6,
        "corr": corr,
        "anchor_source": anchor_source,
        "ref_source": ref_source,
        "ref_encoding": ref_encoding,
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

# Encodings whose payload ka9q's decoder turns into samples.  0 (status packet
# omitted it) would silently decode as F32LE, so it is not accepted.
def _decodable_encodings() -> frozenset:
    from ka9q import Encoding
    return frozenset({
        Encoding.S16LE, Encoding.S16BE, Encoding.F32LE, Encoding.F32BE,
        Encoding.F16LE, Encoding.F16BE, Encoding.MULAW, Encoding.ALAW})


def _check_ref_encoding(enc: int, ssrc=None) -> int:
    if enc not in _decodable_encodings():
        raise SystemExit(
            f"label-check: reference channel (SSRC {ssrc}) reports encoding "
            f"{enc}, which is unknown or not decodable as samples; refusing "
            f"to guess (0 would be parsed as F32LE)")
    return int(enc)


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
    _check_ref_encoding(cands[0].encoding, cands[0].ssrc)
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


# The probe lives on a finite lifetime so a crash leaves a short orphan, and a
# keep-alive (a command carrying LIFETIME; bare polls do not refresh) runs
# during capture.  Removal sets lifetime 1: radiod reaps a channel only when
# its lifetime runs out, so a 0 Hz "removed" channel with lifetime 0
# (infinite) never goes away and the next repeat would be handed it back.
PROBE_LIFETIME_FRAMES = 1000        # ~20 s at 50 frames/s
KEEPALIVE_SEC = 3.0
REAP_TIMEOUT_SEC = 30.0


def _gone(control, ssrc: int) -> bool:
    return control.poll_channel(ssrc, timeout=1.0, allow_idle=True) is None


def _wait_gone(control, ssrc: int, timeout: float | None = None) -> bool:
    """Wait until radiod no longer answers for ``ssrc``; False on timeout."""
    if timeout is None:
        timeout = REAP_TIMEOUT_SEC
    t_end = time.time() + timeout
    while True:
        if _gone(control, ssrc):
            return True
        if time.time() >= t_end:
            logger.error("probe SSRC %d still answering after %.0f s",
                         ssrc, timeout)
            return False
        time.sleep(1.0)


def _rtp_labelled(anchor) -> bool:
    """True when the anchor came from RTP timing, not the host wall clock."""
    return bool(anchor.rtp_referenced) and anchor.source.startswith("rtp_to_utc")


def _reap_probe(control, probe_ssrc: int, ref_ssrc: int) -> bool:
    """Reap our probe channel (lifetime 1, then wait); never hf-timestd's.

    Returns True once radiod no longer answers for the SSRC.
    """
    if probe_ssrc == ref_ssrc:
        raise SystemExit(
            f"label-check: probe SSRC {probe_ssrc} equals the reference "
            f"SSRC; refusing to remove hf-timestd's channel")
    control.set_channel_lifetime(probe_ssrc, 1)
    return _wait_gone(control, probe_ssrc)


def _probe_ssrc(control, freq_hz: float, band) -> int:
    """The SSRC ensure_channel will give the probe (a hash of its params)."""
    from ka9q import allocate_ssrc

    from hfdl_recorder.core.radiod import HFDL_ENCODING, HFDL_PRESET
    return allocate_ssrc(
        frequency_hz=freq_hz, preset=HFDL_PRESET,
        sample_rate=band.samprate_hz, agc=False, gain=0.0,
        destination=control._derive_client_destination(),
        encoding=HFDL_ENCODING, radiod_host=control.status_address)


def _keepalive(multi, ssrc: int, stop: threading.Event) -> None:
    while not stop.wait(KEEPALIVE_SEC):
        try:
            multi.set_channel_lifetime(ssrc, PROBE_LIFETIME_FRAMES)
        except Exception:  # noqa: BLE001 - the capture will time out instead
            logger.exception("probe keep-alive failed")


def _run_repeat(i, *, control, ref, band, seconds, reader, label_fn=_label,
                ref_stream_cls=None, multi_cls=None):
    """One capture + measurement.  Returns a record dict, or None on failure.

    The reference path is subscribe-only: a RadiodStream built straight from
    the discovered ChannelInfo.  It never goes through ensure_channel, so it
    cannot create, modify or remove a channel on hf-timestd's group.
    """
    if ref_stream_cls is None:
        from ka9q import RadiodStream as ref_stream_cls
    if multi_cls is None:
        from ka9q import MultiStream as multi_cls
    from hfdl_recorder.core.radiod import HFDL_ENCODING, HFDL_PRESET

    # A probe left from an earlier run would be handed back by ensure_channel
    # (same SSRC): no fresh RTP origin.  Reap it before creating.
    want = _probe_ssrc(control, float(ref.frequency), band)
    if want == ref.ssrc:
        raise SystemExit(
            f"label-check: probe SSRC {want} equals the reference SSRC; "
            f"refusing to continue")
    if not _gone(control, want):
        logger.warning("probe SSRC %d already present; reaping it", want)
        if not _reap_probe(control, want, ref.ssrc):
            print(f"label-check: repeat {i}: stale probe {want} would not "
                  f"go away; repeat failed", file=sys.stderr)
            return None

    ref_fs = int(ref.sample_rate)
    ref_c = _Collector(int(seconds * ref_fs))
    probe_c = _Collector(int(seconds * band.samprate_hz))
    ref_stream = ref_stream_cls(channel=ref, on_samples=ref_c)
    probe_ms = multi_cls(control=control)
    probe_info = probe_ms.add_channel(
        frequency_hz=float(ref.frequency), preset=HFDL_PRESET,
        sample_rate=band.samprate_hz, encoding=HFDL_ENCODING,
        on_samples=probe_c, lifetime=PROBE_LIFETIME_FRAMES)
    if probe_info.ssrc == ref.ssrc:
        raise SystemExit(
            f"label-check: probe SSRC {probe_info.ssrc} equals the reference "
            f"SSRC (the reference channel); refusing to continue.  The probe "
            f"channel is left on radiod, because removing it would remove "
            f"the reference")
    ok = False
    stop_ka = threading.Event()
    ka = threading.Thread(target=_keepalive,
                          args=(probe_ms, probe_info.ssrc, stop_ka),
                          daemon=True)
    try:
        ka.start()
        probe_ms.start()
        ref_stream.start()
        limit = seconds + 30
        ok = probe_c.done.wait(limit) and ref_c.done.wait(limit)
    finally:
        stop_ka.set()
        ref_stream.stop()
        probe_ms.stop()
        reaped = _reap_probe(control, probe_info.ssrc, ref.ssrc)
    if not reaped:
        print(f"label-check: repeat {i}: probe {probe_info.ssrc} still "
              f"present after {REAP_TIMEOUT_SEC:.0f} s; repeat failed (a "
              f"live probe would be reused with its old RTP origin)",
              file=sys.stderr)
        return None
    if not ok:
        print(f"label-check: repeat {i}: capture timed out", file=sys.stderr)
        return None
    a_ref = label_fn(ref_c, ref, ref_fs, reader)
    a_probe = label_fn(probe_c, probe_info, band.samprate_hz, reader)
    if not (_rtp_labelled(a_ref) and _rtp_labelled(a_probe)):
        print(f"label-check: repeat {i}: DROPPED, label not RTP-derived "
              f"(ref {a_ref.source}, probe {a_probe.source}); a "
              f"wall-clock lag is meaningless", file=sys.stderr)
        return None
    lag, corr = labelled_lag_seconds(
        ref_c.samples(), ref_fs, a_ref.utc,
        probe_c.samples(), band.samprate_hz, a_probe.utc)
    return build_repeat_record(
        band=band.name, probe_fs=band.samprate_hz, ref_fs=ref_fs,
        lag_s=lag, corr=corr, anchor_source=a_probe.source,
        ref_source=a_ref.source, ref_encoding=int(ref.encoding),
        t_level_active=getattr(a_probe.snapshot, "t_level_active", None))


def run_label_check(args) -> int:
    from hamsci_dsp.timing import AuthorityReader
    from ka9q import RadiodControl

    from hfdl_recorder.bands import HFDL_BANDS
    from hfdl_recorder.config import (
        load_config, resolve_config_path, resolve_radiod_block,
        resolve_radiod_status,
    )

    band = HFDL_BANDS.get(args.band)
    if band is None:
        print(f"label-check: unknown band {args.band!r}", file=sys.stderr)
        return 2
    config = load_config(resolve_config_path(instance=None,
                                             explicit_path=args.config))
    status = resolve_radiod_status(resolve_radiod_block(config, args.radiod_id))
    ref = _find_reference(status, args.ref_freq_hz, args.ref_ssrc)
    reader = AuthorityReader()
    control = RadiodControl(status, client_id="hfdl-label-check")
    records: list[dict] = []
    try:
        for i in range(args.repeats):
            rec = _run_repeat(i, control=control, ref=ref, band=band,
                              seconds=args.seconds, reader=reader)
            if rec is None:
                continue
            records.append(rec)
            print(json.dumps(rec) if args.json else
                  f"{band.name} repeat {i}: lag {rec['lag_us']:+.1f} us "
                  f"corr {rec['corr']:.3f} ({rec['anchor_source']})",
                  flush=True)
    finally:
        control.close()
    if records:
        write_corrections(args.out, records)
    return 0 if len(records) == args.repeats else 1
