"""Thin wrapper around ka9q-python ``RadiodControl.ensure_channel`` for HFDL.

Per CLIENT-CONTRACT.md §7, clients never pass ``destination=`` — radiod
+ ka9q-python deterministically derive the multicast group. We read the
resolved address back from ``ChannelInfo`` for inventory purposes.

The radiod HFDL fragment ships every band as ``mode=iq`` ``encoding=s16be``
with band-specific ``samprate``. We hard-code those here so the band
table in :mod:`hfdl_recorder.bands` stays purely descriptive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hfdl_recorder.bands import HfdlBand

logger = logging.getLogger(__name__)


HFDL_PRESET = "iq"
# ka9q.Encoding.F32LE = 4. radiod can serve any encoding the operator
# requests; the constraint here is that ka9q-python's IQ payload parser
# (ka9q/stream.py::_parse_payload, the ``if is_iq:`` branch) hard-codes
# np.float32 LE and does not honour the ``encoding`` parameter for IQ
# channels. Asking for s16be IQ delivers raw big-endian bytes that
# numpy reinterprets as float32 garbage (NaN, denormals, ±FLT_MAX).
# Asking for F32LE matches ka9q-python's parser, so the on_samples
# callback gets the clean unit-scaled IQ we need. Different
# (freq, sample_rate, encoding) tuples produce different SSRCs, so
# the s16be channels declared by the radiod HFDL fragment stay idle
# and don't conflict with the F32LE ones we create on demand.
HFDL_ENCODING = 4

# The "iq" preset's default channel filter is ±5 kHz — sized for narrowband
# audio, not an HFDL band.  Left there, radiod resamples a 10 kHz slice of
# spectrum up to the band samprate and ground stations outside ±5 kHz of
# center are lost.  The filter spans the band's Nyquist less a small guard
# for the channelizer transition.
FILTER_GUARD_HZ = 1500


def band_filter_edges(samprate_hz: int) -> tuple[float, float]:
    """Channel filter (low_edge, high_edge) in Hz for an HFDL band."""
    half = samprate_hz / 2 - FILTER_GUARD_HZ
    return -half, half


@dataclass(frozen=True)
class ResolvedChannel:
    band: HfdlBand
    multicast_address: str
    port: int


def ensure_band_channel(control, band: HfdlBand) -> ResolvedChannel:
    """Provision (or look up) a radiod channel for one HFDL band.

    ``control`` is a ka9q.RadiodControl instance. We pass through to
    ``ensure_channel`` and surface the resolved multicast destination
    so callers can subscribe via MultiStream.
    """
    info = control.ensure_channel(
        frequency_hz=float(band.center_hz),
        preset=HFDL_PRESET,
        sample_rate=band.samprate_hz,
        encoding=HFDL_ENCODING,
    )
    logger.info(
        "Provisioned %s: %d Hz @ %d S/s -> %s:%d",
        band.name, band.center_hz, band.samprate_hz,
        info.multicast_address, info.port,
    )
    return ResolvedChannel(
        band=band,
        multicast_address=info.multicast_address,
        port=info.port,
    )
