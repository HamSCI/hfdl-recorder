"""sigmond Receiver Channels TUI parser for hfdl-recorder.

Loaded by sigmond at TUI time via ``[client_features.receiver_channels]``
in ``deploy.toml``.  The band-center table mirrors what
hfdl_recorder.bands.HFDL_BANDS publishes to radiod (the IQ band
centers radiod tunes per ka9q-radio/config/fragments/hfdl.conf),
inlined here to keep the parser self-contained and lazily-loaded by
sigmond's TUI.  If the central table moves, update both.
"""

from __future__ import annotations

from typing import Optional

from sigmond.ka9q_encoding import ENCODING_INTS


# HFDL band IQ-center frequencies (Hz) — one wide IQ channel per band
# is published by radiod; dumphfdl demodulates per-band sub-channel
# slots from each.
_HFDL_BAND_CENTERS_HZ: dict[str, int] = {
    "HFDL2":   2_980_000,
    "HFDL3":   3_477_000,
    "HFDL4":   4_672_000,
    "HFDL5":   5_587_000,
    "HFDL6":   6_622_000,
    "HFDL8":   8_902_500,
    "HFDL10": 10_061_500,
    "HFDL11": 11_287_000,
    "HFDL13": 13_310_000,
    "HFDL15": 15_025_000,
    "HFDL17": 17_944_000,
    "HFDL21": 21_964_000,
}


def parse_receiver_channels(
    cfg: dict,
) -> tuple[str, set[int], Optional[int]]:
    """Return ``(status_dns, configured_freqs_hz, encoding_int)`` from
    an hfdl-recorder per-instance config.

    dumphfdl consumes complex F32 IQ; HFDL_ENCODING = 4 (F32LE) is
    hardcoded in hfdl_recorder.core.radiod with no operator-facing
    override, so we return that constant unconditionally.
    """
    blocks = cfg.get("radiod") or []
    if isinstance(blocks, dict):
        blocks = [blocks]
    status = ""
    freqs: set[int] = set()
    for b in blocks:
        if not status:
            status = str(b.get("status") or "")
        for name in (b.get("bands") or {}).get("enabled", []) or []:
            hz = _HFDL_BAND_CENTERS_HZ.get(name)
            if hz is not None:
                freqs.add(hz)
    return status, freqs, ENCODING_INTS["f32"]
