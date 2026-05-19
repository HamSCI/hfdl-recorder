"""Spot-log tailer for hfdl-recorder (CONTRACT v0.6 §17).

Watches the per-band JSON spool that `dumphfdl` writes via
`--output decoded:json:file:path=<path>` (one compact JSON object per
line, terminated with `\\n` — see ka9q/dumphfdl `fmtr-json.c:55` /
`EOL(vstr)`).  Parses each new frame, extracts the high-signal fields,
preserves the raw JSON for a future re-parse, and inserts rows into
`hfdl.spots` via `sigmond.hamsci_sink.Writer.from_env()`.

Runs as a daemon thread inside the HfdlRecorder process, parallel to
dumphfdl's own optional airframes.io TCP feed (which is not affected
because that path is dumphfdl-internal, not via this file).

`Writer.from_env()` stages rows into sigmond's local SQLite sink by
default (`/var/lib/sigmond/sink.db`); it resolves to a clean no-op
only when the sink path is unwritable.  This tailer is the
producer-side sink; moving the airframes / SFTP upload path onto
`hs-uploader` is still future work.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Frame parser ────────────────────────────────────────────────────────────

# Keys we recognize at any depth inside the libacars-emitted PDU tree.
# Mapping: libacars-key → hfdl.spots column.  Unknown keys are ignored.
_INTERESTING_KEYS = {
    "icao":           "icao_addr",       # ICAO 24-bit address (int)
    "icao_address":   "icao_addr",       # alt. spelling some payloads use
    "ac_info":        "_ac_info_dict",   # subobject — handled below
    "flight_id":      "flight",
    "flight":         "flight",
    "tail":           "aircraft_reg",
    "reg":            "aircraft_reg",
    "label":          "acars_label",
    "msg_text":       "acars_message",
    "message":        "acars_message",
}


def parse_hfdl_frame(line: str, *, band_name: str) -> Optional[dict]:
    """Parse one `dumphfdl` JSON frame line into an `hfdl.spots` row.

    Returns None on parse failure — callers skip silently.  Best-effort:
    when a nested payload is unrecognized the row is still emitted with
    all known top-level fields and `raw_json` preserved for re-parse.
    """
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    hfdl = obj.get("hfdl") if isinstance(obj, dict) else None
    if not isinstance(hfdl, dict):
        return None

    t_obj = hfdl.get("t") or {}
    try:
        sec = int(t_obj.get("sec", 0))
        usec = int(t_obj.get("usec", 0))
    except (ValueError, TypeError):
        return None
    if sec <= 0:
        return None
    ts = datetime.fromtimestamp(sec + usec / 1_000_000, tz=timezone.utc).replace(tzinfo=None)

    try:
        freq = int(hfdl.get("freq", 0) or 0)
    except (ValueError, TypeError):
        freq = 0

    extracted: dict[str, Any] = {}
    _walk_for_fields(hfdl, extracted)
    direction = _direction_from(hfdl)

    return {
        "time":          ts,
        "band_name":     band_name,
        "station_id":    str(hfdl.get("station", "")),
        "frequency":     freq,
        "frequency_mhz": freq / 1_000_000.0,
        "bit_rate":      _safe_int(hfdl.get("bit_rate")),
        "sig_level":     _safe_float(hfdl.get("sig_level")),
        "noise_level":   _safe_float(hfdl.get("noise_level")),
        "freq_skew":     _safe_float(hfdl.get("freq_skew")),
        "slot":          str(hfdl.get("slot", "")),
        "direction":     direction,
        "ground_station": extracted.get("ground_station", ""),
        "icao_addr":     extracted.get("icao_addr"),
        "flight":        extracted.get("flight", ""),
        "aircraft_reg":  extracted.get("aircraft_reg", ""),
        "acars_label":   extracted.get("acars_label", ""),
        "acars_message": extracted.get("acars_message", ""),
        "position_lat":  extracted.get("position_lat"),
        "position_lon":  extracted.get("position_lon"),
        "position_alt_ft": extracted.get("position_alt_ft"),
        "raw_json":      line,
    }


def _safe_int(v: Any) -> int:
    try:
        return int(v) if v is not None else 0
    except (ValueError, TypeError):
        return 0


def _safe_float(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _direction_from(hfdl: dict) -> str:
    """Recognize uplink/downlink by which PDU subobject is present.

    spdu = squitter (ground-station beacon, broadcast)
    mpdu/lpdu = directional packets — direction tag if libacars emits one
    """
    if "spdu" in hfdl:
        return "downlink"   # squitters originate from the ground station
    for key in ("mpdu", "lpdu"):
        sub = hfdl.get(key)
        if isinstance(sub, dict):
            d = sub.get("dir") or sub.get("direction")
            if isinstance(d, str):
                return d.lower()
            if "src" in sub and "dst" in sub:
                return ""   # ambiguous without explicit dir tag
    return ""


def _walk_for_fields(node: Any, out: dict) -> None:
    """DFS the PDU tree pulling out high-signal fields by key.

    Tolerant of libacars schema drift: unknown keys are ignored, and we
    never crash on unexpected types.  Order of traversal is irrelevant
    because we only set fields that aren't already populated.
    """
    if isinstance(node, dict):
        # Some objects encode the ground station as `{"type":"ground",
        # "name":...,"id":...}` under `src`/`dst` rather than a
        # dedicated `gs` key (LPDU/MPDU directional packets).  Detect
        # that here so downlink/uplink frames both surface ground info.
        if (
            isinstance(node.get("type"), str)
            and node["type"].lower() == "ground"
            and "ground_station" not in out
        ):
            gs_str = node.get("name") or node.get("id")
            if gs_str:
                out["ground_station"] = str(gs_str)

        for key, value in node.items():
            # Position-bearing payloads (HFNPDU position reports).
            if key in ("position", "pos") and isinstance(value, dict):
                _extract_position(value, out)
                continue
            # Ground-station id is `gs` (a dict with id/name) or
            # `ground_station` (rare).  Don't overwrite once set.
            if key in ("gs", "ground_station") and isinstance(value, dict):
                gs_str = value.get("name") or value.get("id")
                if gs_str and "ground_station" not in out:
                    out["ground_station"] = str(gs_str)
                continue
            mapped = _INTERESTING_KEYS.get(key)
            if mapped == "_ac_info_dict" and isinstance(value, dict):
                # ac_info is a nested record: {"icao":..., "tail":...,
                # "operator":..., "type":...}
                if "icao_addr" not in out:
                    icao = value.get("icao") or value.get("icao_address")
                    out["icao_addr"] = _icao_to_int(icao)
                if "aircraft_reg" not in out:
                    reg = value.get("tail") or value.get("reg")
                    if reg:
                        out["aircraft_reg"] = str(reg).strip()
                continue
            if mapped and not isinstance(value, (dict, list)):
                if mapped == "icao_addr" and "icao_addr" not in out:
                    out["icao_addr"] = _icao_to_int(value)
                elif mapped not in out:
                    s = str(value).strip()
                    if s:
                        out[mapped] = s
            # Recurse into sub-trees regardless.
            if isinstance(value, (dict, list)):
                _walk_for_fields(value, out)
    elif isinstance(node, list):
        for item in node:
            _walk_for_fields(item, out)


def _extract_position(pos: dict, out: dict) -> None:
    """Pull lat/lon/alt out of an HFNPDU position record (best-effort)."""
    lat = pos.get("lat") or pos.get("latitude")
    lon = pos.get("lon") or pos.get("longitude")
    alt = pos.get("alt") or pos.get("altitude")
    if lat is not None and "position_lat" not in out:
        try:
            out["position_lat"] = float(lat)
        except (ValueError, TypeError):
            pass
    if lon is not None and "position_lon" not in out:
        try:
            out["position_lon"] = float(lon)
        except (ValueError, TypeError):
            pass
    if alt is not None and "position_alt_ft" not in out:
        try:
            out["position_alt_ft"] = int(alt)
        except (ValueError, TypeError):
            pass


def _icao_to_int(value: Any) -> Optional[int]:
    """ICAO 24-bit addresses appear as int, hex string, or 0x-prefixed."""
    if value is None:
        return None
    if isinstance(value, int):
        return value if 0 <= value < (1 << 24) else None
    s = str(value).strip().lower()
    if not s:
        return None
    try:
        if s.startswith("0x"):
            return int(s, 16)
        # Try decimal first, then hex (libacars often prints 6-hex-digit).
        try:
            return int(s)
        except ValueError:
            return int(s, 16)
    except ValueError:
        return None


# ── Tailer ──────────────────────────────────────────────────────────────────

class ChTailer:
    """One tailer per (radiod, band) JSON file.

    Spawns a daemon thread that polls the file for new frames, parses
    each, and inserts rows into `hfdl.spots` via hamsci_sink.Writer.
    Clean no-op only when the sink path is unwritable.
    """

    POLL_INTERVAL_SEC = 1.0
    FLUSH_INTERVAL_SEC = 30.0

    def __init__(
        self,
        *,
        json_path: Path,
        band_name: str,
        radiod_id: str,
        host_call: str = "",
        host_grid: str = "",
        processing_version: str = "",
        batch_rows: int = 200,
        writer_factory=None,
    ) -> None:
        self._json_path = Path(json_path)
        self._band_name = band_name
        self._radiod_id = radiod_id
        self._host_call = host_call
        self._host_grid = host_grid
        self._processing_version = processing_version
        self._batch_rows = batch_rows
        self._writer_factory = writer_factory or _default_writer_factory
        self._writer = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_pos = 0
        self._last_flush = 0.0
        # Buffer for partial trailing line across reads.
        self._partial = ""

    def start(self) -> None:
        try:
            self._writer = self._writer_factory(self._batch_rows)
        except Exception as e:
            logger.warning("ch_tailer disabled (band=%s): %s", self._band_name, e)
            return
        if self._writer.is_noop:
            logger.debug("ch_tailer band=%s: sink writer is a no-op "
                         "(sink path unwritable)", self._band_name)
        if self._json_path.exists():
            try:
                self._last_pos = self._json_path.stat().st_size
            except OSError:
                self._last_pos = 0
        self._stop.clear()
        self._last_flush = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"ch-tail-{self._radiod_id}-{self._band_name}",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass

    @property
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def health(self) -> str:
        if self._writer is None:
            return "noop"
        return self._writer.health

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.POLL_INTERVAL_SEC):
                self._poll_once()
        except Exception:
            logger.exception("ch_tailer band=%s: unhandled error",
                             self._band_name)

    def _poll_once(self) -> None:
        if self._writer is None:
            return
        try:
            stat = self._json_path.stat()
        except FileNotFoundError:
            return
        size = stat.st_size
        if size < self._last_pos:
            self._last_pos = 0
            self._partial = ""
        if size > self._last_pos:
            try:
                with open(self._json_path, "rb") as fh:
                    fh.seek(self._last_pos)
                    chunk = fh.read(size - self._last_pos)
                self._last_pos = size
            except OSError as e:
                logger.warning("ch_tailer band=%s: read failed: %s",
                               self._band_name, e)
                return
            self._consume(chunk.decode(errors="replace"))

        if (time.monotonic() - self._last_flush) > self.FLUSH_INTERVAL_SEC:
            try:
                self._writer.flush()
            except Exception as e:
                logger.warning("ch_tailer band=%s: flush failed: %s",
                               self._band_name, e)
            self._last_flush = time.monotonic()

    def _consume(self, text: str) -> None:
        # Split into complete lines; any trailing partial line goes back
        # into self._partial and prepends the next read.
        text = self._partial + text
        if "\n" not in text:
            self._partial = text
            return
        lines = text.split("\n")
        self._partial = lines[-1]              # partial trailing fragment
        rows: list[dict] = []
        for line in lines[:-1]:
            row = parse_hfdl_frame(line, band_name=self._band_name)
            if row is None:
                continue
            row["host_call"] = self._host_call
            row["host_grid"] = self._host_grid
            row["radiod_id"] = self._radiod_id
            row["instance"] = self._radiod_id
            row["processing_version"] = self._processing_version
            rows.append(row)
        if rows:
            try:
                self._writer.insert(rows)
            except Exception as e:
                logger.warning("ch_tailer band=%s: insert failed (%d rows): %s",
                               self._band_name, len(rows), e)


def _default_writer_factory(batch_rows: int):
    """Lazy-import sigmond.hamsci_sink.Writer for hfdl.spots."""
    from sigmond.hamsci_sink import Writer
    return Writer.from_env(
        table="spots", mode="hfdl",
        schema_version=1, batch_rows=batch_rows,
    )
