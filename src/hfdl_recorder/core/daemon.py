"""HfdlRecorder: orchestrates one radiod's per-band pipelines.

One ``HfdlRecorder`` per radiod (= one systemd unit). Provisions a
ka9q channel for each enabled band, groups them into ``MultiStream``
instances by multicast destination (typically all bands land on
``hfdl.local`` per the ka9q-radio HFDL fragment, so a single MultiStream),
and supervises one :class:`BandPipeline` per band.
"""

from __future__ import annotations

import logging
import os
import signal
import time

from hfdl_recorder.config import (
    get_enabled_bands,
    resolve_radiod_status,
)
from hfdl_recorder.core.band_pipeline import BandPipeline
from hfdl_recorder.core.ch_tailer import ChTailer
from hfdl_recorder.core.radiod import (
    HFDL_ENCODING,
    HFDL_PRESET,
)

logger = logging.getLogger(__name__)


class HfdlRecorder:
    """Manages all enabled HFDL bands for a single radiod."""

    def __init__(self, config: dict, radiod_block: dict):
        self._config = config
        self._radiod = radiod_block
        self._radiod_id = radiod_block.get("id", "default")

        self._pipelines: list[BandPipeline] = []
        self._multi_streams: list = []
        self._ch_tailers: list[ChTailer] = []
        self._control = None
        self._running = False

    def run(self) -> None:
        self._running = True
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

        try:
            self._provision()
            self._start()
            self._notify_ready()
            self._main_loop()
        except Exception:
            logger.exception("Fatal error in hfdl-recorder")
        finally:
            self._shutdown()

    # -- provisioning --

    def _provision(self) -> None:
        """Resolve radiod, ensure_channel per band, group MultiStreams."""
        from ka9q import MultiStream, RadiodControl

        status = resolve_radiod_status(self._radiod)
        logger.info("Connecting to radiod at %s", status)
        self._control = RadiodControl(status)

        bands = get_enabled_bands(self._radiod)
        if not bands:
            raise ValueError(
                f"No HFDL bands enabled for radiod {self._radiod_id!r}"
            )

        multi_by_group: dict[tuple, object] = {}

        for band in bands:
            logger.info(
                "Provisioning %s (center=%d Hz, sr=%d S/s)",
                band.name, band.center_hz, band.samprate_hz,
            )
            info = self._control.ensure_channel(
                frequency_hz=float(band.center_hz),
                preset=HFDL_PRESET,
                sample_rate=band.samprate_hz,
                agc_enable=0,
                gain=0.0,
                encoding=HFDL_ENCODING,
            )
            # The "iq" preset's default channel filter is ±5 kHz — sized
            # for narrowband audio, not an HFDL band. Without this call,
            # radiod resamples a 10 kHz slice of spectrum up to the band
            # samprate and ground stations outside ±5 kHz of center are
            # lost. Set the filter to span the full band Nyquist with a
            # small guard for the channelizer transition.
            guard_hz = 1500
            self._control.set_filter(
                ssrc=info.ssrc,
                low_edge=-band.samprate_hz / 2 + guard_hz,
                high_edge=+band.samprate_hz / 2 - guard_hz,
            )
            key = (info.multicast_address, info.port)
            multi = multi_by_group.get(key)
            if multi is None:
                multi = MultiStream(control=self._control)
                multi_by_group[key] = multi

            pipeline = BandPipeline(
                band=band,
                radiod_id=self._radiod_id,
                config=self._config,
            )
            pipeline.attach(multi)
            self._pipelines.append(pipeline)

        self._multi_streams = list(multi_by_group.values())
        logger.info(
            "Provisioned %d band(s) across %d multicast group(s) on radiod %s",
            len(self._pipelines), len(self._multi_streams), self._radiod_id,
        )

    def _start(self) -> None:
        for pipeline in self._pipelines:
            try:
                pipeline.start()
            except Exception:
                logger.exception("Failed to start pipeline %s", pipeline.name)
        for multi in self._multi_streams:
            try:
                multi.start()
            except Exception:
                logger.exception("Failed to start MultiStream")
        self._start_ch_tailers()

    def _start_ch_tailers(self) -> None:
        """Start one ChTailer per enabled band — CONTRACT v0.6 §17.

        Each tailer watches the per-band JSON spool dumphfdl writes
        and inserts parsed frames into `hfdl.spots`.  No-op when
        SIGMOND_CLICKHOUSE_URL is unset.  Failure to import / start is
        non-fatal: dumphfdl's own outputs (local JSON, optional
        airframes.io TCP) are unaffected.
        """
        from pathlib import Path as _Path
        paths = self._config.get("paths", {})
        spool_dir = _Path(paths.get("spool_dir", "/var/lib/hfdl-recorder"))
        station = self._config.get("station", {})
        host_call = station.get("call") or station.get("station_id") or ""
        host_grid = station.get("grid_square") or station.get("grid") or ""

        try:
            from hfdl_recorder.version import GIT_INFO
            short = (GIT_INFO or {}).get("short", "")
        except Exception:
            short = ""
        try:
            from importlib.metadata import version as pkg_version
            ver = pkg_version("hfdl-recorder")
        except Exception:
            ver = "0.1.0"
        proc_version = f"{ver}+{short}" if short else ver

        for pipeline in self._pipelines:
            band_name = pipeline.name
            json_path = spool_dir / self._radiod_id / f"{band_name}.json"
            try:
                tailer = ChTailer(
                    json_path=json_path,
                    band_name=band_name,
                    radiod_id=self._radiod_id,
                    host_call=host_call,
                    host_grid=host_grid,
                    processing_version=proc_version,
                )
                tailer.start()
                self._ch_tailers.append(tailer)
            except Exception:
                logger.exception(
                    "ch_tailer band=%s startup failed; dumphfdl outputs unaffected",
                    band_name,
                )

    # -- systemd integration --

    def _notify_ready(self) -> None:
        self._sd_notify(b"READY=1")
        logger.info("sd_notify READY=1 sent")

    def _pet_watchdog(self) -> None:
        self._sd_notify(b"WATCHDOG=1")

    @staticmethod
    def _sd_notify(message: bytes) -> None:
        addr = os.environ.get("NOTIFY_SOCKET")
        if not addr:
            return
        try:
            import socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                if addr.startswith("@"):
                    addr = "\0" + addr[1:]
                sock.connect(addr)
                sock.sendall(message)
            finally:
                sock.close()
        except Exception:
            logger.debug("sd_notify failed", exc_info=True)

    # -- main loop --

    def _main_loop(self) -> None:
        watchdog_usec = os.environ.get("WATCHDOG_USEC")
        pet_interval = (
            int(watchdog_usec) / 1_000_000 / 2
            if watchdog_usec else 30.0
        )
        while self._running:
            time.sleep(min(pet_interval, 5.0))
            self._pet_watchdog()

    def _on_signal(self, signum, frame) -> None:
        logger.info("Received signal %d, shutting down", signum)
        self._running = False

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        for tailer in self._ch_tailers:
            try:
                tailer.stop()
            except Exception:
                logger.exception("Error stopping ch_tailer")
        for multi in self._multi_streams:
            try:
                multi.stop()
            except Exception:
                logger.exception("Error stopping MultiStream")
        for pipeline in self._pipelines:
            try:
                pipeline.stop()
            except Exception:
                logger.exception("Error stopping pipeline %s", pipeline.name)
        if self._control is not None:
            try:
                self._control.close()
            except Exception:
                pass
        logger.info("Shutdown complete")
