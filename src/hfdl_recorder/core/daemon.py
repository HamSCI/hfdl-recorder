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
import threading
import time
from typing import Optional

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

    def __init__(
        self,
        config: dict,
        radiod_block: dict,
        *,
        reporter_id: Optional[str] = None,
    ):
        self._config = config
        self._radiod = radiod_block
        # Phase 6: canonical identifier is the mDNS status name.
        self._radiod_id = resolve_radiod_status(radiod_block)
        # Phase-5 (sigmond MULTI-INSTANCE-ARCHITECTURE.md §3): per-
        # instance reporter ID.  None on legacy single-instance hosts;
        # ChTailer falls back to radiod_id at row construction.
        self._reporter_id = reporter_id

        self._pipelines: list[BandPipeline] = []
        self._multi_streams: list = []
        # (MultiStream, ssrc) pairs for LIFETIME keep-alive — populated
        # at provisioning, consumed by the lifetime thread.
        # (MultiStream, BandPipeline): the keep-alive refreshes the slot's
        # CURRENT SSRC, which MultiStream may re-key on a restore.
        self._lifetime_entries: list[tuple[object, BandPipeline]] = []
        self._ch_tailers: list[ChTailer] = []
        self._control = None
        self._running = False

        # radiod LIFETIME tag (ka9q-python ≥3.13.0).  0 = no LIFETIME
        # tag sent + no keep-alive; >0 = self-destruct after N frames,
        # refreshed at frames/4 cadence while we're alive.
        proc = config.get("processing", {})
        self._radiod_lifetime_frames: int = int(
            proc.get("radiod_lifetime_frames", 0)
        )
        self._lifetime_thread: threading.Thread | None = None

        # Squitter timing windows (off by default).  One keeper per band;
        # each entry: (keeper, ssrc, ChannelCounterMap).
        self._timing_cfg: dict = config.get("timing_squitters", {}) or {}
        self._keepers: list[tuple] = []
        self._timing_thread: threading.Thread | None = None
        self._authority_reader = None

    def run(self) -> None:
        self._running = True
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

        try:
            self._provision()
            self._start()
            self._start_lifetime_keepalive()
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
        # client_id makes ka9q-python derive a per-(client, radiod)
        # multicast destination so HFDL bands never share a multicast
        # group with peer clients on the same radiod.  CONTRACT v0.3
        # §7 / ka9q-python ≥ 3.14.0.
        self._control = RadiodControl(status, client_id="hfdl-recorder")

        bands = get_enabled_bands(self._radiod)
        if not bands:
            raise ValueError(
                f"No HFDL bands enabled for radiod {self._radiod_id!r}"
            )

        # `lifetime=None` when configured to 0 — distinguishes "no
        # LIFETIME tag at all" from "finite N frames".
        lifetime_arg = (
            self._radiod_lifetime_frames
            if self._radiod_lifetime_frames > 0 else None
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
                lifetime=lifetime_arg,
            )
            # Defensive: ensure_channel can return an existing SSRC at
            # the right (freq, preset, sample_rate) but a stale
            # encoding.  Observed on bee1 after the 2026-04-27 switch
            # from HFDL_ENCODING=2 (s16be) → 4 (F32LE): five of seven
            # bands kept reporting encoding=2 to discover_channels
            # while the daemon believed it was running f32.  The
            # consumer (dumphfdl, --sample-format cf32) then read the
            # s16be bytes as float32 and produced silent garbage.
            # Force the encoding when ensure_channel's return diverges
            # from what we asked for, then re-read to confirm.
            current_enc = getattr(info, "encoding", None)
            if current_enc is not None and int(current_enc) != HFDL_ENCODING:
                logger.warning(
                    "%s: ensure_channel returned ssrc=%d with "
                    "encoding=%d, expected %d — forcing via "
                    "set_output_encoding",
                    band.name, info.ssrc, int(current_enc), HFDL_ENCODING,
                )
                try:
                    self._control.set_output_encoding(
                        ssrc=info.ssrc, encoding=HFDL_ENCODING,
                    )
                except Exception:
                    logger.exception(
                        "%s: set_output_encoding(ssrc=%d, %d) failed",
                        band.name, info.ssrc, HFDL_ENCODING,
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
            ssrc = pipeline.attach(multi, lifetime=lifetime_arg)
            if lifetime_arg is not None:
                self._lifetime_entries.append((multi, pipeline))
            self._pipelines.append(pipeline)
            if self._timing_cfg.get("enabled"):
                try:
                    self._add_keeper(pipeline, ssrc)
                except Exception:
                    logger.exception(
                        "%s: squitter window keeper not built; decoding "
                        "unaffected", band.name)

        self._multi_streams = list(multi_by_group.values())
        logger.info(
            "Provisioned %d band(s) across %d multicast group(s) on radiod %s",
            len(self._pipelines), len(self._multi_streams), self._radiod_id,
        )

    # -- squitter timing windows --

    def _add_keeper(self, pipeline: BandPipeline, ssrc: int) -> None:
        """Build one WindowKeeper for this band and hand it to the pipeline.

        Both labels come from the band channel's own status replies,
        gathered off the sample path: a passive listen first, then polls only
        to the slot's CURRENT SSRC and only while ``pipeline.may_poll`` holds.
        ``ssrc`` names the channel at build time, for the log.
        """
        from pathlib import Path as _Path

        from hamsci_dsp.timing import AuthorityReader

        from hfdl_recorder.timing.channel_status import (
            ChannelCounterMap,
            least_late_anchor,
        )
        from hfdl_recorder.timing.window_keeper import WindowKeeper

        if self._authority_reader is None:
            self._authority_reader = AuthorityReader()
        reader = self._authority_reader
        band = pipeline.band
        counter = ChannelCounterMap(band.name, band.samprate_hz, band.center_hz)
        counter.bind(ssrc)
        control = self._control

        def anchor_fn(rtp: int):
            live = pipeline.current_ssrc
            freq = pipeline.channel_freq_hz
            counter.bind(live, freq)
            return least_late_anchor(
                control, live, rtp, band.samprate_hz, freq_hz=freq,
                gate=pipeline.may_poll, authority_reader=reader, counter=counter,
                chain_delay_correction_ns=pipeline.chain_delay_correction_ns)

        def tier_fn():
            snap = reader.read()
            return snap.t_level_active if snap is not None else None

        ts = self._timing_cfg
        keeper = WindowKeeper(
            band=band, fs=band.samprate_hz,
            out_dir=_Path(ts["window_dir"]) / self._radiod_id,
            ground_stations=list(ts["ground_stations"]),
            anchor_fn=anchor_fn, counter_map_fn=counter.current,
            min_free_gb=float(ts["min_free_gb"]),
            retention_days=int(ts["retention_days"]),
            tier_fn=tier_fn,
        )
        pipeline.set_keeper(keeper)
        self._keepers.append((keeper, pipeline, counter))
        logger.info("%s: squitter windows on (ssrc=%d, dir=%s)",
                    band.name, ssrc, keeper.out_dir)

    def _note_squitter(self, frame: dict) -> None:
        # A squitter heard on one band announces frequencies on all of them.
        for keeper, _pipeline, _counter in self._keepers:
            keeper.note_squitter(frame)

    TIMING_POLL_SEC = 60.0
    TIMING_STATS_SEC = 600.0

    def _start_timing(self) -> None:
        if not self._keepers:
            return
        started = []
        for entry in self._keepers:
            keeper = entry[0]
            try:
                keeper.start()
                started.append(entry)
            except Exception:
                logger.exception("squitter window keeper failed to start; "
                                 "decoding unaffected")
                for pipeline in self._pipelines:
                    if pipeline._keeper is keeper:
                        pipeline.set_keeper(None)
        self._keepers = started
        if not self._keepers:
            return
        self._timing_thread = threading.Thread(
            target=self._timing_loop, daemon=True, name="squitter-timing")
        self._timing_thread.start()

    def _refresh_counter_maps(self) -> None:
        """One pass: each band's counter map hears its channel's status.

        A passive listen first; polls only to the slot's current SSRC while
        ``may_poll`` holds (see BandPipeline.may_poll for why).
        """
        from hfdl_recorder.timing.channel_status import gather_status

        for _keeper, pipeline, counter in self._keepers:
            ssrc = pipeline.current_ssrc
            if ssrc is None:
                continue
            freq = pipeline.channel_freq_hz
            counter.bind(ssrc, freq)
            band = pipeline.band
            for reply in gather_status(
                    self._control, ssrc, counter.polls_wanted,
                    rate=band.samprate_hz, freq_hz=freq,
                    gate=pipeline.may_poll):
                counter.add_reply(reply)

    def _timing_loop(self) -> None:
        """Keep each band's counter map fresh; log keeper stats.  Off the
        sample path; at most one status exchange per band per minute."""
        next_poll = 0.0
        next_stats = time.monotonic() + self.TIMING_STATS_SEC
        while self._running:
            now = time.monotonic()
            if now >= next_poll:
                next_poll = now + self.TIMING_POLL_SEC
                try:
                    self._refresh_counter_maps()
                except Exception:
                    logger.exception("counter map refresh failed")
            if now >= next_stats:
                next_stats = now + self.TIMING_STATS_SEC
                for keeper, _pipeline, _counter in self._keepers:
                    logger.info("%s", keeper.stats_line())
            time.sleep(1.0)

    def _start(self) -> None:
        self._start_timing()
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
        and stages parsed frames into `hfdl.spots` via sigmond's local
        SQLite sink.  Resolves to a no-op when the sink path is
        unwritable.  Failure to import / start is non-fatal: dumphfdl's
        own outputs (local JSON, optional airframes.io TCP) are
        unaffected.
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
                    reporter_id=self._reporter_id,
                    host_call=host_call,
                    host_grid=host_grid,
                    processing_version=proc_version,
                    on_squitter=self._note_squitter if self._keepers else None,
                )
                tailer.start()
                self._ch_tailers.append(tailer)
            except Exception:
                logger.exception(
                    "ch_tailer band=%s startup failed; dumphfdl outputs unaffected",
                    band_name,
                )

    # -- LIFETIME keep-alive --

    def _start_lifetime_keepalive(self) -> None:
        """Refresh radiod's LIFETIME on every active SSRC at frames/4 cadence.

        No-op when radiod_lifetime_frames is 0 or no channels opted in.
        Failure to refresh (network blip, radiod restart) must not crash
        the daemon — log and continue; MultiStream's drop/restore path
        re-applies the slot's lifetime when reception resumes.
        """
        if not self._lifetime_entries:
            return
        # Refresh every quarter of the lifetime — gives 4× safety
        # margin against radiod self-destruct if a single refresh is
        # missed.  Floor at 1 s so absurd configs don't busy-loop.
        interval = max(self._radiod_lifetime_frames / 50.0 / 4.0, 1.0)
        logger.info(
            "lifetime keepalive: %d channels, %d frames, refresh every %.1fs",
            len(self._lifetime_entries),
            self._radiod_lifetime_frames,
            interval,
        )
        self._lifetime_thread = threading.Thread(
            target=self._lifetime_loop,
            args=(interval,),
            daemon=True,
            name="lifetime",
        )
        self._lifetime_thread.start()

    def _lifetime_loop(self, interval_sec: float) -> None:
        while self._running:
            time.sleep(interval_sec)
            if not self._running:
                break
            self._refresh_lifetimes()

    def _refresh_lifetimes(self) -> None:
        """One keep-alive pass, to each slot's current SSRC.  A build-time
        SSRC goes stale after a re-key: set_channel_lifetime no-ops for it,
        nothing refreshes the new channel, and radiod reaps it."""
        for multi, pipeline in self._lifetime_entries:
            ssrc = pipeline.current_ssrc
            if ssrc is None:
                continue
            try:
                multi.set_channel_lifetime(ssrc, self._radiod_lifetime_frames)
            except Exception as exc:
                logger.warning(
                    "lifetime keepalive failed (ssrc=%s): %s", ssrc, exc,
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
        for keeper, _pipeline, _counter in self._keepers:
            try:
                keeper.stop()
            except Exception:
                logger.exception("Error stopping window keeper")
        if self._control is not None:
            try:
                self._control.close()
            except Exception:
                pass
        logger.info("Shutdown complete")
