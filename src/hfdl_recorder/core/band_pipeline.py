"""Per-band pipeline: ka9q-python subscription → CF32 → dumphfdl subprocess.

One ``BandPipeline`` is responsible for one HFDL band:

  - Registers the band as a channel on a shared ka9q ``MultiStream``.
  - Spawns one ``dumphfdl --iq-file -`` subprocess.
  - Forwards float32 IQ samples (delivered via ``on_samples``) to the
    subprocess's stdin as interleaved CF32 (32-bit float complex, LE).
  - Writes dumphfdl's stderr to a per-band log file.
  - Restarts the subprocess on exit with exponential backoff (cap 60 s);
    a successful run of >= ``MIN_RUN_SEC`` resets the backoff.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from hfdl_recorder.bands import HfdlBand
from hfdl_recorder.core.feed import build_output_args
from hfdl_recorder.core.radiod import HFDL_ENCODING, HFDL_PRESET

logger = logging.getLogger(__name__)


class BandPipeline:
    """Supervises one (ka9q channel, dumphfdl subprocess) pair."""

    INITIAL_BACKOFF_SEC = 1.0
    MAX_BACKOFF_SEC = 60.0
    # A run lasting at least this long is treated as a healthy session,
    # so the next failure starts the backoff over from the floor.
    MIN_RUN_SEC = 30.0

    def __init__(
        self,
        *,
        band: HfdlBand,
        radiod_id: str,
        config: dict,
    ):
        self._band = band
        self._radiod_id = radiod_id
        self._config = config
        self._station = config.get("station", {})
        self._paths = config.get("paths", {})
        self._sinks = config.get("sinks", {})

        self._stop_event = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._supervisor_thread: threading.Thread | None = None
        self._log_fd = None
        self._multi = None
        self._first_sample_logged = False
        self._bytes_written = 0
        # Optional squitter window keeper (timing/window_keeper.py).  Fed
        # after the dumphfdl write; it never blocks and never raises here.
        self._keeper = None
        self._keeper_failed = False
        # The slot's CURRENT SSRC (MultiStream may re-key it on restore),
        # whether the stream is down, and when the last sample arrived.
        # BandPipeline.may_poll reads these so no status poll ever reaches
        # an SSRC radiod does not have (a poll would mint a channel).
        self._ssrc: int | None = None
        # The frequency the channel reports (ChannelInfo), which status
        # replies must match; band.center_hz until a ChannelInfo says.
        self._channel_freq_hz: float = float(band.center_hz)
        self._dropped = False
        self._last_sample_mono = 0.0

    @property
    def name(self) -> str:
        return self._band.name

    @property
    def band(self) -> HfdlBand:
        return self._band

    def set_keeper(self, keeper) -> None:
        """Hand each block, after dumphfdl's copy, to a window keeper."""
        self._keeper = keeper

    def attach(self, multi, lifetime: int | None = None) -> int:
        """Register our band as a channel in the shared MultiStream.

        Caller is responsible for ``multi.start()`` after all bands attach.
        Returns the SSRC so the daemon can register it for LIFETIME
        keep-alive refreshes.
        """
        self._multi = multi
        info = multi.add_channel(
            frequency_hz=float(self._band.center_hz),
            preset=HFDL_PRESET,
            sample_rate=self._band.samprate_hz,
            encoding=HFDL_ENCODING,
            on_samples=self._on_samples,
            on_stream_dropped=self._on_stream_dropped,
            on_stream_restored=self._on_stream_restored,
            lifetime=lifetime,
        )
        self._ssrc = info.ssrc
        self._note_channel_freq(info)
        return info.ssrc

    # A status poll to this SSRC is safe only while all of these hold.
    POLL_SAMPLE_AGE_SEC = 0.5

    @property
    def current_ssrc(self) -> int | None:
        return self._ssrc

    @property
    def channel_freq_hz(self) -> float:
        return self._channel_freq_hz

    def _note_channel_freq(self, info) -> None:
        f = getattr(info, "frequency", None)
        # 0, None or anything not a number: radiod did not say.
        if isinstance(f, (int, float)) and not isinstance(f, bool) and f > 0:
            self._channel_freq_hz = float(f)

    def may_poll(self, ssrc: int) -> bool:
        """True when a command to ``ssrc`` cannot mint a channel on radiod:
        it is the slot's current SSRC, MultiStream still holds that slot,
        the stream is up, and a sample arrived in the last 0.5 s."""
        if ssrc is None or ssrc != self._ssrc or self._dropped:
            return False
        # Fail closed: a MultiStream without the slot table (or without
        # this SSRC in it) gets no command.
        slots = getattr(self._multi, "_slots", None)
        if slots is None or ssrc not in slots:
            return False
        return time.monotonic() - self._last_sample_mono <= self.POLL_SAMPLE_AGE_SEC

    def start(self) -> None:
        """Open the per-band log and launch the supervisor thread."""
        log_dir = Path(self._paths.get("log_dir", "/var/log/hfdl-recorder"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_fd = open(
            log_dir / f"{self._radiod_id}-{self._band.name}.log", "ab"
        )

        spool_dir = Path(self._paths.get("spool_dir", "/var/lib/hfdl-recorder"))
        (spool_dir / self._radiod_id).mkdir(parents=True, exist_ok=True)

        self._supervisor_thread = threading.Thread(
            target=self._supervise,
            name=f"hfdl-{self._band.name}-sup",
            daemon=True,
        )
        self._supervisor_thread.start()
        logger.info("BandPipeline %s started", self._band.name)

    def stop(self) -> None:
        """Signal shutdown and tear the subprocess down."""
        self._stop_event.set()
        with self._proc_lock:
            proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except OSError:
                pass
        if self._supervisor_thread is not None:
            self._supervisor_thread.join(timeout=10)
        if self._log_fd is not None:
            try:
                self._log_fd.close()
            except OSError:
                pass

    # -- supervisor loop --

    def _supervise(self) -> None:
        backoff = self.INITIAL_BACKOFF_SEC
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self._spawn_dumphfdl()
            except OSError as e:
                logger.error(
                    "%s: failed to spawn dumphfdl: %s", self._band.name, e
                )
                if self._stop_event.wait(backoff):
                    return
                backoff = min(backoff * 2, self.MAX_BACKOFF_SEC)
                continue

            with self._proc_lock:
                proc = self._proc
            assert proc is not None
            proc.wait()
            ran_for = time.monotonic() - started
            with self._proc_lock:
                rc = proc.returncode
                self._proc = None

            if self._stop_event.is_set():
                return

            logger.warning(
                "%s: dumphfdl exited (rc=%s) after %.1fs",
                self._band.name, rc, ran_for,
            )
            if ran_for >= self.MIN_RUN_SEC:
                backoff = self.INITIAL_BACKOFF_SEC
            if self._stop_event.wait(backoff):
                return
            backoff = min(backoff * 2, self.MAX_BACKOFF_SEC)

    def _spawn_dumphfdl(self) -> None:
        argv = self.build_argv()
        logger.debug("%s: spawning %s", self._band.name, " ".join(argv))
        with self._proc_lock:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._log_fd,
                bufsize=0,
            )
        logger.info(
            "%s: dumphfdl started (pid=%d)",
            self._band.name, self._proc.pid,
        )

    def build_argv(self) -> list[str]:
        """Construct the dumphfdl argv for this band.

        Public (no leading underscore) so unit tests can assert on it
        without spawning a process.
        """
        dumphfdl = self._paths.get(
            "dumphfdl", "/opt/git/sigmond/hfdl-recorder/bin/dumphfdl"
        )
        systable = self._paths.get(
            "systable", "/var/lib/hfdl-recorder/systable.conf"
        )
        spool = Path(self._paths.get("spool_dir", "/var/lib/hfdl-recorder"))
        station_id = self._station.get("station_id", "")

        # dumphfdl wants centerfreq in kHz; ``%g`` keeps "21964" or
        # "10061.5" depending on whether the band has a half-kHz offset.
        center_khz = f"{self._band.center_hz / 1000:g}"

        argv: list[str] = [
            dumphfdl,
            "--iq-file", "-",
            "--sample-format", "cf32",
            "--sample-rate", str(self._band.samprate_hz),
            "--centerfreq", center_khz,
        ]
        if station_id:
            argv += ["--station-id", station_id]
        if systable:
            argv += [
                "--system-table", systable,
                "--system-table-save", systable,
            ]

        argv += build_output_args(
            self._sinks,
            spool_dir=spool,
            radiod_id=self._radiod_id,
            band_name=self._band.name,
        )

        argv += [str(int(round(khz))) for khz in self._band.channels_khz]
        return argv

    # -- ka9q-python callbacks (called from MultiStream worker thread) --

    def _on_samples(self, samples, quality=None) -> None:
        if not self._first_sample_logged:
            try:
                arr = np.asarray(samples)
                flat = arr.view(np.float32) if arr.dtype == np.complex64 else arr
                nan_frac = float(np.isnan(flat).sum()) / max(flat.size, 1)
                finite = flat[np.isfinite(flat)]
                peak = float(np.max(np.abs(finite))) if finite.size else 0.0
                rms = float(np.sqrt(np.mean(finite ** 2))) if finite.size else 0.0
                logger.info(
                    "%s: first samples dtype=%s shape=%s nan_frac=%.3f finite_peak=%.4f finite_rms=%.4f first5=%s",
                    self._band.name, arr.dtype, arr.shape, nan_frac, peak, rms,
                    flat[:5].tolist(),
                )
            except Exception as e:
                logger.warning("%s: first-sample debug failed: %s", self._band.name, e)
            self._first_sample_logged = True

        self._write_dumphfdl(samples)
        self._last_sample_mono = time.monotonic()

        keeper = self._keeper
        if keeper is not None:
            rtp = getattr(quality, "delivered_rtp_start", None)
            if rtp is not None:
                gaps = getattr(quality, "batch_gaps", None)
                try:
                    if gaps:
                        keeper.feed(samples, rtp, gap_samples=sum(
                            int(g.duration_samples) for g in gaps))
                    else:
                        keeper.feed(samples, rtp)
                    if self._keeper_failed:
                        self._keeper_failed = False
                        logger.info("%s: window keeper recovered", self._band.name)
                except Exception:  # noqa: BLE001 - decoding must not notice
                    if not self._keeper_failed:
                        self._keeper_failed = True
                        logger.exception(
                            "%s: window keeper failed; dumphfdl unaffected "
                            "(silent until it recovers)", self._band.name)

    def _write_dumphfdl(self, samples) -> None:
        with self._proc_lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                return
            stdin = proc.stdin
        try:
            blob = _f32_iq_to_cf32(samples)
            stdin.write(blob)
            self._bytes_written += len(blob)
        except (BrokenPipeError, OSError) as e:
            # Subprocess is dying; supervisor will notice and restart.
            logger.debug("%s: write to dumphfdl failed: %s", self._band.name, e)

    def _on_stream_dropped(self, *_args, **_kwargs) -> None:
        # No teardown needed on a drop: dumphfdl tolerates an input gap (it
        # just waits on stdin), and we hold no per-channel ring/closure to
        # orphan.  Recovery is owned by dumphfdl + the supervisor; the slot
        # resumes on restore.
        self._dropped = True
        logger.warning("%s: ka9q-radio stream dropped", self._band.name)

    def _on_stream_restored(self, channel_info=None, *_args, **_kwargs) -> None:
        # No re-subscribe: channels are provisioned ONCE at startup and the
        # existing MultiStream slot resumes.  The SSRC usually stays the same
        # across a radiod restart, but MultiStream._attempt_restore re-keys
        # the slot itself when ensure_channel returns a new one.
        # MAINTAINER NOTE: if a mid-flight reprovision path is ever added (a
        # new SSRC per re-ensure_channel), it MUST prune the superseded slot
        # — multi.prune_frequency(freq, keep_ssrc=...) — exactly as
        # wspr-recorder/receiver_manager.py does, or the old slot leaks and
        # the band can go silent on a changed SSRC.
        # MultiStream._attempt_restore re-keys the slot when radiod hands
        # back a new SSRC; follow it, so status polls go to the live one.
        new_ssrc = getattr(channel_info, "ssrc", None)
        if new_ssrc is not None and new_ssrc != self._ssrc:
            logger.warning("%s: stream restored on a new SSRC %s (was %s)",
                           self._band.name, new_ssrc, self._ssrc)
            self._ssrc = new_ssrc
        if channel_info is not None:
            self._note_channel_freq(channel_info)
        self._dropped = False
        logger.info("%s: ka9q-radio stream restored", self._band.name)


def _f32_iq_to_cf32(samples) -> bytes:
    """Convert float32 IQ → CF32 bytes (interleaved float32 LE I/Q pairs).

    dumphfdl's CF32 input is identical in layout to numpy ``complex64``:
    interleaved float32 little-endian I,Q pairs. ``np.nan_to_num`` zeros
    NaN/Inf the resequencer may have inserted on a packet drop so dumphfdl
    never sees a garbage sample.
    """
    arr = np.asarray(samples)
    if arr.dtype == np.complex64:
        flat = arr.view(np.float32)
    elif arr.dtype == np.complex128:
        flat = arr.astype(np.complex64).view(np.float32)
    else:
        flat = arr.astype(np.float32, copy=False).ravel()
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(flat).tobytes()
