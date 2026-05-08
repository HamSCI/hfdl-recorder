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

    @property
    def name(self) -> str:
        return self._band.name

    @property
    def band(self) -> HfdlBand:
        return self._band

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
        return info.ssrc

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
            "dumphfdl", "/opt/hfdl-recorder/bin/dumphfdl"
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
        logger.warning("%s: ka9q-radio stream dropped", self._band.name)

    def _on_stream_restored(self, *_args, **_kwargs) -> None:
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
