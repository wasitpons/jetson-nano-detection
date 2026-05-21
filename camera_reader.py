"""Per-camera RTSP reader using a hardware-decoded GStreamer pipeline.

One thread per camera. We construct a GStreamer pipeline string and feed it
to OpenCV's GStreamer backend. The pipeline does the heavy lifting on the
Tegra hardware (nvv4l2decoder or omxh264dec, then nvvidconv for resize), so
the Python side just calls cap.read() in a loop and publishes the latest
frame into a LatestFrameBuffer (which drops anything older).

Decoder selection: try `decoder_preference` first; on open or first-frame
failure within `open_probe_timeout_s`, try `fallback_decoder`. If both fail,
back off and retry from the top. A broken camera spins in its own retry loop
privately — it never blocks the others.

The active decoder is reported via snapshot() so metrics can show which path
actually got used.
"""

import logging
import threading
import time
from typing import Optional

import cv2

from frame_buffer import LatestFrameBuffer

log = logging.getLogger(__name__)


def build_pipeline(
    *,
    rtsp_url: str,
    decoder: str,
    output_size: int,
    latency_ms: int = 0,
    protocols: str = "tcp",
    drop_on_latency: bool = True,
) -> str:
    """Assemble a low-latency, drop-old GStreamer pipeline string.

        rtspsrc → rtph264depay → h264parse
            → queue max-size-buffers=1 leaky=downstream
            → <decoder> → nvvidconv → resize+convert to BGR
            → appsink drop=true max-buffers=1 sync=false
    """
    drop = "true" if drop_on_latency else "false"
    return (
        f"rtspsrc location={rtsp_url} latency={latency_ms} "
        f"protocols={protocols} drop-on-latency={drop} "
        f"! rtph264depay "
        # config-interval=-1 forces h264parse to insert SPS/PPS in front of
        # every IDR. Without it, the SPS/PPS arrives only at stream start (or
        # every N seconds depending on the server) — if the leaky queue below
        # drops it before nvv4l2decoder sees it, the decoder spins on
        # "Stream format not found, dropping the frame" forever.
        f"! h264parse config-interval=-1 "
        f"! queue max-size-buffers=1 leaky=downstream "
        f"! {decoder} "
        f"! nvvidconv "
        f"! video/x-raw,format=BGRx,width={output_size},height={output_size} "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        # name must contain "appsink" — cv2's GStreamer backend in OpenCV 4.1.1
        # (Jetson JetPack build) substring-matches element names against
        # "appsink"/"opencvsink" to locate the sink. A bare `name=sink`
        # produces the misleading "cannot find appsink in manual pipeline"
        # warning at open() time.
        f"! appsink name=appsink0 drop=true max-buffers=1 sync=false emit-signals=false"
    )


class CameraReader(threading.Thread):
    def __init__(
        self,
        *,
        camera_id: str,
        rtsp_url: str,
        decoder_preference: str,
        fallback_decoder: str,
        output_size: int,
        buffer: LatestFrameBuffer,
        stop_event: threading.Event,
        latency_ms: int = 0,
        protocols: str = "tcp",
        drop_on_latency: bool = True,
        open_probe_timeout_s: float = 8.0,
        reconnect_backoff_max_s: float = 8.0,
    ) -> None:
        super().__init__(name=f"CameraReader[{camera_id}]", daemon=True)
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.decoder_preference = decoder_preference
        self.fallback_decoder = fallback_decoder
        self.output_size = int(output_size)
        self.buffer = buffer
        self.stop_event = stop_event
        self.latency_ms = latency_ms
        self.protocols = protocols
        self.drop_on_latency = drop_on_latency
        self.open_probe_timeout_s = open_probe_timeout_s
        self.reconnect_backoff_max_s = reconnect_backoff_max_s

        # State (read via snapshot()).
        self.active_decoder: Optional[str] = None
        self.frames_read = 0
        self.read_failures = 0
        self.reconnect_count = 0
        self.last_frame_at = 0.0
        self.frame_width = 0
        self.frame_height = 0
        # cap.read() latency tracking — surfaces blocking time per frame.
        # *_total = lifetime, *_window = reset by snapshot() each metrics tick.
        self.read_total_ms = 0.0
        self.read_max_ms = 0.0
        self.read_window_ms = 0.0
        self.read_window_count = 0
        self.read_window_max_ms = 0.0
        self._lock = threading.Lock()

    # ----- pipeline construction & probing -------------------------------------------------

    def _try_open(self, decoder: str) -> Optional[cv2.VideoCapture]:
        """Open a pipeline with `decoder` and probe for one real frame.

        cv2 will happily report isOpened()=True for a pipeline whose upstream
        eventually fails — so we probe for the first frame ourselves rather
        than discover it inside the hot loop later.
        """
        pipeline = build_pipeline(
            rtsp_url=self.rtsp_url,
            decoder=decoder,
            output_size=self.output_size,
            latency_ms=self.latency_ms,
            protocols=self.protocols,
            drop_on_latency=self.drop_on_latency,
        )
        log.info("[%s] opening pipeline (decoder=%s)", self.camera_id, decoder)
        log.debug("[%s] gst-pipeline: %s", self.camera_id, pipeline)

        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            log.warning("[%s] cv2 could not open pipeline with %s", self.camera_id, decoder)
            return None

        deadline = time.monotonic() + self.open_probe_timeout_s
        while time.monotonic() < deadline:
            if self.stop_event.is_set():
                cap.release()
                return None
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                log.info("[%s] first frame ok: %s w=%d h=%d", self.camera_id, decoder, w, h)
                with self._lock:
                    self.frame_width = w
                    self.frame_height = h
                    self.last_frame_at = time.monotonic()
                    self.frames_read += 1
                self.buffer.put(frame, captured_at=self.last_frame_at)
                return cap
            time.sleep(0.05)

        log.warning("[%s] no frame within %.1fs (decoder=%s); giving up",
                    self.camera_id, self.open_probe_timeout_s, decoder)
        cap.release()
        return None

    def _connect(self) -> Optional[cv2.VideoCapture]:
        """Try primary, then fallback. Return live cap or None."""
        for decoder in (self.decoder_preference, self.fallback_decoder):
            if not decoder:
                continue
            if self.stop_event.is_set():
                return None
            cap = self._try_open(decoder)
            if cap is not None:
                with self._lock:
                    self.active_decoder = decoder
                    self.reconnect_count += 1
                return cap
            with self._lock:
                self.read_failures += 1
        return None

    # ----- thread main ---------------------------------------------------------------------

    def run(self) -> None:
        backoff = 0.5
        cap: Optional[cv2.VideoCapture] = None

        while not self.stop_event.is_set():
            if cap is None:
                cap = self._connect()
                if cap is None:
                    log.warning("[%s] all decoders failed; backoff %.1fs",
                                self.camera_id, backoff)
                    if self.stop_event.wait(backoff):
                        break
                    backoff = min(backoff * 2, self.reconnect_backoff_max_s)
                    continue
                backoff = 0.5

            t_read0 = time.monotonic()
            ok, frame = cap.read()
            read_ms = (time.monotonic() - t_read0) * 1000.0
            if not ok or frame is None:
                with self._lock:
                    self.read_failures += 1
                    self.active_decoder = None
                log.warning("[%s] read failed; reconnecting", self.camera_id)
                cap.release()
                cap = None
                continue

            now = time.monotonic()
            with self._lock:
                self.frames_read += 1
                self.last_frame_at = now
                self.read_total_ms += read_ms
                if read_ms > self.read_max_ms:
                    self.read_max_ms = read_ms
                self.read_window_ms += read_ms
                self.read_window_count += 1
                if read_ms > self.read_window_max_ms:
                    self.read_window_max_ms = read_ms
                h, w = frame.shape[:2]
                if w != self.frame_width or h != self.frame_height:
                    log.info("[%s] frame size changed -> %dx%d", self.camera_id, w, h)
                    self.frame_width, self.frame_height = w, h
            self.buffer.put(frame, captured_at=now)

        if cap is not None:
            cap.release()
        with self._lock:
            self.active_decoder = None
        log.info("[%s] reader stopped", self.camera_id)

    # ----- introspection -------------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            avg_total = (self.read_total_ms / self.frames_read) if self.frames_read else 0.0
            avg_window = (
                self.read_window_ms / self.read_window_count
                if self.read_window_count else 0.0
            )
            max_window = self.read_window_max_ms
            # Reset window so each MetricsCollector tick gets a fresh delta.
            self.read_window_ms = 0.0
            self.read_window_count = 0
            self.read_window_max_ms = 0.0
            return {
                "camera_id": self.camera_id,
                "active_decoder": self.active_decoder,
                "frames_read": self.frames_read,
                "read_failures": self.read_failures,
                "reconnect_count": self.reconnect_count,
                "last_frame_at": self.last_frame_at,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "read_avg_ms_total": avg_total,
                "read_max_ms_total": self.read_max_ms,
                "read_avg_ms_window": avg_window,
                "read_max_ms_window": max_window,
            }
