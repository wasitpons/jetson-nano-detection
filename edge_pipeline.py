#!/usr/bin/env python3
"""jetson_runtime entrypoint.

Wires camera readers, scheduler, detector loop, event logger, and metrics
collector. One YOLO inference loop, fair per-camera scheduling, drop-old
buffers. Ctrl+C / SIGTERM → graceful shutdown.

Usage:
    python3 edge_pipeline.py --config config.yaml
"""

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

# Fail fast on unsupported Python. Jetson Nano JetPack 4.6.x ships Python
# 3.6.9; we don't support anything older. Anything newer is fine for dev.
if sys.version_info < (3, 6):
    sys.exit(
        "ERROR: jetson_runtime requires Python 3.6+ "
        "(JetPack 4.6.x ships 3.6.9). Got {}.{}.{}.".format(*sys.version_info[:3])
    )

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera_reader import CameraReader
from detector import Detector, load_detector
from events import EventLogger
from frame_buffer import LatestFrameBuffer
from frame_scheduler import FrameScheduler
from metrics import MetricsCollector

log = logging.getLogger("edge_pipeline")


class DetectorLoop(threading.Thread):
    """The one and only YOLO loop. Pulls frames, infers, emits events."""

    def __init__(
        self,
        detector: Detector,
        scheduler: FrameScheduler,
        event_logger: EventLogger,
        metrics: "MetricsCollector",
        stop_event: threading.Event,
        active_backend: str = "unknown",
    ) -> None:
        super().__init__(name="DetectorLoop", daemon=True)
        self.detector = detector
        self.scheduler = scheduler
        self.event_logger = event_logger
        self.metrics = metrics
        self.stop_event = stop_event
        self.active_backend = active_backend

        self.frames_processed = 0
        self.errors = 0
        self._win_total_ms = 0.0
        self._win_count = 0
        self._win_max_ms = 0.0
        self._lock = threading.Lock()

    def run(self) -> None:
        try:
            log.info("warming up detector=%s", self.detector.name)
            self.detector.warmup()
        except Exception:
            log.exception("detector warmup failed; loop will not start")
            return
        log.info("detector loop started")

        while not self.stop_event.is_set():
            frame = self.scheduler.next_frame(self.stop_event)
            if frame is None:
                continue

            t0 = time.monotonic()
            try:
                dets = self.detector.infer(frame.image)
            except Exception:
                with self._lock:
                    self.errors += 1
                log.exception("inference failed for %s frame_index=%d",
                              frame.camera_id, frame.frame_index)
                continue
            t1 = time.monotonic()
            inference_ms = (t1 - t0) * 1000.0
            end_to_end_ms = (t1 - frame.captured_at) * 1000.0

            with self._lock:
                self.frames_processed += 1
                self._win_count += 1
                self._win_total_ms += inference_ms
                if inference_ms > self._win_max_ms:
                    self._win_max_ms = inference_ms

            self.metrics.record_latency_ms(end_to_end_ms)

            for d in dets:
                try:
                    self.event_logger.emit(
                        camera_id=frame.camera_id,
                        frame_seq=frame.frame_index,
                        capture_ts_mono=frame.captured_at,
                        inference_ms=inference_ms,
                        class_id=d.class_id,
                        class_name=d.class_name,
                        confidence=d.confidence,
                        bbox={"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                              "w": max(0.0, d.x2 - d.x1), "h": max(0.0, d.y2 - d.y1)},
                    )
                except Exception:
                    log.exception("event emit failed")

        try:
            self.detector.close()
        except Exception:
            log.exception("detector close failed")
        log.info("detector loop stopped")

    def snapshot(self) -> dict:
        with self._lock:
            avg = (self._win_total_ms / self._win_count) if self._win_count else 0.0
            mx = self._win_max_ms
            self._win_total_ms = 0.0
            self._win_count = 0
            self._win_max_ms = 0.0
            return {
                "frames_processed": self.frames_processed,
                "errors": self.errors,
                "inference_avg_ms_window": avg,
                "inference_max_ms_window": mx,
                "active_backend": self.active_backend,
            }


def build_detector(config: dict) -> Tuple[Detector, str]:
    """Load the TensorRT detector. Fall back to Fake if requested and TRT fails."""
    det_cfg = config.get("detector") or {}
    allow_fallback = bool(det_cfg.get("allow_fake_fallback", True))
    try:
        return load_detector("tensorrt", det_cfg), "tensorrt"
    except Exception:
        log.exception("tensorrt detector failed to load")
        if not allow_fallback:
            raise
        log.warning("FALLBACK: serving FakeDetector (allow_fake_fallback=true)")
        return load_detector("fake", config), "fake"


def build_runtime(config: dict):
    rtsp_cfg = config.get("rtsp", {})
    scheduler_cfg = config.get("scheduler", {})
    node_id = config.get("node_id", "jetson-unset")

    stop_event = threading.Event()

    # Filter to enabled cameras. Decoder + output size are global on the RTSP
    # block — every camera uses the same hardware decode path.
    decoder_pref = rtsp_cfg.get("decoder_preference", "nvv4l2decoder")
    decoder_fallback = rtsp_cfg.get("fallback_decoder", "omxh264dec")
    output_size = int(rtsp_cfg.get("output_size", 416))

    enabled = [c for c in config["cameras"] if c.get("enabled", True)]
    for c in config["cameras"]:
        if not c.get("enabled", True):
            log.info("camera %s disabled in config; skipping", c["camera_id"])
    if not enabled:
        raise SystemExit("no enabled cameras in config")

    buffers: List[LatestFrameBuffer] = []
    readers: List[CameraReader] = []
    target_fps_map = {}
    for cam in enabled:
        cid = cam["camera_id"]
        buf = LatestFrameBuffer(cid)
        target_fps_map[cid] = float(cam.get("target_fps", 1.0))
        readers.append(CameraReader(
            camera_id=cid,
            rtsp_url=cam["rtsp_url"],
            decoder_preference=decoder_pref,
            fallback_decoder=decoder_fallback,
            output_size=output_size,
            buffer=buf,
            stop_event=stop_event,
            latency_ms=int(rtsp_cfg.get("latency_ms", 0)),
            protocols=str(rtsp_cfg.get("protocols", "tcp")),
            drop_on_latency=bool(rtsp_cfg.get("drop_on_latency", True)),
            open_probe_timeout_s=float(rtsp_cfg.get("open_probe_timeout_s", 8.0)),
            reconnect_backoff_max_s=float(rtsp_cfg.get("reconnect_backoff_max_s", 8.0)),
        ))
        buffers.append(buf)

    scheduler = FrameScheduler(buffers, target_fps_map, **scheduler_cfg)

    detector, active_backend = build_detector(config)
    log.info("detector backend resolved: active=%s", active_backend)

    event_logger = EventLogger(
        jsonl_path=config.get("event_log_path", "logs/detections.jsonl"),
        node_id=node_id,
    )

    detector_loop = DetectorLoop(
        detector=detector, scheduler=scheduler,
        event_logger=event_logger, metrics=None,
        stop_event=stop_event, active_backend=active_backend,
    )
    metrics = MetricsCollector(
        node_id=node_id,
        interval_s=float(config.get("metrics_interval_sec", 5)),
        jsonl_path=config.get("metrics_log_path", "logs/runtime_metrics.jsonl"),
        readers=readers,
        buffers=buffers,
        scheduler=scheduler,
        detector_loop=detector_loop,
        event_logger=event_logger,
        stop_event=stop_event,
    )
    detector_loop.metrics = metrics

    return stop_event, readers, scheduler, detector_loop, metrics, event_logger


def run_forever(config: dict) -> int:
    stop_event, readers, _, detector_loop, metrics, event_logger = build_runtime(config)

    log.info("starting %d camera readers", len(readers))
    for r in readers:
        r.start()
    metrics.start()
    detector_loop.start()

    def _sig(_s, _f):
        log.info("signal received; stopping")
        stop_event.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        while not stop_event.is_set():
            stop_event.wait(1.0)
    finally:
        stop_event.set()
        # Readers are daemon threads possibly blocked inside cv2.VideoCapture;
        # short timeout, interpreter collects laggards on exit.
        for r in readers:
            r.join(timeout=1.0)
        detector_loop.join(timeout=10)
        metrics.join(timeout=5)
        event_logger.close()
        log.info("shutdown complete")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logging.basicConfig(
        level=config.get("logging", {}).get("level", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    return run_forever(config)


if __name__ == "__main__":
    sys.exit(main())
