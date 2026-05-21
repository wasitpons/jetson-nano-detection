"""End-to-end smoke test for jetson_runtime.

Runs the full pipeline with the fake detector against five synthetic
cameras (one intentionally broken/stalled). Validates:

  - drop-old buffering            (fastest camera shows dropped > 0)
  - fair scheduling                (every healthy camera served)
  - broken-camera isolation        (broken cam doesn't block others)
  - events + metrics output        (jsonl files written, contain expected keys)
  - memory + latest_frame_age      (per-spec metrics present)

No cv2 / RTSP / onnxruntime / model needed.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frame_buffer import LatestFrameBuffer
from frame_scheduler import FrameScheduler
from detector.fake_detector import FakeDetector
from events.event_logger import EventLogger
from metrics.metrics_collector import MetricsCollector
from edge_pipeline import DetectorLoop


class FakeReaderThread(threading.Thread):
    """Stand-in for CameraReader — produces zeros at a chosen FPS, no cv2."""

    def __init__(self, camera_id, buffer, fps, stop_event, *, broken=False):
        super().__init__(name=f"FakeReader[{camera_id}]", daemon=True)
        self.camera_id = camera_id
        self.buffer = buffer
        self.interval = 1.0 / fps if fps > 0 else 1e9
        self.stop_event = stop_event
        self.broken = broken
        self.frames_read = 0
        self.read_failures = 0
        self.reconnects = 1 if not broken else 0
        self.last_frame_at = 0.0
        self._lock = threading.Lock()

    def run(self):
        if self.broken:
            # Simulate a camera that never connects: spin counting failures.
            while not self.stop_event.wait(0.5):
                with self._lock:
                    self.read_failures += 1
            return
        while not self.stop_event.wait(self.interval):
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            now = time.monotonic()
            self.buffer.put(img, captured_at=now)
            with self._lock:
                self.frames_read += 1
                self.last_frame_at = now

    def snapshot(self):
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "frames_read": self.frames_read,
                "read_failures": self.read_failures,
                "reconnects": self.reconnects,
                "last_frame_at": self.last_frame_at,
            }


def run(duration_s: float = 5.0) -> int:
    stop = threading.Event()
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    events_path = log_dir / "smoke_events.jsonl"
    metrics_path = log_dir / "smoke_metrics.jsonl"
    # Start clean.
    for p in (events_path, metrics_path):
        if p.exists():
            p.unlink()

    cam_ids = [f"cam-{i:02d}" for i in range(1, 6)]
    # cam-01 super hot (10 fps), cam-02..04 normal (2 fps), cam-05 broken.
    fps_plan = [10.0, 2.0, 2.0, 2.0, 0.0]
    broken = [False, False, False, False, True]
    target_fps = {cid: 1.0 for cid in cam_ids}

    buffers = [LatestFrameBuffer(cid) for cid in cam_ids]
    readers = [
        FakeReaderThread(cid, buf, fps=fps, stop_event=stop, broken=br)
        for cid, buf, fps, br in zip(cam_ids, buffers, fps_plan, broken)
    ]

    scheduler = FrameScheduler(
        buffers, target_fps,
        starvation_grace_ms=3000, stale_frame_factor=2.0,
    )
    detector = FakeDetector({"min_latency_ms": 20, "max_latency_ms": 40,
                             "max_dets_per_frame": 2})
    event_logger = EventLogger(jsonl_path=str(events_path), node_id="smoke-node")

    detector_loop = DetectorLoop(
        detector=detector, scheduler=scheduler,
        event_logger=event_logger, metrics=None, stop_event=stop,
    )
    metrics = MetricsCollector(
        node_id="smoke-node", interval_s=2.0, jsonl_path=str(metrics_path),
        readers=readers, buffers=buffers, scheduler=scheduler,
        detector_loop=detector_loop, event_logger=event_logger, stop_event=stop,
    )
    detector_loop.metrics = metrics

    for r in readers:
        r.start()
    metrics.start()
    detector_loop.start()

    time.sleep(duration_s)
    stop.set()
    for r in readers:
        r.join(timeout=3)
    detector_loop.join(timeout=5)
    metrics.join(timeout=3)
    event_logger.close()

    failures = []

    served = scheduler.snapshot()["served_per_camera"]
    print("served:", served)
    # Every healthy camera should be served at least once.
    for cid, br in zip(cam_ids, broken):
        if br:
            continue
        if served.get(cid, 0) == 0:
            failures.append(f"healthy camera {cid} was never served (broken cam blocked others)")

    # Drop-old: hottest camera must have visible drops.
    hot_snap = buffers[0].snapshot()
    if hot_snap["dropped"] == 0:
        failures.append("hot camera shows zero drops — drop-old buffer not working")

    # Broken camera did not produce events.
    if served.get(cam_ids[4], 0) != 0:
        failures.append("broken camera somehow got served — fake reader bug?")

    # Events file populated.
    if not events_path.exists() or events_path.stat().st_size == 0:
        failures.append("events.jsonl is empty")

    # Metrics file populated and shape is sane.
    if not metrics_path.exists():
        failures.append("metrics.jsonl missing")
    else:
        lines = metrics_path.read_text().splitlines()
        if not lines:
            failures.append("metrics.jsonl empty")
        else:
            sample = json.loads(lines[-1])
            for k in ("memory_mb", "per_camera", "detector", "end_to_end_latency_ms"):
                if k not in sample:
                    failures.append(f"metric line missing key {k}")
            # latest_frame_age_ms must be present for healthy cams.
            for cid, br in zip(cam_ids, broken):
                if br:
                    continue
                pc = sample["per_camera"].get(cid, {})
                if "latest_frame_age_ms" not in pc:
                    failures.append(f"per_camera.{cid} missing latest_frame_age_ms")
            if sample["detector"]["inference_avg_ms"] <= 0:
                failures.append("detector.inference_avg_ms <= 0")
            print("last metric line:")
            print(json.dumps(sample, indent=2, default=str))

    print("detector_loop:", detector_loop.snapshot())
    print("events_written:", event_logger.snapshot()["events_written"])

    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        return 1

    # Tidy up smoke artifacts so they don't pollute real runs.
    for p in (events_path, metrics_path):
        p.unlink(missing_ok=True)
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(run())
