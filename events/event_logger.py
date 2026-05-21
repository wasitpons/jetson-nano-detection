"""JSONL detection-event sink.

One JSON per line, line-buffered. Production wants this shipped to a queue;
for Task 01 a local file is enough. fsync is intentionally off — we'd rather
lose the tail on a power cut than stall the inference loop on every write.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)


class EventLogger:
    def __init__(self, jsonl_path: str, node_id: str, *, stdout: bool = False) -> None:
        self.jsonl_path = jsonl_path
        self.node_id = node_id
        self.stdout = stdout
        os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)) or ".", exist_ok=True)
        self._fh = open(jsonl_path, "a", buffering=1)
        self._lock = threading.Lock()
        self.events_written = 0

    def emit(
        self,
        *,
        camera_id: str,
        frame_seq: int,
        capture_ts_mono: float,
        inference_ms: float,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: dict,
    ) -> None:
        event = {
            "ts": time.time(),
            "node_id": self.node_id,
            "camera_id": camera_id,
            "frame_seq": frame_seq,
            "capture_ts_mono": capture_ts_mono,
            "inference_ms": inference_ms,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox,
        }
        line = json.dumps(event, separators=(",", ":"))
        with self._lock:
            self.events_written += 1
            try:
                self._fh.write(line + "\n")
            except Exception:
                log.exception("event write failed")
            if self.stdout:
                print(line, flush=True)

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                log.exception("close event log failed")

    def snapshot(self) -> dict:
        return {"events_written": self.events_written}
