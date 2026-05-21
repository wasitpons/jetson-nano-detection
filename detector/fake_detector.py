"""Fake detector for smoke tests and demo safety.

Sleeps a configurable Jetson-ish amount and returns 0–2 random vehicle bboxes
so the postprocess/event/metrics chain can be exercised without CUDA, an
engine, or RTSP.
"""

from __future__ import annotations

import random
import time
from typing import List

import numpy as np

from .base import Detection, Detector

_VEHICLE_CLASSES = [
    (2, "car"),
    (3, "motorcycle"),
    (5, "bus"),
    (7, "truck"),
]


class FakeDetector(Detector):
    name = "fake"

    def __init__(self, config: dict) -> None:
        self.min_latency_s = float(config.get("min_latency_ms", 30)) / 1000.0
        self.max_latency_s = float(config.get("max_latency_ms", 60)) / 1000.0
        self.max_dets = int(config.get("max_dets_per_frame", 2))
        self._rng = random.Random(0xC0FFEE)

    def warmup(self) -> None:
        time.sleep(self.min_latency_s)

    def infer(self, image: np.ndarray) -> List[Detection]:
        time.sleep(self._rng.uniform(self.min_latency_s, self.max_latency_s))
        h, w = image.shape[:2]
        n = self._rng.randint(0, self.max_dets)
        out: List[Detection] = []
        for _ in range(n):
            cid, cname = self._rng.choice(_VEHICLE_CLASSES)
            bw = self._rng.uniform(0.1, 0.3) * w
            bh = self._rng.uniform(0.1, 0.3) * h
            x1 = self._rng.uniform(0, w - bw)
            y1 = self._rng.uniform(0, h - bh)
            out.append(Detection(
                class_id=cid,
                class_name=cname,
                confidence=round(self._rng.uniform(0.55, 0.95), 3),
                x1=x1, y1=y1, x2=x1 + bw, y2=y1 + bh,
            ))
        return out
