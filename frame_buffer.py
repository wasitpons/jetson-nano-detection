"""Single-slot, drop-old frame buffer.

The load-shedding boundary between fast hardware-decoded RTSP readers and the
slower YOLO inference loop. Producers always succeed; consumers always get
the freshest frame. There is no queue: a new put() displaces whatever was in
the slot, and the displaced frame is counted as a drop. Counting drops at
this boundary is the proof that the no-backlog invariant holds.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    camera_id: str
    image: np.ndarray
    frame_index: int       # monotonically increasing per camera
    captured_at: float     # monotonic time the reader received the frame from upstream
    updated_at: float      # monotonic time put() last refreshed the slot


class LatestFrameBuffer:
    """One slot per camera. put() never blocks; get_latest() returns newest or None."""

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._slot: Optional[Frame] = None
        self._index = 0
        self._produced = 0
        self._dropped = 0
        self._last_put_ts = 0.0

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def put(self, image: np.ndarray, captured_at: Optional[float] = None) -> Frame:
        """Atomically replace the slot. Counts the displaced frame as a drop."""
        now = time.monotonic()
        cap_ts = captured_at if captured_at is not None else now
        with self._cond:
            if self._slot is not None:
                self._dropped += 1
            self._index += 1
            frame = Frame(
                camera_id=self._camera_id,
                image=image,
                frame_index=self._index,
                captured_at=cap_ts,
                updated_at=now,
            )
            self._slot = frame
            self._produced += 1
            self._last_put_ts = now
            self._cond.notify()
            return frame

    def get_latest(self, consume: bool = True) -> Optional[Frame]:
        with self._cond:
            frame = self._slot
            if consume:
                self._slot = None
            return frame

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "camera_id": self._camera_id,
                "produced": self._produced,
                "dropped": self._dropped,
                "has_pending": self._slot is not None,
                "last_put_ts": self._last_put_ts,
                "current_frame_index": self._index,
            }
