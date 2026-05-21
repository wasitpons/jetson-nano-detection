"""Fair round-robin scheduler with per-camera FPS gating.

Each camera owns its `target_fps`. The scheduler:
  - Skips a camera until at least `1/target_fps` seconds have passed since it
    was last served — so a noisy camera can't crowd out the others.
  - Among cameras eligible AND with a pending frame, picks the one waited
    longest → fairness under uneven production.
  - Drops frames older than `stale_frame_factor / target_fps` seconds — they
    were going to be useless anyway and we shouldn't burn GPU on them.

Pull-based: the inference loop calls `next_frame()` itself. No extra thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from frame_buffer import Frame, LatestFrameBuffer

log = logging.getLogger(__name__)


class FrameScheduler:
    def __init__(
        self,
        buffers: List[LatestFrameBuffer],
        target_fps_per_camera: Dict[str, float],
        *,
        starvation_grace_ms: int = 3000,
        stale_frame_factor: float = 2.0,
        tick_resolution_s: float = 0.05,
    ) -> None:
        self._buffers: Dict[str, LatestFrameBuffer] = {b.camera_id: b for b in buffers}
        self._target_interval: Dict[str, float] = {
            cid: 1.0 / max(target_fps_per_camera[cid], 0.1)
            for cid in self._buffers
        }
        self._stale_age: Dict[str, float] = {
            cid: stale_frame_factor / max(target_fps_per_camera[cid], 0.1)
            for cid in self._buffers
        }
        self._starvation_grace = starvation_grace_ms / 1000.0
        self._tick = tick_resolution_s

        self._last_served: Dict[str, float] = {cid: 0.0 for cid in self._buffers}
        self._served: Dict[str, int] = {cid: 0 for cid in self._buffers}
        self._stale_skipped: Dict[str, int] = {cid: 0 for cid in self._buffers}
        self._starvation_alerts: Dict[str, int] = {cid: 0 for cid in self._buffers}
        self._lock = threading.Lock()

    def next_frame(self, stop_event: threading.Event) -> Optional[Frame]:
        """Block until a fresh, fair, non-stale frame is ready, then return it."""
        while not stop_event.is_set():
            now = time.monotonic()
            best: Optional[tuple[float, str]] = None    # (wait_time, camera_id)

            for cid, buf in self._buffers.items():
                # Per-camera FPS gate.
                if now - self._last_served[cid] < self._target_interval[cid]:
                    continue
                snap = buf.snapshot()
                if not snap["has_pending"]:
                    continue
                wait = now - self._last_served[cid]
                if best is None or wait > best[0]:
                    best = (wait, cid)

            if best is None:
                # Nothing eligible — log starvation hints, then sleep one tick.
                for cid in self._buffers:
                    if now - self._last_served[cid] > self._starvation_grace:
                        # Don't spam — only count, the metrics layer will surface it.
                        with self._lock:
                            self._starvation_alerts[cid] += 1
                if stop_event.wait(self._tick):
                    return None
                continue

            cid = best[1]
            frame = self._buffers[cid].get_latest(consume=True)
            if frame is None:
                continue  # race: producer consumed by another path

            # Stale-frame check — if we held this frame too long, drop it.
            if (time.monotonic() - frame.captured_at) > self._stale_age[cid]:
                with self._lock:
                    self._stale_skipped[cid] += 1
                log.debug("[%s] skipping stale frame_index=%d", cid, frame.frame_index)
                # Don't update _last_served — pretend we never served this one.
                continue

            with self._lock:
                self._last_served[cid] = time.monotonic()
                self._served[cid] += 1
            return frame

        return None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "served_per_camera": dict(self._served),
                "last_served_per_camera": dict(self._last_served),
                "stale_skipped": dict(self._stale_skipped),
                "starvation_alerts": dict(self._starvation_alerts),
            }
