"""Periodic runtime metrics.

Owns no source-of-truth state — pulls snapshots from each component and
computes window deltas. Keeps hot paths lock-free.

Metrics that actually help when you're chasing Jetson bottlenecks:

  processed_fps_total                : total throughput vs target (e.g. 5.0)
  detector.inference_avg_ms          : GPU saturation / model too heavy
  per_camera.latest_frame_age_ms     : reader alive? RTSP stalled?
  per_camera.read_failures_window    : flaky network / wrong RTSP transport
  per_camera.buffer_dropped_window   : reader > scheduler — this is healthy
  scheduler.stale_skipped            : scheduler overloaded, throwing frames
  scheduler.starvation_alerts        : one camera crowding out others
  end_to_end_latency_ms.p95          : capture → emit, total user-visible delay
  memory_mb                          : memory growth / leak
"""

from __future__ import annotations

import json
import logging
import os
import resource
import sys
import threading
import time
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def _read_memory_mb() -> tuple[float, str]:
    """Return (memory_mb, source). Prefer psutil; fall back to resource."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024), "psutil_rss"
    except ImportError:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is KB on Linux, bytes on macOS — make it MB either way.
        divisor = 1024 if sys.platform == "darwin" else 1
        return (ru / 1024) / divisor, "ru_maxrss"


class MetricsCollector(threading.Thread):
    def __init__(
        self,
        *,
        node_id: str,
        interval_s: float,
        jsonl_path: Optional[str],
        readers: list,
        buffers: list,
        scheduler,
        detector_loop,
        event_logger,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="MetricsCollector", daemon=True)
        self.node_id = node_id
        self.interval_s = interval_s
        self.jsonl_path = jsonl_path
        self.readers = readers
        self.buffers = buffers
        self.scheduler = scheduler
        self.detector_loop = detector_loop
        self.event_logger = event_logger
        self.stop_event = stop_event

        self._latencies_ms: List[float] = []
        self._lat_lock = threading.Lock()
        self._prev: Dict[str, dict] = {}

        self._fh = None
        if jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)) or ".", exist_ok=True)
            self._fh = open(jsonl_path, "a", buffering=1)

    def record_latency_ms(self, ms: float) -> None:
        with self._lat_lock:
            self._latencies_ms.append(ms)
            if len(self._latencies_ms) > 10_000:
                self._latencies_ms = self._latencies_ms[-5_000:]

    def _drain_latencies(self) -> List[float]:
        with self._lat_lock:
            out = self._latencies_ms
            self._latencies_ms = []
        return out

    @staticmethod
    def _percentile(sorted_vals: List[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        k = max(0, min(len(sorted_vals) - 1,
                       int(round((p / 100.0) * (len(sorted_vals) - 1)))))
        return sorted_vals[k]

    def _delta(self, key: str, snap: dict, fields: list[str]) -> dict:
        prev = self._prev.get(key, {})
        out = {f: snap.get(f, 0) - prev.get(f, 0) for f in fields}
        self._prev[key] = snap
        return out

    def collect(self) -> dict:
        now_wall = time.time()
        now_mono = time.monotonic()
        window = self.interval_s

        per_cam: Dict[str, dict] = {}

        for r in self.readers:
            s = r.snapshot()
            d = self._delta(f"r:{s['camera_id']}", s,
                            ["frames_read", "read_failures", "reconnect_count"])
            age_ms = ((now_mono - s["last_frame_at"]) * 1000.0
                      if s["last_frame_at"] > 0 else None)
            per_cam.setdefault(s["camera_id"], {}).update({
                "read_fps": d["frames_read"] / window,
                "read_failures_window": d["read_failures"],
                "reconnects_window": d["reconnect_count"],
                "latest_frame_age_ms": age_ms,
                # Surfaced for the TUI watcher (already in r.snapshot()).
                "active_decoder": s.get("active_decoder"),
                "frame_width": s.get("frame_width", 0),
                "frame_height": s.get("frame_height", 0),
            })

        for b in self.buffers:
            s = b.snapshot()
            d = self._delta(f"b:{s['camera_id']}", s, ["produced", "dropped"])
            per_cam.setdefault(s["camera_id"], {}).update({
                "buffer_produced_window": d["produced"],
                "buffer_dropped_window": d["dropped"],
                "buffer_pending": s["has_pending"],
            })

        sched = self.scheduler.snapshot()
        prev_served = self._prev.get("sched", {}).get("served_per_camera", {})
        prev_stale = self._prev.get("sched", {}).get("stale_skipped", {})
        served_delta = {cid: sched["served_per_camera"].get(cid, 0) - prev_served.get(cid, 0)
                        for cid in sched["served_per_camera"]}
        stale_delta = {cid: sched["stale_skipped"].get(cid, 0) - prev_stale.get(cid, 0)
                       for cid in sched["stale_skipped"]}
        self._prev["sched"] = {
            "served_per_camera": dict(sched["served_per_camera"]),
            "stale_skipped": dict(sched["stale_skipped"]),
        }
        for cid, n in served_delta.items():
            per_cam.setdefault(cid, {})["processed_fps"] = n / window
        total_processed = sum(served_delta.values())

        det = self.detector_loop.snapshot()
        det_delta = self._delta("det", det, ["frames_processed", "errors"])
        detector_view = {
            "frames_processed_window": det_delta["frames_processed"],
            "processed_fps_total": total_processed / window,
            "inference_avg_ms": det["inference_avg_ms_window"],
            "inference_max_ms_window": det["inference_max_ms_window"],
            "errors_window": det_delta["errors"],
            "stale_skipped_window": sum(stale_delta.values()),
            # active_backend == "fake" while engine path was set => fallback used.
            "active_backend": det.get("active_backend", "unknown"),
        }
        # The DetectorLoop tracks per-window avg/max itself and resets each pull.

        lat = sorted(self._drain_latencies())
        latency_view = {
            "samples": len(lat),
            "p50_ms": self._percentile(lat, 50),
            "p95_ms": self._percentile(lat, 95),
            "p99_ms": self._percentile(lat, 99),
        }

        evt = self.event_logger.snapshot()
        evt_d = self._delta("evt", evt, ["events_written"])

        mem_mb, mem_src = _read_memory_mb()

        return {
            "ts": now_wall,
            "node_id": self.node_id,
            "window_s": window,
            "per_camera": per_cam,
            "detector": detector_view,
            "end_to_end_latency_ms": latency_view,
            "events_window": evt_d["events_written"],
            "memory_mb": round(mem_mb, 2),
            "memory_source": mem_src,
            "starvation_alerts_alltime": sched["starvation_alerts"],
        }

    def run(self) -> None:
        log.info("metrics collector started (interval=%.1fs)", self.interval_s)
        # Prime baselines so the first window doesn't read as a huge spike.
        for r in self.readers:
            self._prev[f"r:{r.camera_id}"] = r.snapshot()
        for b in self.buffers:
            self._prev[f"b:{b.camera_id}"] = b.snapshot()
        self._prev["det"] = self.detector_loop.snapshot()
        self._prev["evt"] = self.event_logger.snapshot()
        sched = self.scheduler.snapshot()
        self._prev["sched"] = {
            "served_per_camera": dict(sched["served_per_camera"]),
            "stale_skipped": dict(sched["stale_skipped"]),
        }

        while not self.stop_event.wait(self.interval_s):
            try:
                report = self.collect()
            except Exception:
                log.exception("metrics collect failed")
                continue
            line = json.dumps(report, separators=(",", ":"))
            log.info("METRICS %s", line)
            if self._fh is not None:
                try:
                    self._fh.write(line + "\n")
                except Exception:
                    log.exception("metrics write failed")

        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
        log.info("metrics collector stopped")
