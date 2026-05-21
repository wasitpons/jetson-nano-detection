#!/usr/bin/env python3
"""Live TUI watcher for jetson_runtime.

Tails `logs/runtime_metrics.jsonl` + `logs/detections.jsonl` and renders a
3-section dashboard:

  1. System line   — backend, total fps, inference avg, memory, e2e p95,
                     plus a staleness badge if METRICS stops arriving
                     (most common silent-failure signal: runtime hung)
  2. Per-camera    — decoder, fps, frame age, W×H, drops, fails, reconnects
                     colour-coded green/yellow/red against simple thresholds
  3. Recent events — last 10 detections from detections.jsonl

The runtime stays headless. This tool just consumes the JSONL it already
writes — kill it with Ctrl+C and it doesn't touch the runtime.

Usage:
  python3 tools/watch.py
  python3 tools/watch.py --metrics PATH --detections PATH
  python3 tools/watch.py --once     # render one frame then exit (CI / screenshots)
"""

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class JsonlTailer(threading.Thread):
    """Tail a JSONL file, calling `on_line(json_obj)` for each new line.

    Stdlib-only polling tailer so we work on Python 3.6.9 (Jetson) without
    inotify. If the file doesn't exist yet it waits; if it gets truncated
    we re-seek to start.
    """

    def __init__(self, path: Path, on_line, stop_event: threading.Event,
                 poll_s: float = 0.2):
        super().__init__(daemon=True, name=f"Tailer[{path.name}]")
        self.path = path
        self.on_line = on_line
        self.stop_event = stop_event
        self.poll_s = poll_s

    def run(self) -> None:
        fh = None
        try:
            while not self.stop_event.is_set():
                if fh is None:
                    if not self.path.exists():
                        self.stop_event.wait(self.poll_s)
                        continue
                    fh = open(self.path)
                    fh.seek(0, 2)   # end of file — only watch new lines
                line = fh.readline()
                if not line:
                    # Truncation check: stat the file; if size < pos, reopen.
                    try:
                        if self.path.stat().st_size < fh.tell():
                            fh.close()
                            fh = None
                            continue
                    except FileNotFoundError:
                        fh.close()
                        fh = None
                        continue
                    self.stop_event.wait(self.poll_s)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    self.on_line(obj)
                except Exception:
                    # A bad consumer must never kill the tailer.
                    pass
        finally:
            if fh is not None:
                fh.close()


class WatcherState:
    """Latest snapshot + rolling tail of detection events. Threadsafe."""

    def __init__(self, events_max: int = 20):
        self._lock = threading.Lock()
        self.latest_metrics: Optional[dict] = None
        self.metrics_wall_ts: float = 0.0
        self.events: Deque[dict] = deque(maxlen=events_max)

    def push_metrics(self, obj: dict) -> None:
        with self._lock:
            self.latest_metrics = obj
            self.metrics_wall_ts = time.time()

    def push_event(self, obj: dict) -> None:
        with self._lock:
            self.events.append(obj)

    def snapshot(self):
        with self._lock:
            return self.latest_metrics, self.metrics_wall_ts, list(self.events)


# ---- rendering ----------------------------------------------------------------------------

def _fmt_age(age_ms: Optional[float]) -> str:
    if age_ms is None:
        return "n/a"
    if age_ms < 1000:
        return f"{age_ms:>5.0f}ms"
    return f"{age_ms / 1000:>5.2f}s"


def _fps_colour(fps: float, target: float = 1.0) -> str:
    if fps >= 0.8 * target:
        return "green"
    if fps >= 0.5 * target:
        return "yellow"
    return "red"


def _age_colour(age_ms: Optional[float]) -> str:
    if age_ms is None:
        return "red"
    if age_ms < 500:
        return "green"
    if age_ms < 1500:
        return "yellow"
    return "red"


def render_system(metrics: Optional[dict], metrics_wall_ts: float) -> Panel:
    if metrics is None:
        return Panel(Text("waiting for first metrics line…", style="dim"),
                     title="system", border_style="dim")

    det = metrics.get("detector") or {}
    e2e = metrics.get("end_to_end_latency_ms") or {}
    window = float(metrics.get("window_s") or 5)
    age = time.time() - metrics_wall_ts if metrics_wall_ts else 0
    stale = age > 2 * window

    backend = det.get("active_backend") or "unknown"
    backend_colour = "red bold" if backend == "fake" else "green"

    total_fps = det.get("processed_fps_total") or 0
    inf_avg = det.get("inference_avg_ms") or 0
    mem = metrics.get("memory_mb") or 0
    e2e_p95 = e2e.get("p95_ms") or 0
    errors = det.get("errors_window") or 0
    stale_skips = det.get("stale_skipped_window") or 0
    starv = sum((metrics.get("starvation_alerts_alltime") or {}).values())

    txt = Text()
    txt.append(f"backend=", style="dim")
    txt.append(f"{backend}", style=backend_colour)
    if backend == "fake":
        txt.append("  (FALLBACK — engine failed to load)", style="red")
    txt.append(f"   total_fps=", style="dim")
    txt.append(f"{total_fps:.2f}", style="bold")
    txt.append(f"   inf_avg=", style="dim")
    txt.append(f"{inf_avg:.0f}ms")
    txt.append(f"   mem=", style="dim")
    txt.append(f"{mem:.0f}MB")
    txt.append(f"   e2e p95=", style="dim")
    txt.append(f"{e2e_p95:.0f}ms")
    txt.append("\n")
    txt.append(f"errors={errors}  stale_skips={stale_skips}  starvation_total={starv}",
               style="dim")
    txt.append(f"   metrics_age={age:.1f}s", style="red bold" if stale else "dim")
    if stale:
        txt.append("   ⚠ STALE — runtime may be hung", style="red bold")

    return Panel(txt, title="system", border_style="red" if stale else "cyan")


def render_cameras(metrics: Optional[dict]) -> Panel:
    table = Table(expand=True, show_edge=False, padding=(0, 1))
    table.add_column("Camera", style="bold")
    table.add_column("Decoder")
    table.add_column("FPS", justify="right")
    table.add_column("Read ms", justify="right")
    table.add_column("Age", justify="right")
    table.add_column("W×H")
    table.add_column("Produced", justify="right")
    table.add_column("Dropped", justify="right")
    table.add_column("Fails", justify="right")
    table.add_column("Reconn", justify="right")

    if metrics is None:
        table.add_row("(no data yet)", "", "", "", "", "", "", "", "", "",
                      style="dim")
        return Panel(table, title="cameras", border_style="dim")

    per_cam = metrics.get("per_camera") or {}
    if not per_cam:
        table.add_row("(no cameras reporting)", "", "", "", "", "", "", "", "", "",
                      style="dim")
        return Panel(table, title="cameras", border_style="dim")

    for cid in sorted(per_cam.keys()):
        v = per_cam[cid] or {}
        fps = float(v.get("processed_fps") or 0)
        age_ms = v.get("latest_frame_age_ms")
        decoder = v.get("active_decoder") or "(none)"
        w = int(v.get("frame_width") or 0)
        h = int(v.get("frame_height") or 0)
        produced = int(v.get("buffer_produced_window") or 0)
        dropped = int(v.get("buffer_dropped_window") or 0)
        fails = int(v.get("read_failures_window") or 0)
        reconn = int(v.get("reconnects_window") or 0)
        read_avg = float(v.get("read_avg_ms") or 0)
        read_max = float(v.get("read_max_ms") or 0)

        wxh = f"{w}×{h}" if w and h else "—"
        dec_style = "green" if decoder and decoder != "(none)" else "red"
        fail_style = "red" if fails > 0 else "dim"
        read_str = f"{read_avg:>4.1f}/{read_max:.0f}" if read_avg > 0 else "—"

        table.add_row(
            cid,
            Text(decoder, style=dec_style),
            Text(f"{fps:>4.2f}", style=_fps_colour(fps)),
            read_str,
            Text(_fmt_age(age_ms), style=_age_colour(age_ms)),
            wxh,
            f"{produced}",
            f"{dropped}",
            Text(f"{fails}", style=fail_style),
            f"{reconn}",
        )

    return Panel(table, title="cameras", border_style="cyan")


def render_events(events) -> Panel:
    table = Table(expand=True, show_edge=False, padding=(0, 1))
    table.add_column("Time", style="dim")
    table.add_column("Camera", style="bold")
    table.add_column("Class")
    table.add_column("Conf", justify="right")
    table.add_column("BBox WxH")
    if not events:
        table.add_row("(no detections yet)", "", "", "", "", style="dim")
        return Panel(table, title="recent detections", border_style="dim")
    # Newest first.
    for ev in reversed(events[-10:]):
        ts = ev.get("ts")
        when = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "—"
        cls = ev.get("class_name") or "?"
        conf = ev.get("confidence") or 0
        bbox = ev.get("bbox") or {}
        w = bbox.get("w") or 0
        h = bbox.get("h") or 0
        table.add_row(
            when,
            ev.get("camera_id") or "—",
            cls,
            f"{conf:.2f}",
            f"{int(w)}×{int(h)}",
        )
    return Panel(table, title="recent detections", border_style="cyan")


def build_layout(state: WatcherState) -> Layout:
    metrics, ts, events = state.snapshot()
    layout = Layout(name="root")
    layout.split_column(
        Layout(render_system(metrics, ts), name="sys", size=5),
        Layout(render_cameras(metrics), name="cams"),
        Layout(render_events(events), name="events", size=14),
    )
    return layout


def main() -> int:
    parser = argparse.ArgumentParser(description="live TUI for jetson_runtime")
    parser.add_argument("--metrics",
                        default=str(PROJECT_ROOT / "logs" / "runtime_metrics.jsonl"))
    parser.add_argument("--detections",
                        default=str(PROJECT_ROOT / "logs" / "detections.jsonl"))
    parser.add_argument("--once", action="store_true",
                        help="render one frame and exit (CI / screenshots)")
    parser.add_argument("--refresh-hz", type=float, default=2.0)
    args = parser.parse_args()

    state = WatcherState()
    stop_event = threading.Event()

    # Prime state by reading whatever's already in the files (last 100 lines).
    for path, push, take_last in (
        (Path(args.metrics), state.push_metrics, True),
        (Path(args.detections), state.push_event, False),
    ):
        if path.exists():
            try:
                with open(path) as f:
                    lines = f.readlines()
                if take_last:
                    lines = lines[-1:]
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        push(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if take_last and state.metrics_wall_ts == 0:
                    # We primed off existing file — pretend it just arrived so
                    # the staleness clock doesn't immediately fire.
                    state.metrics_wall_ts = time.time()
            except OSError:
                pass

    # Start tailers (background).
    t_metrics = JsonlTailer(Path(args.metrics), state.push_metrics, stop_event)
    t_events = JsonlTailer(Path(args.detections), state.push_event, stop_event)
    t_metrics.start()
    t_events.start()

    console = Console()
    if args.once:
        console.print(build_layout(state))
        stop_event.set()
        return 0

    try:
        with Live(build_layout(state), console=console,
                  refresh_per_second=args.refresh_hz, screen=False) as live:
            while not stop_event.is_set():
                stop_event.wait(1.0 / max(args.refresh_hz, 0.5))
                live.update(build_layout(state))
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        t_metrics.join(timeout=1.0)
        t_events.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
