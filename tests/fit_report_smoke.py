"""Tiny smoke for tools/fit_report.py.

Builds a synthetic 5-window runtime_metrics.jsonl describing a healthy run,
runs fit_report.py against it, asserts the verdict is one of
{PASS, WARNING, FAIL} and that both report files were written.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_metrics_window(idx: int) -> dict:
    return {
        "ts": 1700000000 + idx * 5,
        "node_id": "test",
        "window_s": 5,
        "per_camera": {
            "cam-01": {"processed_fps": 1.0, "latest_frame_age_ms": 100.0,
                       "reconnects_window": 0},
            "cam-02": {"processed_fps": 1.0, "latest_frame_age_ms": 120.0,
                       "reconnects_window": 0},
        },
        "detector": {
            "processed_fps_total": 2.0,
            "inference_avg_ms": 35.0,
            "inference_max_ms_window": 50.0,
            "errors_window": 0,
            "active_backend": "tensorrt",
        },
        "end_to_end_latency_ms": {"p50_ms": 120.0, "p95_ms": 200.0, "p99_ms": 250.0},
        "memory_mb": 100.0 + idx * 0.2,
        "starvation_alerts_alltime": {"cam-01": 0, "cam-02": 0},
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        metrics_path = td / "runtime_metrics.jsonl"
        detections_path = td / "detections.jsonl"
        out_dir = td / "reports"

        with open(metrics_path, "w") as f:
            for i in range(10):
                f.write(json.dumps(make_metrics_window(i)) + "\n")
        detections_path.write_text("")  # empty is fine

        cmd = [
            sys.executable, str(ROOT / "tools" / "fit_report.py"),
            "--metrics", str(metrics_path),
            "--detections", str(detections_path),
            "--out", str(out_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("stdout:", result.stdout)
        if result.stderr:
            print("stderr:", result.stderr)

        md = out_dir / "fit_report.md"
        js = out_dir / "fit_report.json"
        if not md.exists() or not js.exists():
            print("FAIL: report files not written")
            return 1
        payload = json.loads(js.read_text())
        verdict = payload.get("verdict")
        if verdict not in ("PASS", "WARNING", "FAIL"):
            print(f"FAIL: invalid verdict {verdict!r}")
            return 1
        # The synthetic input was healthy → expect PASS.
        if verdict != "PASS":
            print(f"FAIL: expected PASS for healthy synthetic input, got {verdict}")
            print("reasons:", payload.get("reasons"))
            return 1
        print(f"OK verdict={verdict}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
