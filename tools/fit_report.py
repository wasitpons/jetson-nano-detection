#!/usr/bin/env python3
"""Grade a benchmark run as PASS / WARNING / FAIL.

Reads runtime_metrics.jsonl + detections.jsonl, computes per-camera +
system aggregates, then applies the acceptance thresholds from the task
spec. Writes both a machine-readable JSON and a human-readable Markdown
report.

Usage:
  python3 tools/fit_report.py \
    --metrics logs/runtime_metrics.jsonl \
    --detections logs/detections.jsonl \
    --out reports/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional


# --- Thresholds (from task spec) -----------------------------------------------------------

PASS_MIN_CAM_FPS = 0.8
WARN_MIN_CAM_FPS = 0.5
PASS_MAX_E2E_P95_MS = 1500
WARN_MAX_E2E_P95_MS = 3000
TRT_ERROR_WINDOWS_FAIL = 3
MEM_GROWTH_FAIL_MB_PER_WINDOW = 1.0   # sustained positive slope


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def linreg_slope(xs: List[float], ys: List[float]) -> float:
    """Simple least-squares slope; returns 0 on degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def grade(metrics: List[dict], detections: List[dict]) -> dict:
    if not metrics:
        return {
            "verdict": "FAIL",
            "reasons": ["no runtime_metrics.jsonl rows — runtime never produced metrics"],
            "criteria": {}, "per_camera": {}, "system": {}, "summary": {},
        }

    # ---- Per-camera aggregates ----
    cam_ids = sorted({c for m in metrics for c in m.get("per_camera", {}).keys()})
    per_cam: Dict[str, dict] = {}
    for cid in cam_ids:
        fps_vals = []
        age_vals = []
        reconnects_window = 0
        for m in metrics:
            v = m.get("per_camera", {}).get(cid)
            if not v:
                continue
            if v.get("processed_fps") is not None:
                fps_vals.append(float(v["processed_fps"]))
            if v.get("latest_frame_age_ms") is not None:
                age_vals.append(float(v["latest_frame_age_ms"]))
            reconnects_window += int(v.get("reconnects_window", 0) or 0)
        mean_fps = statistics.fmean(fps_vals) if fps_vals else 0.0
        # "current" age = mean of last 5 windows.
        recent_ages = age_vals[-5:] if age_vals else []
        recent_age_mean = statistics.fmean(recent_ages) if recent_ages else None
        # Trend: is latest_frame_age_ms continuously growing?
        age_slope = linreg_slope(list(range(len(age_vals))), age_vals) if len(age_vals) >= 5 else 0.0
        per_cam[cid] = {
            "mean_processed_fps": round(mean_fps, 3),
            "recent_latest_frame_age_ms": (round(recent_age_mean, 1)
                                           if recent_age_mean is not None else None),
            "age_ms_slope_per_window": round(age_slope, 3),
            "reconnects_total_in_run": reconnects_window,
        }

    # ---- Detector + system aggregates ----
    inf_avg_window = []
    inf_max_window = []
    total_fps_vals = []
    e2e_p95_vals = []
    starvation_alltime_last: Dict[str, int] = {}
    starvation_alltime_first: Dict[str, int] = {}
    trt_error_windows = 0
    active_backends_seen = []
    memory_series = []
    final_backend = None

    for m in metrics:
        det = m.get("detector", {}) or {}
        if det.get("inference_avg_ms") is not None:
            inf_avg_window.append(float(det["inference_avg_ms"]))
        if det.get("inference_max_ms_window") is not None:
            inf_max_window.append(float(det["inference_max_ms_window"]))
        if det.get("processed_fps_total") is not None:
            total_fps_vals.append(float(det["processed_fps_total"]))
        if det.get("errors_window", 0) and det["errors_window"] > 0:
            trt_error_windows += 1
        if det.get("active_backend"):
            active_backends_seen.append(det["active_backend"])
            final_backend = det["active_backend"]
        e2e = (m.get("end_to_end_latency_ms") or {})
        if e2e.get("p95_ms") is not None:
            e2e_p95_vals.append(float(e2e["p95_ms"]))

        starv = m.get("starvation_alerts_alltime") or {}
        for k, v in starv.items():
            starvation_alltime_last[k] = v
            starvation_alltime_first.setdefault(k, v)

        if m.get("memory_mb") is not None:
            memory_series.append(float(m["memory_mb"]))

    starvation_window_total = sum(
        starvation_alltime_last.get(k, 0) - starvation_alltime_first.get(k, 0)
        for k in starvation_alltime_last
    )
    fallback_used = any(b == "fake" for b in active_backends_seen[:-1])  # any non-final
    fallback_at_end = (final_backend == "fake")
    memory_slope = (linreg_slope(list(range(len(memory_series))), memory_series)
                    if len(memory_series) >= 3 else 0.0)

    system = {
        "mean_processed_fps_total": round(statistics.fmean(total_fps_vals), 3) if total_fps_vals else 0.0,
        "p50_inference_avg_ms": round(percentile(inf_avg_window, 50), 2),
        "p95_inference_avg_ms": round(percentile(inf_avg_window, 95), 2),
        "p95_inference_max_ms": round(percentile(inf_max_window, 95), 2),
        "p50_end_to_end_p95_ms": round(percentile(e2e_p95_vals, 50), 2),
        "p95_end_to_end_p95_ms": round(percentile(e2e_p95_vals, 95), 2),
        "starvation_events_during_run": int(starvation_window_total),
        "trt_error_windows": int(trt_error_windows),
        "memory_slope_mb_per_window": round(memory_slope, 4),
        "fallback_used_during_run": bool(fallback_used),
        "final_active_backend": final_backend,
        "detection_events_total": len(detections),
    }

    # ---- Criteria evaluation ----
    reasons: List[str] = []
    criteria: Dict[str, dict] = {}

    # FAIL conditions.
    fail = False
    for cid, p in per_cam.items():
        if p["mean_processed_fps"] < WARN_MIN_CAM_FPS:
            reasons.append(f"FAIL: {cid} mean_fps {p['mean_processed_fps']} < {WARN_MIN_CAM_FPS}")
            fail = True
        if (p["age_ms_slope_per_window"] > 0
                and p["recent_latest_frame_age_ms"] is not None
                and p["recent_latest_frame_age_ms"] > 5000):
            reasons.append(f"FAIL: {cid} latest_frame_age_ms growing without bound "
                           f"(slope={p['age_ms_slope_per_window']}, recent={p['recent_latest_frame_age_ms']}ms)")
            fail = True
    if memory_slope > MEM_GROWTH_FAIL_MB_PER_WINDOW:
        reasons.append(f"FAIL: memory growing {memory_slope:.2f} MB/window "
                       f"(> {MEM_GROWTH_FAIL_MB_PER_WINDOW})")
        fail = True
    if trt_error_windows >= TRT_ERROR_WINDOWS_FAIL:
        reasons.append(f"FAIL: TensorRT errors in {trt_error_windows} windows "
                       f"(>= {TRT_ERROR_WINDOWS_FAIL})")
        fail = True

    # WARNING conditions.
    warn = False
    if not fail:
        for cid, p in per_cam.items():
            if WARN_MIN_CAM_FPS <= p["mean_processed_fps"] < PASS_MIN_CAM_FPS:
                reasons.append(f"WARNING: {cid} mean_fps {p['mean_processed_fps']} in [{WARN_MIN_CAM_FPS}, {PASS_MIN_CAM_FPS})")
                warn = True
            if p["reconnects_total_in_run"] > 0:
                reasons.append(f"WARNING: {cid} had {p['reconnects_total_in_run']} reconnects during run")
                warn = True
        if PASS_MAX_E2E_P95_MS <= system["p95_end_to_end_p95_ms"] < WARN_MAX_E2E_P95_MS:
            reasons.append(f"WARNING: p95 end-to-end {system['p95_end_to_end_p95_ms']}ms "
                           f"in [{PASS_MAX_E2E_P95_MS}, {WARN_MAX_E2E_P95_MS})")
            warn = True
        if fallback_used or fallback_at_end:
            reasons.append("WARNING: detector fallback to fake was active during the run")
            warn = True

    # PASS criteria (record results for the markdown table even if we passed).
    criteria["every_camera_fps_ge_0.8"] = {
        "pass": all(p["mean_processed_fps"] >= PASS_MIN_CAM_FPS for p in per_cam.values()) if per_cam else False,
        "detail": {cid: p["mean_processed_fps"] for cid, p in per_cam.items()},
    }
    criteria["p95_total_latency_lt_1.5s"] = {
        "pass": system["p95_end_to_end_p95_ms"] < PASS_MAX_E2E_P95_MS,
        "detail": f"{system['p95_end_to_end_p95_ms']}ms",
    }
    criteria["no_starvation"] = {
        "pass": system["starvation_events_during_run"] == 0,
        "detail": system["starvation_events_during_run"],
    }
    criteria["memory_stable"] = {
        "pass": memory_slope <= MEM_GROWTH_FAIL_MB_PER_WINDOW,
        "detail": f"slope={memory_slope:.3f} MB/window",
    }
    criteria["no_repeated_trt_errors"] = {
        "pass": trt_error_windows < TRT_ERROR_WINDOWS_FAIL,
        "detail": f"{trt_error_windows} window(s)",
    }
    criteria["no_fallback_during_final_benchmark"] = {
        "pass": not fallback_at_end,
        "detail": f"final_backend={final_backend}",
    }

    if fail:
        verdict = "FAIL"
    elif warn or not all(c["pass"] for c in criteria.values()):
        verdict = "WARNING"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "criteria": criteria,
        "per_camera": per_cam,
        "system": system,
        "summary": {
            "metrics_windows": len(metrics),
            "detection_events": len(detections),
            "cameras_seen": list(per_cam.keys()),
        },
    }


def render_markdown(result: dict) -> str:
    v = result["verdict"]
    banner = {"PASS": "✅ PASS", "WARNING": "⚠️ WARNING", "FAIL": "❌ FAIL"}.get(v, v)
    lines = [
        f"# Fit Report — {banner}",
        "",
        f"- metrics windows: **{result['summary']['metrics_windows']}**",
        f"- detection events: **{result['summary']['detection_events']}**",
        f"- cameras: {', '.join(result['summary']['cameras_seen']) or '_none_'}",
        "",
        "## Criteria",
        "",
        "| Criterion | Pass | Detail |",
        "|---|---|---|",
    ]
    for name, c in result["criteria"].items():
        mark = "✔" if c["pass"] else "✘"
        detail = c["detail"]
        if isinstance(detail, dict):
            detail = ", ".join(f"{k}={v}" for k, v in detail.items())
        lines.append(f"| `{name}` | {mark} | {detail} |")
    lines += [
        "",
        "## Per-camera",
        "",
        "| Camera | mean fps | recent frame age (ms) | age slope | reconnects |",
        "|---|---:|---:|---:|---:|",
    ]
    for cid, p in result["per_camera"].items():
        lines.append(
            f"| {cid} | {p['mean_processed_fps']} | "
            f"{p['recent_latest_frame_age_ms']} | "
            f"{p['age_ms_slope_per_window']} | "
            f"{p['reconnects_total_in_run']} |"
        )
    lines += [
        "",
        "## System",
        "",
    ]
    for k, v in result["system"].items():
        lines.append(f"- `{k}`: **{v}**")
    if result["reasons"]:
        lines += ["", "## Reasons", ""]
        for r in result["reasons"]:
            lines.append(f"- {r}")
    lines += [
        "",
        "## Thresholds",
        "",
        f"- PASS: every camera fps ≥ {PASS_MIN_CAM_FPS}; p95 e2e < {PASS_MAX_E2E_P95_MS}ms; no starvation; memory stable; no repeated TRT errors; fallback not used at end",
        f"- WARNING: any camera fps in [{WARN_MIN_CAM_FPS}, {PASS_MIN_CAM_FPS}); p95 e2e in [{PASS_MAX_E2E_P95_MS}, {WARN_MAX_E2E_P95_MS})ms; reconnects; fallback used",
        f"- FAIL: any camera fps < {WARN_MIN_CAM_FPS}; frame age unbounded; memory growth > {MEM_GROWTH_FAIL_MB_PER_WINDOW} MB/window; TRT errors in ≥ {TRT_ERROR_WINDOWS_FAIL} windows",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="logs/runtime_metrics.jsonl")
    parser.add_argument("--detections", default="logs/detections.jsonl")
    parser.add_argument("--out", default="reports/")
    args = parser.parse_args()

    metrics = load_jsonl(Path(args.metrics))
    detections = load_jsonl(Path(args.detections))

    result = grade(metrics, detections)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fit_report.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "fit_report.md").write_text(render_markdown(result))

    print(f"verdict: {result['verdict']}")
    for r in result["reasons"]:
        print(f"  - {r}")
    print(f"wrote {out_dir / 'fit_report.md'}")
    print(f"wrote {out_dir / 'fit_report.json'}")

    return {"PASS": 0, "WARNING": 1, "FAIL": 2}.get(result["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
