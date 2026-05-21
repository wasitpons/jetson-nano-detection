#!/usr/bin/env python3
"""Runtime health gate.

Fails loudly with fix instructions if anything required to load the TensorRT
detector is missing on the box. Designed to run BEFORE `edge_pipeline.py` on
a fresh deployment, so we never have to debug a broken environment from
mid-stream tracebacks.

Exit codes:
  0 — all checks passed
  1 — one or more required checks failed

Usage:
  python3 tools/health_check.py --config ../config.yaml
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

# Allow running from anywhere — resolve project root from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _check_python() -> CheckResult:
    v = sys.version_info
    ok = (v.major == 3 and v.minor == 6)
    return CheckResult(
        "Python 3.6.x",
        ok,
        f"running {v.major}.{v.minor}.{v.micro}",
        "" if ok else "Install JetPack 4.6.x; do not upgrade Python past system version.",
    )


def _check_import(module: str, fix: str, version_attr: str = "__version__") -> CheckResult:
    try:
        m = importlib.import_module(module)
    except Exception as e:
        return CheckResult(f"import {module}", False, f"{type(e).__name__}: {e}", fix)
    ver = getattr(m, version_attr, "?")
    return CheckResult(f"import {module}", True, f"version={ver}", "")


def _check_cv2_nms() -> CheckResult:
    try:
        import cv2
        ok = hasattr(cv2, "dnn") and hasattr(cv2.dnn, "NMSBoxes")
    except Exception as e:
        return CheckResult("cv2.dnn.NMSBoxes", False, f"{type(e).__name__}: {e}",
                           "Use JetPack-provided cv2 (built with dnn module).")
    return CheckResult(
        "cv2.dnn.NMSBoxes", ok,
        "available" if ok else "missing",
        "" if ok else "Rebuild OpenCV with -DBUILD_opencv_dnn=ON.",
    )


def _check_trtexec() -> CheckResult:
    path = shutil.which("trtexec") or (
        "/usr/src/tensorrt/bin/trtexec"
        if Path("/usr/src/tensorrt/bin/trtexec").exists() else None
    )
    if path:
        return CheckResult("trtexec on PATH", True, f"found at {path}", "")
    return CheckResult(
        "trtexec on PATH", False,
        "not found",
        "Add `/usr/src/tensorrt/bin` to PATH, "
        "or `export PATH=/usr/src/tensorrt/bin:$PATH`.",
    )


def _check_file(label: str, path: Optional[str], fix: str) -> CheckResult:
    if not path:
        return CheckResult(label, False, "no path configured", fix)
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if p.exists():
        return CheckResult(label, True, f"{p} ({p.stat().st_size} bytes)", "")
    return CheckResult(label, False, f"missing: {p}", fix)


def _check_engine_loadable(engine_path: Optional[str]) -> CheckResult:
    if not engine_path:
        return CheckResult("engine deserializes", False, "no path configured",
                           "Set detector.tensorrt.engine_path in config.yaml.")
    p = Path(engine_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return CheckResult("engine deserializes", False, f"missing: {p}",
                           "Run tools/build_engine.sh first.")
    try:
        import tensorrt as trt
    except ImportError:
        return CheckResult("engine deserializes", False, "tensorrt not importable",
                           "TensorRT ships with JetPack; verify /usr/lib/python3.6/dist-packages/tensorrt.")
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        with open(p, "rb") as f:
            data = f.read()
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(data)
    except Exception as e:
        return CheckResult("engine deserializes", False, f"{type(e).__name__}: {e}",
                           "Engine corrupted or built for a different TRT version — rebuild.")
    if engine is None:
        return CheckResult("engine deserializes", False, "deserialize returned None",
                           "Rebuild engine with the current TRT (engines are not portable).")
    n = engine.num_bindings
    if n != 2:
        return CheckResult("engine bindings valid", False,
                           f"{n} bindings (expected 1 input + 1 output)",
                           "Re-export ONNX with a single input + single output, then rebuild.")
    return CheckResult("engine deserializes + bindings", True,
                       f"{n} bindings ok", "")


def _load_config_path(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("WARNING: PyYAML not installed; cannot resolve paths from config.")
        return {}
    p = Path(config_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        print(f"WARNING: config not found at {p}")
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--engine", help="override engine path")
    parser.add_argument("--labels", help="override labels path")
    args = parser.parse_args()

    cfg = _load_config_path(args.config)
    det_cfg = cfg.get("detector") or {}
    trt_cfg = det_cfg.get("tensorrt") or {}
    engine_path = args.engine or trt_cfg.get("engine_path")
    labels_path = args.labels or det_cfg.get("labels_path")

    checks: List[Callable[[], CheckResult]] = [
        _check_python,
        lambda: _check_import("tensorrt",
            "TensorRT ships with JetPack; check /usr/lib/python3.6/dist-packages/tensorrt."),
        lambda: _check_import("pycuda",
            "`pip3 install --user pycuda`; needs CUDA toolkit on PATH."),
        lambda: _check_import("cv2",
            "Use JetPack-provided cv2 with GStreamer + Tegra plugins."),
        _check_cv2_nms,
        lambda: _check_import("numpy",
            "`pip3 install --user 'numpy<1.20'` for Python 3.6."),
        _check_trtexec,
        lambda: _check_file("engine file exists", engine_path,
            "Run tools/build_engine.sh <onnx> <engine_out>."),
        lambda: _check_file("labels file exists", labels_path,
            "Add models/coco_labels.txt (one class per line, indexed by class_id)."),
        lambda: _check_engine_loadable(engine_path),
    ]

    results: List[CheckResult] = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as e:
            results.append(CheckResult(fn.__name__ if hasattr(fn, "__name__") else "<check>",
                                       False, f"raised {type(e).__name__}: {e}",
                                       "Investigate exception above."))

    # Pretty-print a table.
    name_w = max(len(r.name) for r in results) + 2
    print("=" * (name_w + 50))
    print(f"{'check'.ljust(name_w)}status  detail")
    print("-" * (name_w + 50))
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        print(f"{r.name.ljust(name_w)}{status}    {r.detail}")
    print("=" * (name_w + 50))

    failures = [r for r in results if not r.ok]
    if failures:
        print("\nFix instructions:")
        for i, r in enumerate(failures, 1):
            print(f"  {i}. [{r.name}] {r.fix}")
        return 1

    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
