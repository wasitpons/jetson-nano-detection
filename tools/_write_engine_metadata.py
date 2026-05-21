#!/usr/bin/env python3
"""Write engine_metadata.json next to a freshly-built TRT engine.

Called by tools/build_engine.sh. Kept separate (instead of inline bash JSON)
because shell-quoting JSON portably is painful and Python is always present
on Jetson.

Probes the system for JetPack/L4T, TensorRT, CUDA, and device model versions.
Missing probes become JSON `null` so downstream consumers don't crash.
"""

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


HEADER_NOTE = (
    "Engines are not portable across Jetson devices, JetPack versions, or "
    "TensorRT versions. Rebuild on the target board."
)


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return None


def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
        if out.stderr.strip():
            return out.stderr.strip()
    except Exception:
        return None
    return None


def probe_jetpack_or_l4t() -> Optional[str]:
    txt = _read_text("/etc/nv_tegra_release")
    if txt:
        return txt.splitlines()[0].strip()
    # JetPack number sometimes lives here.
    for p in ("/etc/jetson_release", "/proc/device-tree/nvidia,dtsfilename"):
        txt = _read_text(p)
        if txt:
            return txt.splitlines()[0].strip()
    return None


def probe_tensorrt_version() -> Optional[str]:
    # Prefer the python module if importable; it's always in sync with the libs.
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        pass
    out = _run(["dpkg", "-l", "tensorrt"])
    if out:
        for line in out.splitlines():
            if line.startswith("ii") and "tensorrt" in line:
                parts = line.split()
                if len(parts) >= 3:
                    return parts[2]
    return None


def probe_cuda_version() -> Optional[str]:
    out = _run(["nvcc", "--version"])
    if out:
        m = re.search(r"release\s+(\d+\.\d+)", out)
        if m:
            return m.group(1)
    txt = _read_text("/usr/local/cuda/version.txt")
    if txt:
        return txt.strip()
    return None


def probe_device_model() -> Optional[str]:
    txt = _read_text("/proc/device-tree/model")
    if txt:
        return txt.strip("\x00 \n")
    return None


def probe_input_size(onnx_path: str) -> Optional[int]:
    """Read input HxW from the ONNX graph. Falls back to None if onnx not installed."""
    try:
        import onnx
    except ImportError:
        return None
    try:
        model = onnx.load(onnx_path)
        shape = model.graph.input[0].type.tensor_type.shape.dim
        # Standard YOLO input is NCHW; H is dim[2], W is dim[3].
        if len(shape) >= 4:
            h = shape[2].dim_value
            w = shape[3].dim_value
            if h == w and h > 0:
                return int(h)
            if h > 0 and w > 0:
                return int(max(h, w))
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--workspace-mb", type=int, required=True)
    parser.add_argument("--trtexec", required=True)
    parser.add_argument("--trtexec-command", required=True)
    parser.add_argument("--build-log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    metadata = {
        "_note": HEADER_NOTE,
        "onnx_path": args.onnx,
        "engine_path": args.engine,
        "input_size": probe_input_size(args.onnx),
        "precision": args.precision,
        "workspace_mb": args.workspace_mb,
        "build_datetime": now,
        "jetpack_or_l4t_version": probe_jetpack_or_l4t(),
        "tensorrt_version": probe_tensorrt_version(),
        "cuda_version": probe_cuda_version(),
        "trtexec_path": shutil.which(args.trtexec) or args.trtexec,
        "trtexec_command": args.trtexec_command,
        "build_log_path": args.build_log,
        "device_model": probe_device_model(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
