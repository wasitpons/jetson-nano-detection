"""YOLOv8 inference via TensorRT + PyCUDA on Jetson.

Build the engine once on the target Jetson using `tools/swap_model.sh`
(wraps trtexec). Engines are NOT portable across Jetson devices, JetPack
versions, or TensorRT versions — rebuild on the target board.

Two Jetson-specific design choices live here:

1) **CUDA managed (unified) memory** — `cuda.managed_empty(..., GLOBAL)` plus
   `host.base.get_device_pointer()` as the binding pointer. Tegra shares
   physical RAM between CPU and GPU, so explicit pinned-host + memcpy_htod/dtoh
   would have us paying the driver to serialise transfers that don't physically
   happen. Managed memory removes them entirely.

2) **Thread-bound CUDA context** — `pycuda.autoinit` binds the primary CUDA
   context to the **calling thread**. The `Detector` is constructed on the
   main thread but `infer()` runs on the `DetectorLoop` thread, so we defer
   the pycuda import + buffer allocation to the first `infer()` call. A
   thread-id check raises if a single detector is shared across threads.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import Detection, Detector
from ._yolo_utils import postprocess_yolov8, preprocess_yolov8

log = logging.getLogger(__name__)


@dataclass
class _Token:
    """Per-frame bookkeeping needed by postprocess to undo the letterbox."""
    scale: float
    pad: Tuple[int, int]
    original_shape: Tuple[int, int]


class TensorRTDetector(Detector):
    name = "tensorrt"

    def __init__(self, config: dict) -> None:
        # tensorrt is needed for engine deserialisation in __init__; pycuda
        # comes later because its primary context binds to the calling thread.
        try:
            import tensorrt as trt  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "tensorrt is required. On Jetson it ships with JetPack. "
                "Off-Jetson: use detector.engine pointing at a real engine, "
                "or set allow_fake_fallback=true."
            ) from e
        self._trt = __import__("tensorrt")
        self._cuda = None
        self._ctx_thread: Optional[int] = None

        self.engine_path: str = config["engine"]
        self.imgsz = int(config.get("imgsz", 416))
        self.conf_thres = float(config.get("conf_thres", 0.35))
        self.iou_thres = float(config.get("iou_thres", 0.5))
        self.vehicle_class_ids = set(config.get("vehicle_class_ids", [2, 3, 5, 7]))
        self.class_names: Dict[int, str] = {
            int(k): v for k, v in config.get("class_names", {}).items()
        }

        self._engine = None
        self._context = None
        self._stream = None
        self._input_binding: Optional[int] = None
        self._output_binding: Optional[int] = None
        self._host_input = None
        self._host_output = None
        self._bindings: List[int] = []
        self._input_shape: Optional[tuple] = None
        self._output_shape: Optional[tuple] = None

        self._cuda_init_lock = threading.Lock()
        self._load_engine()

    # ----- engine bring-up (no CUDA context needed yet) ----------------------

    def _load_engine(self) -> None:
        trt = self._trt
        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(
                f"failed to deserialize engine at {self.engine_path} "
                "(likely built with a different TRT/JetPack version — rebuild)"
            )
        if engine.num_bindings != 2:
            raise RuntimeError(
                f"engine has {engine.num_bindings} bindings; expected exactly 2 "
                "(1 input + 1 output)"
            )

        in_idx, out_idx = None, None
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                in_idx = i
            else:
                out_idx = i
        in_shape = tuple(engine.get_binding_shape(in_idx))
        out_shape = tuple(engine.get_binding_shape(out_idx))
        if any(d < 1 for d in in_shape + out_shape):
            raise RuntimeError(
                f"engine has dynamic shapes (in={in_shape} out={out_shape}); "
                "rebuild with static shapes"
            )

        log.info("trt engine loaded: input=%s output=%s", in_shape, out_shape)
        self._engine = engine
        self._input_binding = in_idx
        self._output_binding = out_idx
        self._input_shape = in_shape
        self._output_shape = out_shape

    # ----- thread-bound CUDA context bring-up --------------------------------

    def _ensure_cuda_context(self) -> None:
        """Initialise the CUDA context, stream, and managed buffers on the
        first inference call from the calling thread."""
        current = threading.get_ident()
        if self._cuda is not None:
            if self._ctx_thread != current:
                raise RuntimeError(
                    f"TensorRTDetector was initialised on thread "
                    f"{self._ctx_thread} but called from {current}; "
                    "CUDA contexts are thread-bound. One detector per thread."
                )
            return

        with self._cuda_init_lock:
            if self._cuda is not None:
                return
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit  # noqa: F401 — binds context to THIS thread
            except ImportError as e:
                raise RuntimeError(
                    "pycuda is required. `pip3 install --user pycuda`."
                ) from e
            self._cuda = cuda
            self._ctx_thread = current

            trt = self._trt
            in_dtype = trt.nptype(self._engine.get_binding_dtype(self._input_binding))
            out_dtype = trt.nptype(self._engine.get_binding_dtype(self._output_binding))

            self._host_input = cuda.managed_empty(
                int(np.prod(self._input_shape)), in_dtype,
                mem_flags=cuda.mem_attach_flags.GLOBAL,
            )
            self._host_output = cuda.managed_empty(
                int(np.prod(self._output_shape)), out_dtype,
                mem_flags=cuda.mem_attach_flags.GLOBAL,
            )
            bindings = [None, None]
            bindings[self._input_binding] = int(self._host_input.base.get_device_pointer())
            bindings[self._output_binding] = int(self._host_output.base.get_device_pointer())
            self._bindings = bindings

            self._context = self._engine.create_execution_context()
            self._stream = cuda.Stream()
            log.info("trt CUDA context initialised on thread=%s (managed memory)", current)

    # ----- inference ---------------------------------------------------------

    def warmup(self) -> None:
        self._ensure_cuda_context()
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(2):
            self.infer(dummy)
        log.info("tensorrt detector warm")

    def infer(self, image: np.ndarray) -> List[Detection]:
        self._ensure_cuda_context()
        blob, scale, pad = preprocess_yolov8(image, self.imgsz)
        np.copyto(
            self._host_input.reshape(self._input_shape),
            np.ascontiguousarray(blob).astype(self._host_input.dtype, copy=False),
        )

        t0 = time.monotonic()
        if not self._context.execute_async_v2(
            bindings=self._bindings, stream_handle=self._stream.handle,
        ):
            raise RuntimeError("trt execute_async_v2 returned False")
        self._stream.synchronize()
        log.debug("trt forward %.2fms", (time.monotonic() - t0) * 1000)

        raw = np.asarray(self._host_output).reshape(self._output_shape)
        return postprocess_yolov8(
            raw,
            original_shape=image.shape[:2],
            scale=scale, pad=pad,
            conf_thres=self.conf_thres, iou_thres=self.iou_thres,
            vehicle_class_ids=self.vehicle_class_ids,
            class_names=self.class_names,
        )

    def close(self) -> None:
        # Managed allocs are freed by their Python wrappers + the CUDA context
        # at interpreter exit; just drop references so a fallback swap doesn't
        # keep the engine alive.
        self._host_input = self._host_output = None
        self._bindings = []
        self._stream = self._context = self._engine = None
