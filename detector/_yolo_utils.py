"""YOLOv8 preprocess + postprocess shared by every backend.

Both the onnxruntime detector and the TensorRT detector run identical
preprocess (letterbox + BGR→RGB + /255 + NCHW float32) and identical
postprocess (xywh→xyxy + class filter + NMS + letterbox-undo). Keeping these
in one module means the two backends can never drift on the YOLO math —
which is the part most likely to introduce silent regressions.
"""

from typing import Dict, Iterable, List, Set, Tuple

import cv2
import numpy as np

from .base import Detection


def letterbox(im: np.ndarray, new_shape: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize keeping aspect ratio, pad to square. Returns (img, scale, (pad_x, pad_y))."""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_x = (new_shape - nw) // 2
    pad_y = (new_shape - nh) // 2
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, r, (pad_x, pad_y)


def preprocess_yolov8(image: np.ndarray, imgsz: int):
    """BGR uint8 H×W×3 → (NCHW float32 blob, scale, (pad_x, pad_y))."""
    lb, scale, (pad_x, pad_y) = letterbox(image, imgsz)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return chw[np.newaxis, ...], scale, (pad_x, pad_y)


def postprocess_yolov8(
    raw_output: np.ndarray,
    *,
    original_shape: Tuple[int, int],
    scale: float,
    pad: Tuple[int, int],
    conf_thres: float,
    iou_thres: float,
    vehicle_class_ids: Iterable[int],
    class_names: Dict[int, str],
) -> List[Detection]:
    """Decode YOLOv8 head → filter → NMS → undo letterbox → Detection list.

    `raw_output` accepts either (1, 84, N) (as exported by Ultralytics) or
    (1, N, 84). The head layout is 4 bbox (xywh-center) + 80 class scores;
    there is no objectness score in v8.
    """
    pad_x, pad_y = pad
    h0, w0 = original_shape
    vehicle_set: Set[int] = set(int(c) for c in vehicle_class_ids)

    preds = raw_output[0] if raw_output.ndim == 3 else raw_output
    # Heuristic: rows should be detections, columns should be 4+nc. If the
    # other axis is larger we transpose. v8 nano: 84 channels, ~8400 anchors.
    if preds.shape[0] < preds.shape[1]:
        preds = preds.transpose(1, 0)
    if preds.size == 0:
        return []

    boxes_xywh = preds[:, :4]
    scores = preds[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]

    mask = (confidences >= conf_thres) & np.isin(class_ids, list(vehicle_set))
    if not mask.any():
        return []
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # NMS over letterbox-space xywh (cv2 expects [x,y,w,h] in pixels).
    nms_in = np.stack([x1, y1, bw, bh], axis=1).astype(np.float32)
    keep = cv2.dnn.NMSBoxes(
        nms_in.tolist(), confidences.astype(np.float32).tolist(),
        conf_thres, iou_thres,
    )
    if keep is None or len(keep) == 0:
        return []
    keep = np.array(keep).flatten()

    # Undo letterbox.
    x1 = (x1[keep] - pad_x) / scale
    y1 = (y1[keep] - pad_y) / scale
    x2 = (x2[keep] - pad_x) / scale
    y2 = (y2[keep] - pad_y) / scale
    x1 = np.clip(x1, 0, w0 - 1)
    y1 = np.clip(y1, 0, h0 - 1)
    x2 = np.clip(x2, 0, w0 - 1)
    y2 = np.clip(y2, 0, h0 - 1)

    out: List[Detection] = []
    for i, k in enumerate(keep):
        cid = int(class_ids[k])
        out.append(Detection(
            class_id=cid,
            class_name=class_names.get(cid, str(cid)),
            confidence=float(confidences[k]),
            x1=float(x1[i]), y1=float(y1[i]), x2=float(x2[i]), y2=float(y2[i]),
        ))
    return out
