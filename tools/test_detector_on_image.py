#!/usr/bin/env python3
"""Golden-image test for the TensorRT detector.

Validates end-to-end (preprocess → infer → NMS → de-letterbox) on a single
image before any RTSP is involved. Prints detections as JSON and writes an
annotated PNG to `outputs/`.

Usage:
  python3 tools/test_detector_on_image.py \
    --engine models/yolov8n_416_fp16.engine \
    --image samples/cars.jpg \
    --labels models/coco_labels.txt

Exit codes:
  0 — at least one detection above --conf
  1 — runtime error (engine load, image read, etc.)
  2 — zero detections (canary regressed)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector.tensorrt_detector import TensorRTDetector

log = logging.getLogger("test_detector_on_image")


def load_labels(path: Path) -> Dict[int, str]:
    """One class name per line; class_id = line index."""
    out: Dict[int, str] = {}
    with open(path) as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                out[idx] = name
    return out


def draw_detections(image: np.ndarray, dets) -> np.ndarray:
    out = image.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, (d.x1, d.y1, d.x2, d.y2))
        color = (0, 200, 0) if d.class_name == "car" else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--vehicle-class-ids", default="2,3,5,7")
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = PROJECT_ROOT / labels_path
    if not labels_path.exists():
        log.error("labels file not found: %s", labels_path)
        return 1
    class_names = load_labels(labels_path)

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    image = cv2.imread(str(image_path))
    if image is None:
        log.error("could not read image: %s", image_path)
        return 1

    vehicle_ids = [int(x) for x in args.vehicle_class_ids.split(",") if x.strip()]

    try:
        detector = TensorRTDetector({
            "engine": args.engine,
            "imgsz": args.imgsz,
            "conf_thres": args.conf,
            "iou_thres": args.iou,
            "vehicle_class_ids": vehicle_ids,
            "class_names": class_names,
        })
    except Exception as e:
        log.error("failed to load detector: %s", e)
        return 1

    detector.warmup()
    dets = detector.infer(image)
    detector.close()

    print(json.dumps([
        {"class_id": d.class_id, "class_name": d.class_name,
         "confidence": round(d.confidence, 4),
         "bbox": {"x1": round(d.x1, 2), "y1": round(d.y1, 2),
                  "x2": round(d.x2, 2), "y2": round(d.y2, 2),
                  "w": round(d.x2 - d.x1, 2), "h": round(d.y2 - d.y1, 2)}}
        for d in dets
    ], indent=2))

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = draw_detections(image, dets)
    out_path = out_dir / f"{image_path.stem}_dets.png"
    cv2.imwrite(str(out_path), annotated)
    log.info("wrote %s (%d detections)", out_path, len(dets))

    return 0 if dets else 2


if __name__ == "__main__":
    sys.exit(main())
