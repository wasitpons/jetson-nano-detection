#!/usr/bin/env bash
# swap_model.sh — drop a new ONNX in, get a live engine pointing at it.
#
# Usage:
#   tools/swap_model.sh <onnx_path> [imgsz=416]
#
# Steps:
#   1. Build a TensorRT FP16 engine from <onnx_path> via trtexec.
#      Engine lands at models/<onnx-basename>_fp16.engine.
#   2. Update config.yaml's detector.engine + detector.imgsz in place
#      (comments preserved).
#   3. Run the golden-image canary (test_detector_on_image.py) against the
#      new engine to confirm it actually decodes detections end-to-end.
#      Skipped quietly if no canary image is present.
#
# Run this ON THE TARGET JETSON. The engine is non-portable across Jetson
# devices, JetPack versions, or TensorRT versions.

set -euo pipefail

ONNX_PATH="${1:-}"
IMGSZ="${2:-416}"

if [[ -z "$ONNX_PATH" ]]; then
  echo "usage: $0 <onnx_path> [imgsz=416]" >&2
  exit 2
fi
if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ERROR: onnx not found at $ONNX_PATH" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="$PROJECT_ROOT/config.yaml"

# Engine path: models/<basename>_fp16.engine
BASENAME="$(basename "${ONNX_PATH%.onnx}")"
ENGINE_OUT="$PROJECT_ROOT/models/${BASENAME}_fp16.engine"

echo "=========================================================================="
echo "swap_model: $ONNX_PATH  →  $ENGINE_OUT  (imgsz=$IMGSZ)"
echo "=========================================================================="

# 1. Build.
"$SCRIPT_DIR/build_engine.sh" "$ONNX_PATH" "$ENGINE_OUT" fp16 1024

# 2. Point config.yaml at it.
RELATIVE_ENGINE="models/${BASENAME}_fp16.engine"
python3 "$SCRIPT_DIR/_set_active_engine.py" "$CONFIG_PATH" "$RELATIVE_ENGINE" "$IMGSZ"

# 3. Golden-image canary (only if we have an image to test against).
CANARY="$PROJECT_ROOT/models/samples/canary.jpg"
LABELS="$PROJECT_ROOT/models/coco_labels.txt"
if [[ -f "$CANARY" && -f "$LABELS" ]]; then
  echo
  echo "Running golden-image canary on $CANARY"
  if python3 "$SCRIPT_DIR/test_detector_on_image.py" \
       --engine "$ENGINE_OUT" \
       --image "$CANARY" \
       --labels "$LABELS" \
       --imgsz "$IMGSZ" \
       --output "$PROJECT_ROOT/outputs/"; then
    echo "Canary: OK"
  else
    rc=$?
    if [[ $rc -eq 2 ]]; then
      echo "WARNING: canary produced zero detections (rc=2). Engine loaded but" >&2
      echo "  the model didn't find anything in the canary image. Inspect" >&2
      echo "  $PROJECT_ROOT/outputs/ before promoting." >&2
    else
      echo "ERROR: canary failed (rc=$rc)" >&2
      exit $rc
    fi
  fi
else
  echo
  echo "(skipping canary: drop a vehicle photo at models/samples/canary.jpg to enable)"
fi

echo
echo "=========================================================================="
echo "Done."
echo "Active engine: $RELATIVE_ENGINE (imgsz=$IMGSZ)"
echo "Runtime will pick this up on next start (python3 edge_pipeline.py)."
echo "=========================================================================="
