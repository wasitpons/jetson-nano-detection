#!/usr/bin/env bash
# build_engine.sh — wrap trtexec to produce a TensorRT engine + metadata json.
#
# Usage:
#   tools/build_engine.sh <onnx_path> <engine_out> [precision=fp16] [workspace_mb=1024]
#
# Most users want tools/swap_model.sh instead — it calls this script then
# updates config.yaml and verifies the engine with the golden-image canary.
#
# Always run this ON THE TARGET JETSON. Engines are NOT portable across
# Jetson devices, JetPack versions, or TensorRT versions.

set -euo pipefail

ONNX="${1:-}"
ENGINE_OUT="${2:-}"
PRECISION="${3:-fp16}"
WORKSPACE_MB="${4:-1024}"

if [[ -z "$ONNX" || -z "$ENGINE_OUT" ]]; then
  echo "usage: $0 <onnx_path> <engine_out> [precision=fp16] [workspace_mb=1024]" >&2
  exit 2
fi
if [[ ! -f "$ONNX" ]]; then
  echo "ERROR: onnx not found at $ONNX" >&2
  exit 2
fi

TRTEXEC="$(command -v trtexec || true)"
if [[ -z "$TRTEXEC" && -x "/usr/src/tensorrt/bin/trtexec" ]]; then
  TRTEXEC="/usr/src/tensorrt/bin/trtexec"
fi
if [[ -z "$TRTEXEC" ]]; then
  echo "ERROR: trtexec not found (try `export PATH=/usr/src/tensorrt/bin:\$PATH`)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/models"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/$(basename "${ENGINE_OUT%.engine}").build.log"
META_OUT="$LOG_DIR/$(basename "${ENGINE_OUT%.engine}").engine_metadata.json"

case "$PRECISION" in
  fp32) PFLAG="" ;;
  fp16) PFLAG="--fp16" ;;
  int8) PFLAG="--int8" ;;
  *) echo "ERROR: unknown precision '$PRECISION' (expected fp32|fp16|int8)" >&2; exit 2 ;;
esac

CMD=("$TRTEXEC" "--onnx=$ONNX" "--saveEngine=$ENGINE_OUT" "--workspace=$WORKSPACE_MB")
[[ -n "$PFLAG" ]] && CMD+=("$PFLAG")

echo "Running: ${CMD[*]}" | tee "$LOG_PATH"
echo "Building TensorRT engine — expect 5–15 min on Jetson Nano." | tee -a "$LOG_PATH"
echo "trtexec progress (also tee'd to $LOG_PATH):" | tee -a "$LOG_PATH"
echo "--------------------------------------------------------------------------"
start_ts=$SECONDS
# Stream trtexec stdout/stderr to the terminal *and* append to the log, so
# the user gets live phase + layer-timing output instead of a 10-minute
# silent stare at the prompt. stdbuf forces line-buffered output (trtexec
# block-buffers when stdout isn't a tty otherwise).
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
else
  "${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
fi
status=${PIPESTATUS[0]}
echo "--------------------------------------------------------------------------"
echo "trtexec done in $((SECONDS - start_ts))s (exit $status)" | tee -a "$LOG_PATH"
if [[ $status -ne 0 ]]; then
  echo "ERROR: trtexec build failed. Full log at $LOG_PATH" >&2
  exit 1
fi
if [[ ! -f "$ENGINE_OUT" ]]; then
  echo "ERROR: trtexec exited 0 but no engine at $ENGINE_OUT" >&2
  exit 1
fi

python3 "$SCRIPT_DIR/_write_engine_metadata.py" \
  --onnx "$ONNX" --engine "$ENGINE_OUT" \
  --precision "$PRECISION" --workspace-mb "$WORKSPACE_MB" \
  --trtexec "$TRTEXEC" --trtexec-command "${CMD[*]}" \
  --build-log "$LOG_PATH" --out "$META_OUT"

echo
echo "Engine:    $ENGINE_OUT"
echo "Metadata:  $META_OUT"
echo "Build log: $LOG_PATH"
echo "REMINDER: engines are not portable. Rebuild on each Jetson / JetPack / TRT change."
