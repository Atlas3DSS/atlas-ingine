#!/usr/bin/env bash
set -euo pipefail

# Standalone Flagship Celebration Video Hillclimb on atlas-ingine (native GPU vision + JSON mode).
# Theme: 10 Years of NVIDIA AI Architecture (2016 Pascal -> 2026 Blackwell),
#        Thinking Machines Lab's Inkling-2-Mini, and Open-Source + Human-AI Collaboration.

PORT="${1:-8199}"
ITERS="${2:-16}"
RENDER_DEV="${3:-CUDA}"
DO_1080P="${4:-1}"
MAX_CTX="${5:-131072}"
OUT_DIR="${6:-./runs/celebration_$(date -u +%Y%m%dT%H%M%SZ)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INGINE_BIN="${INGINE_BIN:-${REPO_ROOT}/native/build-multiarch/apps/serve/ingine-serve}"
MODEL_DIR="${MODEL_DIR:-/mnt/storage/models/Inkling-2-Mini-ingine}"
BLENDER_BIN="${BLENDER_BIN:-$(command -v blender || echo "${HOME}/.local/opt/blender-4.5.14-linux-x64/blender")}"

mkdir -p "$OUT_DIR"
SRV_LOG="${OUT_DIR}/ingine_serve.log"

SRV_PID=""
cleanup() {
  if [ -n "${SRV_PID:-}" ]; then
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
  fi
}
trap cleanup EXIT

"$INGINE_BIN" \
  --engine \
  --model-dir "$MODEL_DIR" \
  --model-id inkling-2-mini \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-ctx "$MAX_CTX" \
  --dense-fp8 \
  --kv-fp8 \
  --spec \
  --spec-threshold 0.8 \
  --park-mb 0 > "$SRV_LOG" 2>&1 &
SRV_PID=$!

for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    tail -n 60 "$SRV_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "ERROR: health timeout" >&2; exit 1; }

EXTRA_ARGS=()
if [ "$DO_1080P" = "1" ]; then
  EXTRA_ARGS+=(--render-1080p --final-samples 64)
fi

PYTHONPATH="${REPO_ROOT}/ledger/examples" python3 -B -u -m blender_celebration_hillclimb.hillclimb \
  --out-dir "$OUT_DIR" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model inkling-2-mini \
  --blender-bin "$BLENDER_BIN" \
  --render-device "$RENDER_DEV" \
  --iters "$ITERS" \
  --preview-samples 24 \
  --temperature 0.6 \
  --author-effort low \
  --critique-effort low \
  --max-tokens 16384 \
  --seed 42 \
  "${EXTRA_ARGS[@]}"
