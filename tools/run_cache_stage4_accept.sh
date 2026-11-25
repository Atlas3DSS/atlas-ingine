#!/usr/bin/env bash
# Stage 4 (#827) Independent Joint Acceptance Gate on Devbox RTX PRO 6000 Blackwell (sm_120a)
# Runs on the combined #825 + #826 SHA:
#   1. GPU Stage 2 batched prefix parity & full edge-case matrix (batch_prefix_parity)
#   2. GPU Stage 3 root-prefix + host-RAM parking + disk provenance (model_cache_stage3 FP8 + Q4)
#   3. GPU B=1 core reuse regression (model_reuse)
#   4. 4-arm matched HTTP server replay acceptance on frozen #824 traces:
#        b1-cold (--no-prefix-cache) vs b1-warm (--park-mb -1 --park-host-mb 2048)
#        batch-cold (--max-batch 4 --no-prefix-cache) vs batch-warm (--max-batch 4)
#      plus automated turn-by-turn output parity & latency/ceiling summary!
set -euo pipefail

REPO="/home/orwel/dev_genius/engines/atlas-ingine-cache-827"
BUILD="$REPO/native/build-sm120a"
SERVE_BIN="$BUILD/apps/serve/ingine-serve"
MODEL_DIR="/mnt/storage/models/Inkling-2-Mini-ingine"
TRACES_DIR="$REPO/results/cache_stage1_traces"
OUT_DIR="$REPO/results/stage4_accept"
PORT=18497
GIT_REV="$(git -C "$REPO" rev-parse --short HEAD)"
FULL_SHA="$(git -C "$REPO" rev-parse HEAD)"

mkdir -p "$OUT_DIR"

echo "=== #827 STAGE 4 JOINT ACCEPTANCE GATE ==="
echo "REPO=$REPO"
echo "COMBINED_SHA=$FULL_SHA ($GIT_REV)"

read -r GPU_UUID GPU_NAME < <(python3 - << 'EOF'
import ctypes
lib = ctypes.CDLL("libcuda.so.1")
assert lib.cuInit(0) == 0
dev = ctypes.c_int(0)
assert lib.cuDeviceGet(ctypes.byref(dev), 0) == 0
uuid_bytes = (ctypes.c_ubyte * 16)()
assert lib.cuDeviceGetUuid(uuid_bytes, dev) == 0
h = bytes(uuid_bytes).hex()
uuid = f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
name_buf = ctypes.create_string_buffer(256)
assert lib.cuDeviceGetName(name_buf, 256, dev) == 0
print(uuid, name_buf.value.decode("utf-8").strip())
EOF
)
echo "GPU=$GPU_NAME ($GPU_UUID)"

echo "=== [1/5] GPU STAGE 2: batch_prefix_parity (full logits + KL + edge matrix) ==="
INGINE_MODEL_DIR="$MODEL_DIR" "$BUILD/tests/batch/batch_prefix_parity"

run_arm() {
  local mode="$1"
  local extra_flags="$2"
  local out_json="$OUT_DIR/stage4_${mode//-/_}.json"
  local srv_log="$OUT_DIR/stage4_${mode//-/_}.server.log"
  rm -f "$srv_log"
  echo "=== [2/5] HTTP REPLAY ARM: $mode ($extra_flags) ==="
  # shellcheck disable=SC2086
  "$SERVE_BIN" --engine --model-dir "$MODEL_DIR" --model-id inkling-2-mini \
    --host 127.0.0.1 --port "$PORT" --max-ctx 65536 --dense-fp8 --kv-fp8 --no-vision \
    $extra_flags > "$srv_log" 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true" RETURN
  local ready=0
  for _ in $(seq 1 720); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: ingine-serve exited early for $mode" >&2
      tail -n 40 "$srv_log" >&2
      return 1
    fi
    sleep 0.5
  done
  if [ "$ready" -ne 1 ]; then
    echo "ERROR: ingine-serve failed to become ready for $mode" >&2
    tail -n 40 "$srv_log" >&2
    return 1
  fi
  PYTHONPATH="$REPO/tools" python3 "$REPO/tools/cache_stage4_accept.py" replay \
    --traces-dir "$TRACES_DIR" \
    --model-dir "$MODEL_DIR" \
    --base-url "http://127.0.0.1:$PORT" \
    --model inkling-2-mini \
    --mode "$mode" \
    --server-log "$srv_log" \
    --git-rev "$GIT_REV" \
    --gpu-name "$GPU_NAME" \
    --gpu-uuid "$GPU_UUID" \
    --serve-flags "--max-ctx 65536 --dense-fp8 --kv-fp8 --no-vision $extra_flags" \
    --max-tokens 16 \
    --output "$out_json"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - RETURN
}

run_arm "b1-cold" "--max-batch 1 --park-mb 0 --no-prefix-cache"
run_arm "b1-warm" "--max-batch 1 --park-mb -1 --park-host-mb 2048"
run_arm "batch-cold" "--max-batch 4 --max-kv-tokens 131072 --no-prefix-cache"
run_arm "batch-warm" "--max-batch 4 --max-kv-tokens 131072"

echo "=== PAIRED COMPARISON SUMMARY ==="
PYTHONPATH="$REPO/tools" python3 "$REPO/tools/cache_stage4_accept.py" compare \
  --stage1-baseline "$REPO/results/cache_stage1_baseline.json" \
  --b1-cold "$OUT_DIR/stage4_b1_cold.json" \
  --b1-warm "$OUT_DIR/stage4_b1_warm.json" \
  --batch-cold "$OUT_DIR/stage4_batch_cold.json" \
  --batch-warm "$OUT_DIR/stage4_batch_warm.json" \
  --output "$OUT_DIR/cache_stage4_accept_summary.json"

echo "=== [3/5] GPU CORE REUSE: model_reuse ==="
INGINE_MODEL_DIR="$MODEL_DIR" "$BUILD/tests/model/model_reuse"

echo "=== [4/5] GPU STAGE 3: model_cache_stage3 (FP8 KV) ==="
INGINE_MODEL_DIR="$MODEL_DIR" "$BUILD/tests/model/model_cache_stage3" --model-dir "$MODEL_DIR"

echo "=== [5/5] GPU STAGE 3: model_cache_stage3 (--kv-q4) ==="
INGINE_MODEL_DIR="$MODEL_DIR" "$BUILD/tests/model/model_cache_stage3" --model-dir "$MODEL_DIR" --kv-q4

echo "=== STAGE4_827_ALL_PASS ==="
