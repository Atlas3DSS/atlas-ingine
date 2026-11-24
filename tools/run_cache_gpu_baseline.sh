#!/usr/bin/env bash
# Runs the matched pre-optimization GPU cache replay baseline (#824) against native `ingine-serve --engine`:
#   1. b1-cold    (--max-batch 1 --park-mb 0 + per-turn cache flush)
#   2. b1-warm    (--max-batch 1, default VRAM parking)
#   3. batch-cold (--max-batch 4 --max-kv-tokens 131072)
set -euo pipefail

REPO="${1:-/home/orwel/dev_genius/engines/atlas-ingine-cache-s1}"
SERVE_BIN="${SERVE_BIN:-$REPO/native/build-s1/apps/serve/ingine-serve}"
MODEL_DIR="${MODEL_DIR:-/mnt/storage/models/Inkling-2-Mini-ingine}"
TRACES_DIR="${TRACES_DIR:-$REPO/results/cache_stage1_traces}"
OUT_DIR="${OUT_DIR:-$REPO/results}"
PORT="${PORT:-18491}"
GIT_REV="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Resolve the ACTUAL physical GPU that CUDA opens as logical device 0 via libcuda (no CUDA context created),
# honoring CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER identically to the CUDA runtime.
read -r GPU_UUID GPU_NAME < <(python3 - << 'EOF'
import ctypes

lib = ctypes.CDLL("libcuda.so.1")
if lib.cuInit(0) != 0:
    raise SystemExit("cuInit(0) failed")
dev = ctypes.c_int(0)
if lib.cuDeviceGet(ctypes.byref(dev), 0) != 0:
    raise SystemExit("cuDeviceGet(0) failed")
uuid_bytes = (ctypes.c_ubyte * 16)()
if lib.cuDeviceGetUuid(uuid_bytes, dev) != 0:
    raise SystemExit("cuDeviceGetUuid failed")
h = bytes(uuid_bytes).hex()
uuid = f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
name_buf = ctypes.create_string_buffer(256)
if lib.cuDeviceGetName(name_buf, 256, dev) != 0:
    raise SystemExit("cuDeviceGetName failed")
print(uuid, name_buf.value.decode("utf-8").strip())
EOF
)
echo "Resolved CUDA logical device 0: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} -> uuid=$GPU_UUID name=$GPU_NAME"

run_arm() {
  local mode="$1"
  local extra_flags="$2"
  local out_json="$OUT_DIR/cache_stage1_gpu_${mode//-/_}.json"
  local srv_log="$OUT_DIR/cache_stage1_gpu_${mode//-/_}.server.log"
  rm -f "$srv_log"
  echo "=== Starting $mode ($extra_flags) on $GPU_NAME ($GPU_UUID, $GIT_REV) ==="
  # shellcheck disable=SC2086
  "$SERVE_BIN" --engine --model-dir "$MODEL_DIR" --model-id inkling-2-mini \
    --host 127.0.0.1 --port "$PORT" --max-ctx 65536 --dense-fp8 --kv-fp8 --no-vision \
    $extra_flags > "$srv_log" 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true" RETURN
  local ready=0
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: ingine-serve exited early for $mode; tail of $srv_log:" >&2
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
  python3 "$REPO/tools/cache_replay_baseline.py" \
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

run_arm "b1-cold" "--max-batch 1 --park-mb 0"
run_arm "b1-warm" "--max-batch 1"
run_arm "batch-cold" "--max-batch 4 --max-kv-tokens 131072"
echo "ALL_GPU_BASELINE_ARMS_DONE"
