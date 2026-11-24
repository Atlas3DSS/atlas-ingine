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

# Resolve the ACTUAL CUDA device selected by CUDA_VISIBLE_DEVICES (index or UUID) — never blindly pick GPU 0.
read -r GPU_UUID GPU_NAME < <(python3 - << 'EOF'
import os, subprocess
cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
sel = cvd.split(",")[0].strip() if cvd else "0"
lines = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
    text=True,
).splitlines()
for line in lines:
    parts = [p.strip() for p in line.split(",", 2)]
    if len(parts) != 3:
        continue
    idx, uuid, name = parts
    if sel == idx or sel == uuid or (sel.startswith("GPU-") and uuid.startswith(sel)):
        print(uuid, name)
        raise SystemExit(0)
raise SystemExit(f"Could not resolve CUDA_VISIBLE_DEVICES={cvd!r} in nvidia-smi output")
EOF
)
echo "Resolved CUDA device: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} -> uuid=$GPU_UUID name=$GPU_NAME"

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
