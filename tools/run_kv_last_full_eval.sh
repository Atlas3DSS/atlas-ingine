#!/usr/bin/env bash
# Paired #829 native GPU qualification launcher (per_pass default vs opt-in L11/L17 last_full).
# Runs on a single physical GPU (B=1, 3 passes, no spec, cold prefix, EXL3-9G/FP8 pack + Q4_0 KV):
#   1. eval_kv_last_full: VRAM table (64K..1M), multi-chunk (512, 2048) NLL/KL + >1024 decode, ckpt isolation
#   2. eval_kv_last_full_tasks.py: paired server-level long-context + coding + tool-use tasks
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/native/build-multiarch}"
MODEL_DIR="${MODEL_DIR:-/mnt/storage/models/Inkling-2-Mini-atlas-exl3-9g-fp8-ingine}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/kv_last_full_829}"
KV_FORMAT="${KV_FORMAT:-q4_0}"
PREFIX_LEN="${PREFIX_LEN:-4096}"
DECODE_LEN="${DECODE_LEN:-1280}"
CHUNKS="${CHUNKS:-512,2048}"
MAX_CTX="${MAX_CTX:-32768}"
TASK_CHUNK="${TASK_CHUNK:-512}"
HOURLY_USD="${HOURLY_USD:-0.0}"

mkdir -p "${OUT_DIR}"

EVAL_BIN="${BUILD_DIR}/tests/model/eval_kv_last_full"
SERVE_BIN="${BUILD_DIR}/apps/serve/ingine-serve"

test -x "${EVAL_BIN}" || { echo "missing ${EVAL_BIN}" >&2; exit 1; }
test -x "${SERVE_BIN}" || { echo "missing ${SERVE_BIN}" >&2; exit 1; }
test -d "${MODEL_DIR}" || { echo "missing model dir ${MODEL_DIR}" >&2; exit 1; }

echo "=== [#829 Step 1/2] Native NLL / KL / >1024 decode / VRAM / checkpoint mode isolation ==="
"${EVAL_BIN}" \
  --model-dir "${MODEL_DIR}" \
  --kv-format "${KV_FORMAT}" \
  --chunks "${CHUNKS}" \
  --prefix-len "${PREFIX_LEN}" \
  --decode-len "${DECODE_LEN}" \
  --out-json "${OUT_DIR}/native_nll_kl.json"

echo "=== [#829 Step 2/2] Paired B=1 server task suite (long-context + coding + tool-use) ==="
python3 -B "${REPO_ROOT}/tools/eval_kv_last_full_tasks.py" \
  --serve-bin "${SERVE_BIN}" \
  --model-dir "${MODEL_DIR}" \
  --kv-format "${KV_FORMAT}" \
  --max-ctx "${MAX_CTX}" \
  --prefill-chunk "${TASK_CHUNK}" \
  --hourly-usd "${HOURLY_USD}" \
  --log-dir "${OUT_DIR}" \
  --out-json "${OUT_DIR}/server_tasks.json"

echo "=== [#829] COMPLETE: ${OUT_DIR}/native_nll_kl.json and ${OUT_DIR}/server_tasks.json ==="
