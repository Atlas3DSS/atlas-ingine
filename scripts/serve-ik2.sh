#!/usr/bin/env bash
# Serve Inkling-2-Mini (UD-IQ4_XS + DSpark drafter) on GPUs 2+3, OpenAI-compatible.
#
# This is the campaign's best measured config, lifted from the benchmark line:
#   inkling-2-mini-atlas-ingine-2026-11-26, HANDOFF.md "Best config"
#   4 x 256K = 1M KV token pool, FP4 values / FP8 keys, 4-way batching, spec decode.
#
# Measured with ingine-bench at 4K depth: 295.3 tok/s aggregate, 311.4 steady
# (77.9 tok/s/stream), prefill 3143 tok/s, verify 23.7 ms.
#
# GPUs 2 and 3 ONLY (CUDA_VISIBLE_DEVICES). The binary is the TP2/ETP build;
# TP_SIZE is compile-time, so device selection is entirely CUDA_VISIBLE_DEVICES.
#
# Usage: serve-ik2.sh [extra ingine-serve args...]
set -euo pipefail

MODEL_DIR=${IK2_MODEL_DIR:-/home/svniak/models/Inkling-2-Mini-atlas-gguf}
BIN=${IK2_SERVE_BIN:-/home/svniak/atlas/atlas-ingine/native/build-sm70-tp2-etp/apps/serve/ingine-serve}
# 0.0.0.0 covers loopback AND the tailnet address (100.127.90.18) in one listener;
# ingine-serve binds a single host (no --plain dual-bind like Maya).
HOST=${IK2_HOST:-0.0.0.0}
PORT=${IK2_PORT:-8090}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3

# <=50 of 56 threads, same pinning the benchmark used (socket 1 memory, both sockets' cores).
exec numactl --membind=1 --physcpubind=14-27,42-55,0-10,28-38 \
  "$BIN" \
    --engine \
    --model-dir "$MODEL_DIR" \
    --host "$HOST" --port "$PORT" \
    --max-ctx 262144 \
    --max-kv-tokens 1048576 \
    --kv-fp4v \
    --prefill-chunk 2048 \
    --spec \
    --no-vision \
    --draft-ctx 8192 \
    --max-batch 4 \
    --max-queue 8 \
    --park-mb 0 \
    --log-level info \
    "$@"
