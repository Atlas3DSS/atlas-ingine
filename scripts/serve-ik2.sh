#!/usr/bin/env bash
# Serve Inkling-2-Mini (UD-IQ4_XS + DSpark drafter) on GPUs 2+3, OpenAI-compatible.
#
# This is the campaign's best measured production config, lifted from:
#   inkling-2-mini-atlas-ingine-2026-11-26, HANDOFF.md "Best config"
#   4 x 256K = 1M KV token pool, FP4 values / FP8 keys, 4-way batching, DSpark spec decode,
#   closed-loop interleaved chunked prefill (mixed read+decode >= 80% of 0R/4D),
#   and zero-copy paged prefix cache in pre-registered pinned host RAM (0 MiB VRAM).
#
# VRAM headroom (2x V100-PCIE-16GB): ~260 MiB free on GPU 2, ~600 MiB free on GPU 3
# (prefill-chunk 1024 avoids the 2 MiB VRAM cliff of chunk 2048 while matching/beating decode).
#
# Measured live HTTP SSE on 2x V100:
#   0R/4D: 79.1-80.4 tok/s/req (316-322 agg)
#   3R/1D: 65.8-69.2 tok/s/req (83-94% of 0R/4D; warm 22.8K prefill ~20,178 tok/s eff)
#   2R/2D: 62.9-73.4 tok/s/req (80-96% of 0R/4D; warm 22.8K prefill ~15,568 tok/s eff)
#   1R/3D: 63.4-68.0 tok/s/req (80-92% of 0R/4D; warm 22.8K prefill ~10,813 tok/s eff)
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
    --prefill-chunk 1024 \
    --prefill-interleave-chunk 512 \
    --mixed-decode-ratio 0.80 \
    --spec \
    --no-vision \
    --draft-ctx 8192 \
    --max-batch 4 \
    --max-queue 8 \
    --park-mb 0 \
    --park-host-mb 2048 \
    --log-level info \
    "$@"
