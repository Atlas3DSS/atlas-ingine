// Native multi-GPU Tensor+Expert Parallel (TP2/TP4 + EP2/EP4) generator for Inkling-2-Mini (INGINE_TP_SIZE > 1).
// Shards attention heads (24Q/8KV -> 12Q/4KV at TP2 or 6Q/2KV at TP4), KV cache (TP_SIZE x aggregate pool),
// routed MoE experts (128 -> 64 or 32 per GPU), and LM head across TP_SIZE GPUs in one process via NCCL
// (FP32 AllReduce / AllGather). Design inspired by knapcio/Inkling-2-Mini-4x-DGX-Spark-TP4 (c0f5f1cd);
// startup byte-exact P2P probe guards against Linux IOMMU translation-mode silent DMAR corruption
// (Cavemanmining Tesla P40 field report, gist 896bd4fa826da665525c9d2f9742bcff).
#pragma once

#include "ingine/generator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ingine::model {

struct TpP2pProbeResult {
  int capable_pairs = 0;            // directed (i -> j) pairs reporting cudaDeviceCanAccessPeer == 1
  int verified_pairs = 0;           // directed pairs verified byte-exact via cudaMemcpyPeer
  bool p2p_enabled = false;         // true iff capable_pairs > 0 and all capable pairs passed
  bool corrupted = false;           // true if any capable pair failed cudaMemcpyPeer or mismatched bytes
  int first_bad_src = -1;
  int first_bad_dst = -1;
  std::size_t first_bad_mismatches = 0;
  std::string detail;
};

// Deterministic non-zero 32-bit probe word for rank `src` at word index `k` (used by startup P2P probe & tests).
constexpr std::uint32_t tp_p2p_probe_word(int src, std::size_t k) {
  return 0xA5C30000u ^ (std::uint32_t(src + 1) * 0x01010101u) ^ std::uint32_t(k * 2654435761u);
}

// Runs a byte-exact 4 KiB cudaMemcpyPeer probe across all P2P-capable directed rank pairs in [0, tp_size).
// On any copy error or byte mismatch (e.g. Linux VT-d/AMD-Vi IOMMU translation mode silent DMAR fault 0x05),
// disables peer access on all pairs (fail-closed) and reports `corrupted = true`.
TpP2pProbeResult probe_and_enable_tp_p2p(int tp_size, bool force_fail = false);

std::unique_ptr<Generator> make_tp_generator(const RuntimeOptions& options);

}  // namespace ingine::model
