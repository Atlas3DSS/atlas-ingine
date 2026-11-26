// Native multi-GPU Tensor+Expert Parallel (TP2/TP4 + EP2/EP4) generator for Inkling-2-Mini (INGINE_TP_SIZE > 1).
// Implements BOTH:
//   - TpBatchGenerator: multi-GPU continuous batching (B = 1..128+, TP_SIZE x paged full-KV pool,
//     TP_SIZE x smaller sliding rings, bucketed CUDA graphs capturing all 48 visits + NCCL FP32 collectives).
//   - TpSpecGenerator:  multi-GPU single-stream + DSpark speculative decoding (M = 1, 2, 4, 8 CUDA graphs + NCCL,
//     prefix checkpoint reuse across all ranks, rank-0 Drafter with zero Drafter changes).
// Upstream credits:
//   - Multi-GPU TP+EP sharding architecture & benchmark suite: knapcio/Inkling-2-Mini-4x-DGX-Spark-TP4 (c0f5f1cd).
//   - Startup byte-exact P2P probe & iommu=pt fail-closed guard: Cavemanmining Tesla P40 multi-GPU field report
//     (gist 896bd4fa826da665525c9d2f9742bcff).
#include "model/tp_engine.h"

#include "core/act.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/ops.h"
#include "engine/log.h"
#include "kernels/attn/attn.h"
#include "kernels/batch/batch.h"
#include "kernels/dense/dense.h"
#include "kernels/draft/draft.h"
#include "kernels/head/head.h"
#include "kernels/moe/moe.h"
#include "kernels/prefill/prefill.h"
#include "model/batch_drafter.h"
#include "model/drafter.h"
#include "model/model_kernels.h"
#include "model/runtime.h"
#include "model/weights_gguf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if INGINE_TP_SIZE > 1
#include <nccl.h>
#endif

namespace ingine::model {

TpP2pProbeResult probe_and_enable_tp_p2p(int tp_size, bool force_fail) {
  TpP2pProbeResult res;
  if (tp_size <= 1) return res;
  constexpr std::size_t kProbeWords = 1024;  // 4 KiB
  constexpr std::size_t kProbeBytes = kProbeWords * sizeof(std::uint32_t);
  constexpr std::uint32_t kPoison = 0xDEADBEEFu;

  std::vector<void*> src_buf(tp_size, nullptr);
  std::vector<void*> dst_buf(tp_size, nullptr);
  std::vector<std::uint32_t> host_poison(kProbeWords, kPoison);
  std::vector<std::uint32_t> host_expected(kProbeWords);
  std::vector<std::uint32_t> host_got(kProbeWords);

  auto cleanup = [&]() {
    for (int r = 0; r < tp_size; ++r) {
      cudaSetDevice(r);
      if (src_buf[r]) cudaFree(src_buf[r]);
      if (dst_buf[r]) cudaFree(dst_buf[r]);
    }
    cudaSetDevice(0);
  };

  try {
    for (int r = 0; r < tp_size; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaMalloc(&src_buf[r], kProbeBytes));
      CUDA_CHECK(cudaMalloc(&dst_buf[r], kProbeBytes));
      for (std::size_t k = 0; k < kProbeWords; ++k) host_expected[k] = tp_p2p_probe_word(r, k);
      CUDA_CHECK(cudaMemcpy(src_buf[r], host_expected.data(), kProbeBytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dst_buf[r], host_poison.data(), kProbeBytes, cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < tp_size && !res.corrupted; ++i) {
      for (int j = 0; j < tp_size && !res.corrupted; ++j) {
        if (i == j) continue;
        int can = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can, i, j));
        if (!can) continue;
        ++res.capable_pairs;

        // Enable peer access on both endpoints (ignore already-enabled).
        CUDA_CHECK(cudaSetDevice(i));
        cudaError_t e1 = cudaDeviceEnablePeerAccess(j, 0);
        if (e1 == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else if (e1 != cudaSuccess) {
          res.corrupted = true;
          res.first_bad_src = i;
          res.first_bad_dst = j;
          res.detail = std::string("cudaDeviceEnablePeerAccess failed: ") + cudaGetErrorString(e1);
          break;
        }
        int can_rev = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_rev, j, i));
        if (can_rev) {
          CUDA_CHECK(cudaSetDevice(j));
          cudaError_t e2 = cudaDeviceEnablePeerAccess(i, 0);
          if (e2 == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
          else if (e2 != cudaSuccess) {
            res.corrupted = true;
            res.first_bad_src = i;
            res.first_bad_dst = j;
            res.detail = std::string("cudaDeviceEnablePeerAccess(rev) failed: ") + cudaGetErrorString(e2);
            break;
          }
        }

        // Poison destination, execute P2P transfer, and verify every byte on host.
        CUDA_CHECK(cudaSetDevice(j));
        CUDA_CHECK(cudaMemcpy(dst_buf[j], host_poison.data(), kProbeBytes, cudaMemcpyHostToDevice));
        const cudaError_t pe = cudaMemcpyPeer(dst_buf[j], j, src_buf[i], i, kProbeBytes);
        const cudaError_t se = (pe == cudaSuccess) ? cudaDeviceSynchronize() : pe;
        if (se != cudaSuccess) {
          cudaGetLastError();
          res.corrupted = true;
          res.first_bad_src = i;
          res.first_bad_dst = j;
          res.detail = std::string("cudaMemcpyPeer failed: ") + cudaGetErrorString(se);
          break;
        }
        CUDA_CHECK(cudaMemcpy(host_got.data(), dst_buf[j], kProbeBytes, cudaMemcpyDeviceToHost));
        if (force_fail) host_got[0] ^= 0x00000001u;

        std::size_t mismatches = 0;
        for (std::size_t k = 0; k < kProbeWords; ++k) {
          if (host_got[k] != tp_p2p_probe_word(i, k)) ++mismatches;
        }
        if (mismatches > 0) {
          res.corrupted = true;
          res.first_bad_src = i;
          res.first_bad_dst = j;
          res.first_bad_mismatches = mismatches;
          res.detail = "silent P2P corruption (" + std::to_string(mismatches) + "/" +
                       std::to_string(kProbeWords) + " u32 words mismatched)";
          break;
        }
        ++res.verified_pairs;
      }
    }
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();

  if (res.corrupted) {
    // Fail-closed: disable peer access across all pairs so nothing touches broken PCIe P2P.
    for (int i = 0; i < tp_size; ++i) {
      cudaSetDevice(i);
      for (int j = 0; j < tp_size; ++j) {
        if (i == j) continue;
        cudaError_t e = cudaDeviceDisablePeerAccess(j);
        if (e != cudaSuccess) cudaGetLastError();
      }
    }
    cudaSetDevice(0);
    res.p2p_enabled = false;
  } else {
    res.p2p_enabled = (res.capable_pairs > 0);
  }
  return res;
}

void validate_tp_runtime_options(const RuntimeOptions& options, int tp_size) {
  const std::string tag = "TP" + std::to_string(tp_size);
  if (options.max_batch < 1) throw std::invalid_argument(tag + ": max_batch must be >= 1");
  if (options.max_ctx < 16) throw std::invalid_argument(tag + ": max_ctx must be >= 16");
  if (options.kv_q4 || options.kv_paged || options.kv_vram_mb > 0 || !options.kv_swap_dir.empty()) {
    throw std::invalid_argument(
        tag + " mode supports BF16/FP16 or --kv-fp8 KV cache (not --kv-q4, --kv-paged, --kv-vram-mb, or --kv-swap-dir)");
  }
  if (options.kv_cache_mode != KvCacheMode::PerPass) {
    throw std::invalid_argument(tag + " mode supports only --kv-cache-mode per_pass (not last_full)");
  }
  if (options.kv_fp4v && !options.kv_fp8) throw std::invalid_argument(tag + ": --kv-fp4v needs --kv-fp8");
  if (options.kv_fp4v && options.max_batch == 1 && options.spec)
    throw std::invalid_argument(tag + ": --kv-fp4v is for the batch generator (not --spec with --max-batch 1)");
  if (options.spec && options.max_batch > 1) {
    if (tp_size != 2) throw std::invalid_argument(tag + ": batched --spec currently requires TP2");
    if (options.max_batch > kernels::draft::kMaxBatchSlots)
      throw std::invalid_argument(tag + ": batched --spec supports max_batch <= " +
                                  std::to_string(kernels::draft::kMaxBatchSlots));
    if (options.spec_max_m < 2 || options.spec_max_m > 4)
      throw std::invalid_argument(tag + ": --spec-max-m must be in 2..4");
  }
  if (options.vision) {
    throw std::invalid_argument(tag + " mode does not support --vision");
  }
  if (options.offload_vram_mb > 0 || !options.expert_ranking.empty() || !options.dump_router_stats.empty()) {
    throw std::invalid_argument(tag + " mode does not support dynamic MoE offloading or router stats dumping");
  }
  if (!options.kv_ckpt_path.empty() || !options.kv_ckpt_resume.empty() || options.kv_ckpt_every > 0) {
    throw std::invalid_argument(tag + " mode does not support disk KV checkpoints");
  }
  if (options.enable_persona_hooks || !options.refusal_map.empty()) {
    throw std::invalid_argument(tag + " mode does not support persona hooks or refusal maps");
  }
}

#if INGINE_TP_SIZE <= 1

std::unique_ptr<Generator> make_tp_generator(const RuntimeOptions&) {
  throw std::logic_error("make_tp_generator called in INGINE_TP_SIZE=1 build");
}

#else

namespace {

namespace attn = kernels::attn;
namespace bk = kernels::batch;
namespace dense = kernels::dense;
namespace head = kernels::head;
namespace kd = kernels::draft;
namespace moe = kernels::moe;
namespace pk = kernels::prefill;

constexpr int H = HIDDEN;
constexpr int kHeadChunk = 32768;

void validate_tp_request(const Prompt& prompt, const Sampling& sampling) {
  if (!prompt.images.empty()) throw std::invalid_argument("TP mode does not support vision inputs");
  if (sampling.persona) throw std::invalid_argument("TP mode does not support persona profiles");
  if (prompt.ids.empty()) throw std::invalid_argument("prompt is empty");
  ops::validate_token_ids_i32(prompt.ids.data(), int(prompt.ids.size()), VOCAB);
  ops::validate_token_ids_i32(sampling.forced.data(), int(sampling.forced.size()), VOCAB);
}

#define NCCL_CHECK(expr)                                                                                      \
  do {                                                                                                        \
    const ncclResult_t _nr = (expr);                                                                          \
    if (_nr != ncclSuccess) {                                                                                 \
      throw std::runtime_error(std::string("NCCL error in ") + #expr + ": " + ncclGetErrorString(_nr));       \
    }                                                                                                         \
  } while (0)

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
const ActT* bf16p(const Tensor& t) { return static_cast<const ActT*>(t.data); }
const ActT* bf16p(const void* p) { return static_cast<const ActT*>(p); }

dense::Matrix matrix(const Dense& d) {
  return dense::Matrix{d.w.data, static_cast<const float*>(d.scale.data), d.rows(),
                       static_cast<const std::int64_t*>(d.blk_off.data), static_cast<const std::int32_t*>(d.blk_soff.data),
                       static_cast<const std::uint16_t*>(d.q8_scale.data)};
}

std::vector<int> proj_layout(bool full, bool single_input) {
  std::vector<int> t;
  for (int s = 0; s < num_segs(full); ++s) {
    t.push_back(SEG_BEGIN[s]);
    t.push_back(SEG_BEGIN[s + 1]);
    t.push_back(single_input ? 0 : SEG_INPUT[s]);
  }
  return t;
}

DeviceBuffer upload_ints(const std::vector<int>& v) {
  DeviceBuffer b(v.size() * sizeof(int));
  b.copy_from_host(v.data(), v.size() * sizeof(int));
  return b;
}

std::vector<const float*> blocks_of(int l, int p, const float* B0, const float* B1, float* const* B2, float* const* E) {
  if (l < 1) return {};
  if (l < 7) return {B0};
  if (l < 13) return {B0, B1};
  if (l < 19) return {B0, B1, B2[l < LOOP_END ? p : 2]};
  return {B0, B1, B2[2], E[2]};
}

head::SamplingParams to_head_params(const Sampling& sampling) {
  head::SamplingParams p;
  p.temperature = sampling.temperature;
  p.top_k = sampling.top_k;
  p.top_p = sampling.top_p;
  for (std::int32_t id : sampling.banned_tokens) {
    if (id >= 0 && p.banned.count < head::BannedTokens::kMax) p.banned.ids[p.banned.count++] = id;
  }
  return p;
}

bool same_banned(const head::BannedTokens& a, const head::BannedTokens& b) {
  if (a.count != b.count) return false;
  for (int i = 0; i < a.count; ++i)
    if (a.ids[i] != b.ids[i]) return false;
  return true;
}

std::string format_uuid(const cudaUUID_t& u) {
  char buf[64];
  const auto* b = reinterpret_cast<const unsigned char*>(u.bytes);
  std::snprintf(buf, sizeof(buf),
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
  return buf;
}

// ----------------------------------------------------------------------------- TpWorkerPool (4-GPU NCCL coordinator)

constexpr int kBlocksAr = kernels::batch::P2pAllReduce2::kBlocks;

class TpWorkerPool {
 public:
  struct RankInfo {
    int rank = 0;
    int device = 0;
    std::string name;
    std::string uuid;
    std::string pci_bus;
    std::size_t vram_total_mib = 0;
    std::size_t vram_free_init_mib = 0;
    std::atomic<std::uint64_t> prefill_steps{0};
    std::atomic<std::uint64_t> decode_steps{0};
    std::atomic<std::uint64_t> allreduce_calls{0};
    std::atomic<std::uint64_t> allgather_calls{0};
    kernels::batch::P2pAllReduce2 p2p_ar;  // TP2 decode all-reduce over peer access (inactive: NCCL)
  };

  TpWorkerPool() {
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count < TP_SIZE) {
      throw std::runtime_error("TP" + std::to_string(TP_SIZE) + " requires " + std::to_string(TP_SIZE) +
                               " visible CUDA GPUs, found " + std::to_string(dev_count));
    }
    int devs[TP_SIZE];
    for (int r = 0; r < TP_SIZE; ++r) {
      devs[r] = r;
      CUDA_CHECK(cudaSetDevice(r));
      cudaDeviceProp prop{};
      CUDA_CHECK(cudaGetDeviceProperties(&prop, r));
      char pci[32];
      CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), r));
      std::size_t free_b = 0, tot_b = 0;
      CUDA_CHECK(cudaMemGetInfo(&free_b, &tot_b));
      info_[r].rank = r;
      info_[r].device = r;
      info_[r].name = prop.name;
      info_[r].uuid = format_uuid(prop.uuid);
      info_[r].pci_bus = pci;
      info_[r].vram_total_mib = tot_b >> 20;
      info_[r].vram_free_init_mib = free_b >> 20;
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[r], cudaStreamNonBlocking));
    }
    // Byte-exact startup P2P probe before enabling peer access / initializing NCCL.
    // Guards against Linux VT-d/AMD-Vi IOMMU translation-mode silent DMAR corruption where
    // cudaDeviceCanAccessPeer returns 1 and cudaMemcpyPeer returns cudaSuccess while corrupting bytes.
    auto& log = ingine::engine::server_log();
    const char* force_p2p_fail = std::getenv("INGINE_TP_FORCE_P2P_PROBE_FAIL");
    const bool force_fail = force_p2p_fail && force_p2p_fail[0] == '1';
    p2p_probe_ = probe_and_enable_tp_p2p(TP_SIZE, force_fail);
    if (p2p_probe_.corrupted) {
      // Fail closed: force NCCL off PCIe P2P onto host-staged SHM before ncclCommInitAll.
      setenv("NCCL_P2P_DISABLE", "1", 1);
      log.warn(
          "TP{}/EP{} P2P byte-exact probe FAILED on pair {}->{} ({}); disabling CUDA P2P and setting "
          "NCCL_P2P_DISABLE=1 (host-staged fallback). On Linux hosts with VT-d/AMD-Vi IOMMU enabled, "
          "check dmesg for DMAR faults and add 'iommu=pt' to the kernel cmdline to restore fast P2P.",
          TP_SIZE, TP_SIZE, p2p_probe_.first_bad_src, p2p_probe_.first_bad_dst, p2p_probe_.detail);
    } else if (p2p_probe_.p2p_enabled) {
      log.info("TP{}/EP{} P2P byte-exact probe: PASS ({}/{} directed pairs verified, 4096 B/pair)",
               TP_SIZE, TP_SIZE, p2p_probe_.verified_pairs, p2p_probe_.capable_pairs);
    } else {
      log.info("TP{}/EP{} P2P probe: 0 P2P-capable pairs (cross-socket/PCIe root); NCCL using host-staged SHM",
               TP_SIZE, TP_SIZE);
    }
    CUDA_CHECK(cudaSetDevice(0));
    NCCL_CHECK(ncclCommInitAll(comms_, TP_SIZE, devs));
    // TP2 decode all-reduces over peer access: a [B <= 8, H] FP32 vector a call, 96 slots (2 a visit); a 24 us NCCL
    // ring all-reduce over PCIe became one 8-block kernel (INGINE_TP_P2P_AR=0: NCCL).
    {
      const char* env = std::getenv("INGINE_TP_P2P_AR");
      if (TP_SIZE == 2 && p2p_probe_.p2p_enabled && !p2p_probe_.corrupted && !(env && env[0] == '0')) {
        constexpr int kSlots = 2 * NUM_VISITS + 1, kCap = 16 * HIDDEN;
        constexpr std::size_t nf = std::size_t(kSlots) * kBlocksAr;
        for (int r = 0; r < TP_SIZE; ++r) {
          CUDA_CHECK(cudaSetDevice(r));
          auto& a = info_[r].p2p_ar;
          CUDA_CHECK(cudaMalloc(&a.my_recv, std::size_t(kSlots) * kCap * sizeof(float)));
          CUDA_CHECK(cudaMalloc(&a.my_flag, nf * sizeof(unsigned)));
          CUDA_CHECK(cudaMalloc(&a.my_seq, nf * sizeof(unsigned)));
          CUDA_CHECK(cudaMemset(a.my_flag, 0, nf * sizeof(unsigned)));
          CUDA_CHECK(cudaMemset(a.my_seq, 0, nf * sizeof(unsigned)));
          a.slots = kSlots, a.cap = kCap;
        }
        for (int r = 0; r < TP_SIZE; ++r) {
          info_[r].p2p_ar.peer_recv = info_[1 - r].p2p_ar.my_recv;
          info_[r].p2p_ar.peer_flag = info_[1 - r].p2p_ar.my_flag;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        log.info("TP2 decode all-reduce: CUDA peer access ({} slots x {} floats), NCCL for prefill", kSlots, kCap);
      }
    }

    for (int r = 0; r < TP_SIZE; ++r) {
      const int e_per = EXPERTS / TP_SIZE;
      const int v0 = r * LOCAL_VOCAB_CHUNK;
      const int v1 = std::min(UNPADDED_VOCAB, v0 + LOCAL_VOCAB_CHUNK);
      log.info("TP{}/EP{} rank {}/{}: dev={} {} ({}, {}) Q=[{}..{}) KV=[{}..{}) experts=[{}..{}) vocab=[{}..{}) free={}MiB",
               TP_SIZE, TP_SIZE, r, TP_SIZE, info_[r].device, info_[r].name, info_[r].uuid, info_[r].pci_bus,
               r * HEADS, (r + 1) * HEADS, r * KV_HEADS, (r + 1) * KV_HEADS,
               INGINE_TP_ETP ? 0 : r * e_per, INGINE_TP_ETP ? EXPERTS : (r + 1) * e_per, v0, v1,
               info_[r].vram_free_init_mib);
      if (INGINE_TP_ETP)
        log.info("TP{} rank {}: expert tensor parallel, intermediate rows [{}..{}) of every expert", TP_SIZE, r,
                 r * INTER_LOCAL, (r + 1) * INTER_LOCAL);
    }

    for (int r = 0; r < TP_SIZE; ++r) {
      threads_[r] = std::thread([this, r] { worker_loop(r); });
    }
  }

  ~TpWorkerPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      ++gen_;
    }
    cv_start_.notify_all();
    for (int r = 0; r < TP_SIZE; ++r) {
      if (threads_[r].joinable()) threads_[r].join();
    }
    for (int r = 0; r < TP_SIZE; ++r) {
      cudaSetDevice(r);
      if (streams_[r]) cudaStreamSynchronize(streams_[r]);
      auto& a = info_[r].p2p_ar;
      if (a.my_recv) cudaFree(a.my_recv), cudaFree(a.my_flag), cudaFree(a.my_seq);
      if (comms_[r]) ncclCommDestroy(comms_[r]);
      if (streams_[r]) cudaStreamDestroy(streams_[r]);
    }
  }

  void run_all(std::function<void(int)> fn) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      fn_ = std::move(fn);
      for (auto& ep : errors_) ep = nullptr;
      done_count_ = 0;
      ++gen_;
    }
    cv_start_.notify_all();
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] { return done_count_ == TP_SIZE; });
    for (int r = 0; r < TP_SIZE; ++r) {
      if (errors_[r]) std::rethrow_exception(errors_[r]);
    }
  }

  cudaStream_t stream(int r) const { return streams_[r]; }
  ncclComm_t comm(int r) const { return comms_[r]; }
  RankInfo& info(int r) { return info_[r]; }
  const RankInfo& info(int r) const { return info_[r]; }

  std::string summary(const std::string& mode, const std::string& extra = "") const {
    std::ostringstream os;
    const char* p2p_state = p2p_probe_.corrupted ? "FALLBACK(NCCL_P2P_DISABLE)"
                            : (p2p_probe_.p2p_enabled ? "PASS" : "NONE(SHM)");
    os << "TP" << TP_SIZE << "/EP" << TP_SIZE << " (" << mode << ") p2p_probe=" << p2p_state;
    if (!extra.empty()) os << " " << extra;
    for (int r = 0; r < TP_SIZE; ++r) {
      const auto& in = info_[r];
      os << "\n  rank=" << r << " uuid=" << in.uuid << " pci=" << in.pci_bus
         << " Q=" << (r * HEADS) << ".." << ((r + 1) * HEADS - 1)
         << " KV=" << (r * KV_HEADS) << ".." << ((r + 1) * KV_HEADS - 1)
         << " experts=" << (INGINE_TP_ETP ? 0 : r * (EXPERTS / TP_SIZE)) << ".."
         << (INGINE_TP_ETP ? EXPERTS - 1 : (r + 1) * (EXPERTS / TP_SIZE) - 1)
         << " prefill_steps=" << in.prefill_steps.load()
         << " decode_steps=" << in.decode_steps.load()
         << " allreduce=" << in.allreduce_calls.load()
         << " allgather=" << in.allgather_calls.load();
    }
    return os.str();
  }

 private:
  void worker_loop(int r) {
    CUDA_CHECK(cudaSetDevice(r));
    std::uint64_t seen = 0;
    for (;;) {
      std::function<void(int)> task;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_start_.wait(lk, [&] { return gen_ != seen; });
        seen = gen_;
        if (stop_) return;
        task = fn_;
      }
      try {
        CUDA_CHECK(cudaSetDevice(r));
        task(r);
      } catch (...) {
        errors_[r] = std::current_exception();
      }
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (++done_count_ == TP_SIZE) cv_done_.notify_one();
      }
    }
  }

  std::array<RankInfo, TP_SIZE> info_;
  TpP2pProbeResult p2p_probe_;
  std::array<cudaStream_t, TP_SIZE> streams_{};
  ncclComm_t comms_[TP_SIZE] = {};
  std::array<std::thread, TP_SIZE> threads_;

  std::mutex mu_;
  std::condition_variable cv_start_, cv_done_;
  std::uint64_t gen_ = 0;
  int done_count_ = 0;
  bool stop_ = false;
  std::function<void(int)> fn_;
  std::array<std::exception_ptr, TP_SIZE> errors_{};
};

// ----------------------------------------------------------------------------- TpPrefiller (per-rank sharded prefill)

class TpPrefiller {
 public:
  TpPrefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream, ncclComm_t comm,
              TpWorkerPool::RankInfo& rinfo)
      : w_(w), abl_(abl), max_chunk_(max_chunk), stream_(stream), comm_(comm), rinfo_(rinfo),
        gemm_(std::make_unique<pk::Gemm>()) {
    const int L = WINDOW + max_chunk;
    kpos_rows_ = (L + pk::kAttnKeyTile - 1) / pk::kAttnKeyTile * pk::kAttnKeyTile;
    const std::size_t C = std::size_t(max_chunk);
    auto alloc = [&](std::size_t bytes) {
      storage_.emplace_back(bytes);
      storage_.back().fill(0);
      return storage_.back().p;
    };
    auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
    auto bf16 = [&](std::size_t n) { return static_cast<ActT*>(alloc(n * 2)); };
    auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * 4)); };

    B0_ = f32(C * H), B1_ = f32(C * H);
    for (int p = 0; p < PASSES; ++p) B2_[p] = f32(C * H), E_[p] = f32(C * H);
    hid_ = f32(C * H), part_ = f32(C * H), y_ = f32(C * H), y2_ = f32(C * H);
    msgs_ = f32(4 * C * H);
    mixf_ = f32(3 * C * H), xin_ = bf16(3 * C * H);
    mixm_ = f32(C * H), xm_ = bf16(C * H), xf_ = bf16(C * H);
    rel_ = bf16(C * HEADS * REL_EXTENT);
    proj_ = bf16(C * PROJ_ROWS_FULL);
    q_ = bf16(C * HEADS * HEAD_DIM);
    rel16_ = bf16(C * HEADS * D_REL);
    attn_out_ = bf16(C * HEADS * HEAD_DIM);
    wo_f32_ = f32(C * H);
    krows_ = bf16(std::size_t(kpos_rows_) * KV_DIM);
    vrows_ = bf16(std::size_t(kpos_rows_) * KV_DIM);
    kpos_ = i32(kpos_rows_);
    bounds_ = static_cast<int4*>(alloc(std::size_t(kpos_rows_ / pk::kAttnKeyTile) * sizeof(int4)));
    gin_ = bf16(3 * C * 2 * H);
    g16_ = bf16(C * GATE_RANK);
    fpre_ = f32(C * H), gpre_ = f32(C * H);
    router_logits_ = f32(C * EXPERTS);
    wts_ = f32(C * TOP_K);
    idx_ = i32(C * TOP_K);
    counts_ = i32(EXPERTS);
    tiles_n_ = i32(1);
    pair_tok_ = i32(C * TOP_K);
    pair_dst_ = i32(C * TOP_K);
    tiles_ = static_cast<int4*>(alloc(std::size_t(pk::moe_tiles_capacity(max_chunk)) * sizeof(int4)));
    act_ = bf16(C * TOP_K * INTER);
    down_ = bf16(C * TOP_K * H);
    moe_f32_ = f32(C * H);
    local_logits_ = f32(LOCAL_VOCAB_CHUNK);
    gathered_logits_ = f32(std::size_t(TP_SIZE) * LOCAL_VOCAB_CHUNK);
    wscratch_ = (w.dense_fp8 || w.dense_q8) ? bf16(std::max(std::size_t(PROJ_ROWS_FULL) * H, std::size_t(H) * HEADS * HEAD_DIM)) : nullptr;
    ids_ = static_cast<std::int64_t*>(alloc(C * 8));
  }

  void run(State& st, const std::int32_t* tokens, int T, int pos0, float* logits_out, Tensor* taps) {
    const Weights& w = w_;
    const cudaStream_t s = stream_;
    const std::size_t slab = std::size_t(T) * H;
    const float* attn_dirs = f32p(abl_.attn_dirs);
    const float* attn_w = f32p(abl_.attn_w);
    const float* mlp_dirs = f32p(abl_.mlp_dirs);
    const float* mlp_w = f32p(abl_.mlp_w);
    ActT* tap_out = taps ? static_cast<ActT*>(taps->data) : nullptr;

    auto mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write, const float* abl_r,
                   const float* abl_w, float* partial_out, const float* queries, int Q, const float* norm_w,
                   ActT* out_bf16, float* out_f32) {
      dense::MixParams p;
      p.blocks = blocks.empty() ? nullptr : block_table(blocks);
      p.nblocks = int(blocks.size());
      p.base = base;
      p.write = write;
      p.abl_r = write ? abl_r : nullptr;
      p.abl_d = write && abl_.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;
      p.abl_w = write ? abl_w : nullptr;
      p.partial_out = partial_out;
      p.queries = queries;
      p.Q = Q;
      p.norm_w = norm_w;
      p.out_bf16 = out_bf16;
      p.out_f32 = out_f32;
      p.M = T;
      dense::mix(p, false, s);
    };
    auto bf16_rows = [&](const Dense& d) -> const ActT* {
      if (d.blocked()) {
        dense::dequant_blocked(matrix(d), d.cols(), wscratch_, s);
        return wscratch_;
      }
      if (!d.fp8()) return bf16p(d.w);
      dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, s);
      return wscratch_;
    };
    auto gates = [&](const pk::GateRows& rows, int G, const Tensor& down, const Tensor& forget16, const Tensor& gain16,
                     const float* norm_w, ActT* out) {
      pk::gate_inputs(rows, G, T, gin_, s);
      for (int g = 0; g < G; ++g) {
        const ActT* gd = bf16p(down) + std::size_t(g) * GATE_RANK * 2 * H;
        gemm_->run(gin_ + std::size_t(g) * T * 2 * H, 2 * H, gd, 2 * H, g16_, GATE_RANK, false, T, GATE_RANK, 2 * H, s);
        gemm_->run(g16_, GATE_RANK, bf16p(forget16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, fpre_, H, true, T, H,
                   GATE_RANK, s);
        gemm_->run(g16_, GATE_RANK, bf16p(gain16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, gpre_, H, true, T, H,
                   GATE_RANK, s);
        pk::gate_apply(rows.h[g], rows.msg[g], fpre_, gpre_, norm_w + std::size_t(g) * H, T, out + std::size_t(g) * slab,
                       s);
      }
    };

    pk::widen_ids(tokens, T, ids_, IMAGE_PLACEHOLDER_ID, s);
    if (w.embed_q8()) dense::embed_q8(ids_, T, w.embed.data, f32p(w.embed_norm), B0_, false, s);
    else if (w.embed_q4()) dense::embed_q4(ids_, T, VOCAB, w.embed.data, f32p(w.embed_norm), B0_, false, s);
    else dense::embed(ids_, T, bf16p(w.embed), f32p(w.embed_norm), B0_, false, s);

    int prev_v = -1;
    int tap = 0;
    for (int v = 0; v < NUM_VISITS; ++v) {
      const int l = VISITS[v].layer, p = VISITS[v].pass;
      const Layer& layer = w.layers[l];
      const bool msgs = is_loop_layer(l) && p > 0;
      const bool boundary = l % BLOCK == 0;
      const std::vector<const float*> blocks = blocks_of(l, p, B0_, B1_, B2_, E_);

      float* hid;
      const float* base;
      const float* write;
      if (v == 0) {
        hid = B0_, base = B0_, write = nullptr;
      } else if (l == LOOP_START && p > 0) {
        hid = B1_, base = B1_, write = nullptr;
      } else {
        hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[2] : hid_;
        base = part_, write = y2_;
      }
      const int pv = prev_v >= 0 ? prev_v : 0;
      const float* r_prev = mlp_dirs + std::size_t(pv) * H;
      const float* w_prev = mlp_w + 4 * std::size_t(pv);
      if (msgs) {
        const int i = l - LOOP_START;
        const float* src_base = i == 0 ? part_ : E_[p - 1];
        const float* src_write = i == 0 ? y2_ : nullptr;
        mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
            f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
        if (i == 0) write = nullptr;
      }

      const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
      const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
      float* partial_out = hid != base ? hid : nullptr;
      if (msgs) {
        mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_);
        pk::GateRows rows;
        for (int g = 0; g < 3; ++g) rows.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_ + g * slab;
        gates(rows, 3, layer.qkv_gate_down, layer.qkv_gate_forget16, layer.qkv_gate_gain16, f32p(layer.attn_norm3), xin_);
      } else {
        mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
            f32p(layer.attn_norm_mix), xin_, nullptr);
      }
      if (tap < 3 && v == TAP_VISITS[tap]) {
        if (tap_out) pk::store_tap(hid, T, tap_out + std::size_t(tap) * H, std::int64_t(NUM_TAPS) * H, s);
        ++tap;
      }

      // ---- Sharded attention projections
      const int R = proj_rows(layer.full);
      const ActT* wp = bf16_rows(layer.wproj);
      const bool single = !layer.full && !msgs;
      if (single) {
        gemm_->run(xin_, H, wp, H, proj_, R, false, T, R, H, s);
      } else {
        struct Part { int begin, end, input; };
        const Part parts[] = {{SEG_BEGIN[SEG_Q], SEG_BEGIN[SEG_K], 0},
                              {SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], 1},
                              {SEG_BEGIN[SEG_V], SEG_BEGIN[SEG_K_DYN], 2},
                              {SEG_BEGIN[SEG_K_DYN], SEG_BEGIN[SEG_V_DYN], 1},
                              {SEG_BEGIN[SEG_V_DYN], SEG_BEGIN[NUM_SEGS], 2}};
        for (const Part& part : parts) {
          if (part.begin >= R) break;
          gemm_->run(xin_ + std::size_t(part.input) * slab, H, wp + std::size_t(part.begin) * H, H, proj_ + part.begin,
                     R, false, T, part.end - part.begin, H, s);
        }
      }

      // ---- Sharded attention (HEADS=6, KV_HEADS=2 per GPU)
      pk::AttnPrepParams ap;
      ap.proj = proj_, ap.ld = R, ap.T = T, ap.pos0 = pos0, ap.full = layer.full;
      ap.q_scale = f32p(layer.q_scale), ap.r_scale = f32p(layer.r_scale);
      ap.k_w = f32p(layer.full ? layer.k_static : layer.k_conv);
      ap.v_w = f32p(layer.full ? layer.v_static : layer.v_conv);
      ap.conv_k = st.conv_k_at(v), ap.conv_v = st.conv_v_at(v);
      ap.q = q_, ap.rel = rel16_;
      pk::AttnParams at;
      at.q = q_, at.T = T, at.pos0 = pos0, at.rel = rel_, at.proj = proj_, at.ld = R, at.out = attn_out_;
      if (layer.full && st.k_pool) {
        // TP batch pool: the chunk's rows go straight into the slot's pages, attention reads the pages
        const int fi = FULL_INDEX[v];
        ap.k_out = krows_, ap.v_out = vrows_, ap.out_fp8 = st.kv_fp8;
        pk::attn_prep(ap, s);
        bk::write_full_kv_rows(krows_, vrows_, pos0, T, fi, st.kv_fp8, st.kv_v4, st.pool_slot, st.pool_page_table,
                               st.pool_max_pages, st.k_pool, st.v_pool, s);
        pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, s);
        at.k = st.k_pool, at.v = st.v_pool, at.kv_fp8 = st.kv_fp8, at.kv_v4 = st.kv_v4;
        at.kv_page_row = st.pool_page_table + std::size_t(st.pool_slot) * st.pool_max_pages;
        at.kv_page_visit = fi, at.kv_page_visits = NUM_FULL_VISITS;
        at.L = pos0 + T, at.window = 0;
        pk::attention(at, s);
      } else if (layer.full) {
        const int fi = FULL_INDEX[v];
        const std::size_t es = st.kv_fp8 ? 1 : 2;
        void* kc = st.k_full_at(fi).data;
        void* vc = st.v_full_at(fi).data;
        ap.k_out = static_cast<char*>(kc) + std::size_t(pos0) * KV_DIM * es;
        ap.v_out = static_cast<char*>(vc) + std::size_t(pos0) * KV_DIM * es;
        ap.out_fp8 = st.kv_fp8;
        pk::attn_prep(ap, s);
        pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, s);
        at.k = kc, at.v = vc, at.kv_fp8 = st.kv_fp8, at.L = pos0 + T, at.window = 0;
        pk::attention(at, s);
      } else {
        const int ri = RING_INDEX[v];
        pk::ring_gather(bf16p(st.k_ring_at(ri)), bf16p(st.v_ring_at(ri)), st.ring_pos_at(ri), pos0, T, krows_, vrows_,
                        kpos_, kpos_rows_, s);
        ap.k_out = krows_ + std::size_t(WINDOW) * KV_DIM;
        ap.v_out = vrows_ + std::size_t(WINDOW) * KV_DIM;
        pk::attn_prep(ap, s);
        pk::rel_logits(rel16_, f32p(layer.rel_proj), T, rel_, s);
        const int L = ((WINDOW + T + pk::kAttnKeyTile - 1) / pk::kAttnKeyTile) * pk::kAttnKeyTile;
        pk::key_tile_bounds(kpos_, L, bounds_, s);
        at.k = krows_, at.v = vrows_, at.kpos = kpos_, at.bounds = bounds_, at.L = L, at.window = WINDOW;
        pk::attention(at, s);
        pk::ring_update(krows_, vrows_, pos0, T, static_cast<ActT*>(st.k_ring_at(ri).data),
                        static_cast<ActT*>(st.v_ring_at(ri).data), st.ring_pos_at(ri), s);
      }
      pk::conv_state_update(st.conv_k_at(v), proj_ + SEG_BEGIN[SEG_K], R, T, KV_DIM, s);
      pk::conv_state_update(st.conv_v_at(v), proj_ + SEG_BEGIN[SEG_V], R, T, KV_DIM, s);

      // ---- Sharded wo + NCCL AllReduce (FP32 [T, H]) + static_conv
      const ActT* wo = bf16_rows(layer.wo);
      gemm_->run(attn_out_, HEADS * HEAD_DIM, wo, HEADS * HEAD_DIM, wo_f32_, H, true, T, H, HEADS * HEAD_DIM, s);
      NCCL_CHECK(ncclAllReduce(wo_f32_, wo_f32_, std::size_t(T) * H, ncclFloat32, ncclSum, comm_, s));
      rinfo_.allreduce_calls.fetch_add(1, std::memory_order_relaxed);
      pk::static_conv_f32(wo_f32_, H, T, H, f32p(layer.attn_conv), st.conv_attn_at(v), y_, s);
      pk::conv_state_update_f32(st.conv_attn_at(v), wo_f32_, H, T, H, s);

      // ---- MLP read + gate
      std::vector<const float*> mlp_blocks = blocks;
      if (boundary) mlp_blocks.push_back(hid);
      const float* mbase = boundary ? nullptr : hid;
      if (msgs) {
        mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
            f32p(layer.mlp_query), 1, nullptr, nullptr, mixm_);
        pk::GateRows rows;
        rows.h[0] = mixm_, rows.msg[0] = msgs_ + 3 * slab;
        gates(rows, 1, layer.mlp1_gate_down, layer.mlp1_gate_forget16, layer.mlp1_gate_gain16, f32p(layer.mlp_norm1), xm_);
      } else {
        mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
            f32p(layer.mlp_query), 1, f32p(layer.mlp_norm1), xm_, nullptr);
      }

      // ---- EP4 MoE + NCCL AllReduce (FP32 [T, H]) + static_conv
      gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, T, EXPERTS, H, s);
      CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), s));
      pk::moe_topk(router_logits_, f32p(layer.router_bias), T, idx_, wts_, counts_, s);
      const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_tok_, pair_dst_};
      pk::moe_route(idx_, T, work, s);
      const Experts& ex = layer.experts;
      pk::MoeExperts me;
      me.fmt = static_cast<int>(ex.w13_fmt);
      me.w2_fmt = static_cast<int>(ex.w2_fmt);
      me.w13_factor = f32p(ex.w13_factor);
      me.w2_factor = f32p(ex.w2_factor);
      me.w13_ptrs = static_cast<const std::uint8_t* const*>(ex.w13_ptrs.data);
      me.w13_scale_ptrs = static_cast<const std::uint8_t* const*>(ex.w13_scale_ptrs.data);
      me.w2_ptrs = static_cast<const std::uint8_t* const*>(ex.w2_ptrs.data);
      me.w2_scale_ptrs = static_cast<const std::uint8_t* const*>(ex.w2_scale_ptrs.data);
      me.active_mask = static_cast<const std::uint8_t*>(ex.active_mask.data);
      pk::moe_gate_up(me, xm_, T, work, act_, s);
      pk::moe_down(me, act_, T, work, down_, s);
      pk::moe_combine_ep(down_, wts_, idx_, me.active_mask, T, moe_f32_, s);
      NCCL_CHECK(ncclAllReduce(moe_f32_, moe_f32_, std::size_t(T) * H, ncclFloat32, ncclSum, comm_, s));
      rinfo_.allreduce_calls.fetch_add(1, std::memory_order_relaxed);
      pk::static_conv_f32(moe_f32_, H, T, H, f32p(layer.mlp_conv), st.conv_mlp_at(v), y2_, s);
      pk::conv_state_update_f32(st.conv_mlp_at(v), moe_f32_, H, T, H, s);
      prev_v = v;
    }

    // ---- Final read + sharded LM head + AllGather
    {
      dense::MixParams fp;
      const std::vector<const float*> fblocks = {B0_, B1_, B2_[2], E_[2]};
      fp.blocks = block_table(fblocks);
      fp.nblocks = int(fblocks.size());
      fp.base = part_;
      fp.write = y2_;
      fp.abl_r = mlp_dirs + std::size_t(prev_v) * H;
      fp.abl_d = abl_.detect ? fp.abl_r + std::size_t(NUM_VISITS) * H : nullptr;
      fp.abl_w = mlp_w + 4 * std::size_t(prev_v);
      fp.queries = f32p(w.final_query);
      fp.Q = 1;
      fp.norm_w = f32p(w.final_norm);
      fp.out_bf16 = xf_;
      fp.M = T;
      if (tap_out) {
        fp.out_copy = tap_out + std::size_t(tap) * H;
        fp.out_copy_stride = std::int64_t(NUM_TAPS) * H;
      }
      dense::mix(fp, false, s);
    }
    if (w.unembed.blocked())  // (Q6_K head kept as int8 codes: the block-table GEMV)
      dense::head(xf_ + std::size_t(T - 1) * H, 1, matrix(w.unembed), local_logits_, false, s);
    else
      pk::head_logits(xf_ + std::size_t(T - 1) * H, w.unembed.w.data, f32p(w.unembed.scale), LOCAL_VOCAB_CHUNK,
                      local_logits_, s);
    NCCL_CHECK(ncclAllGather(local_logits_, gathered_logits_, LOCAL_VOCAB_CHUNK, ncclFloat32, comm_, s));
    rinfo_.allgather_calls.fetch_add(1, std::memory_order_relaxed);
    rinfo_.prefill_steps.fetch_add(1, std::memory_order_relaxed);
    if (logits_out) {
      CUDA_CHECK(cudaMemcpyAsync(logits_out, gathered_logits_, std::size_t(UNPADDED_VOCAB) * sizeof(float),
                                 cudaMemcpyDeviceToDevice, s));
    }
    st.length = pos0 + T;
  }

 private:
  const float* const* block_table(const std::vector<const float*>& blocks) {
    auto it = tables_.find(blocks);
    if (it == tables_.end()) {
      DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
      if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
      it = tables_.emplace(blocks, std::move(b)).first;
    }
    return static_cast<const float* const*>(it->second.p);
  }

  const Weights& w_;
  const Ablation& abl_;
  int max_chunk_;
  cudaStream_t stream_;
  ncclComm_t comm_;
  TpWorkerPool::RankInfo& rinfo_;
  std::unique_ptr<pk::Gemm> gemm_;
  int kpos_rows_ = 0;
  std::vector<DeviceBuffer> storage_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;

  float *B0_, *B1_, *B2_[PASSES], *E_[PASSES], *hid_, *part_, *y_, *y2_, *msgs_;
  float *mixf_, *mixm_, *wo_f32_, *fpre_, *gpre_, *router_logits_, *wts_, *moe_f32_;
  float *local_logits_, *gathered_logits_;
  ActT *xin_, *xm_, *xf_, *rel_, *proj_, *q_, *rel16_, *attn_out_, *krows_, *vrows_;
  ActT *gin_, *g16_, *act_, *down_, *wscratch_;
  int *kpos_, *idx_, *counts_, *tiles_n_, *pair_tok_, *pair_dst_;
  int4 *bounds_, *tiles_;
  std::int64_t* ids_;
};

// ----------------------------------------------------------------------------- TpBatchStatePool (per-rank sharded KV)

struct TpBatchStatePool {
  int max_batch = 1;
  int max_ctx = 0;
  bool kv_fp8 = false;
  int max_pages_per_slot = 0;
  int num_pages = 0;

  std::size_t ring_slot_stride = std::size_t(NUM_SLIDING_VISITS) * WINDOW * KV_DIM;
  std::size_t ring_pos_slot_stride = std::size_t(NUM_SLIDING_VISITS) * WINDOW;
  std::size_t conv_kv_slot_stride = std::size_t(NUM_VISITS) * (CONV_K - 1) * KV_DIM;
  std::size_t conv_h_slot_stride = std::size_t(NUM_VISITS) * (CONV_K - 1) * HIDDEN;

  DeviceBuffer k_ring, v_ring, ring_pos;
  DeviceBuffer conv_k, conv_v, conv_attn, conv_mlp;
  DeviceBuffer slot_pos;
  DeviceBuffer k_full_pool, v_full_pool, page_table;
  bool kv_v4 = false;  // FP4 V pages (FP8 K)

  // Bytes of one 64-token page across the full visits: K (FP8 or 16-bit) + V (FP4, FP8 or 16-bit).
  static std::size_t page_bytes(bool fp8, bool v4) {
    const std::size_t rows = std::size_t(NUM_FULL_VISITS) * bk::kPageSize * KV_DIM;
    const std::size_t es = fp8 ? 1 : 2;
    return rows * es + (v4 ? rows / 2 : rows * es);
  }

  void init_buffers(int max_b, int ctx, bool fp8, bool v4, int pages) {
    max_batch = max_b;
    max_ctx = ctx;
    kv_fp8 = fp8;
    kv_v4 = v4;
    max_pages_per_slot = (ctx + bk::kPageSize - 1) / bk::kPageSize;
    num_pages = pages;

    k_ring = DeviceBuffer(std::size_t(max_b) * ring_slot_stride * sizeof(ActT));
    v_ring = DeviceBuffer(std::size_t(max_b) * ring_slot_stride * sizeof(ActT));
    ring_pos = DeviceBuffer(std::size_t(max_b) * ring_pos_slot_stride * sizeof(std::int64_t));
    conv_k = DeviceBuffer(std::size_t(max_b) * conv_kv_slot_stride * sizeof(float));
    conv_v = DeviceBuffer(std::size_t(max_b) * conv_kv_slot_stride * sizeof(float));
    conv_attn = DeviceBuffer(std::size_t(max_b) * conv_h_slot_stride * sizeof(float));
    conv_mlp = DeviceBuffer(std::size_t(max_b) * conv_h_slot_stride * sizeof(float));
    slot_pos = DeviceBuffer(std::size_t(max_b) * sizeof(std::int64_t));

    k_ring.fill(0); v_ring.fill(0); ring_pos.fill(0xff);
    conv_k.fill(0); conv_v.fill(0); conv_attn.fill(0); conv_mlp.fill(0); slot_pos.fill(0);

    // No max_ctx prefill scratch: prefill writes its rows straight into the slot's pages.
    const std::size_t es = fp8 ? 1 : 2;
    const std::size_t page_kv_bytes = std::size_t(NUM_FULL_VISITS) * bk::kPageSize * KV_DIM * es;
    k_full_pool = DeviceBuffer(std::size_t(num_pages) * page_kv_bytes);
    v_full_pool = DeviceBuffer(std::size_t(num_pages) * (v4 ? page_kv_bytes / 2 : page_kv_bytes));
    k_full_pool.fill(0); v_full_pool.fill(0);

    page_table = DeviceBuffer(std::size_t(max_b) * max_pages_per_slot * sizeof(std::int32_t));
    page_table.fill(0xff);
  }

  static int compute_fit_pages(int max_b, int ctx, bool fp8, int max_kv_tokens) {
    const int max_pages_per_slot = (ctx + bk::kPageSize - 1) / bk::kPageSize;
    const int max_possible_pages = max_b * max_pages_per_slot;
    const std::size_t bytes_per_page = page_bytes(fp8, false);
    if (max_kv_tokens > 0) {
      return std::min(max_possible_pages, std::max(max_pages_per_slot, (max_kv_tokens + bk::kPageSize - 1) / bk::kPageSize));
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // Reserve ~3.5 GiB for sliding rings/scratch/activations/NCCL/CUDA graphs (minus scratch already allocated before calling this)
    const std::size_t reserve = std::min<std::size_t>(std::size_t(3072) << 20, free_bytes / 4);
    const std::size_t avail = free_bytes > reserve ? free_bytes - reserve : 0;
    const int fit = int(avail / bytes_per_page);
    return std::min(max_possible_pages, std::max(std::min(max_possible_pages, max_pages_per_slot), fit));
  }

  State prefill_view(int s) const {
    State st;
    st.max_ctx = max_ctx;
    st.kv_fp8 = kv_fp8;
    const DType full_dt = kv_fp8 ? DType::FP8_E4M3FN : kActDType;
    st.k_ring = rm(static_cast<ActT*>(k_ring.p) + std::size_t(s) * ring_slot_stride, kActDType,
                   {NUM_SLIDING_VISITS, WINDOW, KV_HEADS, HEAD_DIM});
    st.v_ring = rm(static_cast<ActT*>(v_ring.p) + std::size_t(s) * ring_slot_stride, kActDType,
                   {NUM_SLIDING_VISITS, WINDOW, KV_HEADS, HEAD_DIM});
    st.ring_pos = rm(static_cast<std::int64_t*>(ring_pos.p) + std::size_t(s) * ring_pos_slot_stride, DType::I64,
                     {NUM_SLIDING_VISITS, WINDOW});
    (void)full_dt;
    st.k_pool = k_full_pool.p, st.v_pool = v_full_pool.p, st.kv_v4 = kv_v4;
    st.pool_page_table = static_cast<const std::int32_t*>(page_table.p);
    st.pool_slot = s, st.pool_max_pages = max_pages_per_slot;
    st.conv_k = rm(static_cast<float*>(conv_k.p) + std::size_t(s) * conv_kv_slot_stride, DType::FP32,
                   {NUM_VISITS, CONV_K - 1, KV_DIM});
    st.conv_v = rm(static_cast<float*>(conv_v.p) + std::size_t(s) * conv_kv_slot_stride, DType::FP32,
                   {NUM_VISITS, CONV_K - 1, KV_DIM});
    st.conv_attn = rm(static_cast<float*>(conv_attn.p) + std::size_t(s) * conv_h_slot_stride, DType::FP32,
                      {NUM_VISITS, CONV_K - 1, HIDDEN});
    st.conv_mlp = rm(static_cast<float*>(conv_mlp.p) + std::size_t(s) * conv_h_slot_stride, DType::FP32,
                     {NUM_VISITS, CONV_K - 1, HIDDEN});
    st.length = 0;
    return st;
  }
};

// ----------------------------------------------------------------------------- TpBatchDecoder (per-rank sharded B=1..max_batch)

struct TpVerifyStepInput {
  int R_cap = 0;
  int S = 0;
  std::array<std::int64_t, 16> tokens{};
  std::array<std::int32_t, kd::kMaxBatchSlots> active_slots{};
  std::array<std::int32_t, kd::kMaxBatchSlots> slot_row0{};
  std::array<std::int32_t, kd::kMaxBatchSlots> slot_m{};
  std::array<std::int32_t, 16> row_slot{};
  std::array<std::int32_t, 16> row_off{};
  std::array<std::int32_t, 16> row_base{};
  std::array<bk::BannedList, 16> banned{};
};

class TpBatchDecoder {
 public:
  TpBatchDecoder(const Weights& w, const Ablation& abl, TpBatchStatePool& pool, int max_batch, int rank,
                 ncclComm_t comm, TpWorkerPool::RankInfo& rinfo, bool spec_enabled = false, int spec_max_m = 4)
      : w_(w), abl_(abl), pool_(pool), max_batch_(max_batch), spec_enabled_(spec_enabled),
        spec_max_m_(spec_enabled ? spec_max_m : 1),
        max_rows_(spec_enabled ? max_batch * spec_max_m : max_batch),
        rank_(rank), comm_(comm), rinfo_(rinfo),
        gemm_(std::make_unique<pk::Gemm>()),
        tokens_host_(std::size_t(max_rows_) * sizeof(std::int64_t)),
        slots_host_(std::size_t(max_rows_) * sizeof(std::int32_t)),
        banned_host_(std::size_t(max_rows_) * sizeof(bk::BannedList)),
        verify_host_(spec_enabled ? sizeof(TpVerifyStepInput) : 64),
        commit_host_(std::size_t(std::max(1, max_batch)) * sizeof(std::int32_t)) {
    auto alloc = [&](std::size_t bytes) {
      storage_.emplace_back(bytes);
      storage_.back().fill(0);
      return storage_.back().p;
    };
    const std::size_t B_slot = std::size_t(max_batch);
    const std::size_t B = std::size_t(max_rows_);
    auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * sizeof(float))); };
    auto bf16 = [&](std::size_t n) { return static_cast<ActT*>(alloc(n * sizeof(ActT))); };
    auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * sizeof(int))); };

    tokens_dev_ = static_cast<std::int64_t*>(alloc(B * sizeof(std::int64_t)));
    slot_ids_dev_ = static_cast<std::int32_t*>(alloc(B * sizeof(std::int32_t)));
    B0_ = f32(B * H), B1_ = f32(B * H);
    for (int p = 0; p < PASSES; ++p) B2_[p] = f32(B * H), E_[p] = f32(B * H);
    hid_ = f32(B * H), part_ = f32(B * H), y_ = f32(B * H), y2_ = f32(B * H);
    const bool blk_dense = w.layers[0].wproj.blocked();
    const bool need_gemm_gates = blk_dense ? (B > 16) : (B > 8);
    xin_ = bf16(3 * B * H), xm_ = bf16(B * H), xf_ = bf16(B * H);
    xin_chunk_ = (w.dense_fp8 && B > 8) ? bf16(3 * 8 * H) : (blk_dense && B > 16 ? bf16(3 * 16 * H) : nullptr);
    mixf_ = f32(3 * B * H), msgs_ = f32(4 * B * H), mixm_ = f32(B * H);
    proj_fp32_ = f32(std::min<std::size_t>(B, (w.dense_fp8 || blk_dense) ? 32 : 8) * PROJ_ROWS_FULL);
    proj_act_ = need_gemm_gates ? bf16(B * PROJ_ROWS_FULL) : nullptr;
    attn_out_ = bf16(B * HEADS * HEAD_DIM);
    wo_fp32_ = f32(B * H);
    gin_ = need_gemm_gates ? bf16(3 * B * 2 * H) : nullptr;
    g16_ = need_gemm_gates ? bf16(B * GATE_RANK) : nullptr;
    fpre_ = need_gemm_gates ? f32(B * H) : nullptr;
    gpre_ = need_gemm_gates ? f32(B * H) : nullptr;
    router_logits_ = f32(B * EXPERTS);
    idx_ = i32(B * TOP_K);
    wts_ = f32(B * TOP_K);
    counts_ = i32(EXPERTS);
    tiles_n_ = i32(1);
    pair_token_ = i32(B * TOP_K);
    pair_dst_ = i32(B * TOP_K);
    tiles_ = static_cast<int4*>(alloc(std::size_t(pk::moe_tiles_capacity(max_rows_)) * sizeof(int4)));
    route_done_ = static_cast<unsigned*>(alloc(sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(route_done_, 0, sizeof(unsigned)));
    act_ = bf16(B * TOP_K * INTER);
    down_ = bf16(B * TOP_K * H);
    moe_f32_ = f32(B * H);
    local_logits_ = f32(B * LOCAL_VOCAB_CHUNK);
    // gathered_logits_ / logits_ are only used by non-spec temperature>0 run_step (<= max_batch_ rows).
    gathered_logits_ = f32(std::size_t(TP_SIZE) * B_slot * LOCAL_VOCAB_CHUNK);
    logits_ = (rank_ == 0) ? f32(B_slot * UNPADDED_VOCAB) : nullptr;
    if (max_rows_ <= 128) {
      banned_dev_ = static_cast<bk::BannedList*>(alloc(B * sizeof(bk::BannedList)));
      argmax_ws_ = alloc(bk::p2p_argmax2_workspace_bytes(max_rows_));
      argmax_out_dev_ = static_cast<std::int32_t*>(alloc(B * sizeof(std::int32_t)));
    }
    head_act_ = B > 8 ? bf16(B * kHeadChunk) : nullptr;
    wscratch_ = (w.dense_fp8 && B > 32) ? bf16(std::size_t(kHeadChunk) * H)
                : (w.dense_q8 && B > 32) ? bf16(std::max(std::size_t(PROJ_ROWS_FULL) * H, std::size_t(H) * HEADS * HEAD_DIM))
                                         : nullptr;
    attn_ws_ = DeviceBuffer(bk::attn_workspace_bytes(max_rows_));
    attn_ws_.fill(0);  // (split-attention tickets start at zero)

    if (spec_enabled_) {
      if (rank_ == 0) taps_dev_ = bf16(B * NUM_TAPS * H);
      step_kv_dev_ = bf16(B * 2 * KV_DIM);
      hist_k_dev_ = f32(std::size_t(NUM_VISITS) * B * KV_DIM);
      hist_v_dev_ = f32(std::size_t(NUM_VISITS) * B * KV_DIM);
      hist_attn_dev_ = f32(std::size_t(NUM_VISITS) * B * H);
      hist_mlp_dev_ = f32(std::size_t(NUM_VISITS) * B * H);

      verify_backup_.conv_slot_stride[0] = std::size_t(NUM_VISITS) * 2 * KV_DIM;
      verify_backup_.conv_slot_stride[1] = std::size_t(NUM_VISITS) * 2 * KV_DIM;
      verify_backup_.conv_slot_stride[2] = std::size_t(NUM_VISITS) * 2 * H;
      verify_backup_.conv_slot_stride[3] = std::size_t(NUM_VISITS) * 2 * H;
      verify_backup_.conv[0] = f32(B_slot * verify_backup_.conv_slot_stride[0]);
      verify_backup_.conv[1] = f32(B_slot * verify_backup_.conv_slot_stride[1]);
      verify_backup_.conv[2] = f32(B_slot * verify_backup_.conv_slot_stride[2]);
      verify_backup_.conv[3] = f32(B_slot * verify_backup_.conv_slot_stride[3]);
      verify_backup_.kv_slot_stride = std::size_t(NUM_SLIDING_VISITS) * bk::kMaxVerifyRows * KV_DIM;
      verify_backup_.pos_slot_stride = std::size_t(NUM_SLIDING_VISITS) * bk::kMaxVerifyRows;
      verify_backup_.k = bf16(B_slot * verify_backup_.kv_slot_stride);
      verify_backup_.v = bf16(B_slot * verify_backup_.kv_slot_stride);
      verify_backup_.pos = static_cast<std::int64_t*>(alloc(B_slot * verify_backup_.pos_slot_stride * sizeof(std::int64_t)));

      state_ptrs_.conv[0] = static_cast<float*>(pool_.conv_k.p);
      state_ptrs_.conv[1] = static_cast<float*>(pool_.conv_v.p);
      state_ptrs_.conv[2] = static_cast<float*>(pool_.conv_attn.p);
      state_ptrs_.conv[3] = static_cast<float*>(pool_.conv_mlp.p);
      state_ptrs_.conv_slot_stride[0] = pool_.conv_kv_slot_stride;
      state_ptrs_.conv_slot_stride[1] = pool_.conv_kv_slot_stride;
      state_ptrs_.conv_slot_stride[2] = pool_.conv_h_slot_stride;
      state_ptrs_.conv_slot_stride[3] = pool_.conv_h_slot_stride;
      state_ptrs_.k_ring = static_cast<ActT*>(pool_.k_ring.p);
      state_ptrs_.v_ring = static_cast<ActT*>(pool_.v_ring.p);
      state_ptrs_.ring_pos = static_cast<std::int64_t*>(pool_.ring_pos.p);
      state_ptrs_.ring_slot_stride = pool_.ring_slot_stride;
      state_ptrs_.ring_pos_slot_stride = pool_.ring_pos_slot_stride;
      state_ptrs_.num_visits = NUM_VISITS;
      state_ptrs_.num_sliding_visits = NUM_SLIDING_VISITS;

      active_slots_dev_ = i32(B_slot);
      slot_row0_dev_ = i32(B_slot);
      slot_m_dev_ = i32(B_slot);
      commit_counts_dev_ = i32(B_slot);
      row_slot_dev_ = i32(B);
      row_off_dev_ = i32(B);
      row_base_dev_ = i32(B);
    }

    segments_[0] = upload_ints(proj_layout(false, true));
    segments_[1] = upload_ints(proj_layout(false, false));
    full_segments_ = upload_ints(proj_layout(true, false));
  }

  ~TpBatchDecoder() { clear_graphs(); }

  bool p2p_argmax_enabled() const {
    static const bool env_on = [] {
      const char* e = std::getenv("INGINE_TP_P2P_ARGMAX");
      return !(e && e[0] == '0');
    }();
    return env_on && TP_SIZE == 2 && rinfo_.p2p_ar.active() && 2 * NUM_VISITS < rinfo_.p2p_ar.slots &&
           argmax_ws_ != nullptr && 2 * max_rows_ <= rinfo_.p2p_ar.cap;
  }

  const std::int32_t* argmax_out_dev() const { return argmax_out_dev_; }
  const ActT* taps_dev() const { return taps_dev_; }
  const std::int32_t* active_slots_dev() const { return active_slots_dev_; }
  const std::int32_t* slot_row0_dev() const { return slot_row0_dev_; }
  const std::int32_t* slot_m_dev() const { return slot_m_dev_; }

  void warmup_verify_bucket(int R_cap, cudaStream_t stream) {
    TpVerifyStepInput in{};
    in.R_cap = R_cap;
    in.S = 0;
    in.active_slots.fill(-1);
    in.slot_row0.fill(0);
    in.slot_m.fill(0);
    in.row_slot.fill(-1);
    in.row_off.fill(0);
    in.row_base.fill(0);
    upload_verify_input(in, R_cap, stream);
    decode_verify_eager(R_cap, stream);
  }

  bool capture_verify_bucket(int R_cap, cudaStream_t stream) {
    cudaGraph_t graph = nullptr;
    const cudaError_t berr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (berr != cudaSuccess) return false;
    try {
      decode_verify_eager(R_cap, stream);
    } catch (...) {
      cudaStreamEndCapture(stream, &graph);
      if (graph) cudaGraphDestroy(graph);
      return false;
    }
    const cudaError_t eerr = cudaStreamEndCapture(stream, &graph);
    if (eerr != cudaSuccess || !graph) {
      if (graph) cudaGraphDestroy(graph);
      return false;
    }
    cudaGraphExec_t exec = nullptr;
    const cudaError_t ierr = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (ierr != cudaSuccess || !exec) return false;
    CUDA_CHECK(cudaGraphUpload(exec, stream));
    auto old = graphs_verify_.find(R_cap);
    if (old != graphs_verify_.end() && old->second) cudaGraphExecDestroy(old->second);
    graphs_verify_[R_cap] = exec;
    return true;
  }

  void run_verify_step(const TpVerifyStepInput& in, int R_cap, cudaStream_t stream) {
    upload_verify_input(in, R_cap, stream);
    if (const auto it = graphs_verify_.find(R_cap); it != graphs_verify_.end()) {
      CUDA_CHECK(cudaGraphLaunch(it->second, stream));
    } else {
      decode_verify_eager(R_cap, stream);
    }
    rinfo_.decode_steps.fetch_add(1, std::memory_order_relaxed);
    rinfo_.allreduce_calls.fetch_add(2 * NUM_VISITS, std::memory_order_relaxed);
  }

  void run_verify_rollback(int S, const std::int32_t* commit_dev, cudaStream_t stream) {
    const float* hist[4] = {hist_k_dev_, hist_v_dev_, hist_attn_dev_, hist_mlp_dev_};
    bk::batch_verify_rollback(state_ptrs_, verify_backup_, hist, max_rows_, active_slots_dev_, slot_row0_dev_,
                              slot_m_dev_, commit_dev, S, static_cast<const std::int64_t*>(pool_.slot_pos.p), stream);
    bk::batched_advance_pos_by(active_slots_dev_, commit_dev, S, static_cast<std::int64_t*>(pool_.slot_pos.p), stream);
  }

  void run_verify_rollback_host(int S, const std::int32_t* commit_h, cudaStream_t stream) {
    std::memcpy(commit_host_.data(), commit_h, std::size_t(S) * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpyAsync(commit_counts_dev_, commit_host_.data(), std::size_t(S) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    run_verify_rollback(S, commit_counts_dev_, stream);
  }

  void warmup_bucket(int B, cudaStream_t stream, bool greedy = false) {
    std::vector<std::int64_t> ztok(max_batch_, 0);
    std::vector<std::int32_t> nslot(max_batch_, -1);
    CUDA_CHECK(cudaMemcpyAsync(tokens_dev_, ztok.data(), ztok.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_ids_dev_, nslot.data(), nslot.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    if (greedy && banned_dev_) {
      std::vector<bk::BannedList> zb(max_batch_);
      CUDA_CHECK(cudaMemcpyAsync(banned_dev_, zb.data(), zb.size() * sizeof(bk::BannedList), cudaMemcpyHostToDevice, stream));
    }
    decode_eager(B, greedy, stream);
  }

  bool capture_bucket(int B, cudaStream_t stream, bool greedy = false) {
    cudaGraph_t graph = nullptr;
    const cudaError_t berr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (berr != cudaSuccess) return false;
    try {
      decode_eager(B, greedy, stream);
    } catch (...) {
      cudaStreamEndCapture(stream, &graph);
      if (graph) cudaGraphDestroy(graph);
      return false;
    }
    const cudaError_t eerr = cudaStreamEndCapture(stream, &graph);
    if (eerr != cudaSuccess || !graph) {
      if (graph) cudaGraphDestroy(graph);
      return false;
    }
    cudaGraphExec_t exec = nullptr;
    const cudaError_t ierr = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (ierr != cudaSuccess || !exec) return false;
    CUDA_CHECK(cudaGraphUpload(exec, stream));
    auto& gm = greedy ? graphs_greedy_ : graphs_;
    auto old = gm.find(B);
    if (old != gm.end() && old->second) cudaGraphExecDestroy(old->second);
    gm[B] = exec;
    return true;
  }

  void clear_graphs() {
    for (auto& [b, exec] : graphs_)
      if (exec) cudaGraphExecDestroy(exec);
    graphs_.clear();
    for (auto& [b, exec] : graphs_greedy_)
      if (exec) cudaGraphExecDestroy(exec);
    graphs_greedy_.clear();
    for (auto& [b, exec] : graphs_verify_)
      if (exec) cudaGraphExecDestroy(exec);
    graphs_verify_.clear();
  }

  bool has_logits_graph(int B_active) const { return graphs_.lower_bound(B_active) != graphs_.end(); }

  const float* run_step(std::span<const BatchSlotStep> inputs, cudaStream_t stream) {
    const int B_active = int(inputs.size());
    auto it = graphs_.lower_bound(B_active);
    const int B_cap = (it != graphs_.end()) ? it->first : B_active;

    auto* ht = static_cast<std::int64_t*>(tokens_host_.data());
    auto* hs = static_cast<std::int32_t*>(slots_host_.data());
    for (int i = 0; i < B_active; ++i) {
      ht[i] = inputs[i].token;
      hs[i] = inputs[i].slot;
    }
    for (int i = B_active; i < B_cap; ++i) {
      ht[i] = 0;
      hs[i] = -1;
    }
    CUDA_CHECK(cudaMemcpyAsync(tokens_dev_, ht, std::size_t(B_cap) * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_ids_dev_, hs, std::size_t(B_cap) * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));

    if (it != graphs_.end()) {
      CUDA_CHECK(cudaGraphLaunch(it->second, stream));
    } else {
      decode_eager(B_active, false, stream);
    }
    rinfo_.decode_steps.fetch_add(1, std::memory_order_relaxed);
    rinfo_.allreduce_calls.fetch_add(2 * NUM_VISITS, std::memory_order_relaxed);
    rinfo_.allgather_calls.fetch_add(1, std::memory_order_relaxed);
    return logits_;
  }

  const std::int32_t* run_step_greedy(std::span<const BatchSlotStep> inputs, std::span<const bk::BannedList> banned,
                                      cudaStream_t stream) {
    const int B_active = int(inputs.size());
    auto it = graphs_greedy_.lower_bound(B_active);
    const int B_cap = (it != graphs_greedy_.end()) ? it->first : B_active;

    auto* ht = static_cast<std::int64_t*>(tokens_host_.data());
    auto* hs = static_cast<std::int32_t*>(slots_host_.data());
    auto* hb = static_cast<bk::BannedList*>(banned_host_.data());
    for (int i = 0; i < B_active; ++i) {
      ht[i] = inputs[i].token;
      hs[i] = inputs[i].slot;
      hb[i] = banned[i];
    }
    for (int i = B_active; i < B_cap; ++i) {
      ht[i] = 0;
      hs[i] = -1;
      hb[i] = bk::BannedList{};
    }
    CUDA_CHECK(cudaMemcpyAsync(tokens_dev_, ht, std::size_t(B_cap) * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_ids_dev_, hs, std::size_t(B_cap) * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(banned_dev_, hb, std::size_t(B_cap) * sizeof(bk::BannedList), cudaMemcpyHostToDevice, stream));

    if (it != graphs_greedy_.end()) {
      CUDA_CHECK(cudaGraphLaunch(it->second, stream));
    } else {
      decode_eager(B_active, true, stream);
    }
    rinfo_.decode_steps.fetch_add(1, std::memory_order_relaxed);
    rinfo_.allreduce_calls.fetch_add(2 * NUM_VISITS, std::memory_order_relaxed);
    return argmax_out_dev_;
  }

 private:
  const float* const* block_table(const std::vector<const float*>& blocks) {
    auto it = tables_.find(blocks);
    if (it == tables_.end()) {
      DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
      if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
      it = tables_.emplace(blocks, std::move(b)).first;
    }
    return static_cast<const float* const*>(it->second.p);
  }

  void decode_eager(int B, bool greedy, cudaStream_t stream) {
    const Weights& w = w_;
    const Ablation& abl = abl_;
    const std::size_t slab = std::size_t(B) * H;
    const bool use_gemv = (B <= 8) || (B <= 16 && w.layers[0].wproj.blocked());

    const float* attn_dirs = f32p(abl.attn_dirs);
    const float* attn_w = f32p(abl.attn_w);
    const float* mlp_dirs = f32p(abl.mlp_dirs);
    const float* mlp_w = f32p(abl.mlp_w);

    auto run_mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write,
                       const float* abl_r, const float* abl_w, float* partial_out, const float* queries, int Q,
                       const float* norm_w, ActT* out_bf16, float* out_f32) {
      dense::MixParams p;
      p.blocks = blocks.empty() ? nullptr : block_table(blocks);
      p.nblocks = int(blocks.size());
      p.base = base;
      p.write = write;
      p.abl_r = write ? abl_r : nullptr;
      p.abl_d = write && abl.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;
      p.abl_w = write ? abl_w : nullptr;
      p.partial_out = partial_out;
      p.queries = queries;
      p.Q = Q;
      p.norm_w = norm_w;
      p.out_bf16 = out_bf16;
      p.out_f32 = out_f32;
      p.M = B;
      dense::mix(p, false, stream);
    };
    auto bf16_rows = [&](const Dense& d) -> const ActT* {
      if (d.blocked()) {
        dense::dequant_blocked(matrix(d), d.cols(), wscratch_, stream);
        return wscratch_;
      }
      if (!d.fp8()) return bf16p(d.w);
      dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, stream);
      return wscratch_;
    };
    auto run_gates_gemm = [&](const pk::GateRows& rows, int G, const Tensor& down, const Tensor& forget16,
                              const Tensor& gain16, const float* norm_w, ActT* out) {
      pk::gate_inputs(rows, G, B, gin_, stream);
      for (int g = 0; g < G; ++g) {
        const ActT* gd = bf16p(down) + std::size_t(g) * GATE_RANK * 2 * H;
        gemm_->run(gin_ + std::size_t(g) * B * 2 * H, 2 * H, gd, 2 * H, g16_, GATE_RANK, false, B, GATE_RANK, 2 * H, stream);
        gemm_->run(g16_, GATE_RANK, bf16p(forget16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, fpre_, H, true, B, H, GATE_RANK, stream);
        gemm_->run(g16_, GATE_RANK, bf16p(gain16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, gpre_, H, true, B, H, GATE_RANK, stream);
        pk::gate_apply(rows.h[g], rows.msg[g], fpre_, gpre_, norm_w + std::size_t(g) * H, B, out + std::size_t(g) * slab, stream);
      }
    };

    if (w.embed_q8()) dense::embed_q8(tokens_dev_, B, w.embed.data, f32p(w.embed_norm), B0_, false, stream);
    else dense::embed(tokens_dev_, B, bf16p(w.embed), f32p(w.embed_norm), B0_, false, stream);

    int prev_v = -1;
    for (int v = 0; v < NUM_VISITS; ++v) {
      const int l = VISITS[v].layer, p = VISITS[v].pass;
      const Layer& layer = w.layers[l];
      const bool msgs = is_loop_layer(l) && p > 0;
      const bool boundary = (l % BLOCK == 0);
      const std::vector<const float*> blocks = blocks_of(l, p, B0_, B1_, B2_, E_);

      float* hid;
      const float* base;
      const float* write;
      if (v == 0) {
        hid = B0_, base = B0_, write = nullptr;
      } else if (l == LOOP_START && p > 0) {
        hid = B1_, base = B1_, write = nullptr;
      } else {
        hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[2] : hid_;
        base = part_, write = y2_;
      }
      const int pv = prev_v >= 0 ? prev_v : 0;
      const float* r_prev = mlp_dirs + std::size_t(pv) * H;
      const float* w_prev = mlp_w + 4 * std::size_t(pv);
      if (msgs) {
        const int i = l - LOOP_START;
        const float* src_base = i == 0 ? part_ : E_[p - 1];
        const float* src_write = i == 0 ? y2_ : nullptr;
        run_mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
                f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
        if (i == 0) write = nullptr;
      }

      const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
      const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
      float* partial_out = hid != base ? hid : nullptr;
      if (msgs) {
        run_mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_);
        if (use_gemv) {
          dense::message_gates(mixf_, !layer.full, msgs_, 3, B, bf16p(layer.qkv_gate_down),
                               layer.qkv_gate_forget16.data, layer.qkv_gate_gain16.data, false, f32p(layer.attn_norm3),
                               xin_, nullptr, false, stream);
        } else {
          pk::GateRows rows;
          for (int g = 0; g < 3; ++g) rows.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_ + g * slab;
          run_gates_gemm(rows, 3, layer.qkv_gate_down, layer.qkv_gate_forget16, layer.qkv_gate_gain16,
                         f32p(layer.attn_norm3), xin_);
        }
      } else {
        run_mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
                f32p(layer.attn_norm_mix), xin_, nullptr);
      }

      // ---- Sharded attention projections
      const int R = proj_rows(layer.full);
      const bool single = !layer.full && !msgs;
      const bool proj_gemv = use_gemv || ((layer.wproj.fp8() || layer.wproj.blocked()) && B <= 32);
      if (proj_gemv) {
        const DeviceBuffer& seg = layer.full ? full_segments_ : segments_[single ? 0 : 1];
        const int max_m = layer.wproj.blocked() ? 16 : 8;
        if (B <= max_m) {
          dense::proj(xin_, B, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full), proj_fp32_, false, stream);
        } else {
          for (int b0 = 0; b0 < B; b0 += max_m) {
            const int cb = std::min(max_m, B - b0);
            const ActT* xptr = xin_ + std::size_t(b0) * H;
            if (!single) {
              for (int g = 0; g < 3; ++g) {
                CUDA_CHECK(cudaMemcpyAsync(xin_chunk_ + std::size_t(g) * cb * H,
                                           xin_ + std::size_t(g) * slab + std::size_t(b0) * H,
                                           std::size_t(cb) * H * sizeof(ActT), cudaMemcpyDeviceToDevice, stream));
              }
              xptr = xin_chunk_;
            }
            dense::proj(xptr, cb, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full),
                        proj_fp32_ + std::size_t(b0) * R, false, stream);
          }
        }
      } else {
        const ActT* wp = bf16_rows(layer.wproj);
        if (single) {
          gemm_->run(xin_, H, wp, H, proj_act_, R, false, B, R, H, stream);
        } else {
          struct Part { int begin, end, input; };
          const Part parts[] = {{SEG_BEGIN[SEG_Q], SEG_BEGIN[SEG_K], 0},
                                {SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], 1},
                                {SEG_BEGIN[SEG_V], SEG_BEGIN[SEG_K_DYN], 2},
                                {SEG_BEGIN[SEG_K_DYN], SEG_BEGIN[SEG_V_DYN], 1},
                                {SEG_BEGIN[SEG_V_DYN], SEG_BEGIN[NUM_SEGS], 2}};
          for (const Part& part : parts) {
            if (part.begin >= R) break;
            gemm_->run(xin_ + std::size_t(part.input) * slab, H, wp + std::size_t(part.begin) * H, H,
                       proj_act_ + part.begin, R, false, B, part.end - part.begin, H, stream);
          }
        }
      }

      // ---- Sharded per-slot attention (HEADS=6, KV_HEADS=2 per GPU)
      bk::BatchedAttnArgs a;
      a.proj = proj_gemv ? static_cast<const void*>(proj_fp32_) : static_cast<const void*>(proj_act_);
      a.proj_fp32 = proj_gemv;
      a.B = B;
      a.slot_ids = slot_ids_dev_;
      a.slot_pos = static_cast<const std::int64_t*>(pool_.slot_pos.p);
      a.q_scale = f32p(layer.q_scale);
      a.r_scale = f32p(layer.r_scale);
      a.rel_proj = f32p(layer.rel_proj);
      a.conv_k_visit = static_cast<float*>(pool_.conv_k.p) + std::size_t(v) * (CONV_K - 1) * KV_DIM;
      a.conv_v_visit = static_cast<float*>(pool_.conv_v.p) + std::size_t(v) * (CONV_K - 1) * KV_DIM;
      a.conv_slot_stride = pool_.conv_kv_slot_stride;
      a.workspace = attn_ws_.p;
      a.workspace_size = attn_ws_.bytes;
      a.out = attn_out_;
      if (layer.full) {
        a.k_conv = f32p(layer.k_static);
        a.v_conv = f32p(layer.v_static);
        a.k_full_pool = pool_.k_full_pool.p;
        a.v_full_pool = pool_.v_full_pool.p;
        a.page_table = static_cast<const std::int32_t*>(pool_.page_table.p);
        a.max_pages_per_slot = pool_.max_pages_per_slot;
        a.full_visit_idx = FULL_INDEX[v];
        a.max_ctx = pool_.max_ctx;
        a.cache_fp8 = pool_.kv_fp8;
        a.cache_v4 = pool_.kv_v4;
        a.window = 0;
        a.log_scaling = true;
      } else {
        const int ri = RING_INDEX[v];
        a.k_conv = f32p(layer.k_conv);
        a.v_conv = f32p(layer.v_conv);
        a.k_ring_visit = static_cast<ActT*>(pool_.k_ring.p) + std::size_t(ri) * WINDOW * KV_DIM;
        a.v_ring_visit = static_cast<ActT*>(pool_.v_ring.p) + std::size_t(ri) * WINDOW * KV_DIM;
        a.ring_pos_visit = static_cast<std::int64_t*>(pool_.ring_pos.p) + std::size_t(ri) * WINDOW;
        a.ring_slot_stride = pool_.ring_slot_stride;
        a.ring_pos_slot_stride = pool_.ring_pos_slot_stride;
        a.cache_fp8 = false;
        a.window = WINDOW;
        a.log_scaling = false;
      }
      bk::batched_attn_decode(a, stream);

      // ---- Sharded wo (K = HEADS*HEAD_DIM = 768) + NCCL AllReduce + per-slot static conv
      const bool wo_gemv = use_gemv || ((layer.wo.fp8() || layer.wo.blocked()) && B <= 32);
      if (wo_gemv) {
        const int max_m = layer.wo.blocked() ? 16 : 8;
        for (int b0 = 0; b0 < B; b0 += max_m) {
          const int cb = std::min(max_m, B - b0);
          dense::gemv(attn_out_ + std::size_t(b0) * HEADS * HEAD_DIM, cb, HEADS * HEAD_DIM, matrix(layer.wo),
                      wo_fp32_ + std::size_t(b0) * H, false, stream);
        }
      } else {
        const ActT* wo = bf16_rows(layer.wo);
        gemm_->run(attn_out_, HEADS * HEAD_DIM, wo, HEADS * HEAD_DIM, wo_fp32_, H, true, B, H, HEADS * HEAD_DIM, stream);
      }
      if (rinfo_.p2p_ar.active() && B * H <= rinfo_.p2p_ar.cap && 2 * v + 1 < rinfo_.p2p_ar.slots)
        bk::p2p_allreduce2(wo_fp32_, B * H, rinfo_.p2p_ar, 2 * v, stream);
      else
        NCCL_CHECK(ncclAllReduce(wo_fp32_, wo_fp32_, std::size_t(B) * H, ncclFloat32, ncclSum, comm_, stream));
      float* conv_attn_v = static_cast<float*>(pool_.conv_attn.p) + std::size_t(v) * (CONV_K - 1) * H;
      bk::batched_static_conv_step(wo_fp32_, true, H, B, H, slot_ids_dev_, f32p(layer.attn_conv), conv_attn_v,
                                   pool_.conv_h_slot_stride, y_, stream);

      // ---- MLP read + gate
      std::vector<const float*> mlp_blocks = blocks;
      if (boundary) mlp_blocks.push_back(hid);
      const float* mbase = boundary ? nullptr : hid;
      if (msgs) {
        run_mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
                f32p(layer.mlp_query), 1, nullptr, nullptr, mixm_);
        if (use_gemv) {
          dense::message_gates(mixm_, false, msgs_ + 3 * slab, 1, B, bf16p(layer.mlp1_gate_down),
                               layer.mlp1_gate_forget16.data, layer.mlp1_gate_gain16.data, false,
                               f32p(layer.mlp_norm1), xm_, nullptr, false, stream);
        } else {
          pk::GateRows rows;
          rows.h[0] = mixm_, rows.msg[0] = msgs_ + 3 * slab;
          run_gates_gemm(rows, 1, layer.mlp1_gate_down, layer.mlp1_gate_forget16, layer.mlp1_gate_gain16,
                         f32p(layer.mlp_norm1), xm_);
        }
      } else {
        run_mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
                f32p(layer.mlp_query), 1, f32p(layer.mlp_norm1), xm_, nullptr);
      }

      // ---- EP4 MoE + NCCL AllReduce + per-slot static conv
      const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
      if (!pk::moe_route_decode(xm_, B, f32p(layer.router), f32p(layer.router_bias), idx_, wts_, router_logits_,
                                route_done_, work, stream)) {
        gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, B, EXPERTS, H, stream);
        CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), stream));
        pk::moe_topk(router_logits_, f32p(layer.router_bias), B, idx_, wts_, counts_, stream);
        pk::moe_route(idx_, B, work, stream);
      }
      const Experts& ex = layer.experts;
      pk::MoeExperts me;
      me.fmt = static_cast<int>(ex.w13_fmt);
      me.w2_fmt = static_cast<int>(ex.w2_fmt);
      me.w13_factor = f32p(ex.w13_factor);
      me.w2_factor = f32p(ex.w2_factor);
      me.w13_ptrs = static_cast<const std::uint8_t* const*>(ex.w13_ptrs.data);
      me.w13_scale_ptrs = static_cast<const std::uint8_t* const*>(ex.w13_scale_ptrs.data);
      me.w2_ptrs = static_cast<const std::uint8_t* const*>(ex.w2_ptrs.data);
      me.w2_scale_ptrs = static_cast<const std::uint8_t* const*>(ex.w2_scale_ptrs.data);
      me.active_mask = static_cast<const std::uint8_t*>(ex.active_mask.data);
      pk::moe_gate_up(me, xm_, B, work, act_, stream);
      pk::moe_down(me, act_, B, work, down_, stream);
      pk::moe_combine_ep(down_, wts_, idx_, me.active_mask, B, moe_f32_, stream);
      if (rinfo_.p2p_ar.active() && B * H <= rinfo_.p2p_ar.cap && 2 * v + 1 < rinfo_.p2p_ar.slots)
        bk::p2p_allreduce2(moe_f32_, B * H, rinfo_.p2p_ar, 2 * v + 1, stream);
      else
        NCCL_CHECK(ncclAllReduce(moe_f32_, moe_f32_, std::size_t(B) * H, ncclFloat32, ncclSum, comm_, stream));
      float* conv_mlp_v = static_cast<float*>(pool_.conv_mlp.p) + std::size_t(v) * (CONV_K - 1) * H;
      bk::batched_static_conv_step(moe_f32_, true, H, B, H, slot_ids_dev_, f32p(layer.mlp_conv), conv_mlp_v,
                                   pool_.conv_h_slot_stride, y2_, stream);
      prev_v = v;
    }

    // ---- Final read + sharded LM head + NCCL AllGather + unpack on rank 0
    run_mix({B0_, B1_, B2_[2], E_[2]}, part_, y2_, mlp_dirs + std::size_t(prev_v) * H,
            mlp_w + 4 * std::size_t(prev_v), nullptr, f32p(w.final_query), 1, f32p(w.final_norm), xf_, nullptr);
    const bool head_gemv = use_gemv || (w.unembed.fp8() && B <= 32) || w.unembed.blocked();
    if (head_gemv) {
      const int max_m = w.unembed.blocked() ? 16 : 8;
      for (int b0 = 0; b0 < B; b0 += max_m) {
        const int cb = std::min(max_m, B - b0);
        dense::head(xf_ + std::size_t(b0) * H, cb, matrix(w.unembed), local_logits_ + std::size_t(b0) * LOCAL_VOCAB_CHUNK,
                    false, stream);
      }
    } else {
      for (int v0 = 0; v0 < LOCAL_VOCAB_CHUNK; v0 += kHeadChunk) {
        const int cv = std::min(kHeadChunk, LOCAL_VOCAB_CHUNK - v0);
        const ActT* wchunk;
        if (w.unembed.fp8()) {
          const auto* q = static_cast<const std::uint8_t*>(w.unembed.w.data) + std::size_t(v0) * H;
          dense::dequant_rows(q, f32p(w.unembed.scale) + v0, cv, H, wscratch_, stream);
          wchunk = wscratch_;
        } else {
          wchunk = bf16p(w.unembed.w) + std::size_t(v0) * H;
        }
        gemm_->run(xf_, H, wchunk, H, head_act_, cv, false, B, cv, H, stream);
        bk::batched_head_softcap(head_act_, B, cv, v0, LOCAL_VOCAB_CHUNK, local_logits_, stream);
      }
    }
    if (greedy && p2p_argmax_enabled()) {
      const int v0 = rank_ * LOCAL_VOCAB_CHUNK;
      const int local_V = std::min(LOCAL_VOCAB_CHUNK, UNPADDED_VOCAB - v0);
      bk::p2p_argmax2(local_logits_, B, local_V, LOCAL_VOCAB_CHUNK, v0, banned_dev_, argmax_ws_, rinfo_.p2p_ar,
                      2 * NUM_VISITS, argmax_out_dev_, stream);
    } else {
      NCCL_CHECK(ncclAllGather(local_logits_, gathered_logits_, std::size_t(B) * LOCAL_VOCAB_CHUNK, ncclFloat32, comm_, stream));
      if (rank_ == 0) {
        unpack_tp_logits(gathered_logits_, B, TP_SIZE, LOCAL_VOCAB_CHUNK, UNPADDED_VOCAB, logits_, stream);
      }
    }
    bk::batched_advance_pos(slot_ids_dev_, B, static_cast<std::int64_t*>(pool_.slot_pos.p), stream);
  }

  void upload_verify_input(const TpVerifyStepInput& in, int R_cap, cudaStream_t stream) {
    auto* vh = static_cast<TpVerifyStepInput*>(verify_host_.data());
    *vh = in;
    CUDA_CHECK(cudaMemcpyAsync(tokens_dev_, vh->tokens.data(), std::size_t(R_cap) * sizeof(std::int64_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(active_slots_dev_, vh->active_slots.data(), std::size_t(max_batch_) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_row0_dev_, vh->slot_row0.data(), std::size_t(max_batch_) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_m_dev_, vh->slot_m.data(), std::size_t(max_batch_) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(row_slot_dev_, vh->row_slot.data(), std::size_t(R_cap) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(row_off_dev_, vh->row_off.data(), std::size_t(R_cap) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(row_base_dev_, vh->row_base.data(), std::size_t(R_cap) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(banned_dev_, vh->banned.data(), std::size_t(R_cap) * sizeof(bk::BannedList),
                               cudaMemcpyHostToDevice, stream));
  }

  void decode_verify_eager(int R_cap, cudaStream_t stream) {
    const Weights& w = w_;
    const Ablation& abl = abl_;
    const int B = R_cap;
    const std::size_t slab = std::size_t(B) * H;
    const bool use_gemv = (B <= 8) || (B <= 16 && w.layers[0].wproj.blocked());

    const float* attn_dirs = f32p(abl.attn_dirs);
    const float* attn_w = f32p(abl.attn_w);
    const float* mlp_dirs = f32p(abl.mlp_dirs);
    const float* mlp_w = f32p(abl.mlp_w);

    auto run_mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write,
                       const float* abl_r, const float* abl_w, float* partial_out, const float* queries, int Q,
                       const float* norm_w, ActT* out_bf16, float* out_f32, ActT* tap_new = nullptr,
                       ActT* tap_out_copy = nullptr) {
      dense::MixParams p;
      p.blocks = blocks.empty() ? nullptr : block_table(blocks);
      p.nblocks = int(blocks.size());
      p.base = base;
      p.write = write;
      p.abl_r = write ? abl_r : nullptr;
      p.abl_d = write && abl.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;
      p.abl_w = write ? abl_w : nullptr;
      p.partial_out = partial_out;
      p.queries = queries;
      p.Q = Q;
      p.norm_w = norm_w;
      p.out_bf16 = out_bf16;
      p.out_f32 = out_f32;
      if (tap_new) {
        p.new_bf16 = tap_new;
        p.new_bf16_stride = std::int64_t(NUM_TAPS) * H;
      }
      if (tap_out_copy) {
        p.out_copy = tap_out_copy;
        p.out_copy_stride = std::int64_t(NUM_TAPS) * H;
      }
      p.M = B;
      dense::mix(p, false, stream);
    };

    // Save conv states + sliding ring slots about to be written for rollback
    bk::batch_verify_save(state_ptrs_, verify_backup_, active_slots_dev_, slot_m_dev_, max_batch_,
                          static_cast<const std::int64_t*>(pool_.slot_pos.p), stream);

    if (w.embed_q8()) dense::embed_q8(tokens_dev_, B, w.embed.data, f32p(w.embed_norm), B0_, false, stream);
    else dense::embed(tokens_dev_, B, bf16p(w.embed), f32p(w.embed_norm), B0_, false, stream);

    int prev_v = -1, tap = 0;
    for (int v = 0; v < NUM_VISITS; ++v) {
      const int l = VISITS[v].layer, p = VISITS[v].pass;
      const Layer& layer = w.layers[l];
      const bool msgs = is_loop_layer(l) && p > 0;
      const bool boundary = (l % BLOCK == 0);
      const std::vector<const float*> blocks = blocks_of(l, p, B0_, B1_, B2_, E_);

      float* hid;
      const float* base;
      const float* write;
      if (v == 0) {
        hid = B0_, base = B0_, write = nullptr;
      } else if (l == LOOP_START && p > 0) {
        hid = B1_, base = B1_, write = nullptr;
      } else {
        hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[2] : hid_;
        base = part_, write = y2_;
      }
      const int pv = prev_v >= 0 ? prev_v : 0;
      const float* r_prev = mlp_dirs + std::size_t(pv) * H;
      const float* w_prev = mlp_w + 4 * std::size_t(pv);
      if (msgs) {
        const int i = l - LOOP_START;
        const float* src_base = i == 0 ? part_ : E_[p - 1];
        const float* src_write = i == 0 ? y2_ : nullptr;
        run_mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
                f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
        if (i == 0) write = nullptr;
      }

      ActT* tap_ptr = nullptr;
      if (tap < 3 && v == TAP_VISITS[tap]) {
        if (taps_dev_) tap_ptr = taps_dev_ + std::size_t(tap) * H;
        ++tap;
      }

      const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
      const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
      float* partial_out = hid != base ? hid : nullptr;
      if (msgs) {
        run_mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_, tap_ptr);
        dense::message_gates(mixf_, !layer.full, msgs_, 3, B, bf16p(layer.qkv_gate_down),
                             layer.qkv_gate_forget16.data, layer.qkv_gate_gain16.data, false, f32p(layer.attn_norm3),
                             xin_, nullptr, false, stream);
      } else {
        run_mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
                f32p(layer.attn_norm_mix), xin_, nullptr, tap_ptr);
      }

      const int R = proj_rows(layer.full);
      const bool single = !layer.full && !msgs;
      const DeviceBuffer& seg = layer.full ? full_segments_ : segments_[single ? 0 : 1];
      const int max_m = layer.wproj.blocked() ? 16 : 8;
      if (B <= max_m) {
        dense::proj(xin_, B, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full), proj_fp32_,
                    false, stream);
      } else {
        for (int b0 = 0; b0 < B; b0 += max_m) {
          const int cb = std::min(max_m, B - b0);
          const ActT* xptr = xin_ + std::size_t(b0) * H;
          if (!single) {
            for (int g = 0; g < 3; ++g) {
              CUDA_CHECK(cudaMemcpyAsync(xin_chunk_ + std::size_t(g) * cb * H,
                                         xin_ + std::size_t(g) * slab + std::size_t(b0) * H,
                                         std::size_t(cb) * H * sizeof(ActT), cudaMemcpyDeviceToDevice, stream));
            }
            xptr = xin_chunk_;
          }
          dense::proj(xptr, cb, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full),
                      proj_fp32_ + std::size_t(b0) * R, false, stream);
        }
      }

      // Multi-row sharded per-slot attention (no pos advance; records hist_k/hist_v for rollback)
      bk::BatchedAttnRowsArgs a;
      a.proj = proj_fp32_;
      a.proj_fp32 = true;
      a.R = B;
      a.S = max_batch_;
      a.active_slots = active_slots_dev_;
      a.slot_row0 = slot_row0_dev_;
      a.slot_m = slot_m_dev_;
      a.row_slot = row_slot_dev_;
      a.row_off = row_off_dev_;
      a.row_base = row_base_dev_;
      a.slot_pos = static_cast<const std::int64_t*>(pool_.slot_pos.p);
      a.q_scale = f32p(layer.q_scale);
      a.r_scale = f32p(layer.r_scale);
      a.rel_proj = f32p(layer.rel_proj);
      a.conv_k_visit = static_cast<float*>(pool_.conv_k.p) + std::size_t(v) * (CONV_K - 1) * KV_DIM;
      a.conv_v_visit = static_cast<float*>(pool_.conv_v.p) + std::size_t(v) * (CONV_K - 1) * KV_DIM;
      a.conv_slot_stride = pool_.conv_kv_slot_stride;
      a.step_kv = step_kv_dev_;
      a.hist_k_visit = hist_k_dev_ + std::size_t(v) * max_rows_ * KV_DIM;
      a.hist_v_visit = hist_v_dev_ + std::size_t(v) * max_rows_ * KV_DIM;
      a.workspace = attn_ws_.p;
      a.workspace_size = attn_ws_.bytes;
      a.out = attn_out_;
      if (layer.full) {
        a.k_conv = f32p(layer.k_static);
        a.v_conv = f32p(layer.v_static);
        a.k_full_pool = pool_.k_full_pool.p;
        a.v_full_pool = pool_.v_full_pool.p;
        a.page_table = static_cast<const std::int32_t*>(pool_.page_table.p);
        a.max_pages_per_slot = pool_.max_pages_per_slot;
        a.full_visit_idx = FULL_INDEX[v];
        a.max_ctx = pool_.max_ctx;
        a.cache_fp8 = pool_.kv_fp8;
        a.cache_v4 = pool_.kv_v4;
        a.window = 0;
        a.log_scaling = true;
      } else {
        const int ri = RING_INDEX[v];
        a.k_conv = f32p(layer.k_conv);
        a.v_conv = f32p(layer.v_conv);
        a.k_ring_visit = static_cast<ActT*>(pool_.k_ring.p) + std::size_t(ri) * WINDOW * KV_DIM;
        a.v_ring_visit = static_cast<ActT*>(pool_.v_ring.p) + std::size_t(ri) * WINDOW * KV_DIM;
        a.ring_pos_visit = static_cast<std::int64_t*>(pool_.ring_pos.p) + std::size_t(ri) * WINDOW;
        a.ring_slot_stride = pool_.ring_slot_stride;
        a.ring_pos_slot_stride = pool_.ring_pos_slot_stride;
        a.cache_fp8 = false;
        a.window = WINDOW;
        a.log_scaling = false;
      }
      bk::batched_attn_decode_rows(a, stream);

      // wo + P2P AR + multi-row static conv (with hist_attn)
      const int wo_max_m = layer.wo.blocked() ? 16 : 8;
      for (int b0 = 0; b0 < B; b0 += wo_max_m) {
        const int cb = std::min(wo_max_m, B - b0);
        dense::gemv(attn_out_ + std::size_t(b0) * HEADS * HEAD_DIM, cb, HEADS * HEAD_DIM, matrix(layer.wo),
                    wo_fp32_ + std::size_t(b0) * H, false, stream);
      }
      if (rinfo_.p2p_ar.active() && B * H <= rinfo_.p2p_ar.cap && 2 * v + 1 < rinfo_.p2p_ar.slots)
        bk::p2p_allreduce2(wo_fp32_, B * H, rinfo_.p2p_ar, 2 * v, stream);
      else
        NCCL_CHECK(ncclAllReduce(wo_fp32_, wo_fp32_, std::size_t(B) * H, ncclFloat32, ncclSum, comm_, stream));
      float* conv_attn_v = static_cast<float*>(pool_.conv_attn.p) + std::size_t(v) * (CONV_K - 1) * H;
      bk::batched_static_conv_rows(wo_fp32_, true, H, max_batch_, H, active_slots_dev_, slot_row0_dev_, slot_m_dev_,
                                   f32p(layer.attn_conv), conv_attn_v, pool_.conv_h_slot_stride, y_,
                                   hist_attn_dev_ + std::size_t(v) * max_rows_ * H, stream);

      // MLP read + gate
      std::vector<const float*> mlp_blocks = blocks;
      if (boundary) mlp_blocks.push_back(hid);
      const float* mbase = boundary ? nullptr : hid;
      if (msgs) {
        run_mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
                f32p(layer.mlp_query), 1, nullptr, nullptr, mixm_);
        dense::message_gates(mixm_, false, msgs_ + 3 * slab, 1, B, bf16p(layer.mlp1_gate_down),
                             layer.mlp1_gate_forget16.data, layer.mlp1_gate_gain16.data, false,
                             f32p(layer.mlp_norm1), xm_, nullptr, false, stream);
      } else {
        run_mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
                f32p(layer.mlp_query), 1, f32p(layer.mlp_norm1), xm_, nullptr);
      }

      // MoE + P2P AR + multi-row static conv (with hist_mlp)
      const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
      if (!pk::moe_route_decode(xm_, B, f32p(layer.router), f32p(layer.router_bias), idx_, wts_, router_logits_,
                                route_done_, work, stream)) {
        gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, B, EXPERTS, H, stream);
        CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), stream));
        pk::moe_topk(router_logits_, f32p(layer.router_bias), B, idx_, wts_, counts_, stream);
        pk::moe_route(idx_, B, work, stream);
      }
      const Experts& ex = layer.experts;
      pk::MoeExperts me;
      me.fmt = static_cast<int>(ex.w13_fmt);
      me.w2_fmt = static_cast<int>(ex.w2_fmt);
      me.w13_factor = f32p(ex.w13_factor);
      me.w2_factor = f32p(ex.w2_factor);
      me.w13_ptrs = static_cast<const std::uint8_t* const*>(ex.w13_ptrs.data);
      me.w13_scale_ptrs = static_cast<const std::uint8_t* const*>(ex.w13_scale_ptrs.data);
      me.w2_ptrs = static_cast<const std::uint8_t* const*>(ex.w2_ptrs.data);
      me.w2_scale_ptrs = static_cast<const std::uint8_t* const*>(ex.w2_scale_ptrs.data);
      me.active_mask = static_cast<const std::uint8_t*>(ex.active_mask.data);
      pk::moe_gate_up(me, xm_, B, work, act_, stream);
      pk::moe_down(me, act_, B, work, down_, stream);
      pk::moe_combine_ep(down_, wts_, idx_, me.active_mask, B, moe_f32_, stream);
      if (rinfo_.p2p_ar.active() && B * H <= rinfo_.p2p_ar.cap && 2 * v + 1 < rinfo_.p2p_ar.slots)
        bk::p2p_allreduce2(moe_f32_, B * H, rinfo_.p2p_ar, 2 * v + 1, stream);
      else
        NCCL_CHECK(ncclAllReduce(moe_f32_, moe_f32_, std::size_t(B) * H, ncclFloat32, ncclSum, comm_, stream));
      float* conv_mlp_v = static_cast<float*>(pool_.conv_mlp.p) + std::size_t(v) * (CONV_K - 1) * H;
      bk::batched_static_conv_rows(moe_f32_, true, H, max_batch_, H, active_slots_dev_, slot_row0_dev_, slot_m_dev_,
                                   f32p(layer.mlp_conv), conv_mlp_v, pool_.conv_h_slot_stride, y2_,
                                   hist_mlp_dev_ + std::size_t(v) * max_rows_ * H, stream);
      prev_v = v;
    }

    // Final mix (writes tap 3 on rank 0) + sharded LM head + distributed P2P argmax
    run_mix({B0_, B1_, B2_[2], E_[2]}, part_, y2_, mlp_dirs + std::size_t(prev_v) * H,
            mlp_w + 4 * std::size_t(prev_v), nullptr, f32p(w.final_query), 1, f32p(w.final_norm), xf_, nullptr,
            nullptr, taps_dev_ ? (taps_dev_ + std::size_t(3) * H) : nullptr);
    const int head_max_m = w.unembed.blocked() ? 16 : 8;
    for (int b0 = 0; b0 < B; b0 += head_max_m) {
      const int cb = std::min(head_max_m, B - b0);
      dense::head(xf_ + std::size_t(b0) * H, cb, matrix(w.unembed),
                  local_logits_ + std::size_t(b0) * LOCAL_VOCAB_CHUNK, false, stream);
    }
    const int v0 = rank_ * LOCAL_VOCAB_CHUNK;
    const int local_V = std::min(LOCAL_VOCAB_CHUNK, UNPADDED_VOCAB - v0);
    bk::p2p_argmax2(local_logits_, B, local_V, LOCAL_VOCAB_CHUNK, v0, banned_dev_, argmax_ws_, rinfo_.p2p_ar,
                    2 * NUM_VISITS, argmax_out_dev_, stream);
    (void)use_gemv;
  }

  const Weights& w_;
  const Ablation& abl_;
  TpBatchStatePool& pool_;
  int max_batch_;
  bool spec_enabled_ = false;
  int spec_max_m_ = 1;
  int max_rows_ = 1;
  int rank_;
  ncclComm_t comm_;
  TpWorkerPool::RankInfo& rinfo_;
  std::unique_ptr<pk::Gemm> gemm_;

  std::vector<DeviceBuffer> storage_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;
  std::array<DeviceBuffer, 2> segments_;
  DeviceBuffer full_segments_, attn_ws_;
  std::map<int, cudaGraphExec_t> graphs_, graphs_greedy_, graphs_verify_;
  PinnedHostBuffer tokens_host_, slots_host_, banned_host_, verify_host_, commit_host_;

  std::int64_t* tokens_dev_ = nullptr;
  std::int32_t* slot_ids_dev_ = nullptr;
  bk::BannedList* banned_dev_ = nullptr;
  void* argmax_ws_ = nullptr;
  std::int32_t* argmax_out_dev_ = nullptr;
  ActT* taps_dev_ = nullptr;
  ActT* step_kv_dev_ = nullptr;
  float *hist_k_dev_ = nullptr, *hist_v_dev_ = nullptr, *hist_attn_dev_ = nullptr, *hist_mlp_dev_ = nullptr;
  bk::BatchVerifyBackup verify_backup_{};
  bk::BatchStatePoolPtrs state_ptrs_{};
  std::int32_t *active_slots_dev_ = nullptr, *slot_row0_dev_ = nullptr, *slot_m_dev_ = nullptr,
               *commit_counts_dev_ = nullptr;
  std::int32_t *row_slot_dev_ = nullptr, *row_off_dev_ = nullptr, *row_base_dev_ = nullptr;
  float *B0_, *B1_, *B2_[PASSES], *E_[PASSES], *hid_, *part_, *y_, *y2_;
  ActT *xin_, *xin_chunk_, *xm_, *xf_;
  float *mixf_, *msgs_, *mixm_, *proj_fp32_, *wo_fp32_, *fpre_, *gpre_, *router_logits_, *wts_, *moe_f32_;
  unsigned* route_done_ = nullptr;  // moe_route_decode's last-CTA counter
  ActT *proj_act_, *attn_out_, *gin_, *g16_, *act_, *down_, *head_act_, *wscratch_;
  int *idx_, *counts_, *tiles_n_, *pair_token_, *pair_dst_;
  int4* tiles_;
  float *local_logits_, *gathered_logits_, *logits_;
};

// ----------------------------------------------------------------------------- TpBatchGenerator (continuous batching TP4/EP4)

// INGINE_MEM_LOG=1: log free device memory after each start-up stage (fitting long-context pools)
#define MEMLOG(r, what)                                                                                    \
  do {                                                                                                     \
    static const bool on_ = [] { const char* e = std::getenv("INGINE_MEM_LOG"); return e && e[0] == '1'; }(); \
    if (on_) {                                                                                             \
      std::size_t f_ = 0, t_ = 0;                                                                          \
      CUDA_CHECK(cudaDeviceSynchronize());                                                                 \
      CUDA_CHECK(cudaMemGetInfo(&f_, &t_));                                                                \
      ingine::engine::server_log().info("TP rank {} after {}: {} MiB free", r, what, f_ >> 20);            \
    }                                                                                                      \
  } while (0)

class TpBatchGenerator final : public Generator {
 public:
  explicit TpBatchGenerator(const RuntimeOptions& options)
      : options_(options),
        spec_active_(options.spec && options.max_batch <= kd::kMaxBatchSlots && TP_SIZE == 2),
        spec_max_m_(spec_active_ ? std::clamp(options.spec_max_m, 2, 4) : 1),
        sampled_host_(std::size_t(std::max(1, options.max_batch)) * sizeof(std::int32_t)),
        spec_accept_host_(std::size_t(3 * std::max(1, options.max_batch)) * sizeof(std::int32_t)),
        spec_draft_host_(std::size_t(std::max(1, options.max_batch)) * kd::GAMMA * (sizeof(std::int32_t) + sizeof(float))) {
    validate_tp_runtime_options(options, TP_SIZE);
    prefill_chunk_ = std::min(options.prefill_chunk, options.max_ctx);
    max_pages_per_slot_ = (options.max_ctx + bk::kPageSize - 1) / bk::kPageSize;

    // Step 1: load sharded weights + ablation + prefiller on all 4 ranks in parallel
    std::array<int, TP_SIZE> rank_fit_pages{};
    pool_.run_all([&](int r) {
      auto& rk = ranks_[r];
      rk.weights = load_weights_sharded(options_.model_dir, options_.dense_fp8, pool_.stream(r), r, TP_SIZE, false);
      MEMLOG(r, "weights");
      rk.ablation = load_ablation(options_.ablation);
      rk.prefiller = std::make_unique<TpPrefiller>(rk.weights, rk.ablation, prefill_chunk_, pool_.stream(r),
                                                   pool_.comm(r), pool_.info(r));
      MEMLOG(r, "prefiller");
      rk.chunk_tokens = DeviceBuffer(std::size_t(prefill_chunk_) * sizeof(std::int32_t));
      if (r == 0) {
        prefill_logits_ = DeviceBuffer(std::size_t(UNPADDED_VOCAB) * sizeof(float));
        sampled_dev_ = DeviceBuffer(std::size_t(options_.max_batch) * sizeof(std::int32_t));
      }
      // Compute how many pages fit after accounting for sliding rings + scratch full KV
      const std::size_t fixed_kv =
          std::size_t(options_.max_batch) *
              (2 * std::size_t(NUM_SLIDING_VISITS) * WINDOW * KV_DIM * sizeof(ActT) +
               std::size_t(NUM_SLIDING_VISITS) * WINDOW * sizeof(std::int64_t) +
               2 * std::size_t(NUM_VISITS) * (CONV_K - 1) * (KV_DIM + HIDDEN) * sizeof(float));
      std::size_t free_b = 0, tot_b = 0;
      CUDA_CHECK(cudaMemGetInfo(&free_b, &tot_b));
      const std::size_t reserve = std::size_t(2560) << 20;
      const std::size_t avail = free_b > (fixed_kv + reserve) ? (free_b - fixed_kv - reserve) : 0;
      const std::size_t bytes_per_page = TpBatchStatePool::page_bytes(options_.kv_fp8, options_.kv_fp4v);
      const int max_possible = options_.max_batch * max_pages_per_slot_;
      if (options_.max_kv_tokens > 0) {
        rank_fit_pages[r] = std::min(max_possible, std::max(max_pages_per_slot_, (options_.max_kv_tokens + bk::kPageSize - 1) / bk::kPageSize));
      } else {
        rank_fit_pages[r] = std::min(max_possible, std::max(std::min(max_possible, max_pages_per_slot_), int(avail / bytes_per_page)));
      }
    });

    num_pages_ = *std::min_element(rank_fit_pages.begin(), rank_fit_pages.end());
    if (num_pages_ < 1) throw std::runtime_error("TP BatchStatePool: insufficient VRAM for paged full-KV pool");
    ingine::engine::server_log().info("TP{}/EP{} paged full-KV pool: {} pages ({} tokens, max_batch={}, max_ctx={}, kv={})",
                                      TP_SIZE, TP_SIZE, num_pages_, num_pages_ * bk::kPageSize, options_.max_batch, options_.max_ctx,
                                      options_.kv_fp4v ? "fp8k+fp4v" : (options_.kv_fp8 ? "fp8" : "fp16"));

    free_pages_.resize(num_pages_);
    for (int i = 0; i < num_pages_; ++i) free_pages_[i] = num_pages_ - 1 - i;
    slot_pages_.assign(options_.max_batch, {});
    slot_pos_host_.assign(options_.max_batch, 0);
    slot_in_use_.assign(options_.max_batch, false);
    slot_meta_.resize(options_.max_batch);

    // Step 2: allocate uniform paged pools + batch decoders on all ranks
    pool_.run_all([&](int r) {
      auto& rk = ranks_[r];
      rk.kv_pool.init_buffers(options_.max_batch, options_.max_ctx, options_.kv_fp8, options_.kv_fp4v, num_pages_);
      MEMLOG(r, "kv_pool");
      rk.decoder = std::make_unique<TpBatchDecoder>(rk.weights, rk.ablation, rk.kv_pool, options_.max_batch, r,
                                                    pool_.comm(r), pool_.info(r), spec_active_, spec_max_m_);
      MEMLOG(r, "decoder");
      CUDA_CHECK(cudaStreamSynchronize(pool_.stream(r)));
    });

    if (spec_active_ && !ranks_[0].decoder->p2p_argmax_enabled()) {
      throw std::runtime_error("TP2 batched --spec requires P2P argmax (ensure P2P is enabled and INGINE_TP_P2P_ARGMAX != 0)");
    }

    // Step 2b: construct BatchDrafter on rank 0 (peer-accessible HeadShards, no full_unembed duplicate)
    if (spec_active_) {
      CUDA_CHECK(cudaSetDevice(0));
      drafter_weights_.embed = ranks_[0].weights.embed;
      drafter_weights_.embed_norm = ranks_[0].weights.embed_norm;
      kd::HeadShards hs{};
      hs.n_shards = TP_SIZE;
      hs.rows_per_shard = LOCAL_VOCAB_CHUNK;
      hs.total_rows = UNPADDED_VOCAB;
      for (int r = 0; r < TP_SIZE; ++r) {
        const auto& u = ranks_[r].weights.unembed;
        hs.shard[r] = dense::Matrix{
            u.w.data, u.fp8() ? static_cast<const float*>(u.scale.data) : nullptr, u.rows(),
            static_cast<const std::int64_t*>(u.blk_off.data),
            static_cast<const std::int32_t*>(u.blk_soff.data),
            static_cast<const std::uint16_t*>(u.q8_scale.data)};
      }
      const int dlen = spec_max_m_ - 1;
      batch_drafter_ = std::make_unique<BatchDrafter>(
          options_.model_dir, drafter_weights_, options_.max_batch, options_.max_ctx, options_.draft_ctx,
          pool_.stream(0), options_.spec_shortlist, hs, spec_max_m_, dlen);
      batch_drafter_->capture();
      prefill_taps_ = DeviceBuffer(std::size_t(prefill_chunk_) * NUM_TAPS * HIDDEN * sizeof(ActT));
      spec_accept_dev_ = DeviceBuffer(std::size_t(3 * options_.max_batch) * sizeof(std::int32_t));
      CUDA_CHECK(cudaEventCreateWithFlags(&accept_ready_ev_, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreate(&ev_v0_));
      CUDA_CHECK(cudaEventCreate(&ev_v1_));
      CUDA_CHECK(cudaEventCreate(&ev_d1_));
      MEMLOG(0, "batch_drafter");
      CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
    }

    // Step 3: warmup & capture buckets across all ranks concurrently.
    const char* no_graph = std::getenv("INGINE_TP_NO_GRAPH");
    enable_graphs_ = !(no_graph && no_graph[0] == '1');
    for (int b = 1; b < options_.max_batch; b <<= 1) buckets_.push_back(b);
    for (int mid : {24, 48, 96})
      if (mid < options_.max_batch) buckets_.push_back(mid);
    buckets_.push_back(options_.max_batch);
    std::sort(buckets_.begin(), buckets_.end());
    buckets_.erase(std::unique(buckets_.begin(), buckets_.end()), buckets_.end());
    const bool greedy_init = ranks_[0].decoder->p2p_argmax_enabled();
    if (spec_active_) {
      // In batched spec mode, capture fine-grained verify buckets (step 2 up to R_max) instead of
      // pre-capturing non-spec greedy buckets (those capture lazily if a non-spec step is ever called).
      const int R_max = options_.max_batch * spec_max_m_;
      for (int r = 2; r <= R_max; r += 2) verify_buckets_.push_back(r);
      if (verify_buckets_.empty() || verify_buckets_.back() != R_max) verify_buckets_.push_back(R_max);
      for (int r : verify_buckets_) bucket_ms_ema_[r] = 13.0f + 1.15f * float(r);
      capture_verify_buckets_unlocked();
      greedy_graphs_ready_ = false;
      logits_graphs_ready_ = false;
    } else {
      capture_buckets_unlocked(greedy_init);
      greedy_graphs_ready_ = greedy_init;
      logits_graphs_ready_ = !greedy_init;
    }
    pool_.run_all([&](int r) { MEMLOG(r, "graphs"); });
    ingine::engine::server_log().info("{}", offload_summary());
  }

  ~TpBatchGenerator() override {
    if (spec_active_) {
      cudaSetDevice(0);
      if (accept_ready_ev_) cudaEventDestroy(accept_ready_ev_);
      if (ev_v0_) cudaEventDestroy(ev_v0_);
      if (ev_v1_) cudaEventDestroy(ev_v1_);
      if (ev_d1_) cudaEventDestroy(ev_d1_);
      batch_drafter_.reset();
    }
    pool_.run_all([&](int r) {
      cudaStreamSynchronize(pool_.stream(r));
      ranks_[r].decoder.reset();
      ranks_[r].prefiller.reset();
    });
  }

  std::string model_id() const override {
    return options_.ablation.empty() ? "inkling-2-mini" : "inkling-2-mini-abliterated";
  }
  int max_model_len() const override { return options_.max_ctx; }
  bool supports_images() const override { return false; }
  int max_batch_size() const override { return options_.max_batch; }

  bool supports_spec_batch() const override { return batch_drafter_ != nullptr; }

  void reset_spec_metrics() override {
    std::lock_guard<std::mutex> lk(mu_);
    spec_steps_ = 0;
    spec_slot_rounds_ = 0;
    spec_accepted_tot_ = 0;
    spec_m_hist_.fill(0);
    spec_verify_ms_tot_ = 0.0;
    spec_draft_ms_tot_ = 0.0;
    spec_sfull_steps_ = 0;
    spec_sfull_toks_ = 0;
    spec_sfull_ms_ = 0.0;
  }

  std::string spec_metrics_summary() const override {
    std::lock_guard<std::mutex> lk(mu_);
    if (!batch_drafter_ || spec_steps_ == 0) return {};
    const double tok_per_verify = double(spec_accepted_tot_ + spec_slot_rounds_) / double(spec_slot_rounds_);
    const double v_ms = spec_verify_ms_tot_ / double(spec_steps_);
    const double d_ms = spec_draft_ms_tot_ / double(spec_steps_);
    char buf[320];
    if (spec_sfull_steps_ > 0 && spec_sfull_ms_ > 0.0) {
      const double steady_agg = double(spec_sfull_toks_) / (spec_sfull_ms_ * 1e-3);
      const double steady_req = steady_agg / double(options_.max_batch);
      std::snprintf(buf, sizeof buf,
                    "spec: %.2f tok/verify, verify=%.2fms, draft=%.2fms, steady(B=%d)=%.1f (%.1f/req), m=[1:%lld,2:%lld,3:%lld,4:%lld]",
                    tok_per_verify, v_ms, d_ms, options_.max_batch, steady_agg, steady_req,
                    static_cast<long long>(spec_m_hist_[1]), static_cast<long long>(spec_m_hist_[2]),
                    static_cast<long long>(spec_m_hist_[3]), static_cast<long long>(spec_m_hist_[4]));
    } else {
      std::snprintf(buf, sizeof buf,
                    "spec: %.2f tok/verify, verify=%.2fms, draft=%.2fms, m=[1:%lld,2:%lld,3:%lld,4:%lld]",
                    tok_per_verify, v_ms, d_ms, static_cast<long long>(spec_m_hist_[1]),
                    static_cast<long long>(spec_m_hist_[2]), static_cast<long long>(spec_m_hist_[3]),
                    static_cast<long long>(spec_m_hist_[4]));
    }
    return buf;
  }

  std::string offload_summary() const override {
    return pool_.summary(batch_drafter_ ? "batch-spec" : "batch",
                         "max_batch=" + std::to_string(options_.max_batch) +
                             " max_ctx=" + std::to_string(options_.max_ctx) +
                             " kv_pool_tokens=" + std::to_string(num_pages_ * bk::kPageSize) +
                             (batch_drafter_ ? (" draft_ctx=" + std::to_string(batch_drafter_->ctx()) +
                                                " max_m=" + std::to_string(spec_max_m_))
                                             : ""));
  }

  bool can_admit_slot(int prompt_tokens, int /*max_new_tokens*/) const override {
    std::lock_guard<std::mutex> lk(mu_);
    if (free_slot_count_unlocked() <= 0) return false;
    const int need = (prompt_tokens + 1 + bk::kPageSize - 1) / bk::kPageSize;
    return int(free_pages_.size()) >= need;
  }

  int start_slot(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                 std::int32_t& first_token, GenerateStats& stats) override {
    validate_tp_request(full_prompt, sampling);
    const std::span<const std::int32_t> prompt = full_prompt.ids;
    const int n = int(prompt.size());
    if (n >= options_.max_ctx) throw std::invalid_argument("prompt exceeds max_ctx");

    std::lock_guard<std::mutex> lk(mu_);
    const int need_pages = (n + 1 + bk::kPageSize - 1) / bk::kPageSize;
    if (free_slot_count_unlocked() <= 0 || int(free_pages_.size()) < need_pages) return -1;
    const int s = alloc_slot_unlocked();
    if (s < 0) return -1;
    if (!ensure_pages_all_unlocked(s, n + 1)) {
      free_slot_unlocked(s);
      return -1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
      const std::size_t chunk = std::size_t(prefill_chunk_);
      pool_.run_all([&](int r) {
        auto& rk = ranks_[r];
        State st = rk.kv_pool.prefill_view(s);
        reset_state(st, pool_.stream(r));
        if (r == 0 && batch_drafter_) batch_drafter_->reset_slot(s);
        for (std::size_t start = 0; start < std::size_t(n); start += chunk) {
          const int T = int(std::min(chunk, std::size_t(n) - start));
          CUDA_CHECK(cudaMemcpyAsync(rk.chunk_tokens.p, prompt.data() + start, std::size_t(T) * sizeof(std::int32_t),
                                     cudaMemcpyHostToDevice, pool_.stream(r)));
          float* out_l = (r == 0) ? static_cast<float*>(prefill_logits_.p) : nullptr;
          ActT* tap_p = (r == 0 && batch_drafter_) ? static_cast<ActT*>(prefill_taps_.p) : nullptr;
          Tensor tap_view = tap_p ? rm(tap_p, kActDType, {T, NUM_TAPS, HIDDEN}) : Tensor{};
          rk.prefiller->run(st, static_cast<const std::int32_t*>(rk.chunk_tokens.p), T, int(start), out_l,
                            tap_p ? &tap_view : nullptr);
          if (r == 0 && batch_drafter_) batch_drafter_->commit_slot(s, tap_p, T, int(start));
        }
        const std::int64_t pos64 = n;
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int64_t*>(rk.kv_pool.slot_pos.p) + s, &pos64, sizeof(pos64),
                                   cudaMemcpyHostToDevice, pool_.stream(r)));
      });
      slot_pos_host_[s] = n;

      // Sample first token on rank 0
      CUDA_CHECK(cudaSetDevice(0));
      SlotMeta& meta = slot_meta_[s];
      meta.params = to_head_params(sampling);
      meta.rng.seed = sampling.seed ? *sampling.seed : (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
      meta.rng.counter = 0;
      meta.forced.assign(sampling.forced.begin(), sampling.forced.end());
      meta.forced_idx = 0;
      meta.spec_eligible = (batch_drafter_ != nullptr && sampling.temperature <= 0.f && sampling.forced.empty());
      meta.has_draft = false;

      auto* sampled = static_cast<std::int32_t*>(sampled_dev_.p);
      head::sample(static_cast<const float*>(prefill_logits_.p), 1, UNPADDED_VOCAB, UNPADDED_VOCAB, meta.params,
                   meta.rng, sampled, pool_.stream(0));
      CUDA_CHECK(cudaMemcpyAsync(sampled_host_.data(), sampled, sizeof(std::int32_t), cudaMemcpyDeviceToHost, pool_.stream(0)));
      CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
      first_token = static_cast<const std::int32_t*>(sampled_host_.data())[0];
      if (!meta.forced.empty()) {
        first_token = meta.forced[0];
        meta.forced_idx = 1;
      }

      if (meta.spec_eligible) {
        batch_drafter_->note_tokens(prompt);
        batch_drafter_->note_tokens(std::span<const std::int32_t>(&first_token, 1));
        const int act1[1] = {s};
        const std::int32_t anc1[1] = {first_token};
        const int pos1[1] = {n};
        batch_drafter_->draft_slots(act1, anc1, pos1, 0.f);
        const int dlen = batch_drafter_->draft_len();
        auto* dtok_h = static_cast<std::int32_t*>(spec_draft_host_.data());
        auto* dconf_h = reinterpret_cast<float*>(dtok_h + std::size_t(options_.max_batch) * kd::GAMMA);
        CUDA_CHECK(cudaMemcpyAsync(dtok_h, batch_drafter_->slot_tokens_dev(s), std::size_t(dlen) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, pool_.stream(0)));
        CUDA_CHECK(cudaMemcpyAsync(dconf_h, batch_drafter_->slot_conf_dev(s), std::size_t(dlen) * sizeof(float),
                                   cudaMemcpyDeviceToHost, pool_.stream(0)));
        CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
        for (int k = 0; k < dlen; ++k) {
          meta.draft_tokens[k] = dtok_h[k];
          meta.draft_conf[k] = dconf_h[k];
        }
        meta.has_draft = true;
      }

      stats.prompt_tokens = n;
      stats.reused_tokens = 0;
      stats.reuse = Reuse::None;
      stats.prefill_seconds = seconds_since(t0);
      return s;
    } catch (...) {
      pool_.run_all([&](int r) { cudaStreamSynchronize(pool_.stream(r)); });
      free_slot_unlocked(s);
      throw;
    }
  }

  void step_slots(std::span<const BatchSlotStep> inputs, std::span<std::int32_t> out_tokens) override {
    const int B = int(inputs.size());
    if (B == 0) return;
    if (B > options_.max_batch || out_tokens.size() < inputs.size())
      throw std::invalid_argument("step_slots: invalid batch size or output size");
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<bool> seen(slot_in_use_.size(), false);
    int extra_pages = 0;
    for (const auto& input : inputs) {
      const int s = input.slot;
      if (s < 0 || s >= options_.max_batch || !slot_in_use_[s]) throw std::invalid_argument("step_slots: bad slot");
      if (seen[s]) throw std::invalid_argument("step_slots: duplicate slot in batch");
      seen[s] = true;
      ops::validate_token_ids_i32(&input.token, 1, VOCAB);
      const int pos = slot_pos_host_[s];
      if (pos >= options_.max_ctx) throw std::runtime_error("step_slots: slot reached max_ctx");
      const int needed = (pos + 1 + bk::kPageSize - 1) / bk::kPageSize;
      if (needed > max_pages_per_slot_) throw std::runtime_error("step_slots: paged KV pool exhausted");
      extra_pages += std::max(0, needed - int(slot_pages_[s].size()));
    }
    if (extra_pages > int(free_pages_.size())) throw std::runtime_error("step_slots: paged KV pool exhausted");
    for (const auto& input : inputs) {
      ensure_pages_all_unlocked(input.slot, slot_pos_host_[input.slot] + 1);
    }

    bool all_greedy_temp = true;
    for (int i = 0; i < B; ++i) {
      if (slot_meta_[inputs[i].slot].params.temperature > 0.0f) { all_greedy_temp = false; break; }
    }

    if (all_greedy_temp && ranks_[0].decoder->p2p_argmax_enabled()) {
      if (enable_graphs_ && !greedy_graphs_ready_) {
        capture_buckets_unlocked(true);
        greedy_graphs_ready_ = true;
      }
      std::vector<bk::BannedList> banned(B);
      for (int i = 0; i < B; ++i) {
        const auto& sb = slot_meta_[inputs[i].slot].params.banned;
        banned[i].count = sb.count;
        for (int k = 0; k < sb.count && k < bk::BannedList::kMax; ++k) banned[i].ids[k] = sb.ids[k];
      }
      const std::int32_t* argmax_dev = nullptr;
      pool_.run_all([&](int r) {
        const std::int32_t* out = ranks_[r].decoder->run_step_greedy(inputs, banned, pool_.stream(r));
        if (r == 0) argmax_dev = out;
      });
      for (int i = 0; i < B; ++i) slot_pos_host_[inputs[i].slot] += 1;

      CUDA_CHECK(cudaSetDevice(0));
      CUDA_CHECK(cudaMemcpyAsync(sampled_host_.data(), argmax_dev, std::size_t(B) * sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost, pool_.stream(0)));
      CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
    } else {
      if (enable_graphs_ && !logits_graphs_ready_) {
        capture_buckets_unlocked(false);
        logits_graphs_ready_ = true;
      }
      const float* logits = nullptr;
      pool_.run_all([&](int r) {
        const float* l = ranks_[r].decoder->run_step(inputs, pool_.stream(r));
        if (r == 0) logits = l;
      });
      for (int i = 0; i < B; ++i) slot_pos_host_[inputs[i].slot] += 1;

      CUDA_CHECK(cudaSetDevice(0));
      bool all_greedy = all_greedy_temp;
      const head::BannedTokens& b0 = slot_meta_[inputs[0].slot].params.banned;
      if (all_greedy) {
        for (int i = 0; i < B; ++i) {
          if (!same_banned(slot_meta_[inputs[i].slot].params.banned, b0)) { all_greedy = false; break; }
        }
      }
      auto* sampled = static_cast<std::int32_t*>(sampled_dev_.p);
      if (all_greedy) {
        head::argmax(logits, B, UNPADDED_VOCAB, UNPADDED_VOCAB, sampled, pool_.stream(0), b0);
      } else {
        for (int i = 0; i < B; ++i) {
          SlotMeta& meta = slot_meta_[inputs[i].slot];
          head::sample(logits + std::size_t(i) * UNPADDED_VOCAB, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, meta.params,
                       meta.rng, sampled + i, pool_.stream(0));
        }
      }
      CUDA_CHECK(cudaMemcpyAsync(sampled_host_.data(), sampled, std::size_t(B) * sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost, pool_.stream(0)));
      CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
    }
    const auto* host_out = static_cast<const std::int32_t*>(sampled_host_.data());
    for (int i = 0; i < B; ++i) {
      SlotMeta& meta = slot_meta_[inputs[i].slot];
      std::int32_t tok = host_out[i];
      if (meta.forced_idx < meta.forced.size()) tok = meta.forced[meta.forced_idx++];
      out_tokens[i] = tok;
    }
  }

  void step_slots_multi(std::span<const BatchSlotStep> inputs, std::span<BatchSlotOutput> outputs) override {
    const int S = int(inputs.size());
    if (S == 0) return;
    if (S > options_.max_batch || outputs.size() < inputs.size())
      throw std::invalid_argument("step_slots_multi: invalid batch size or output size");

    bool use_spec = (batch_drafter_ != nullptr);
    if (use_spec) {
      std::lock_guard<std::mutex> lk(mu_);
      for (int i = 0; i < S; ++i) {
        const int s = inputs[i].slot;
        if (s < 0 || s >= options_.max_batch || !slot_in_use_[s] || !slot_meta_[s].spec_eligible ||
            !slot_meta_[s].has_draft) {
          use_spec = false;
          break;
        }
      }
    }
    if (!use_spec) {
      std::vector<std::int32_t> next(S, 0);
      step_slots(inputs, next);
      for (int i = 0; i < S; ++i) {
        outputs[i] = BatchSlotOutput{};
        outputs[i].tokens[0] = next[i];
        outputs[i].count = 1;
        outputs[i].verify_width = 1;
        outputs[i].accepted = 0;
      }
      return;
    }

    std::lock_guard<std::mutex> lk(mu_);
    const int dlen = batch_drafter_->draft_len();
    std::array<int, kd::kMaxBatchSlots> active_slots{};
    std::array<int, kd::kMaxBatchSlots> m_slot{};
    std::array<int, kd::kMaxBatchSlots> m_cap{};
    std::vector<bool> seen(slot_in_use_.size(), false);

    // 1) Validate slots and choose per-slot verify width m_s in [1, min(spec_max_m_, dlen + 1, room)].
    static const bool use_ev_policy = [] {
      const char* e = std::getenv("INGINE_BATCH_SPEC_EV");
      return !(e && e[0] == '0');
    }();
    int R_sum = 0;
    for (int i = 0; i < S; ++i) {
      const int s = inputs[i].slot;
      if (s < 0 || s >= options_.max_batch || !slot_in_use_[s]) throw std::invalid_argument("step_slots_multi: bad slot");
      if (seen[s]) throw std::invalid_argument("step_slots_multi: duplicate slot in batch");
      seen[s] = true;
      ops::validate_token_ids_i32(&inputs[i].token, 1, VOCAB);
      const int pos = slot_pos_host_[s];
      const int room = options_.max_ctx - pos;
      if (room <= 0) throw std::runtime_error("step_slots_multi: slot reached max_ctx");
      active_slots[i] = s;
      m_cap[i] = std::min({spec_max_m_, dlen + 1, room});
      m_slot[i] = 1;
    }

    int R_cap = S;
    float predicted_acc_sum = 0.f;
    if (options_.spec_threshold <= 0.f) {
      for (int i = 0; i < S; ++i) {
        m_slot[i] = m_cap[i];
        R_sum += m_slot[i];
      }
      R_cap = R_sum;
      for (int b : verify_buckets_) {
        if (b >= R_sum) { R_cap = b; break; }
      }
    } else if (options_.spec_threshold > 1.0f) {
      R_sum = S;
      R_cap = R_sum;
      for (int b : verify_buckets_) {
        if (b >= R_sum) { R_cap = b; break; }
      }
    } else if (use_ev_policy) {
      // Expected-throughput optimal bucket selection:
      // For each candidate bucket R_cap >= S, assign the K = R_cap - S extra verify rows to the highest
      // marginal cumulative acceptance probabilities P_s(j) across all active slots, then maximize E[tok]/T_ema(R_cap).
      struct Cand {
        float prob;
        int slot_i;
      };
      std::array<Cand, 16> cands{};
      int ncands = 0;
      for (int i = 0; i < S; ++i) {
        const SlotMeta& meta = slot_meta_[active_slots[i]];
        float cum = 1.0f;
        for (int j = 1; j < m_cap[i]; ++j) {
          cum *= std::clamp(meta.draft_conf[j - 1] * spec_alpha_cal_, 0.0f, 1.0f);
          cands[ncands++] = {cum, i};
        }
      }
      std::sort(cands.begin(), cands.begin() + ncands, [](const Cand& a, const Cand& b) { return a.prob > b.prob; });

      int best_R_cap = verify_buckets_.empty() ? S : verify_buckets_.back();
      int best_K = 0;
      float best_score = -1.f;
      for (int b : verify_buckets_) {
        if (b < S) continue;
        const int K = std::min(b - S, ncands);
        float ev_tok = float(S);
        for (int k = 0; k < K; ++k) ev_tok += cands[k].prob;
        const auto it_t = bucket_ms_ema_.find(b);
        const float t_ms = (it_t != bucket_ms_ema_.end()) ? it_t->second : (14.0f + 1.15f * float(b));
        const float score = ev_tok / std::max(t_ms, 1.0f);
        if (score > best_score) {
          best_score = score;
          best_R_cap = b;
          best_K = K;
        }
      }
      R_cap = best_R_cap;
      R_sum = S + best_K;
      for (int k = 0; k < best_K; ++k) {
        ++m_slot[cands[k].slot_i];
        predicted_acc_sum += cands[k].prob;
      }
    } else {
      for (int i = 0; i < S; ++i) {
        const SlotMeta& meta = slot_meta_[active_slots[i]];
        int m = 1;
        while (m < m_cap[i] && meta.draft_conf[m - 1] >= options_.spec_threshold) ++m;
        m_slot[i] = m;
        R_sum += m;
      }
      R_cap = R_sum;
      for (int b : verify_buckets_) {
        if (b >= R_sum) { R_cap = b; break; }
      }
      while (R_sum < R_cap) {
        int best_i = -1;
        float best_c = -1.f;
        for (int i = 0; i < S; ++i) {
          if (m_slot[i] < m_cap[i]) {
            const float c = slot_meta_[active_slots[i]].draft_conf[m_slot[i] - 1];
            if (c > best_c) { best_c = c; best_i = i; }
          }
        }
        if (best_i < 0) break;
        ++m_slot[best_i];
        ++R_sum;
      }
    }

    // 2) Ensure full-attention KV pages for slot_pos + m_slot[i] across all ranks (shrink m if pool is tight).
    auto calc_extra_pages = [&]() {
      int extra = 0;
      for (int i = 0; i < S; ++i) {
        const int s = active_slots[i];
        const int needed = (slot_pos_host_[s] + m_slot[i] + bk::kPageSize - 1) / bk::kPageSize;
        extra += std::max(0, needed - int(slot_pages_[s].size()));
      }
      return extra;
    };
    while (calc_extra_pages() > int(free_pages_.size())) {
      bool shrank = false;
      for (int i = 0; i < S; ++i) {
        if (m_slot[i] > 1) { --m_slot[i]; shrank = true; break; }
      }
      if (!shrank) throw std::runtime_error("step_slots_multi: paged KV pool exhausted");
    }
    R_sum = 0;
    for (int i = 0; i < S; ++i) {
      ensure_pages_all_unlocked(active_slots[i], slot_pos_host_[active_slots[i]] + m_slot[i]);
      R_sum += m_slot[i];
    }
    R_cap = R_sum;
    for (int b : verify_buckets_) {
      if (b >= R_sum) { R_cap = b; break; }
    }

    // 3) Pack verify input descriptor
    TpVerifyStepInput vin{};
    vin.R_cap = R_cap;
    vin.S = S;
    vin.active_slots.fill(-1);
    vin.slot_row0.fill(0);
    vin.slot_m.fill(0);
    vin.row_slot.fill(-1);
    vin.row_off.fill(0);
    vin.row_base.fill(0);
    int cursor = 0;
    int max_P_upper = 0;
    for (int i = 0; i < S; ++i) {
      const int s = active_slots[i];
      const int m = m_slot[i];
      const SlotMeta& meta = slot_meta_[s];
      bk::BannedList bl{};
      bl.count = meta.params.banned.count;
      for (int k = 0; k < bl.count && k < bk::BannedList::kMax; ++k) bl.ids[k] = meta.params.banned.ids[k];
      vin.active_slots[i] = s;
      vin.slot_row0[i] = cursor;
      vin.slot_m[i] = m;
      vin.tokens[cursor] = inputs[i].token;
      vin.row_slot[cursor] = s;
      vin.row_off[cursor] = 0;
      vin.row_base[cursor] = cursor;
      vin.banned[cursor] = bl;
      for (int j = 1; j < m; ++j) {
        vin.tokens[cursor + j] = meta.draft_tokens[j - 1];
        vin.row_slot[cursor + j] = s;
        vin.row_off[cursor + j] = j;
        vin.row_base[cursor + j] = cursor;
        vin.banned[cursor + j] = bl;
      }
      cursor += m;
      max_P_upper = std::max(max_P_upper, slot_pos_host_[s] + m);
    }

    const std::span<const int> active_span(active_slots.data(), std::size_t(S));
    const int MB = options_.max_batch;
    auto* acc_dev = static_cast<std::int32_t*>(spec_accept_dev_.p);
    auto* com_dev = acc_dev + MB;
    auto* nxt_dev = acc_dev + 2 * MB;
    auto* acc_h = static_cast<std::int32_t*>(spec_accept_host_.data());
    auto* com_h = acc_h + MB;
    auto* nxt_h = acc_h + 2 * MB;
    auto* dtok_h = static_cast<std::int32_t*>(spec_draft_host_.data());
    auto* dconf_h = reinterpret_cast<float*>(dtok_h + std::size_t(MB) * kd::GAMMA);

    // 4) Launch target verify on both GPUs + zero-bubble accept -> rollback -> BatchDrafter on rank 0
    pool_.run_all([&](int r) {
      cudaStream_t st = pool_.stream(r);
      if (r == 0) CUDA_CHECK(cudaEventRecord(ev_v0_, st));
      ranks_[r].decoder->run_verify_step(vin, R_cap, st);
      if (r == 0) {
        CUDA_CHECK(cudaEventRecord(ev_v1_, st));
        batch_drafter_->upload_active_desc(active_span);
        kd::batch_greedy_accept(ranks_[0].decoder->argmax_out_dev(), ranks_[0].decoder->active_slots_dev(),
                                ranks_[0].decoder->slot_row0_dev(), ranks_[0].decoder->slot_m_dev(),
                                static_cast<const std::int64_t*>(ranks_[0].kv_pool.slot_pos.p),
                                ranks_[0].decoder->taps_dev(), batch_drafter_->slots_dev(), S, acc_dev, com_dev,
                                nxt_dev, st);
        CUDA_CHECK(cudaMemcpyAsync(acc_h, acc_dev, std::size_t(3 * MB) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaEventRecord(accept_ready_ev_, st));
        ranks_[0].decoder->run_verify_rollback(S, com_dev, st);
        batch_drafter_->step_commit_and_draft(active_span, max_P_upper, /*desc_already_uploaded=*/true);
        for (int i = 0; i < S; ++i) {
          const int s = active_slots[i];
          CUDA_CHECK(cudaMemcpyAsync(dtok_h + std::size_t(i) * kd::GAMMA, batch_drafter_->slot_tokens_dev(s),
                                     std::size_t(dlen) * sizeof(std::int32_t), cudaMemcpyDeviceToHost, st));
          CUDA_CHECK(cudaMemcpyAsync(dconf_h + std::size_t(i) * kd::GAMMA, batch_drafter_->slot_conf_dev(s),
                                     std::size_t(dlen) * sizeof(float), cudaMemcpyDeviceToHost, st));
        }
        CUDA_CHECK(cudaEventRecord(ev_d1_, st));
      }
    });

    // 5) Overlap rank 1 rollback behind rank 0's BatchDrafter (wakes right after batch_greedy_accept)
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventSynchronize(accept_ready_ev_));
    CUDA_CHECK(cudaSetDevice(1));
    ranks_[1].decoder->run_verify_rollback_host(S, com_h, pool_.stream(1));

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaStreamSynchronize(pool_.stream(1)));
    CUDA_CHECK(cudaSetDevice(0));

    float v_ms = 0.f, d_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&v_ms, ev_v0_, ev_v1_));
    CUDA_CHECK(cudaEventElapsedTime(&d_ms, ev_v1_, ev_d1_));
    spec_steps_ += 1;
    spec_slot_rounds_ += S;
    spec_verify_ms_tot_ += double(v_ms);
    spec_draft_ms_tot_ += double(d_ms);
    // Update per-bucket step time EMA (for EV-optimal bucket selection across depths)
    const float step_ms = v_ms + d_ms;
    if (auto it_b = bucket_ms_ema_.find(R_cap); it_b != bucket_ms_ema_.end()) {
      it_b->second = 0.85f * it_b->second + 0.15f * step_ms;
    } else {
      bucket_ms_ema_[R_cap] = step_ms;
    }

    // 6) Emit accepted drafts + next token per slot, update positions, shortlist, and next draft buffers
    std::array<std::int32_t, 16> emitted_all{};
    int emitted_n = 0;
    int step_acc_sum = 0;
    for (int i = 0; i < S; ++i) {
      const int s = active_slots[i];
      const int acc = acc_h[i];
      const int com = com_h[i];
      step_acc_sum += acc;
      const std::int32_t next_tok = nxt_h[i];
      SlotMeta& meta = slot_meta_[s];
      BatchSlotOutput out{};
      for (int j = 0; j < acc; ++j) {
        out.tokens[j] = meta.draft_tokens[j];
        emitted_all[emitted_n++] = meta.draft_tokens[j];
      }
      out.tokens[acc] = next_tok;
      emitted_all[emitted_n++] = next_tok;
      out.count = com;
      out.verify_width = m_slot[i];
      out.accepted = acc;
      outputs[i] = out;

      slot_pos_host_[s] += com;
      batch_drafter_->set_slot_length(s, slot_pos_host_[s]);
      for (int k = 0; k < dlen; ++k) {
        meta.draft_tokens[k] = dtok_h[std::size_t(i) * kd::GAMMA + k];
        meta.draft_conf[k] = dconf_h[std::size_t(i) * kd::GAMMA + k];
      }
      meta.has_draft = true;
      spec_accepted_tot_ += acc;
      if (m_slot[i] >= 1 && m_slot[i] <= 4) spec_m_hist_[m_slot[i]] += 1;
    }
    if (emitted_n > 0) {
      batch_drafter_->note_tokens(std::span<const std::int32_t>(emitted_all.data(), std::size_t(emitted_n)));
    }
    if (S == options_.max_batch) {
      spec_sfull_steps_ += 1;
      spec_sfull_toks_ += emitted_n;
      spec_sfull_ms_ += double(step_ms);
    }
    if (predicted_acc_sum >= 0.5f) {
      const float ratio = std::clamp(float(step_acc_sum) / predicted_acc_sum, 0.6f, 1.25f);
      spec_alpha_cal_ = std::clamp(0.95f * spec_alpha_cal_ + 0.05f * (spec_alpha_cal_ * ratio), 0.75f, 1.05f);
    }
  }

  void release_slot(int slot) override {
    std::lock_guard<std::mutex> lk(mu_);
    free_slot_unlocked(slot);
  }

  GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate(Prompt{prompt, {}}, max_new_tokens, sampling, stop_ids, on_tokens, PrefillCallback{});
  }
  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate(full_prompt, max_new_tokens, sampling, stop_ids, on_tokens, PrefillCallback{});
  }
  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                         const PrefillCallback& on_prefill) override {
    return generate_from_slots(full_prompt, max_new_tokens, sampling, stop_ids, on_tokens, on_prefill);
  }

 private:
  struct RankCtx {
    Weights weights;
    Ablation ablation;
    TpBatchStatePool kv_pool;
    std::unique_ptr<TpPrefiller> prefiller;
    std::unique_ptr<TpBatchDecoder> decoder;
    DeviceBuffer chunk_tokens;
  };
  struct SlotMeta {
    head::SamplingParams params{};
    head::SamplerRng rng{};
    std::vector<std::int32_t> forced;
    std::size_t forced_idx = 0;
    bool spec_eligible = false;
    bool has_draft = false;
    std::array<std::int32_t, kd::GAMMA> draft_tokens{};
    std::array<float, kd::GAMMA> draft_conf{};
  };

  int free_slot_count_unlocked() const {
    int n = 0;
    for (bool u : slot_in_use_) if (!u) ++n;
    return n;
  }
  int alloc_slot_unlocked() {
    for (int s = 0; s < options_.max_batch; ++s) {
      if (!slot_in_use_[s]) {
        slot_in_use_[s] = true;
        slot_pages_[s].clear();
        slot_pos_host_[s] = 0;
        return s;
      }
    }
    return -1;
  }
  bool ensure_pages_all_unlocked(int s, int total_tokens) {
    const int needed = (total_tokens + bk::kPageSize - 1) / bk::kPageSize;
    if (needed > max_pages_per_slot_) return false;
    auto& pages = slot_pages_[s];
    const int cur = int(pages.size());
    if (needed <= cur) return true;
    const int add = needed - cur;
    if (int(free_pages_.size()) < add) return false;
    for (int i = 0; i < add; ++i) {
      pages.push_back(free_pages_.back());
      free_pages_.pop_back();
    }
    // Mirror exact page table update to all 4 ranks
    pool_.run_all([&](int r) {
      auto& kv = ranks_[r].kv_pool;
      std::int32_t* dev_row = static_cast<std::int32_t*>(kv.page_table.p) + std::size_t(s) * max_pages_per_slot_;
      CUDA_CHECK(cudaMemcpyAsync(dev_row + cur, pages.data() + cur, std::size_t(add) * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, pool_.stream(r)));
    });
    return true;
  }
  void free_slot_unlocked(int s) {
    if (s < 0 || s >= options_.max_batch || !slot_in_use_[s]) return;
    for (int p : slot_pages_[s]) free_pages_.push_back(p);
    slot_pages_[s].clear();
    slot_pos_host_[s] = 0;
    slot_in_use_[s] = false;
    slot_meta_[s].has_draft = false;
    slot_meta_[s].spec_eligible = false;
    if (batch_drafter_ && free_slot_count_unlocked() == options_.max_batch) {
      CUDA_CHECK(cudaSetDevice(0));
      batch_drafter_->reset_context_shortlist();
    }
  }

  void capture_verify_buckets_unlocked() {
    for (int R_cap : verify_buckets_) {
      for (int w = 0; w < 2; ++w) {
        pool_.run_all([&](int r) {
          ranks_[r].decoder->warmup_verify_bucket(R_cap, pool_.stream(r));
          CUDA_CHECK(cudaStreamSynchronize(pool_.stream(r)));
        });
      }
      if (enable_graphs_) {
        std::array<bool, TP_SIZE> ok{};
        pool_.run_all([&](int r) {
          ok[r] = ranks_[r].decoder->capture_verify_bucket(R_cap, pool_.stream(r));
          CUDA_CHECK(cudaStreamSynchronize(pool_.stream(r)));
        });
        const bool all_ok = std::all_of(ok.begin(), ok.end(), [](bool v) { return v; });
        if (!all_ok) {
          ingine::engine::server_log().warn("TP{}/EP{} verify CUDA graph capture failed for R={} — falling back to eager",
                                            TP_SIZE, TP_SIZE, R_cap);
          pool_.run_all([&](int r) { ranks_[r].decoder->clear_graphs(); });
          enable_graphs_ = false;
          break;
        }
      }
    }
  }

  void capture_buckets_unlocked(bool greedy) {
    for (int B : buckets_) {
      for (int w = 0; w < 2; ++w) {
        pool_.run_all([&](int r) {
          ranks_[r].decoder->warmup_bucket(B, pool_.stream(r), greedy);
          CUDA_CHECK(cudaStreamSynchronize(pool_.stream(r)));
        });
      }
      if (enable_graphs_) {
        std::array<bool, TP_SIZE> ok{};
        pool_.run_all([&](int r) {
          ok[r] = ranks_[r].decoder->capture_bucket(B, pool_.stream(r), greedy);
          CUDA_CHECK(cudaStreamSynchronize(pool_.stream(r)));
        });
        const bool all_ok = std::all_of(ok.begin(), ok.end(), [](bool v) { return v; });
        if (!all_ok) {
          ingine::engine::server_log().warn("TP{}/EP{} CUDA graph capture failed for B={} (greedy={}) — falling back to eager",
                                            TP_SIZE, TP_SIZE, B, greedy ? 1 : 0);
          pool_.run_all([&](int r) { ranks_[r].decoder->clear_graphs(); });
          enable_graphs_ = false;
          break;
        }
      }
    }
  }

  RuntimeOptions options_;
  bool spec_active_ = false;
  int spec_max_m_ = 1;
  TpWorkerPool pool_;
  std::array<RankCtx, TP_SIZE> ranks_;
  int prefill_chunk_ = 3072;
  int max_pages_per_slot_ = 0;
  int num_pages_ = 0;
  bool enable_graphs_ = true;
  bool greedy_graphs_ready_ = false;
  bool logits_graphs_ready_ = false;
  std::vector<int> buckets_, verify_buckets_;
  DeviceBuffer prefill_logits_, sampled_dev_, prefill_taps_, spec_accept_dev_;
  PinnedHostBuffer sampled_host_, spec_accept_host_, spec_draft_host_;
  Weights drafter_weights_;
  std::unique_ptr<BatchDrafter> batch_drafter_;
  cudaEvent_t accept_ready_ev_ = nullptr, ev_v0_ = nullptr, ev_v1_ = nullptr, ev_d1_ = nullptr;
  std::int64_t spec_steps_ = 0, spec_slot_rounds_ = 0, spec_accepted_tot_ = 0;
  std::int64_t spec_sfull_steps_ = 0, spec_sfull_toks_ = 0;
  std::array<std::int64_t, 5> spec_m_hist_{};
  double spec_verify_ms_tot_ = 0.0, spec_draft_ms_tot_ = 0.0, spec_sfull_ms_ = 0.0;
  std::map<int, float> bucket_ms_ema_;
  float spec_alpha_cal_ = 0.98f;
  std::vector<int> free_pages_;
  std::vector<std::vector<int>> slot_pages_;
  std::vector<int> slot_pos_host_;
  std::vector<bool> slot_in_use_;
  std::vector<SlotMeta> slot_meta_;
  mutable std::mutex mu_;
};

// ----------------------------------------------------------------------------- TpDecoder (M = 1, 2, 4, 8 for TP4 + DSpark --spec)

class TpDecoder {
 public:
  TpDecoder(const Weights& w, const Ablation& abl, State& st, int rank, ncclComm_t comm, TpWorkerPool::RankInfo& rinfo)
      : w_(w), abl_(abl), st_(st), rank_(rank), comm_(comm), rinfo_(rinfo),
        gemm_(std::make_unique<pk::Gemm>()),
        attn_ws_(attn::workspace_bytes(8)),
        wo_f32_(std::size_t(8) * H * sizeof(float)),
        moe_f32_(std::size_t(8) * H * sizeof(float)),
        router_logits_(std::size_t(8) * EXPERTS * sizeof(float)),
        counts_(EXPERTS * sizeof(int)),
        tiles_n_(sizeof(int)),
        pair_token_(std::size_t(8) * TOP_K * sizeof(int)),
        pair_dst_(std::size_t(8) * TOP_K * sizeof(int)),
        tiles_(std::size_t(pk::moe_tiles_capacity(8)) * sizeof(int4)),
        route_done_(sizeof(unsigned)),
        down_act_(std::size_t(8) * TOP_K * H * sizeof(ActT)),
        local_logits_(std::size_t(8) * LOCAL_VOCAB_CHUNK * sizeof(float)),
        gathered_logits_(std::size_t(TP_SIZE) * 8 * LOCAL_VOCAB_CHUNK * sizeof(float)) {
    attn_ws_.fill(0);
    wo_f32_.fill(0);
    moe_f32_.fill(0);
    counts_.fill(0);
    tiles_n_.fill(0);
    route_done_.fill(0);
    segments_[0] = upload_ints(proj_layout(false, true));
    segments_[1] = upload_ints(proj_layout(false, false));
    full_segments_ = upload_ints(proj_layout(true, false));
  }

  void decode(Buffers& b, const std::int64_t* pos, bool history, cudaStream_t stream) {
    const int m = b.m;
    const Weights& w = w_;
    const Ablation& abl = abl_;
    State& st = st_;
    const std::size_t slab = std::size_t(m) * H;
    auto f32m = [](const Tensor& t) { return static_cast<float*>(t.data); };
    auto bf16m = [](const Tensor& t) { return static_cast<ActT*>(t.data); };

    auto mix_params = [&](const std::vector<const float*>& blocks, const float* base, const float* write,
                          const float* abl_r, const float* abl_w, float* partial_out, const Tensor* queries,
                          const float* norm_w, ActT* out_bf16, float* out_f32) {
      dense::MixParams p;
      p.blocks = blocks.empty() ? nullptr : reinterpret_cast<const float* const*>(block_table(blocks));
      p.nblocks = int(blocks.size());
      p.base = base;
      p.write = write;
      p.abl_r = write ? abl_r : nullptr;
      p.abl_d = write && abl.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;
      p.abl_w = write ? abl_w : nullptr;
      p.partial_out = partial_out;
      p.queries = queries ? f32p(*queries) : nullptr;
      p.Q = queries ? queries->ne[1] : 1;
      p.norm_w = norm_w;
      p.out_bf16 = out_bf16;
      p.out_f32 = out_f32;
      p.M = m;
      return p;
    };

    if (w.embed_q8()) {
      dense::embed_q8(static_cast<const std::int64_t*>(b.token.data), m, w.embed.data, f32p(w.embed_norm), f32m(b.B0), false, stream);
    } else if (w.embed_q4()) {
      dense::embed_q4(static_cast<const std::int64_t*>(b.token.data), m, VOCAB, w.embed.data, f32p(w.embed_norm), f32m(b.B0), false, stream);
    } else {
      dense::embed(static_cast<const std::int64_t*>(b.token.data), m, static_cast<const ActT*>(w.embed.data), f32p(w.embed_norm), f32m(b.B0), false, stream);
    }

    int prev_v = -1, tap = 0;
    const float* attn_dirs = f32p(abl.attn_dirs);
    const float* attn_w = f32p(abl.attn_w);
    const float* mlp_dirs = f32p(abl.mlp_dirs);
    const float* mlp_w = f32p(abl.mlp_w);
    ActT* taps = bf16m(b.taps);

    for (int v = 0; v < NUM_VISITS; ++v) {
      const int l = VISITS[v].layer, p = VISITS[v].pass;
      const Layer& layer = w.layers[l];
      const bool loop = is_loop_layer(l);
      const bool msgs = loop && p > 0;
      const bool boundary = l % BLOCK == 0;
      const std::vector<const float*> blocks = b.blocks(l, p);

      float* hid;
      const float* base;
      const float* write;
      if (v == 0) {
        hid = f32m(b.B0), base = f32p(b.B0), write = nullptr;
      } else if (l == LOOP_START && p > 0) {
        hid = f32m(b.B1), base = f32p(b.B1), write = nullptr;
      } else {
        hid = l == 6 ? f32m(b.B1) : l == 12 ? f32m(b.B2[p]) : l == 18 ? f32m(b.E[2]) : f32m(b.hid);
        base = f32p(b.part), write = f32p(b.y2);
      }
      const int pv = prev_v >= 0 ? prev_v : 0;
      const float* r_prev = mlp_dirs + std::size_t(pv) * H;
      const float* w_prev = mlp_w + 4 * std::size_t(pv);
      dense::MixParams msg_mix;
      if (msgs) {
        const int i = l - LOOP_START;
        const float* src_base = i == 0 ? f32p(b.part) : f32p(b.E[p - 1]);
        const float* src_write = i == 0 ? f32p(b.y2) : nullptr;
        const Tensor mq = rm(f32m(w.message_queries) + std::size_t(i) * 4 * H, DType::FP32, {4, H});
        msg_mix = mix_params({f32p(b.B2[p - 1])}, src_base, src_write, r_prev, w_prev,
                             i == 0 ? f32m(b.E[p - 1]) : nullptr, &mq, nullptr, nullptr, f32m(b.msgs));
        if (i == 0) write = nullptr;
      }
      const Tensor* queries = layer.attn_query.data ? &layer.attn_query : nullptr;
      float* partial_out = hid != base ? hid : nullptr;
      dense::MixParams ap =
          msgs ? mix_params(blocks, base, write, r_prev, w_prev, partial_out, queries, nullptr, nullptr, f32m(b.mixf))
               : mix_params(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr,
                            f32p(layer.attn_norm_mix), bf16m(b.xin), nullptr);
      if (tap < 3 && v == TAP_VISITS[tap]) {
        ap.new_bf16 = taps + std::size_t(tap) * H;
        ap.new_bf16_stride = std::int64_t(NUM_TAPS) * H;
        ++tap;
      }
      if (msgs) dense::mix_pair(ap, msg_mix, false, stream);
      else dense::mix(ap, false, stream);
      if (msgs)
        dense::message_gates(f32p(b.mixf), !layer.full, f32p(b.msgs), 3, m,
                             static_cast<const ActT*>(layer.qkv_gate_down.data), layer.qkv_gate_forget16.data,
                             layer.qkv_gate_gain16.data, false, f32p(layer.attn_norm3), bf16m(b.xin), nullptr, false, stream);

      // ---- Sharded attention
      const bool single = !layer.full && !msgs;
      const DeviceBuffer& seg = layer.full ? full_segments_ : segments_[single ? 0 : 1];
      const Tensor& proj_out = layer.full ? b.proj_full : b.proj_sliding;
      const int R = proj_rows(layer.full);
      dense::proj(bf16m(b.xin), m, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full),
                  f32m(proj_out), false, stream);

      attn::DecodeArgs a;
      a.proj = f32p(proj_out);
      a.M = m;
      a.q_scale = f32p(layer.q_scale);
      a.r_scale = f32p(layer.r_scale);
      a.rel_proj = f32p(layer.rel_proj);
      a.conv_k_state = st.conv_k_at(v);
      a.conv_v_state = st.conv_v_at(v);
      a.pos = pos;
      a.workspace = attn_ws_.p;
      a.workspace_size = attn_ws_.bytes;
      a.out = bf16m(b.attn_out);
      if (layer.full) {
        const int fi = FULL_INDEX[v];
        a.k_conv = f32p(layer.k_static);
        a.v_conv = f32p(layer.v_static);
        a.k_cache = st.k_full_at(fi).data;
        a.v_cache = st.v_full_at(fi).data;
        a.cache_fp8 = st.kv_fp8;
        a.cache_q4 = false;
        a.cache_rows = st.max_ctx;
        a.kpos = nullptr;
        a.window = 0;
        a.log_scaling = true;
      } else {
        const int ri = RING_INDEX[v];
        a.k_conv = f32p(layer.k_conv);
        a.v_conv = f32p(layer.v_conv);
        a.k_cache = st.k_ring_at(ri).data;
        a.v_cache = st.v_ring_at(ri).data;
        a.cache_fp8 = false;
        a.cache_q4 = false;
        a.cache_rows = WINDOW;
        a.kpos = st.ring_pos_at(ri);
        a.window = WINDOW;
        a.log_scaling = false;
      }
      attn::attn_decode(a, stream);

      // ---- Sharded wo + P2P/NCCL AllReduce + static_conv_decode
      dense::gemv(bf16m(b.attn_out), m, HEADS * HEAD_DIM, matrix(layer.wo), static_cast<float*>(wo_f32_.p), false, stream);
      if (rinfo_.p2p_ar.active() && m * H <= rinfo_.p2p_ar.cap && 2 * v + 1 < rinfo_.p2p_ar.slots)
        bk::p2p_allreduce2(static_cast<float*>(wo_f32_.p), m * H, rinfo_.p2p_ar, 2 * v, stream);
      else
        NCCL_CHECK(ncclAllReduce(wo_f32_.p, wo_f32_.p, std::size_t(m) * H, ncclFloat32, ncclSum, comm_, stream));
      dense::static_conv_decode(static_cast<const float*>(wo_f32_.p), m, f32p(layer.attn_conv), st.conv_attn_at(v),
                                f32m(b.y), history ? f32m(b.hist_attn) + std::size_t(v) * m * H : nullptr, stream);

      // ---- MLP read + gate
      std::vector<const float*> mlp_blocks = blocks;
      if (boundary) mlp_blocks.push_back(hid);
      const float* mbase = boundary ? nullptr : hid;
      dense::MixParams mp =
          msgs ? mix_params(mlp_blocks, mbase, f32p(b.y), attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), f32m(b.part),
                            &layer.mlp_query, nullptr, nullptr, f32m(b.mixm))
               : mix_params(mlp_blocks, mbase, f32p(b.y), attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), f32m(b.part),
                            &layer.mlp_query, f32p(layer.mlp_norm1), bf16m(b.xm), nullptr);
      if (history) {
        int c = 0;
        for (auto [dst, s] : {std::pair{&b.hist_k, SEG_K}, std::pair{&b.hist_v, SEG_V}})
          mp.copies[c++] = dense::MixRowCopy{f32p(proj_out) + SEG_BEGIN[s], R, f32m(*dst) + std::size_t(v) * m * KV_DIM,
                                             KV_DIM, KV_DIM};
      }
      dense::mix(mp, false, stream);
      if (msgs)
        dense::message_gates(f32p(b.mixm), false, f32p(b.msgs) + 3 * slab, 1, m,
                             static_cast<const ActT*>(layer.mlp1_gate_down.data), layer.mlp1_gate_forget16.data,
                             layer.mlp1_gate_gain16.data, false, f32p(layer.mlp_norm1), bf16m(b.xm), nullptr, false, stream);

      // ---- TP/EP MoE (wgemm / Volta decode_v70) + P2P/NCCL AllReduce + static_conv_decode
      const pk::MoeWork work{static_cast<int*>(counts_.p), static_cast<int*>(tiles_n_.p),
                             static_cast<int4*>(tiles_.p), static_cast<int*>(pair_token_.p),
                             static_cast<int*>(pair_dst_.p)};
      if (!pk::moe_route_decode(bf16m(b.xm), m, f32p(layer.router), f32p(layer.router_bias),
                                static_cast<int*>(b.idx.data), f32m(b.wts), static_cast<float*>(router_logits_.p),
                                static_cast<unsigned*>(route_done_.p), work, stream)) {
        gemm_->run(bf16m(b.xm), H, static_cast<const ActT*>(layer.router_bf16.data), H,
                   static_cast<float*>(router_logits_.p), EXPERTS, true, m, EXPERTS, H, stream);
        CUDA_CHECK(cudaMemsetAsync(counts_.p, 0, EXPERTS * sizeof(int), stream));
        pk::moe_topk(static_cast<const float*>(router_logits_.p), f32p(layer.router_bias), m,
                     static_cast<int*>(b.idx.data), f32m(b.wts), static_cast<int*>(counts_.p), stream);
        pk::moe_route(static_cast<const int*>(b.idx.data), m, work, stream);
      }
      const Experts& ex = layer.experts;
      pk::MoeExperts me;
      me.fmt = static_cast<int>(ex.w13_fmt);
      me.w2_fmt = static_cast<int>(ex.w2_fmt);
      me.w13_factor = f32p(ex.w13_factor);
      me.w2_factor = f32p(ex.w2_factor);
      me.w13_ptrs = static_cast<const std::uint8_t* const*>(ex.w13_ptrs.data);
      me.w13_scale_ptrs = static_cast<const std::uint8_t* const*>(ex.w13_scale_ptrs.data);
      me.w2_ptrs = static_cast<const std::uint8_t* const*>(ex.w2_ptrs.data);
      me.w2_scale_ptrs = static_cast<const std::uint8_t* const*>(ex.w2_scale_ptrs.data);
      me.active_mask = static_cast<const std::uint8_t*>(ex.active_mask.data);
      pk::moe_gate_up(me, bf16m(b.xm), m, work, bf16m(b.act), stream);
      pk::moe_down(me, bf16m(b.act), m, work, static_cast<ActT*>(down_act_.p), stream);
      pk::moe_combine_ep(static_cast<const ActT*>(down_act_.p), f32p(b.wts), static_cast<const int*>(b.idx.data),
                         me.active_mask, m, static_cast<float*>(moe_f32_.p), stream);
      if (rinfo_.p2p_ar.active() && m * H <= rinfo_.p2p_ar.cap && 2 * v + 1 < rinfo_.p2p_ar.slots)
        bk::p2p_allreduce2(static_cast<float*>(moe_f32_.p), m * H, rinfo_.p2p_ar, 2 * v + 1, stream);
      else
        NCCL_CHECK(ncclAllReduce(moe_f32_.p, moe_f32_.p, std::size_t(m) * H, ncclFloat32, ncclSum, comm_, stream));
      dense::static_conv_decode(static_cast<const float*>(moe_f32_.p), m, f32p(layer.mlp_conv), st.conv_mlp_at(v),
                                f32m(b.y2), history ? f32m(b.hist_mlp) + std::size_t(v) * m * H : nullptr, stream);
      prev_v = v;
    }

    // ---- Final read + sharded LM head + NCCL AllGather + unpack on rank 0
    const Tensor final_norm1 = rm(w.final_norm.data, DType::FP32, {1, H});
    dense::MixParams fp = mix_params({f32p(b.B0), f32p(b.B1), f32p(b.B2[2]), f32p(b.E[2])}, f32p(b.part), f32p(b.y2),
                                     mlp_dirs + std::size_t(prev_v) * H, mlp_w + 4 * std::size_t(prev_v), nullptr, &w.final_query,
                                     f32p(final_norm1), bf16m(b.xf), nullptr);
    fp.out_copy = taps + std::size_t(tap) * H;
    fp.out_copy_stride = std::int64_t(NUM_TAPS) * H;
    dense::mix(fp, false, stream);
    dense::head(bf16m(b.xf), m, matrix(w.unembed), static_cast<float*>(local_logits_.p), false, stream);
    NCCL_CHECK(ncclAllGather(local_logits_.p, gathered_logits_.p, std::size_t(m) * LOCAL_VOCAB_CHUNK, ncclFloat32, comm_, stream));
    if (rank_ == 0) {
      unpack_tp_logits(static_cast<const float*>(gathered_logits_.p), m, TP_SIZE, LOCAL_VOCAB_CHUNK, UNPADDED_VOCAB,
                       f32m(b.logits), stream);
    }
  }

 private:
  const std::int64_t* block_table(const std::vector<const float*>& blocks) {
    auto it = tables_.find(blocks);
    if (it == tables_.end()) {
      DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
      if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
      it = tables_.emplace(blocks, std::move(b)).first;
    }
    return static_cast<const std::int64_t*>(it->second.p);
  }

  const Weights& w_;
  const Ablation& abl_;
  State& st_;
  int rank_;
  ncclComm_t comm_;
  TpWorkerPool::RankInfo& rinfo_;
  std::unique_ptr<pk::Gemm> gemm_;
  DeviceBuffer attn_ws_, wo_f32_, moe_f32_, router_logits_, counts_, tiles_n_, pair_token_, pair_dst_, tiles_,
      route_done_, down_act_, local_logits_, gathered_logits_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;
  std::array<DeviceBuffer, 2> segments_;
  DeviceBuffer full_segments_;
};

// ----------------------------------------------------------------------------- TpSpecGenerator (TP4 + DSpark --spec)

class TpSpecGenerator final : public Generator {
 public:
  explicit TpSpecGenerator(const RuntimeOptions& options)
      : options_(options), host_token_(sizeof(std::int32_t)), host_m_(sizeof(std::int32_t)),
        host_drafts_(kd::GAMMA * sizeof(std::int32_t)), host_accept_(sizeof(kd::AcceptResult)) {
    validate_tp_runtime_options(options, TP_SIZE);
    prefill_chunk_ = std::min(options.prefill_chunk, options.max_ctx);
    const std::vector<int> widths = options.spec ? std::vector<int>{1, 2, 4, 8} : std::vector<int>{1};

    const bool use_head_shards = is_gguf_model(options_.model_dir);
    pool_.run_all([&](int r) {
      auto& rk = ranks_[r];
      rk.weights = load_weights_sharded(options_.model_dir, options_.dense_fp8, pool_.stream(r), r, TP_SIZE,
                                        /*keep_full_unembed=*/(r == 0 && options_.spec && !use_head_shards));
      rk.ablation = load_ablation(options_.ablation);
      rk.state = make_state(options_.max_ctx, options_.kv_fp8, false);
      rk.prefiller = std::make_unique<TpPrefiller>(rk.weights, rk.ablation, prefill_chunk_, pool_.stream(r),
                                                   pool_.comm(r), pool_.info(r));
      rk.decoder = std::make_unique<TpDecoder>(rk.weights, rk.ablation, rk.state, r, pool_.comm(r), pool_.info(r));
      rk.chunk_tokens = DeviceBuffer(std::size_t(prefill_chunk_) * sizeof(std::int32_t));
      rk.pos = DeviceBuffer(sizeof(std::int64_t));
      rk.pos.fill(0);
      rk.token = DeviceBuffer(8 * sizeof(std::int64_t));
      rk.token.fill(0);
      for (int m : widths) {
        Buffers b = make_buffers(m);
        b.token = rm(rk.token.p, DType::I64, {m});
        rk.buffers.emplace(m, std::move(b));
      }
      // Verify backup for rollback
      std::size_t bbytes = 0;
      auto align256 = [](std::size_t n) { return (n + 255) / 256 * 256; };
      std::size_t off_conv[4];
      for (int i = 0; i < 4; ++i) {
        off_conv[i] = bbytes;
        bbytes = align256(bbytes + std::size_t(NUM_VISITS) * 3 * CONV_COLS[i] * sizeof(float));
      }
      const std::size_t off_k = bbytes;
      bbytes = align256(bbytes + std::size_t(NUM_SLIDING_VISITS) * 8 * KV_DIM * sizeof(ActT));
      const std::size_t off_v = bbytes;
      bbytes = align256(bbytes + std::size_t(NUM_SLIDING_VISITS) * 8 * KV_DIM * sizeof(ActT));
      const std::size_t off_p = bbytes;
      bbytes = align256(bbytes + std::size_t(NUM_SLIDING_VISITS) * 8 * sizeof(std::int64_t));
      rk.backup_storage = DeviceBuffer(bbytes);
      auto* bp = static_cast<char*>(rk.backup_storage.p);
      for (int i = 0; i < 4; ++i) rk.backup.conv[i] = reinterpret_cast<float*>(bp + off_conv[i]);
      rk.backup.k = reinterpret_cast<ActT*>(bp + off_k);
      rk.backup.v = reinterpret_cast<ActT*>(bp + off_v);
      rk.backup.pos = reinterpret_cast<std::int64_t*>(bp + off_p);
      // Prompt checkpoint buffer (sliding rings + ring_pos + 4 conv states)
      rk.ckpt = DeviceBuffer(state_ckpt_bytes(rk.state));
      CUDA_CHECK(cudaStreamSynchronize(pool_.stream(r)));
    });

    // Capture decode CUDA graphs for M in widths
    const char* no_graph = std::getenv("INGINE_TP_NO_GRAPH");
    const bool enable_graphs = !(no_graph && no_graph[0] == '1');
    for (int m : widths) {
      const bool hist = (m > 1);
      for (int w = 0; w < 2; ++w) {
        pool_.run_all([&](int r) {
          auto& rk = ranks_[r];
          rk.decoder->decode(rk.buffers.at(m), static_cast<const std::int64_t*>(rk.pos.p), hist, pool_.stream(r));
          CUDA_CHECK(cudaStreamSynchronize(pool_.stream(r)));
        });
      }
      if (enable_graphs) {
        std::array<bool, TP_SIZE> ok{};
        pool_.run_all([&](int r) {
          auto& rk = ranks_[r];
          cudaGraph_t g = nullptr;
          if (cudaStreamBeginCapture(pool_.stream(r), cudaStreamCaptureModeRelaxed) != cudaSuccess) { ok[r] = false; return; }
          try {
            rk.decoder->decode(rk.buffers.at(m), static_cast<const std::int64_t*>(rk.pos.p), hist, pool_.stream(r));
          } catch (...) {
            cudaStreamEndCapture(pool_.stream(r), &g);
            if (g) cudaGraphDestroy(g);
            ok[r] = false;
            return;
          }
          if (cudaStreamEndCapture(pool_.stream(r), &g) != cudaSuccess || !g) {
            if (g) cudaGraphDestroy(g);
            ok[r] = false;
            return;
          }
          cudaGraphExec_t exec = nullptr;
          const cudaError_t ie = cudaGraphInstantiate(&exec, g, 0);
          cudaGraphDestroy(g);
          if (ie != cudaSuccess || !exec) { ok[r] = false; return; }
          CUDA_CHECK(cudaGraphUpload(exec, pool_.stream(r)));
          rk.graphs[m] = exec;
          ok[r] = true;
          CUDA_CHECK(cudaStreamSynchronize(pool_.stream(r)));
        });
        if (!std::all_of(ok.begin(), ok.end(), [](bool v) { return v; })) {
          ingine::engine::server_log().warn("TP{}/EP{} spec graph capture failed for M={} — falling back to eager", TP_SIZE, TP_SIZE, m);
          pool_.run_all([&](int r) {
            for (auto& [k, e] : ranks_[r].graphs) if (e) cudaGraphExecDestroy(e);
            ranks_[r].graphs.clear();
          });
          break;
        }
      }
    }

    // Rank 0 owns sampling buffers + Drafter (using rank 0's full_unembed view)
    CUDA_CHECK(cudaSetDevice(0));
    logits_ = DeviceBuffer(std::size_t(UNPADDED_VOCAB) * sizeof(float));
    ckpt_logits_ = DeviceBuffer(std::size_t(UNPADDED_VOCAB) * sizeof(float));
    token_dev_ = DeviceBuffer(sizeof(std::int32_t));
    width_dev_ = DeviceBuffer(sizeof(std::int32_t));
    if (options_.spec) {
      drafter_weights_.embed = ranks_[0].weights.embed;
      drafter_weights_.embed_norm = ranks_[0].weights.embed_norm;
      kd::HeadShards hs{};
      if (use_head_shards) {
        hs.n_shards = TP_SIZE;
        hs.rows_per_shard = LOCAL_VOCAB_CHUNK;
        hs.total_rows = UNPADDED_VOCAB;
        for (int r = 0; r < TP_SIZE; ++r) {
          const auto& u = ranks_[r].weights.unembed;
          hs.shard[r] = dense::Matrix{
              u.w.data, u.fp8() ? static_cast<const float*>(u.scale.data) : nullptr, u.rows(),
              static_cast<const std::int64_t*>(u.blk_off.data),
              static_cast<const std::int32_t*>(u.blk_soff.data),
              static_cast<const std::uint16_t*>(u.q8_scale.data)};
        }
      } else {
        drafter_weights_.unembed = ranks_[0].weights.full_unembed;
      }
      drafter_ = std::make_unique<Drafter>(options_.model_dir, drafter_weights_, options_.max_ctx, pool_.stream(0),
                                           Drafter::SHORTLIST_ROWS, hs);
      drafter_->capture();
      taps_ = DeviceBuffer(std::size_t(prefill_chunk_) * NUM_TAPS * HIDDEN * sizeof(ActT));
      ckpt_drafter_ = DeviceBuffer(drafter_->checkpoint_bytes());
      CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
    }
    ingine::engine::server_log().info("{}", offload_summary());
  }

  ~TpSpecGenerator() override {
    CUDA_CHECK(cudaSetDevice(0));
    drafter_.reset();
    pool_.run_all([&](int r) {
      cudaStreamSynchronize(pool_.stream(r));
      for (auto& [k, e] : ranks_[r].graphs) if (e) cudaGraphExecDestroy(e);
      ranks_[r].graphs.clear();
      ranks_[r].decoder.reset();
      ranks_[r].prefiller.reset();
    });
  }

  std::string model_id() const override {
    return options_.ablation.empty() ? "inkling-2-mini" : "inkling-2-mini-abliterated";
  }
  int max_model_len() const override { return options_.max_ctx; }
  bool supports_images() const override { return false; }

  std::string offload_summary() const override {
    return pool_.summary(options_.spec ? "spec-b1" : "plain-b1",
                         "max_ctx=" + std::to_string(options_.max_ctx) +
                             (options_.spec ? " drafter=DSpark" : " drafter=off"));
  }

  GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate(Prompt{prompt, {}}, max_new_tokens, sampling, stop_ids, on_tokens, PrefillCallback{});
  }
  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate(full_prompt, max_new_tokens, sampling, stop_ids, on_tokens, PrefillCallback{});
  }

  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                         const PrefillCallback& on_prefill) override {
    validate_tp_request(full_prompt, sampling);
    const std::span<const std::int32_t> prompt = full_prompt.ids;
    if (!sampling.forced.empty()) max_new_tokens = std::min(max_new_tokens, int(sampling.forced.size()));
    const int n = int(prompt.size());
    const int reserve = drafter_ ? 8 : 0;
    if (n + std::int64_t(max_new_tokens) + reserve > options_.max_ctx) {
      max_new_tokens = options_.max_ctx - n - reserve;
      if (max_new_tokens <= 0) throw std::invalid_argument("prompt exceeds max_ctx");
    }

    std::lock_guard<std::mutex> lk(mu_);
    GenerateStats stats;
    head::SamplerRng rng;
    rng.seed = sampling.seed ? *sampling.seed : (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();

    auto t0 = std::chrono::steady_clock::now();
    try {
      prefill_all(full_prompt, stats);
      stats.prompt_tokens = n;
      std::int32_t first = sample_r0(static_cast<const float*>(logits_.p), sampling, rng);
      if (!sampling.forced.empty()) first = sampling.forced[0];
      stats.prefill_seconds = seconds_since(t0);
      if (on_prefill) on_prefill(stats);

      t0 = std::chrono::steady_clock::now();
      if (max_new_tokens > 0) {
        if (drafter_) {
          drafter_->set_seed(rng.seed);
          generate_spec(first, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats);
        } else {
          generate_plain(first, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats);
        }
      }
      pool_.run_all([&](int r) { CUDA_CHECK(cudaStreamSynchronize(pool_.stream(r))); });
    } catch (...) {
      pool_.run_all([&](int r) { cudaStreamSynchronize(pool_.stream(r)); });
      tokens_in_state_.clear();
      ckpt_ids_.clear();
      if (drafter_) drafter_->reset();
      throw;
    }
    stats.decode_seconds = seconds_since(t0);
    return stats;
  }

 private:
  struct RankCtx {
    Weights weights;
    Ablation ablation;
    State state;
    std::unique_ptr<TpPrefiller> prefiller;
    std::unique_ptr<TpDecoder> decoder;
    DeviceBuffer chunk_tokens, pos, token, backup_storage, ckpt;
    VerifyBackup backup{};
    std::map<int, Buffers> buffers;
    std::map<int, cudaGraphExec_t> graphs;

    StatePtrs state_ptrs() const {
      return StatePtrs{{static_cast<float*>(state.conv_k.data), static_cast<float*>(state.conv_v.data),
                        static_cast<float*>(state.conv_attn.data), static_cast<float*>(state.conv_mlp.data)},
                       static_cast<ActT*>(state.k_ring.data), static_cast<ActT*>(state.v_ring.data),
                       static_cast<std::int64_t*>(state.ring_pos.data)};
    }
  };

  static std::size_t state_ckpt_bytes(const State& st) {
    return st.k_ring.bytes() + st.v_ring.bytes() + st.ring_pos.bytes() +
           st.conv_k.bytes() + st.conv_v.bytes() + st.conv_attn.bytes() + st.conv_mlp.bytes();
  }
  static void save_state_ckpt(const State& st, void* dst, cudaStream_t s) {
    char* p = static_cast<char*>(dst);
    for (const Tensor* t : {&st.k_ring, &st.v_ring, &st.ring_pos, &st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp}) {
      CUDA_CHECK(cudaMemcpyAsync(p, t->data, t->bytes(), cudaMemcpyDeviceToDevice, s));
      p += t->bytes();
    }
  }
  static std::size_t restore_state_ckpt(State& st, const void* src, cudaStream_t s) {
    const char* p = static_cast<const char*>(src);
    std::size_t tot = 0;
    for (Tensor* t : {&st.k_ring, &st.v_ring, &st.ring_pos, &st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp}) {
      CUDA_CHECK(cudaMemcpyAsync(t->data, p, t->bytes(), cudaMemcpyDeviceToDevice, s));
      p += t->bytes();
      tot += t->bytes();
    }
    return tot;
  }

  void prefill_all(const Prompt& request, GenerateStats& stats) {
    const auto prompt = request.ids;
    const bool cache_enabled = options_.prefix_cache && !request.bypass_prefix_cache;
    // The full KV prefix lives in the active rank states. A cold prefill overwrites it.
    if (!cache_enabled || request.cache_namespace != ckpt_namespace_) ckpt_ids_.clear();
    const int n = int(prompt.size());
    auto is_prefix = [&](const std::vector<std::int32_t>& cached) {
      return !cached.empty() && int(cached.size()) <= n &&
             std::equal(cached.begin(), cached.end(), prompt.begin());
    };
    int start = 0;
    if (is_prefix(ckpt_ids_) && int(ckpt_ids_.size()) == n) {
      // Exact prompt checkpoint hit!
      std::size_t restored = 0;
      pool_.run_all([&](int r) {
        const std::size_t b = restore_state_ckpt(ranks_[r].state, ranks_[r].ckpt.p, pool_.stream(r));
        ranks_[r].state.length = n;
        if (r == 0) {
          restored = b;
          if (drafter_) restored += drafter_->restore_checkpoint(ckpt_drafter_.p);
          CUDA_CHECK(cudaMemcpyAsync(logits_.p, ckpt_logits_.p, logits_.bytes, cudaMemcpyDeviceToDevice, pool_.stream(0)));
          restored += logits_.bytes;
        }
      });
      tokens_in_state_ = ckpt_ids_;
      stats.reused_tokens = n;
      stats.reuse = Reuse::Checkpoint;
      stats.restored_bytes = restored;
      return;
    }
    if (is_prefix(ckpt_ids_)) {
      std::size_t restored = 0;
      pool_.run_all([&](int r) {
        const std::size_t b = restore_state_ckpt(ranks_[r].state, ranks_[r].ckpt.p, pool_.stream(r));
        ranks_[r].state.length = int(ckpt_ids_.size());
        if (r == 0) {
          restored = b;
          if (drafter_) restored += drafter_->restore_checkpoint(ckpt_drafter_.p);
        }
      });
      start = int(ckpt_ids_.size());
      tokens_in_state_ = ckpt_ids_;
      stats.reused_tokens = start;
      stats.reuse = Reuse::Checkpoint;
      stats.restored_bytes = restored;
    } else {
      pool_.run_all([&](int r) {
        reset_state(ranks_[r].state, pool_.stream(r));
      });
      if (drafter_) {
        CUDA_CHECK(cudaSetDevice(0));
        drafter_->reset();
      }
      tokens_in_state_.clear();
      stats.reused_tokens = 0;
      stats.reuse = Reuse::None;
    }

    const int chunk = prefill_chunk_;
    for (int pos = start; pos < n; pos += chunk) {
      const int T = std::min(chunk, n - pos);
      pool_.run_all([&](int r) {
        auto& rk = ranks_[r];
        CUDA_CHECK(cudaMemcpyAsync(rk.chunk_tokens.p, prompt.data() + pos, std::size_t(T) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, pool_.stream(r)));
        float* out_l = (r == 0) ? static_cast<float*>(logits_.p) : nullptr;
        Tensor tap_view;
        Tensor* tap_ptr = nullptr;
        if (r == 0 && drafter_) {
          tap_view = rm(taps_.p, kActDType, {T, NUM_TAPS, HIDDEN});
          tap_ptr = &tap_view;
        }
        rk.prefiller->run(rk.state, static_cast<const std::int32_t*>(rk.chunk_tokens.p), T, pos, out_l, tap_ptr);
        if (r == 0 && drafter_) {
          drafter_->commit(static_cast<const ActT*>(taps_.p), T, pos);
        }
      });
    }
    tokens_in_state_.assign(prompt.begin(), prompt.end());
    if (drafter_) {
      CUDA_CHECK(cudaSetDevice(0));
      drafter_->note_tokens(prompt);
    }
    if (cache_enabled) {
      pool_.run_all([&](int r) {
        save_state_ckpt(ranks_[r].state, ranks_[r].ckpt.p, pool_.stream(r));
        if (r == 0) {
          if (drafter_) drafter_->save_checkpoint(ckpt_drafter_.p);
          CUDA_CHECK(cudaMemcpyAsync(ckpt_logits_.p, logits_.p, logits_.bytes, cudaMemcpyDeviceToDevice, pool_.stream(0)));
        }
      });
      ckpt_ids_ = tokens_in_state_;
      ckpt_namespace_ = request.cache_namespace;
    }
  }

  std::int32_t sample_r0(const float* logits, const Sampling& sampling, head::SamplerRng& rng) {
    CUDA_CHECK(cudaSetDevice(0));
    const head::SamplingParams p = to_head_params(sampling);
    head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, p, rng, static_cast<std::int32_t*>(token_dev_.p), pool_.stream(0));
    CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), token_dev_.p, sizeof(std::int32_t), cudaMemcpyDeviceToHost, pool_.stream(0)));
    CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
    return *static_cast<const std::int32_t*>(host_token_.data());
  }

  void run_decode_step(const std::int32_t* toks, int m, int pos, bool save_bk) {
    pool_.run_all([&](int r) {
      auto& rk = ranks_[r];
      set_step_inputs(toks, m, pos, static_cast<std::int64_t*>(rk.token.p), static_cast<std::int64_t*>(rk.pos.p), pool_.stream(r));
      if (save_bk) verify_save(rk.state_ptrs(), rk.backup, nullptr, pos, m, pool_.stream(r));
      auto it = rk.graphs.find(m);
      if (it != rk.graphs.end()) {
        CUDA_CHECK(cudaGraphLaunch(it->second, pool_.stream(r)));
      } else {
        rk.decoder->decode(rk.buffers.at(m), static_cast<const std::int64_t*>(rk.pos.p), m > 1, pool_.stream(r));
      }
      pool_.info(r).decode_steps.fetch_add(1, std::memory_order_relaxed);
      pool_.info(r).allreduce_calls.fetch_add(2 * NUM_VISITS, std::memory_order_relaxed);
      pool_.info(r).allgather_calls.fetch_add(1, std::memory_order_relaxed);
    });
  }

  void rollback_all(int pos, int commit, int m) {
    if (commit >= m) return;
    pool_.run_all([&](int r) {
      auto& rk = ranks_[r];
      const Buffers& b = rk.buffers.at(m);
      const float* hist[4] = {f32p(b.hist_k), f32p(b.hist_v), f32p(b.hist_attn), f32p(b.hist_mlp)};
      verify_rollback(rk.state_ptrs(), rk.backup, hist, nullptr, pos, nullptr, commit, m, pool_.stream(r));
    });
  }

  void generate_plain(std::int32_t tok, int max_new_tokens, const Sampling& sampling, head::SamplerRng& rng,
                      std::span<const std::int32_t> stop_ids,
                      const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats) {
    int P = int(tokens_in_state_.size());
    for (int emitted = 1; emitted <= max_new_tokens; ++emitted) {
      stats.generated_tokens = emitted;
      const bool stop = std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end();
      const bool go = on_tokens(std::span<const std::int32_t>(&tok, 1));
      if (stop || emitted == max_new_tokens || !go) break;
      run_decode_step(&tok, 1, P, false);
      tokens_in_state_.push_back(tok);
      ++P;
      tok = sample_r0(f32p(ranks_[0].buffers.at(1).logits), sampling, rng);
      if (std::size_t(emitted) < sampling.forced.size()) tok = sampling.forced[emitted];
    }
  }

  void generate_spec(std::int32_t token, int max_new_tokens, const Sampling& sampling, head::SamplerRng& rng,
                     std::span<const std::int32_t> stop_ids,
                     const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats) {
    const bool forced = !sampling.forced.empty();
    const bool greedy = sampling.temperature <= 0.f || forced;
    const head::SamplingParams params = to_head_params(sampling);
    const int V = drafter_->vocab();
    const Drafter::AcceptScratch& scratch = drafter_->accept_scratch();
    const Widths widths{3, {2, 4, 8}};

    int P = int(tokens_in_state_.size());
    int emitted = 0, rounds = 0, accepted_tot = 0;
    std::vector<std::int32_t> out{token};
    int round_P = -1, round_m = 0, round_acc = 0;

    CUDA_CHECK(cudaSetDevice(0));
    auto dr = drafter_->draft(token, P, sampling.temperature);

    for (;;) {
      const std::int32_t anchor = out.back();
      std::size_t k = 0;
      bool done = false;
      while (k < out.size()) {
        const std::int32_t t = out[k++];
        ++emitted;
        if (std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end() || emitted == max_new_tokens) {
          done = true;
          break;
        }
      }
      stats.generated_tokens = emitted;
      const bool go = on_tokens(std::span<const std::int32_t>(out.data(), k));
      CUDA_CHECK(cudaSetDevice(0));
      drafter_->note_tokens(std::span<const std::int32_t>(out.data(), k));
      if (done || !go) {
        const int keep = int(k) - 1;
        if (round_P >= 0 && keep < round_acc) {
          rollback_all(round_P, keep + 1, round_m);
          CUDA_CHECK(cudaSetDevice(0));
          drafter_->commit(ptr<const ActT>(ranks_[0].buffers.at(round_m).taps), keep + 1, round_P);
          tokens_in_state_.resize(tokens_in_state_.size() - std::size_t(round_acc - keep));
        }
        break;
      }

      // Select verify width on rank 0 & read draft tokens
      CUDA_CHECK(cudaSetDevice(0));
      verify_select_eager(dr.conf, dr.tokens, options_.spec_threshold, widths,
                          static_cast<std::int64_t*>(ranks_[0].token.p),
                          static_cast<std::int32_t*>(width_dev_.p),
                          static_cast<std::int32_t*>(host_m_.device_ptr()), pool_.stream(0));
      CUDA_CHECK(cudaMemcpyAsync(host_drafts_.data(), dr.tokens, kd::GAMMA * sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost, pool_.stream(0)));
      CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
      const int m = *static_cast<const std::int32_t*>(host_m_.data());
      const auto* drafts = static_cast<const std::int32_t*>(host_drafts_.data());

      std::int32_t step_toks[8];
      step_toks[0] = anchor;
      for (int i = 1; i < m; ++i) step_toks[i] = drafts[i - 1];

      const auto t_verify0 = std::chrono::steady_clock::now();
      // Run TP verify step across all ranks (with backup for rollback)
      run_decode_step(step_toks, m, P, true);

      // Accept on rank 0
      CUDA_CHECK(cudaSetDevice(0));
      const float* logits = f32p(ranks_[0].buffers.at(m).logits);
      const int n_draft = m - 1;
      std::int32_t next_tok = 0;
      int acc = 0;
      if (greedy) {
        head::argmax(logits, m, V, V, scratch.target, pool_.stream(0), params.banned);
        if (forced) {
          CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
          while (acc < n_draft && std::size_t(emitted + acc) < sampling.forced.size() &&
                 drafts[acc] == sampling.forced[emitted + acc]) {
            ++acc;
          }
          next_tok = std::size_t(emitted + acc) < sampling.forced.size() ? sampling.forced[emitted + acc] : 0;
        } else {
          kd::greedy_accept(scratch.target, dr.tokens, n_draft, scratch.result, pool_.stream(0));
          CUDA_CHECK(cudaMemcpyAsync(host_accept_.data(), scratch.result, sizeof(kd::AcceptResult),
                                     cudaMemcpyDeviceToHost, pool_.stream(0)));
          CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
          const auto* ar = static_cast<const kd::AcceptResult*>(host_accept_.data());
          acc = ar->accepted;
          next_tok = ar->next;
        }
      } else {
        head::uniform(kd::GAMMA, rng, scratch.uniforms, pool_.stream(0));
        head::sampling_probs(logits, m, V, V, params, scratch.probs, pool_.stream(0));
        kd::sampled_accept(scratch.probs, dr.logits, V, dr.tokens, n_draft, sampling.temperature, dr.qstats,
                           scratch.uniforms, scratch.ws, scratch.resid, scratch.result, pool_.stream(0));
        head::sample_from_probs(scratch.resid, 1, V, V, rng, static_cast<std::int32_t*>(token_dev_.p), pool_.stream(0));
        CUDA_CHECK(cudaMemcpyAsync(host_accept_.data(), scratch.result, sizeof(kd::AcceptResult),
                                   cudaMemcpyDeviceToHost, pool_.stream(0)));
        CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), token_dev_.p, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, pool_.stream(0)));
        CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
        acc = static_cast<const kd::AcceptResult*>(host_accept_.data())->accepted;
        next_tok = *static_cast<const std::int32_t*>(host_token_.data());
      }
      const float verify_ms =
          forced ? float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_verify0).count())
                 : 0.f;

      const int commit = acc + 1;
      rollback_all(P, commit, m);
      CUDA_CHECK(cudaSetDevice(0));
      const auto t_draft0 = std::chrono::steady_clock::now();
      drafter_->commit(ptr<const ActT>(ranks_[0].buffers.at(m).taps), commit, P);

      round_P = P;
      round_m = m;
      round_acc = acc;
      ++rounds;
      accepted_tot += acc;
      out.clear();
      tokens_in_state_.push_back(anchor);
      for (int i = 0; i < acc; ++i) {
        out.push_back(drafts[i]);
        tokens_in_state_.push_back(drafts[i]);
      }
      out.push_back(next_tok);
      P += commit;
      dr = drafter_->draft(next_tok, P, sampling.temperature);
      if (forced) {
        CUDA_CHECK(cudaStreamSynchronize(pool_.stream(0)));
        const float draft_ms =
            float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_draft0).count());
        int all = 0;
        while (all < kd::GAMMA && std::size_t(emitted + all) < sampling.forced.size() &&
               drafts[all] == sampling.forced[emitted + all]) {
          ++all;
        }
        stats.spec_trace.push_back({m, acc, all, verify_ms, draft_ms});
      }
    }
    stats.spec_rounds = rounds;
    stats.spec_accepted = accepted_tot;
  }

  RuntimeOptions options_;
  TpWorkerPool pool_;
  std::array<RankCtx, TP_SIZE> ranks_;
  int prefill_chunk_ = 3072;
  Weights drafter_weights_;
  std::unique_ptr<Drafter> drafter_;
  DeviceBuffer logits_, ckpt_logits_, ckpt_drafter_, token_dev_, width_dev_, taps_;
  PinnedHostBuffer host_token_;
  MappedHostBuffer host_m_;
  PinnedHostBuffer host_drafts_, host_accept_;
  std::vector<std::int32_t> tokens_in_state_, ckpt_ids_;
  std::string ckpt_namespace_;
  std::mutex mu_;
};

}  // namespace

std::unique_ptr<Generator> make_tp_generator(const RuntimeOptions& options) {
  validate_tp_runtime_options(options, TP_SIZE);
  if (options.max_batch > 1 || !options.spec) {
    return std::make_unique<TpBatchGenerator>(options);
  }
  return std::make_unique<TpSpecGenerator>(options);
}

#endif

}  // namespace ingine::model
