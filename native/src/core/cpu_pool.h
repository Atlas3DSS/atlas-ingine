// Zero-jitter flat futex / spin-barrier thread pool for CPU inference & hybrid MoE execution.
// Adapted from alesha-pro/llm.asm 02-baremetal-os epoch spin-barrier dispatch (100% CUDA-free).
#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ingine::ops::cpu {

struct CpuTopologyInfo {
  int logical_cpus = 1;
  int physical_cores = 1;
  int caller_cpu = -1;
  int caller_numa_node = -1;
  std::vector<int> primary_cpus;
  std::vector<int> smt_cpus;
  std::vector<int> ordered_cpus;
};

// Discovers allowed logical/physical CPUs and NUMA topology within the current affinity mask.
CpuTopologyInfo discover_cpu_topology() noexcept;

// Defaults to available physical cores. Linux affinity also bounds an INGINE_CPU_THREADS override.
int default_cpu_pool_threads() noexcept;

// RAII thread-affinity pin within the caller's allowed Linux CPU mask; restores original mask on exit.
class ScopedThreadPin {
 public:
  explicit ScopedThreadPin(int cpu_id) noexcept;
  ~ScopedThreadPin() noexcept;

  ScopedThreadPin(const ScopedThreadPin&) = delete;
  ScopedThreadPin& operator=(const ScopedThreadPin&) = delete;
  ScopedThreadPin(ScopedThreadPin&& other) noexcept;
  ScopedThreadPin& operator=(ScopedThreadPin&& other) noexcept;

  [[nodiscard]] bool pinned() const noexcept { return pinned_; }
  [[nodiscard]] int cpu() const noexcept { return cpu_; }

 private:
  void release() noexcept;

  void* saved_mask_ = nullptr;
  std::size_t mask_bytes_ = 0;
  int cpu_ = -1;
  bool pinned_ = false;
};

struct CpuPoolOptions {
  int num_threads = 0;   // <= 0 selects default_cpu_pool_threads()
  int spin_iters = -1;   // < 0 uses INGINE_CPU_POOL_SPIN or adaptive 0 under oversubscription
  int pin_workers = -1;  // < 0 respects INGINE_CPU_PIN (enabled by default when not oversubscribed)
};

// Test hook: injects std::system_error(EAGAIN) before spawning worker `worker_index` (0-based; negative disables).
void fail_next_cpu_pool_worker_start(int worker_index) noexcept;

class CpuThreadPool {
 public:
  using TaskFn = void (*)(int tid, int nthreads, void* ctx);

  explicit CpuThreadPool(int num_threads = default_cpu_pool_threads(), int spin_iters = -1);
  explicit CpuThreadPool(CpuPoolOptions opts);
  ~CpuThreadPool();

  CpuThreadPool(const CpuThreadPool&) = delete;
  CpuThreadPool& operator=(const CpuThreadPool&) = delete;

  static CpuThreadPool& global();

  [[nodiscard]] int num_threads() const noexcept { return num_threads_; }
  [[nodiscard]] int effective_spin_iters() const noexcept { return spin_iters_; }
  [[nodiscard]] bool oversubscribed() const noexcept { return oversubscribed_; }
  [[nodiscard]] bool workers_pinned() const noexcept { return workers_pinned_; }
  [[nodiscard]] int primary_numa_node() const noexcept { return primary_numa_node_; }
  [[nodiscard]] const std::vector<int>& assigned_cpus() const noexcept { return assigned_cpus_; }

  // Dispatches `fn(tid, active_threads, ctx)` for tid in [0, active_threads).
  // The caller executes tid = 0 and waits for every worker to acknowledge the dispatch.
  void dispatch(int n_threads, TaskFn fn, void* ctx);

  template <typename Fn>
  void parallel_for(int n_items, int min_items_per_thread, Fn&& fn) {
    if (n_items <= 0) return;
    const int min_chunk = min_items_per_thread > 0 ? min_items_per_thread : 1;
    const int wanted = 1 + (n_items - 1) / min_chunk;
    const int active = wanted < num_threads_ ? wanted : num_threads_;
    if (active <= 1) {
      fn(0, n_items);
      return;
    }
    struct Ctx {
      int total;
      std::remove_reference_t<Fn>* fn_ptr;
    } ctx{n_items, &fn};
    dispatch(
        active,
        [](int tid, int nt, void* raw) {
          auto* c = static_cast<Ctx*>(raw);
          const int base = (c->total / nt) * tid;
          const int rem = c->total % nt;
          const int begin = base + (tid < rem ? tid : rem);
          const int end = begin + (c->total / nt) + (tid < rem ? 1 : 0);
          if (begin < end) {
            (*c->fn_ptr)(begin, end);
          }
        },
        &ctx);
  }

 private:
  void init(int num_threads, int spin_iters, int pin_workers);
  void stop_workers() noexcept;
  void worker_loop(int tid, int cpu_id);
  void capture_exception() noexcept;

  struct alignas(64) SharedState {
    std::atomic<std::uint32_t> epoch{0};
    std::atomic<bool> stop{false};
    std::atomic<int> active_threads{1};
    TaskFn fn = nullptr;
    void* ctx = nullptr;

    alignas(64) std::atomic<std::uint32_t> done{0};
    alignas(64) std::atomic<std::uint32_t> sleeping{0};
  };

  int num_threads_ = 1;
  int spin_iters_ = 4000;
  int primary_numa_node_ = -1;
  bool oversubscribed_ = false;
  bool workers_pinned_ = false;
  std::vector<int> assigned_cpus_;
  SharedState state_{};
  std::mutex dispatch_mu_;
  std::mutex err_mu_;
  std::atomic<bool> has_err_{false};
  std::atomic<std::uint32_t> ready_workers_{0};
  std::atomic<std::uint32_t> pinned_workers_{0};
  std::exception_ptr err_;
  std::vector<std::thread> workers_;
};

}  // namespace ingine::ops::cpu
