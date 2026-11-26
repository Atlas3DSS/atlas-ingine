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

// Defaults to available physical cores. Linux affinity also bounds an INGINE_CPU_THREADS override.
int default_cpu_pool_threads() noexcept;

class CpuThreadPool {
 public:
  using TaskFn = void (*)(int tid, int nthreads, void* ctx);

  explicit CpuThreadPool(int num_threads = default_cpu_pool_threads(), int spin_iters = -1);
  ~CpuThreadPool();

  CpuThreadPool(const CpuThreadPool&) = delete;
  CpuThreadPool& operator=(const CpuThreadPool&) = delete;

  static CpuThreadPool& global();

  [[nodiscard]] int num_threads() const noexcept { return num_threads_; }

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
  void stop_workers() noexcept;
  void worker_loop(int tid);
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
  SharedState state_{};
  std::mutex dispatch_mu_;
  std::mutex err_mu_;
  std::atomic<bool> has_err_{false};
  std::exception_ptr err_;
  std::vector<std::thread> workers_;
};

}  // namespace ingine::ops::cpu
