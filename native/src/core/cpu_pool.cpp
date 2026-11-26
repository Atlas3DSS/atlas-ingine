#include "core/cpu_pool.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <new>
#include <set>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__linux__)
#include <linux/futex.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace ingine::ops::cpu {
namespace {

thread_local bool t_in_worker_pool = false;

struct InPoolGuard {
  bool prev;
  InPoolGuard() noexcept : prev(t_in_worker_pool) { t_in_worker_pool = true; }
  ~InPoolGuard() noexcept { t_in_worker_pool = prev; }
};

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

inline void futex_wait_eq(std::atomic<std::uint32_t>* addr, std::uint32_t expected) noexcept {
  static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t), "futex requires 32-bit atomic");
#if defined(__linux__)
  (void)::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(addr), FUTEX_WAIT_PRIVATE, expected,
                  nullptr, nullptr, 0);
#else
  addr->wait(expected, std::memory_order_acquire);
#endif
}

inline void futex_wake_all(std::atomic<std::uint32_t>* addr) noexcept {
#if defined(__linux__)
  (void)::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(addr), FUTEX_WAKE_PRIVATE, INT_MAX,
                  nullptr, nullptr, 0);
#else
  addr->notify_all();
#endif
}

int parse_env_int(const char* name, int fallback) noexcept {
  const char* v = std::getenv(name);
  if (!v || v[0] == '\0') return fallback;
  char* end = nullptr;
  const long val = std::strtol(v, &end, 10);
  if (end == v || val < 0 || val > 1000000L) return fallback;
  return static_cast<int>(val);
}

}  // namespace

int default_cpu_pool_threads() noexcept {
  const int env_n = parse_env_int("INGINE_CPU_THREADS", 0);
  const unsigned hw = std::thread::hardware_concurrency();
  int fallback = env_n > 0 ? std::min(env_n, 64) : static_cast<int>(hw ? std::min(hw, 64u) : 4);

#if defined(__linux__)
  try {
    // A kernel affinity mask can exceed libc's fixed-size cpu_set_t.
    const auto free_set = [](cpu_set_t* mask) { CPU_FREE(mask); };
    for (int capacity = CPU_SETSIZE; ; capacity *= 2) {
      const auto bytes = CPU_ALLOC_SIZE(capacity);
      std::unique_ptr<cpu_set_t, decltype(free_set)> allowed(CPU_ALLOC(capacity), free_set);
      if (!allowed) break;
      if (::sched_getaffinity(0, bytes, allowed.get()) != 0) {
        if (errno == EINVAL && capacity <= INT_MAX / 2) continue;
        break;
      }
      const int logical = CPU_COUNT_S(bytes, allowed.get());
      if (logical <= 0) break;
      fallback = std::min(logical, 64);
      if (env_n > 0) return std::min(env_n, fallback);

      std::set<std::string> physical_cores;
      int unknown_cores = 0;
      for (int cpu = 0; cpu < capacity; ++cpu) {
        if (!CPU_ISSET_S(cpu, bytes, allowed.get())) continue;
        std::ifstream topology("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                               "/topology/thread_siblings_list");
        std::string siblings;
        if (std::getline(topology, siblings) && !siblings.empty()) {
          physical_cores.insert(siblings);
        } else {
          ++unknown_cores;
        }
      }
      return std::min(fallback, static_cast<int>(physical_cores.size()) + unknown_cores);
    }
  } catch (const std::bad_alloc&) {
    // Topology discovery is optional when the process is under memory pressure.
  }
#endif
  return fallback;
}

CpuThreadPool::CpuThreadPool(int num_threads, int spin_iters) {
  num_threads_ = std::clamp(num_threads > 0 ? num_threads : 1, 1, 64);
  if (spin_iters >= 0) {
    spin_iters_ = spin_iters;
  } else {
    spin_iters_ = parse_env_int("INGINE_CPU_POOL_SPIN", 4000);
  }
  if (num_threads_ > 1) {
    workers_.reserve(static_cast<std::size_t>(num_threads_ - 1));
    try {
      for (int tid = 1; tid < num_threads_; ++tid) {
        workers_.emplace_back([this, tid]() { worker_loop(tid); });
      }
    } catch (...) {
      stop_workers();
      throw;
    }
  }
}

CpuThreadPool::~CpuThreadPool() {
  stop_workers();
}

void CpuThreadPool::stop_workers() noexcept {
  if (!workers_.empty()) {
    {
      std::lock_guard<std::mutex> lock(dispatch_mu_);
      state_.stop.store(true, std::memory_order_seq_cst);
      state_.epoch.fetch_add(1, std::memory_order_seq_cst);
      futex_wake_all(&state_.epoch);
    }
    for (std::thread& w : workers_) {
      if (w.joinable()) w.join();
    }
  }
}

CpuThreadPool& CpuThreadPool::global() {
  static CpuThreadPool pool(default_cpu_pool_threads());
  return pool;
}

void CpuThreadPool::capture_exception() noexcept {
  std::lock_guard<std::mutex> lk(err_mu_);
  if (!err_) {
    err_ = std::current_exception();
    has_err_.store(true, std::memory_order_release);
  }
}

void CpuThreadPool::worker_loop(int tid) {
  t_in_worker_pool = true;
  std::uint32_t local_epoch = 0;

  for (;;) {
    std::uint32_t cur = state_.epoch.load(std::memory_order_acquire);
    if (cur == local_epoch && !state_.stop.load(std::memory_order_acquire)) {
      for (int s = 0; s < spin_iters_; ++s) {
        cpu_relax();
        cur = state_.epoch.load(std::memory_order_acquire);
        if (cur != local_epoch || state_.stop.load(std::memory_order_acquire)) break;
      }
    }
    while (cur == local_epoch && !state_.stop.load(std::memory_order_acquire)) {
      state_.sleeping.fetch_add(1, std::memory_order_seq_cst);
      cur = state_.epoch.load(std::memory_order_seq_cst);
      if (cur == local_epoch && !state_.stop.load(std::memory_order_seq_cst)) {
        futex_wait_eq(&state_.epoch, local_epoch);
      }
      state_.sleeping.fetch_sub(1, std::memory_order_acq_rel);
      cur = state_.epoch.load(std::memory_order_acquire);
    }
    if (state_.stop.load(std::memory_order_acquire)) break;

    local_epoch = cur;
    const int active = state_.active_threads.load(std::memory_order_relaxed);
    if (tid < active) {
      try {
        state_.fn(tid, active, state_.ctx);
      } catch (...) {
        capture_exception();
      }
    }
    state_.done.fetch_add(1, std::memory_order_release);
  }
  t_in_worker_pool = false;
}

void CpuThreadPool::dispatch(int n_threads, TaskFn fn, void* ctx) {
  if (!fn) return;
  if (n_threads <= 1 || num_threads_ <= 1 || t_in_worker_pool) {
    fn(0, 1, ctx);
    return;
  }

  std::lock_guard<std::mutex> lock(dispatch_mu_);
  const int active = std::clamp(n_threads, 1, num_threads_);
  state_.fn = fn;
  state_.ctx = ctx;
  state_.active_threads.store(active, std::memory_order_relaxed);
  state_.done.store(0, std::memory_order_relaxed);
  has_err_.store(false, std::memory_order_relaxed);
  err_ = nullptr;

  state_.epoch.fetch_add(1, std::memory_order_seq_cst);
  if (state_.sleeping.load(std::memory_order_seq_cst) > 0) {
    futex_wake_all(&state_.epoch);
  }

  try {
    InPoolGuard guard;
    fn(0, active, ctx);
  } catch (...) {
    capture_exception();
  }

  const std::uint32_t expected_workers = static_cast<std::uint32_t>(num_threads_ - 1);
  while (state_.done.load(std::memory_order_acquire) < expected_workers) {
    cpu_relax();
  }
  if (has_err_.load(std::memory_order_acquire)) {
    std::exception_ptr rethrow_err = err_;
    err_ = nullptr;
    has_err_.store(false, std::memory_order_relaxed);
    std::rethrow_exception(rethrow_err);
  }
}

}  // namespace ingine::ops::cpu
