#include "core/cpu_pool.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__linux__)
#include <dirent.h>
#include <linux/futex.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace ingine::ops::cpu {
namespace {

thread_local bool t_in_worker_pool = false;
std::atomic<int> g_fail_worker_start{-1};

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
  if (end == v || *end != '\0' || val < 0 || val > 1000000L) return fallback;
  return static_cast<int>(val);
}

bool cpu_pin_enabled() noexcept {
  const char* v = std::getenv("INGINE_CPU_PIN");
  if (!v || v[0] == '\0') return true;
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0) {
    return false;
  }
  return true;
}

#if defined(__linux__)
int read_cpu_numa_node(int cpu) noexcept {
  const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
  if (DIR* d = ::opendir(base.c_str())) {
    int found = -1;
    while (dirent* ent = ::readdir(d)) {
      if (std::strncmp(ent->d_name, "node", 4) == 0 &&
          std::isdigit(static_cast<unsigned char>(ent->d_name[4]))) {
        char* end = nullptr;
        const long n = std::strtol(ent->d_name + 4, &end, 10);
        if (end != ent->d_name + 4 && *end == '\0' && n >= 0 && n <= 1024) {
          found = static_cast<int>(n);
          break;
        }
      }
    }
    ::closedir(d);
    if (found >= 0) return found;
  }
  std::ifstream pkg(base + "/topology/physical_package_id");
  int pkg_id = 0;
  if (pkg >> pkg_id && pkg_id >= 0) return pkg_id;
  return 0;
}

std::string trim_ascii(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  std::size_t first = 0;
  while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) ++first;
  return first > 0 ? s.substr(first) : s;
}

bool pin_current_thread_to_cpu(int cpu_id) noexcept {
  if (cpu_id < 0) return false;
  const auto free_set = [](cpu_set_t* mask) { CPU_FREE(mask); };
  for (int capacity = CPU_SETSIZE; ; capacity *= 2) {
    const auto bytes = CPU_ALLOC_SIZE(capacity);
    std::unique_ptr<cpu_set_t, decltype(free_set)> allowed(CPU_ALLOC(capacity), free_set);
    if (!allowed) return false;
    if (::sched_getaffinity(0, bytes, allowed.get()) != 0) {
      if (errno == EINVAL && capacity <= INT_MAX / 2) continue;
      return false;
    }
    if (cpu_id >= capacity || !CPU_ISSET_S(cpu_id, bytes, allowed.get())) {
      return false;
    }
    std::unique_ptr<cpu_set_t, decltype(free_set)> target(CPU_ALLOC(capacity), free_set);
    if (!target) return false;
    CPU_ZERO_S(bytes, target.get());
    CPU_SET_S(cpu_id, bytes, target.get());
    return ::sched_setaffinity(0, bytes, target.get()) == 0;
  }
}
#endif

}  // namespace

CpuTopologyInfo discover_cpu_topology() noexcept {
  CpuTopologyInfo info;
  const unsigned hw = std::thread::hardware_concurrency();
  info.logical_cpus = static_cast<int>(hw > 0 ? hw : 4u);
  info.physical_cores = info.logical_cpus;

#if defined(__linux__)
  try {
    unsigned gcpu = 0, gnode = 0;
    if (::syscall(SYS_getcpu, &gcpu, &gnode, nullptr) == 0) {
      info.caller_cpu = static_cast<int>(gcpu);
      info.caller_numa_node = static_cast<int>(gnode);
    } else {
      const int cur = ::sched_getcpu();
      if (cur >= 0) info.caller_cpu = cur;
    }

    const auto free_set = [](cpu_set_t* mask) { CPU_FREE(mask); };
    std::vector<int> allowed_cpus;
    for (int capacity = CPU_SETSIZE; ; capacity *= 2) {
      const auto bytes = CPU_ALLOC_SIZE(capacity);
      std::unique_ptr<cpu_set_t, decltype(free_set)> allowed(CPU_ALLOC(capacity), free_set);
      if (!allowed) break;
      if (::sched_getaffinity(0, bytes, allowed.get()) != 0) {
        if (errno == EINVAL && capacity <= INT_MAX / 2) continue;
        break;
      }
      const int count = CPU_COUNT_S(bytes, allowed.get());
      if (count <= 0) break;
      allowed_cpus.reserve(static_cast<std::size_t>(count));
      for (int cpu = 0; cpu < capacity; ++cpu) {
        if (CPU_ISSET_S(cpu, bytes, allowed.get())) allowed_cpus.push_back(cpu);
      }
      break;
    }

    if (allowed_cpus.empty()) return info;
    info.logical_cpus = static_cast<int>(allowed_cpus.size());

    if (info.caller_numa_node < 0 && info.caller_cpu >= 0) {
      info.caller_numa_node = read_cpu_numa_node(info.caller_cpu);
    }

    struct CoreGroup {
      int numa_node = 0;
      std::vector<int> cpus;
    };
    std::map<std::string, CoreGroup> by_core;
    std::map<int, std::string> cpu_to_core_key;
    for (int cpu : allowed_cpus) {
      std::ifstream topology("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                             "/topology/thread_siblings_list");
      std::string siblings;
      std::string key;
      if (std::getline(topology, siblings)) {
        key = trim_ascii(std::move(siblings));
      }
      if (key.empty()) {
        key = "_cpu_" + std::to_string(cpu);
      }
      auto [it, inserted] = by_core.emplace(key, CoreGroup{});
      if (inserted) {
        it->second.numa_node = read_cpu_numa_node(cpu);
      }
      it->second.cpus.push_back(cpu);
      cpu_to_core_key.emplace(cpu, key);
    }

    struct CpuNodePair {
      int cpu;
      int node;
    };
    std::vector<CpuNodePair> primaries;
    std::vector<CpuNodePair> smts;
    primaries.reserve(by_core.size());
    for (const auto& [key, group] : by_core) {
      (void)key;
      if (group.cpus.empty()) continue;
      primaries.push_back(CpuNodePair{group.cpus[0], group.numa_node});
      for (std::size_t i = 1; i < group.cpus.size(); ++i) {
        smts.push_back(CpuNodePair{group.cpus[i], group.numa_node});
      }
    }

    const int pref_node = info.caller_numa_node >= 0 ? info.caller_numa_node : read_cpu_numa_node(allowed_cpus[0]);
    if (info.caller_numa_node < 0) info.caller_numa_node = pref_node;

    auto cmp = [pref_node](const CpuNodePair& a, const CpuNodePair& b) {
      const bool a_local = (a.node == pref_node);
      const bool b_local = (b.node == pref_node);
      if (a_local != b_local) return a_local > b_local;
      if (a.node != b.node) return a.node < b.node;
      return a.cpu < b.cpu;
    };
    std::sort(primaries.begin(), primaries.end(), cmp);
    std::sort(smts.begin(), smts.end(), cmp);

    info.primary_cpus.reserve(primaries.size());
    for (const auto& p : primaries) info.primary_cpus.push_back(p.cpu);
    info.smt_cpus.reserve(smts.size());
    for (const auto& s : smts) info.smt_cpus.push_back(s.cpu);
    info.physical_cores = std::max(1, static_cast<int>(info.primary_cpus.size()));

    info.ordered_cpus.reserve(allowed_cpus.size());
    auto in_vec = [](const std::vector<int>& v, int c) {
      return std::find(v.begin(), v.end(), c) != v.end();
    };

    if (info.caller_cpu >= 0 && in_vec(info.primary_cpus, info.caller_cpu)) {
      info.ordered_cpus.push_back(info.caller_cpu);
      for (int c : info.primary_cpus) {
        if (c != info.caller_cpu) info.ordered_cpus.push_back(c);
      }
      for (int c : info.smt_cpus) info.ordered_cpus.push_back(c);
    } else if (info.caller_cpu >= 0 && in_vec(info.smt_cpus, info.caller_cpu)) {
      int sibling_primary = -1;
      if (auto it_key = cpu_to_core_key.find(info.caller_cpu); it_key != cpu_to_core_key.end()) {
        if (auto it_grp = by_core.find(it_key->second); it_grp != by_core.end() && !it_grp->second.cpus.empty()) {
          sibling_primary = it_grp->second.cpus[0];
        }
      }
      info.ordered_cpus.push_back(info.caller_cpu);
      for (int c : info.primary_cpus) {
        if (c != sibling_primary) info.ordered_cpus.push_back(c);
      }
      for (int c : info.smt_cpus) {
        if (c != info.caller_cpu) info.ordered_cpus.push_back(c);
      }
      if (sibling_primary >= 0 && sibling_primary != info.caller_cpu) {
        info.ordered_cpus.push_back(sibling_primary);
      }
    } else {
      for (int c : info.primary_cpus) info.ordered_cpus.push_back(c);
      for (int c : info.smt_cpus) info.ordered_cpus.push_back(c);
    }
  } catch (...) {
    // Topology discovery is best-effort and must remain noexcept.
  }
#endif

  return info;
}

int default_cpu_pool_threads() noexcept {
  const int env_n = parse_env_int("INGINE_CPU_THREADS", 0);
  const CpuTopologyInfo topo = discover_cpu_topology();
  const int max_allowed = std::clamp(topo.logical_cpus, 1, 64);
  if (env_n > 0) return std::min(env_n, max_allowed);
  return std::min(std::clamp(topo.physical_cores, 1, 64), max_allowed);
}

ScopedThreadPin::ScopedThreadPin(int cpu_id) noexcept {
#if defined(__linux__)
  if (cpu_id < 0) return;
  for (int capacity = CPU_SETSIZE; ; capacity *= 2) {
    const auto bytes = CPU_ALLOC_SIZE(capacity);
    cpu_set_t* saved = CPU_ALLOC(capacity);
    if (!saved) return;
    if (::sched_getaffinity(0, bytes, saved) != 0) {
      CPU_FREE(saved);
      if (errno == EINVAL && capacity <= INT_MAX / 2) continue;
      return;
    }
    if (cpu_id >= capacity || !CPU_ISSET_S(cpu_id, bytes, saved)) {
      CPU_FREE(saved);
      return;
    }
    cpu_set_t* target = CPU_ALLOC(capacity);
    if (!target) {
      CPU_FREE(saved);
      return;
    }
    CPU_ZERO_S(bytes, target);
    CPU_SET_S(cpu_id, bytes, target);
    if (::sched_setaffinity(0, bytes, target) == 0) {
      CPU_FREE(target);
      saved_mask_ = saved;
      mask_bytes_ = bytes;
      cpu_ = cpu_id;
      pinned_ = true;
      return;
    }
    CPU_FREE(target);
    CPU_FREE(saved);
    return;
  }
#else
  (void)cpu_id;
#endif
}

ScopedThreadPin::~ScopedThreadPin() noexcept { release(); }

ScopedThreadPin::ScopedThreadPin(ScopedThreadPin&& other) noexcept
    : saved_mask_(std::exchange(other.saved_mask_, nullptr)),
      mask_bytes_(std::exchange(other.mask_bytes_, 0)),
      cpu_(std::exchange(other.cpu_, -1)),
      pinned_(std::exchange(other.pinned_, false)) {}

ScopedThreadPin& ScopedThreadPin::operator=(ScopedThreadPin&& other) noexcept {
  if (this != &other) {
    release();
    saved_mask_ = std::exchange(other.saved_mask_, nullptr);
    mask_bytes_ = std::exchange(other.mask_bytes_, 0);
    cpu_ = std::exchange(other.cpu_, -1);
    pinned_ = std::exchange(other.pinned_, false);
  }
  return *this;
}

void ScopedThreadPin::release() noexcept {
#if defined(__linux__)
  if (saved_mask_ != nullptr) {
    auto* mask = static_cast<cpu_set_t*>(saved_mask_);
    if (pinned_) {
      (void)::sched_setaffinity(0, mask_bytes_, mask);
    }
    CPU_FREE(mask);
  }
#endif
  saved_mask_ = nullptr;
  mask_bytes_ = 0;
  cpu_ = -1;
  pinned_ = false;
}

void fail_next_cpu_pool_worker_start(int worker_index) noexcept {
  g_fail_worker_start.store(worker_index, std::memory_order_release);
}

CpuThreadPool::CpuThreadPool(int num_threads, int spin_iters) {
  init(num_threads > 0 ? num_threads : 1, spin_iters, -1);
}

CpuThreadPool::CpuThreadPool(CpuPoolOptions opts) {
  const int n = opts.num_threads > 0 ? opts.num_threads : default_cpu_pool_threads();
  init(n, opts.spin_iters, opts.pin_workers);
}

void CpuThreadPool::init(int num_threads, int spin_iters, int pin_workers) {
  num_threads_ = std::clamp(num_threads > 0 ? num_threads : 1, 1, 64);
  const CpuTopologyInfo topo = discover_cpu_topology();
  primary_numa_node_ = topo.caller_numa_node;
  oversubscribed_ = (num_threads_ > topo.physical_cores) || (num_threads_ > topo.logical_cpus);

  if (spin_iters >= 0) {
    spin_iters_ = spin_iters;
  } else {
    const int env_spin = parse_env_int("INGINE_CPU_POOL_SPIN", -1);
    if (env_spin >= 0) {
      spin_iters_ = env_spin;
    } else {
      spin_iters_ = oversubscribed_ ? 0 : 4000;
    }
  }

  const bool want_pin = (pin_workers >= 0) ? (pin_workers != 0) : cpu_pin_enabled();
  const bool can_pin = want_pin && num_threads_ > 1 && num_threads_ <= topo.logical_cpus &&
                       static_cast<int>(topo.ordered_cpus.size()) >= num_threads_;
  assigned_cpus_.assign(static_cast<std::size_t>(num_threads_), -1);
  if (can_pin) {
    for (int i = 0; i < num_threads_; ++i) {
      assigned_cpus_[static_cast<std::size_t>(i)] = topo.ordered_cpus[static_cast<std::size_t>(i)];
    }
  }

  if (num_threads_ > 1) {
    workers_.reserve(static_cast<std::size_t>(num_threads_ - 1));
    const int fail_at = g_fail_worker_start.exchange(-1, std::memory_order_acq_rel);
    try {
      for (int tid = 1; tid < num_threads_; ++tid) {
        const int worker_idx = tid - 1;
        if (worker_idx == fail_at) {
          throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again),
                                  "injected CpuThreadPool worker startup failure");
        }
        const int cpu_id = assigned_cpus_[static_cast<std::size_t>(tid)];
        workers_.emplace_back([this, tid, cpu_id]() { worker_loop(tid, cpu_id); });
      }
    } catch (...) {
      stop_workers();
      throw;
    }

    const std::uint32_t target_ready = static_cast<std::uint32_t>(num_threads_ - 1);
    while (ready_workers_.load(std::memory_order_acquire) < target_ready) {
      cpu_relax();
    }
    workers_pinned_ = can_pin && (pinned_workers_.load(std::memory_order_acquire) == target_ready);
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

void CpuThreadPool::worker_loop(int tid, int cpu_id) {
#if defined(__linux__)
  if (cpu_id >= 0 && pin_current_thread_to_cpu(cpu_id)) {
    pinned_workers_.fetch_add(1, std::memory_order_relaxed);
  }
#else
  (void)cpu_id;
#endif
  ready_workers_.fetch_add(1, std::memory_order_release);

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
    if (spin_iters_ == 0) {
      std::this_thread::yield();
    } else {
      cpu_relax();
    }
  }
  if (has_err_.load(std::memory_order_acquire)) {
    std::exception_ptr rethrow_err = err_;
    err_ = nullptr;
    has_err_.store(false, std::memory_order_relaxed);
    std::rethrow_exception(rethrow_err);
  }
}

}  // namespace ingine::ops::cpu
