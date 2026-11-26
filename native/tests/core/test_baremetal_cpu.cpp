// Bare-metal CPU fast-wins verification and A/B microbenchmarks:
//   1. 2 MiB transparent huge pages (madvise(MADV_HUGEPAGE) + fallback + /proc/self/smaps)
//   2. Flat futex/spin-barrier CpuThreadPool (correctness, nested guard, exceptions, sleep/wake)
//   3. SIMD Q4_0 / Q8_0 / EXL3 microkernels + monotonic prefetch (bit-exact parity & FP64 oracle)
//   4. Optional --bench mode (excluded from CTest): hugepage TLB, barrier jitter, GEMV throughput
//
// 100% CUDA-free.

#include "common/test_util.h"
#include "core/act.h"
#include "core/arena.h"
#include "core/cpu_kernels.h"
#include "core/cpu_pool.h"
#include "core/hal.h"
#include "core/host_memory.h"
#include "core/hugepage.h"
#include "core/ops.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    const char* prev = ::getenv(name_);
    if (prev != nullptr) {
      had_prev_ = true;
      prev_ = prev;
    }
    if (value != nullptr) {
      ::setenv(name_, value, 1);
    } else {
      ::unsetenv(name_);
    }
  }
  ~ScopedEnvVar() {
    if (had_prev_) {
      ::setenv(name_, prev_.c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }
  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

 private:
  const char* name_;
  bool had_prev_ = false;
  std::string prev_;
};

bool thp_system_enabled() {
  std::ifstream in("/sys/kernel/mm/transparent_hugepage/enabled");
  if (!in) return false;
  std::string s;
  std::getline(in, s);
  return s.find("[never]") == std::string::npos;
}

bool thp_always_enabled() {
  std::ifstream in("/sys/kernel/mm/transparent_hugepage/enabled");
  if (!in) return false;
  std::string s;
  std::getline(in, s);
  return s.find("[always]") != std::string::npos;
}

std::uint64_t read_mem_available_bytes() {
  std::ifstream in("/proc/meminfo");
  std::string key, unit;
  std::uint64_t val = 0;
  while (in >> key >> val >> unit) {
    if (key == "MemAvailable:") return val * 1024ull;
  }
  return 0;
}

std::uint32_t f32_bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

void fill_pattern(std::uint8_t* p, std::size_t n, std::uint64_t seed) {
  for (std::size_t i = 0; i < n; i += 64) {
    const std::uint64_t v = seed ^ (i * 0x9e3779b97f4a7c15ull);
    const std::size_t chunk = std::min<std::size_t>(8, n - i);
    std::memcpy(p + i, &v, chunk);
    if (n - i > 8) p[i + n - i - 1 < 63 ? n - i - 1 : 63] = static_cast<std::uint8_t>(v >> 8);
  }
  // Ensure every 4 KiB page and last byte is dirtied & uniquely tagged.
  for (std::size_t off = 0; off < n; off += 4096) {
    p[off] = static_cast<std::uint8_t>((seed + off / 4096) & 0xffu);
  }
  if (n > 0) p[n - 1] = static_cast<std::uint8_t>((seed ^ 0xA5u) & 0xffu);
}

void verify_pattern(const std::uint8_t* p, std::size_t n, std::uint64_t seed) {
  for (std::size_t off = 0; off < n; off += 4096) {
    INGINE_CHECK(p[off] == static_cast<std::uint8_t>((seed + off / 4096) & 0xffu));
  }
  if (n > 0) {
    INGINE_CHECK(p[n - 1] == static_cast<std::uint8_t>((seed ^ 0xA5u) & 0xffu));
  }
}

bool is_2m_aligned(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) % ingine::kHugePageAlignBytes) == 0;
}

void test_hugepage_allocations() {
  std::printf("[1/4] Testing 2 MiB huge-page allocations and fallback...\n");
  constexpr std::size_t kFourMiB = 4ull << 20;
  const bool thp_sys = thp_system_enabled();
  const bool hp_env = ingine::hugepages_enabled();
  std::printf("  sysfs THP enabled=%s, hugepages_enabled()=%s, MemAvailable=%.1f GiB\n",
              thp_sys ? "yes" : "no", hp_env ? "yes" : "no",
              static_cast<double>(read_mem_available_bytes()) / (1024.0 * 1024.0 * 1024.0));

  // HostRegion (4 MiB)
  {
    ingine::HostRegion region(kFourMiB);
    INGINE_CHECK(region.data() != nullptr);
    INGINE_CHECK(region.size() == kFourMiB);
    INGINE_CHECK(is_2m_aligned(region.data()));
    INGINE_CHECK(region.hugepage_advised() == hp_env);
    fill_pattern(reinterpret_cast<std::uint8_t*>(region.data()), kFourMiB, 0x11223344ull);
    region.register_with_cuda();
    verify_pattern(reinterpret_cast<const std::uint8_t*>(region.data()), kFourMiB, 0x11223344ull);

    const auto sm = ingine::query_range_smaps_hugepages(region.data(), kFourMiB);
    INGINE_CHECK(sm.readable);
    INGINE_CHECK(sm.rss_bytes >= kFourMiB);
    std::printf("  HostRegion 4MiB: rss=%.1f MiB AnonHugePages=%.1f MiB\n",
                sm.rss_bytes / 1048576.0, sm.anon_huge_pages_bytes / 1048576.0);
    if (thp_sys && hp_env && read_mem_available_bytes() > (512ull << 20)) {
      if (sm.anon_huge_pages_bytes == 0) {
        std::printf("  note: kernel reported AnonHugePages=0 under current fragmentation (acceptable fallback)\n");
      }
    }
  }

  // PinnedHostBuffer (4 MiB)
  {
    ingine::PinnedHostBuffer pinned(kFourMiB);
    INGINE_CHECK(pinned.data() != nullptr);
    INGINE_CHECK(pinned.size() == kFourMiB);
    INGINE_CHECK(is_2m_aligned(pinned.data()));
    INGINE_CHECK(pinned.hugepage_advised() == hp_env);
    fill_pattern(static_cast<std::uint8_t*>(pinned.data()), kFourMiB, 0x55667788ull);
    verify_pattern(static_cast<const std::uint8_t*>(pinned.data()), kFourMiB, 0x55667788ull);
    const auto sm = ingine::query_range_smaps_hugepages(pinned.data(), kFourMiB);
    INGINE_CHECK(sm.readable && sm.rss_bytes >= kFourMiB);
    std::printf("  PinnedHostBuffer 4MiB: rss=%.1f MiB AnonHugePages=%.1f MiB\n",
                sm.rss_bytes / 1048576.0, sm.anon_huge_pages_bytes / 1048576.0);

    // Fault-injection parity (#826/#827): injected OOM must trip even on >= 2 MiB THP PinnedHostBuffer.
    ingine::fail_next_pinned_allocations(1);
    bool threw = false;
    try {
      ingine::PinnedHostBuffer should_fail(kFourMiB);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    ingine::fail_next_pinned_allocations(0);
    INGINE_CHECK(threw);
    ingine::PinnedHostBuffer recovers(kFourMiB);
    INGINE_CHECK(recovers.data() != nullptr);
  }

  // MappedHostBuffer (4 MiB)
  {
    ingine::MappedHostBuffer mapped(kFourMiB);
    INGINE_CHECK(mapped.data() != nullptr);
    INGINE_CHECK(mapped.size() == kFourMiB);
    INGINE_CHECK(is_2m_aligned(mapped.data()));
    INGINE_CHECK(mapped.hugepage_advised() == hp_env);
    fill_pattern(static_cast<std::uint8_t*>(mapped.data()), kFourMiB, 0x99aabbccull);
    verify_pattern(static_cast<const std::uint8_t*>(mapped.data()), kFourMiB, 0x99aabbccull);
    const auto sm = ingine::query_range_smaps_hugepages(mapped.data(), kFourMiB);
    INGINE_CHECK(sm.readable && sm.rss_bytes >= kFourMiB);
    std::printf("  MappedHostBuffer 4MiB: rss=%.1f MiB AnonHugePages=%.1f MiB\n",
                sm.rss_bytes / 1048576.0, sm.anon_huge_pages_bytes / 1048576.0);
  }

#if !defined(INGINE_BACKEND_CUDA) || !INGINE_BACKEND_CUDA
  // CPU backend DeviceBuffer + DeviceArena (>= 2 MiB)
  {
    ingine::DeviceBuffer dbuf(kFourMiB);
    INGINE_CHECK(dbuf.p != nullptr && dbuf.bytes == kFourMiB);
    INGINE_CHECK(is_2m_aligned(dbuf.p));
    INGINE_CHECK(dbuf.hugepage_advised() == hp_env);
    std::vector<std::uint8_t> src(kFourMiB), dst(kFourMiB, 0);
    fill_pattern(src.data(), kFourMiB, 0xdeadbeefull);
    dbuf.copy_from_host(src.data(), kFourMiB);
    dbuf.copy_to_host(dst.data(), kFourMiB);
    verify_pattern(dst.data(), kFourMiB, 0xdeadbeefull);

    ingine::DeviceArena arena(kFourMiB);
    INGINE_CHECK(arena.base() != nullptr);
    INGINE_CHECK(is_2m_aligned(arena.base()));
    INGINE_CHECK(arena.hugepage_advised() == hp_env);
  }
#endif

  // Explicit opt-out via INGINE_DISABLE_HUGEPAGES=1
  {
    ScopedEnvVar disable("INGINE_DISABLE_HUGEPAGES", "1");
    INGINE_CHECK(!ingine::hugepages_enabled());
    ingine::HostRegion nohp_region(kFourMiB);
    INGINE_CHECK(nohp_region.data() != nullptr);
    INGINE_CHECK(!nohp_region.hugepage_advised());
    fill_pattern(reinterpret_cast<std::uint8_t*>(nohp_region.data()), kFourMiB, 0x42424242ull);
    const auto sm = ingine::query_range_smaps_hugepages(nohp_region.data(), kFourMiB);
    INGINE_CHECK(sm.readable && sm.rss_bytes >= kFourMiB);
    if (!thp_always_enabled()) {
      INGINE_CHECK(sm.anon_huge_pages_bytes == 0);
    }

    ingine::PinnedHostBuffer nohp_pinned(kFourMiB);
    INGINE_CHECK(nohp_pinned.data() != nullptr);
    INGINE_CHECK(!nohp_pinned.hugepage_advised());
    std::printf("  INGINE_DISABLE_HUGEPAGES=1 verified: hugepage_advised=false, AnonHugePages=%.1f MiB\n",
                sm.anon_huge_pages_bytes / 1048576.0);
  }

  // NUMA-local allocation (SYS_mbind + /proc/self/numa_maps) and INGINE_DISABLE_NUMA_BIND=1 opt-out
  {
    const int cur_node = ingine::current_numa_node();
    if (cur_node >= 0) {
      ingine::HugePageMapping bound =
          ingine::alloc_hugepage_anon(kFourMiB, true, cur_node, ingine::NumaPolicy::Bind);
      INGINE_CHECK(bound.ptr != nullptr);
      fill_pattern(static_cast<std::uint8_t*>(bound.ptr), kFourMiB, 0x13572468ull);
      verify_pattern(static_cast<const std::uint8_t*>(bound.ptr), kFourMiB, 0x13572468ull);
      const auto nm = ingine::query_range_numa_nodes(bound.ptr, kFourMiB);
      std::printf("  NUMA Bind(node=%d): bound=%s readable=%s total=%.1f MiB node%d=%.1f MiB\n",
                  cur_node, bound.numa_bound ? "yes" : "no", nm.readable ? "yes" : "no",
                  nm.total_bytes() / 1048576.0, cur_node,
                  (nm.node_bytes.count(cur_node) ? nm.node_bytes.at(cur_node) : 0ull) / 1048576.0);
      if (bound.numa_bound && nm.readable && nm.total_bytes() > 0) {
        INGINE_CHECK(nm.total_bytes() >= kFourMiB);
        INGINE_CHECK(nm.node_bytes.count(cur_node) > 0 && nm.node_bytes.at(cur_node) >= kFourMiB);
      }
      ingine::free_hugepage_anon(bound);
    }

    ScopedEnvVar disable_numa("INGINE_DISABLE_NUMA_BIND", "1");
    INGINE_CHECK(!ingine::numa_bind_enabled());
    ingine::HostRegion no_numa(kFourMiB);
    INGINE_CHECK(no_numa.data() != nullptr);
    INGINE_CHECK(!no_numa.numa_bound());
    INGINE_CHECK(no_numa.numa_node() == -1);
  }
}

void test_cpu_thread_pool() {
  std::printf("[2/4] Testing futex/spin-barrier CpuThreadPool (affinity, SMT/cpuset spin=0, unwind)...\n");
  using namespace ingine::ops::cpu;

  const CpuTopologyInfo topo = discover_cpu_topology();
  const int detected = default_cpu_pool_threads();
  std::printf("  default_cpu_pool_threads()=%d, logical=%d, physical=%d, caller_cpu=%d, caller_node=%d, ISA=%s\n",
              detected, topo.logical_cpus, topo.physical_cores, topo.caller_cpu, topo.caller_numa_node,
              cpu_isa_name(detect_cpu_isa()));
  INGINE_CHECK(detected >= 1);
  INGINE_CHECK(topo.logical_cpus >= topo.physical_cores);
  INGINE_CHECK(topo.physical_cores >= 1);

#ifdef __linux__
  INGINE_CHECK(static_cast<int>(topo.ordered_cpus.size()) == topo.logical_cpus);

  auto capture_affinity = [](std::size_t& out_bytes) -> cpu_set_t* {
    for (int cap = CPU_SETSIZE; ; cap *= 2) {
      out_bytes = CPU_ALLOC_SIZE(cap);
      cpu_set_t* m = CPU_ALLOC(cap);
      INGINE_CHECK(m != nullptr);
      if (::sched_getaffinity(0, out_bytes, m) == 0) return m;
      CPU_FREE(m);
      INGINE_CHECK(errno == EINVAL && cap <= INT_MAX / 2);
    }
  };

  // ScopedThreadPin save/restore + out-of-cpuset refusal
  {
    std::size_t before_bytes = 0, after_bytes = 0;
    cpu_set_t* before = capture_affinity(before_bytes);
    const int target_cpu = !topo.ordered_cpus.empty() ? topo.ordered_cpus[0] : ::sched_getcpu();
    {
      ScopedThreadPin pin(target_cpu);
      INGINE_CHECK(pin.pinned());
      INGINE_CHECK(pin.cpu() == target_cpu);
      INGINE_CHECK(::sched_getcpu() == target_cpu);
    }
    cpu_set_t* after = capture_affinity(after_bytes);
    INGINE_CHECK(before_bytes == after_bytes && CPU_EQUAL_S(before_bytes, before, after));
    CPU_FREE(before);
    CPU_FREE(after);

    ScopedThreadPin bad_pin(999999);
    INGINE_CHECK(!bad_pin.pinned());
  }

  // Worker sibling-0 pinning + verification that caller thread affinity stays intact
  {
    const int nt_pin = std::min(4, topo.physical_cores);
    if (nt_pin >= 2) {
      std::size_t before_bytes = 0, after_bytes = 0;
      cpu_set_t* before = capture_affinity(before_bytes);
      CpuThreadPool pinned_pool(nt_pin);
      INGINE_CHECK(pinned_pool.workers_pinned());
      INGINE_CHECK(!pinned_pool.oversubscribed());
      INGINE_CHECK(pinned_pool.effective_spin_iters() == 4000);
      INGINE_CHECK(static_cast<int>(pinned_pool.assigned_cpus().size()) == nt_pin);

      std::atomic<int> worker_mismatches{0};
      struct PinCheckCtx {
        const std::vector<int>* assigned;
        std::atomic<int>* mismatches;
      } pctx{&pinned_pool.assigned_cpus(), &worker_mismatches};
      for (int iter = 0; iter < 10; ++iter) {
        pinned_pool.dispatch(nt_pin,
                             [](int tid, int /*nt*/, void* raw) {
                               auto* c = static_cast<PinCheckCtx*>(raw);
                               if (tid > 0 && ::sched_getcpu() != (*c->assigned)[static_cast<std::size_t>(tid)]) {
                                 c->mismatches->fetch_add(1, std::memory_order_relaxed);
                               }
                             },
                             &pctx);
      }
      INGINE_CHECK(worker_mismatches.load() == 0);

      cpu_set_t* after = capture_affinity(after_bytes);
      INGINE_CHECK(before_bytes == after_bytes && CPU_EQUAL_S(before_bytes, before, after));
      CPU_FREE(before);
      CPU_FREE(after);
    }
  }

  // Pin only a short-lived thread to 1 CPU: tests default_cpu_pool_threads() cap AND adaptive spin_iters=0.
  std::thread([] {
    const int cpu = ::sched_getcpu();
    INGINE_CHECK(cpu >= 0);
    const auto bytes = CPU_ALLOC_SIZE(cpu + 1);
    const auto free_set = [](cpu_set_t* mask) { CPU_FREE(mask); };
    std::unique_ptr<cpu_set_t, decltype(free_set)> mask(CPU_ALLOC(cpu + 1), free_set);
    INGINE_CHECK(mask != nullptr);
    CPU_ZERO_S(bytes, mask.get());
    CPU_SET_S(cpu, bytes, mask.get());
    INGINE_CHECK(::sched_setaffinity(0, bytes, mask.get()) == 0);
    ScopedEnvVar automatic("INGINE_CPU_THREADS", nullptr);
    INGINE_CHECK(default_cpu_pool_threads() == 1);
    ScopedEnvVar oversized("INGINE_CPU_THREADS", "64");
    INGINE_CHECK(default_cpu_pool_threads() == 1);

    ScopedEnvVar clear_spin("INGINE_CPU_POOL_SPIN", nullptr);
    CpuThreadPool oversub_pool(2);
    INGINE_CHECK(oversub_pool.oversubscribed());
    INGINE_CHECK(oversub_pool.effective_spin_iters() == 0);
    INGINE_CHECK(!oversub_pool.workers_pinned());
    std::atomic<int> sum{0};
    oversub_pool.parallel_for(32, 1, [&](int b, int e) {
      for (int i = b; i < e; ++i) sum.fetch_add(i + 1, std::memory_order_relaxed);
    });
    INGINE_CHECK(sum.load() == 32 * 33 / 2);
  }).join();

  // SMT-sibling sharing detection: restrict a child thread to 2 SMT siblings of 1 physical core if present.
  {
    std::map<std::string, std::vector<int>> sib_groups;
    int smt_c0 = -1, smt_c1 = -1;
    for (int cpu : topo.ordered_cpus) {
      std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
      std::string s;
      if (std::getline(f, s) && !s.empty()) {
        auto& g = sib_groups[s];
        g.push_back(cpu);
        if (g.size() == 2 && smt_c0 < 0) {
          smt_c0 = g[0];
          smt_c1 = g[1];
        }
      }
    }
    if (smt_c0 >= 0 && smt_c1 >= 0) {
      std::thread([smt_c0, smt_c1]() {
        const int maxc = std::max(smt_c0, smt_c1) + 1;
        const auto bytes = CPU_ALLOC_SIZE(maxc);
        const auto free_set = [](cpu_set_t* mask) { CPU_FREE(mask); };
        std::unique_ptr<cpu_set_t, decltype(free_set)> mask(CPU_ALLOC(maxc), free_set);
        INGINE_CHECK(mask != nullptr);
        CPU_ZERO_S(bytes, mask.get());
        CPU_SET_S(smt_c0, bytes, mask.get());
        CPU_SET_S(smt_c1, bytes, mask.get());
        INGINE_CHECK(::sched_setaffinity(0, bytes, mask.get()) == 0);
        ScopedEnvVar clear_spin("INGINE_CPU_POOL_SPIN", nullptr);
        const CpuTopologyInfo smt_topo = discover_cpu_topology();
        INGINE_CHECK(smt_topo.logical_cpus == 2);
        INGINE_CHECK(smt_topo.physical_cores == 1);
        CpuThreadPool smt_pool(2);
        INGINE_CHECK(smt_pool.oversubscribed());
        INGINE_CHECK(smt_pool.effective_spin_iters() == 0);
        std::atomic<int> acc{0};
        smt_pool.parallel_for(32, 1, [&](int b, int e) {
          for (int i = b; i < e; ++i) acc.fetch_add(i + 1, std::memory_order_relaxed);
        });
        INGINE_CHECK(acc.load() == 32 * 33 / 2);
      }).join();
    }
  }
#endif

  // Deterministic partial worker startup unwind & recovery
  {
    fail_next_cpu_pool_worker_start(0);
    bool threw0 = false;
    try {
      CpuThreadPool p0(4);
    } catch (const std::system_error&) { threw0 = true; }
    INGINE_CHECK(threw0);

    fail_next_cpu_pool_worker_start(2);
    bool threw2 = false;
    try {
      CpuThreadPool p2(4);
    } catch (const std::system_error&) { threw2 = true; }
    INGINE_CHECK(threw2);

    CpuThreadPool healthy(4);
    std::atomic<int> sum{0};
    healthy.parallel_for(128, 1, [&](int b, int e) {
      for (int i = b; i < e; ++i) sum.fetch_add(i + 1, std::memory_order_relaxed);
    });
    INGINE_CHECK(sum.load() == 128 * 129 / 2);
  }

  CpuThreadPool boundary_pool(2);
  std::atomic<int> chunks{0};
  boundary_pool.parallel_for(INT_MAX, INT_MAX - 1, [&](int begin, int end) {
    INGINE_CHECK((begin == 0 && end == INT_MAX / 2 + 1) ||
                 (begin == INT_MAX / 2 + 1 && end == INT_MAX));
    ++chunks;
  });
  INGINE_CHECK(chunks == 2);

  for (int nt : {1, 2, 4, 8, 16}) {
    CpuThreadPool pool(nt, /*spin_iters=*/2000);
    INGINE_CHECK(pool.num_threads() == nt);

    constexpr int kN = 1024;
    std::vector<std::uint64_t> out(kN, 0);
    for (int iter = 0; iter < 500; ++iter) {
      pool.parallel_for(kN, 8, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
          out[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(i) * 3u + static_cast<std::uint64_t>(iter);
        }
      });
      if (iter == 0 || iter == 499) {
        for (int i = 0; i < kN; ++i) {
          INGINE_CHECK(out[static_cast<std::size_t>(i)] ==
                       static_cast<std::uint64_t>(i) * 3u + static_cast<std::uint64_t>(iter));
        }
      }
    }

    // Force workers into futex sleep, then wake cleanly.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    std::atomic<int> woken_sum{0};
    pool.parallel_for(nt * 4, 1, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) woken_sum.fetch_add(i + 1, std::memory_order_relaxed);
    });
    const int expected_sum = (nt * 4) * (nt * 4 + 1) / 2;
    INGINE_CHECK(woken_sum.load() == expected_sum);

    // Nested parallel_for inside a worker must fall back inline without deadlock.
    std::vector<int> nested(64, 0);
    pool.parallel_for(8, 1, [&](int b0, int e0) {
      for (int i = b0; i < e0; ++i) {
        pool.parallel_for(8, 1, [&](int b1, int e1) {
          for (int j = b1; j < e1; ++j) nested[static_cast<std::size_t>(i * 8 + j)] = i * 10 + j;
        });
      }
    });
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        INGINE_CHECK(nested[static_cast<std::size_t>(i * 8 + j)] == i * 10 + j);
      }
    }

    // Worker exception propagation (skip worker throw check for nt == 1? Even caller tid=0 propagates!).
    bool caught = false;
    try {
      pool.parallel_for(nt * 2, 1, [&](int begin, int /*end*/) {
        if (begin > 0 || nt == 1) throw std::runtime_error("pool test boom");
      });
    } catch (const std::runtime_error& ex) {
      caught = (std::string(ex.what()) == "pool test boom");
    }
    INGINE_CHECK(caught);
  }
}

void test_q4_0_and_ops_parity() {
  std::printf("[3/4] Testing bit-exact Q4_0 decode + embed_q4_rmsnorm parity...\n");
  using namespace ingine::ops;
  using namespace ingine::ops::cpu;
  using namespace ingine::test;

  const CpuIsa simd_isa = detect_cpu_isa();

  // Edge-case fixture per row (H=256 = 8 blocks of 32):
  // block 0: scale +0.0 (0x0000)
  // block 1: scale -0.0 (0x8000)
  // block 2: scale +Inf (0x7C00) -> sanitized to 0.0f
  // block 3: scale -Inf (0xFC00) -> sanitized to 0.0f
  // block 4: scale  NaN (0x7E01) -> sanitized to 0.0f
  // block 5: smallest subnormal (0x0001)
  // block 6: negative normal -1.0 (0xBC00)
  // block 7: largest finite FP16 65504.0 (0x7BFF)
  constexpr int kH = 256;
  constexpr int kRowBytes = kH / 2 + (kH / 32) * 2;
  std::vector<std::uint8_t> special_row(kRowBytes, 0);
  const std::uint16_t special_scales[8] = {0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x0001, 0xBC00, 0x7BFF};
  Rng rng(20261125ull);
  for (int i = 0; i < kH / 2; ++i) {
    special_row[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(rng.next() & 0xffu);
  }
  std::memcpy(special_row.data() + kH / 2, special_scales, sizeof(special_scales));

  std::vector<float> ref(kH), got_scalar(kH), got_simd(kH);
  ingine::ops::decode_q4_0_row(special_row.data(), kH, ref.data());
  decode_q4_0_row_simd(special_row.data(), kH, got_scalar.data(), CpuIsa::Scalar);
  decode_q4_0_row_simd(special_row.data(), kH, got_simd.data(), simd_isa);
  for (int i = 0; i < kH; ++i) {
    INGINE_CHECK(f32_bits(got_scalar[static_cast<std::size_t>(i)]) == f32_bits(ref[static_cast<std::size_t>(i)]));
    INGINE_CHECK(f32_bits(got_simd[static_cast<std::size_t>(i)]) == f32_bits(ref[static_cast<std::size_t>(i)]));
  }

  // 64 random rows bit-exact against scalar_q4_0_unpack_row
  constexpr int kRows = 64;
  std::vector<std::uint8_t> table(static_cast<std::size_t>(kRows) * kRowBytes);
  std::vector<float> scalar_table(static_cast<std::size_t>(kRows) * kH);
  for (int r = 0; r < kRows; ++r) {
    auto vals = randn(rng, kH, 1.5);
    scalar_q4_0_pack_row(vals.data(), 1, kH, table.data() + static_cast<std::size_t>(r) * kRowBytes);
    scalar_q4_0_unpack_row(table.data() + static_cast<std::size_t>(r) * kRowBytes, 1, kH,
                           scalar_table.data() + static_cast<std::size_t>(r) * kH);
  }
  std::vector<float> ops_decoded(static_cast<std::size_t>(kRows) * kH);
  decode_q4_0_rows(table.data(), kRows, kH, ops_decoded.data(), &CpuThreadPool::global());
  for (std::size_t i = 0; i < ops_decoded.size(); ++i) {
    INGINE_CHECK(f32_bits(ops_decoded[i]) == f32_bits(scalar_table[i]));
  }
#if !defined(INGINE_BACKEND_CUDA) || !INGINE_BACKEND_CUDA
  std::fill(ops_decoded.begin(), ops_decoded.end(), 0.0f);
  ingine::ops::decode_q4_0(table.data(), kRows, kH, ops_decoded.data(), ingine::hal::StreamHandle{});
  for (std::size_t i = 0; i < ops_decoded.size(); ++i) {
    INGINE_CHECK(f32_bits(ops_decoded[i]) == f32_bits(scalar_table[i]));
  }
#endif

  // embed_q4_rmsnorm_cpu (both backends) + ops::run_embed_q4_rmsnorm (CPU backend) vs FP64 oracle
  constexpr int kToks = 12;
  std::vector<std::int64_t> tids = {0, 7, 63, 15, 31, 42, 3, 19, 55, 11, 28, 60};
  auto norm_w = randn(rng, kH, 0.7);
  std::vector<double> unrounded_ref, rounded_ref;
  std::vector<float> out_rmsnorm(static_cast<std::size_t>(kToks) * kH);

  for (auto rounding : {ActRounding::None, ActRounding::Bf16Rn, ActRounding::Fp16RnSat}) {
    const bool fp16_flag = (rounding == ActRounding::Fp16RnSat);
    scalar_embed_q4_rmsnorm_fp64(tids.data(), kToks, table.data(), kRows, norm_w.data(), kH, 1e-6,
                                 fp16_flag, nullptr, &unrounded_ref, &rounded_ref);
    EmbedQ4RmsNormDesc desc{};
    desc.vocab_size = kRows;
    desc.hidden_size = kH;
    desc.tokens = kToks;
    desc.weight_row_bytes = kRowBytes;
    desc.out_row_stride = kH;
    desc.eps = 1e-6f;
    desc.output_rounding = rounding;
    desc.token_bounds = TokenBoundsMode::DeferredFlag;

    std::uint32_t bad_flag = 99u;
    EmbedQ4RmsNormArgs args{};
    args.token_ids = tids.data();
    args.table_q4 = table.data();
    args.norm_weight = norm_w.data();
    args.out = out_rmsnorm.data();
    args.invalid_token_flag = &bad_flag;

    const auto* ref_vec = (rounding == ActRounding::None) ? &unrounded_ref : &rounded_ref;
    const double tol = (rounding == ActRounding::None) ? 1e-5 : (rounding == ActRounding::Bf16Rn ? 1e-2 : 2e-3);
    const OracleCheck chk{"embed_q4_rmsnorm", tol, tol, rounding == ActRounding::None ? 1.0 : 0.02};

    embed_q4_rmsnorm_cpu(desc, args, &CpuThreadPool::global());
    INGINE_CHECK(chk(out_rmsnorm, *ref_vec, rounding == ActRounding::None ? nullptr : &rounded_ref));

#if !defined(INGINE_BACKEND_CUDA) || !INGINE_BACKEND_CUDA
    std::fill(out_rmsnorm.begin(), out_rmsnorm.end(), 0.0f);
    const auto path = ingine::ops::run_embed_q4_rmsnorm(desc, args, ingine::hal::StreamHandle{});
    INGINE_CHECK(path == DispatchPath::Native);
    INGINE_CHECK(bad_flag == 0u);
    INGINE_CHECK(chk(out_rmsnorm, *ref_vec, rounding == ActRounding::None ? nullptr : &rounded_ref));
#endif
  }

#if !defined(INGINE_BACKEND_CUDA) || !INGINE_BACKEND_CUDA
  // Out-of-range DeferredFlag check on CPU HAL dispatcher
  {
    std::vector<std::int64_t> bad_tids = tids;
    bad_tids[3] = kRows;
    EmbedQ4RmsNormDesc desc{};
    desc.vocab_size = kRows;
    desc.hidden_size = kH;
    desc.tokens = kToks;
    desc.weight_row_bytes = kRowBytes;
    desc.out_row_stride = kH;
    desc.eps = 1e-6f;
    desc.token_bounds = TokenBoundsMode::DeferredFlag;
    std::uint32_t bad_flag = 0u;
    EmbedQ4RmsNormArgs args{};
    args.token_ids = bad_tids.data();
    args.table_q4 = table.data();
    args.norm_weight = norm_w.data();
    args.out = out_rmsnorm.data();
    args.invalid_token_flag = &bad_flag;
    ingine::ops::run_embed_q4_rmsnorm(desc, args, ingine::hal::StreamHandle{});
    INGINE_CHECK(bad_flag == 1u);
  }
#endif
}

void build_exl3_fixture(ingine::test::Rng& rng, int rows, int cols, int bits,
                        std::vector<std::uint8_t>& codes, std::vector<std::uint16_t>& rin,
                        std::vector<std::uint16_t>& rout) {
  const std::size_t codes_bytes =
      static_cast<std::size_t>(rows / 16) * static_cast<std::size_t>(cols / 16) * static_cast<std::size_t>(32 * bits);
  codes.resize(codes_bytes);
  for (std::size_t i = 0; i < codes_bytes; ++i) codes[i] = static_cast<std::uint8_t>(rng.next() & 0xffu);
  rin.resize(static_cast<std::size_t>(cols));
  rout.resize(static_cast<std::size_t>(rows));
  for (int c = 0; c < cols; ++c) {
    rin[static_cast<std::size_t>(c)] =
        ingine::act::host_float_to_fp16_bits(static_cast<float>(0.15 + 0.25 * rng.uniform()));
  }
  for (int r = 0; r < rows; ++r) {
    rout[static_cast<std::size_t>(r)] =
        ingine::act::host_float_to_fp16_bits(static_cast<float>(0.15 + 0.25 * rng.uniform()));
  }
}

void test_gemv_q4_q8_and_exl3_kernels() {
  std::printf("[4/4] Testing Q4_0, Q8_0, FWHT, EXL3 (2-bit/3-bit), and SwiGLU MoE kernels...\n");
  using namespace ingine::ops::cpu;
  using namespace ingine::test;

  const CpuIsa simd_isa = detect_cpu_isa();
  CpuThreadPool& pool = CpuThreadPool::global();
  Rng rng(0x5A17FACE1234ull);

  constexpr int kN = 128;
  constexpr int kK = 512;

  // Q4_0 GEMV
  {
    const int row_bytes = kK / 2 + (kK / 32) * 2;
    std::vector<std::uint8_t> w_q4(static_cast<std::size_t>(kN) * row_bytes);
    std::vector<float> w_deq(static_cast<std::size_t>(kN) * kK);
    for (int r = 0; r < kN; ++r) {
      auto v = randn(rng, kK, 0.4);
      scalar_q4_0_pack_row(v.data(), 1, kK, w_q4.data() + static_cast<std::size_t>(r) * row_bytes);
      scalar_q4_0_unpack_row(w_q4.data() + static_cast<std::size_t>(r) * row_bytes, 1, kK,
                             w_deq.data() + static_cast<std::size_t>(r) * kK);
    }
    auto x = randn(rng, kK, 0.5);
    std::vector<double> ref(kN, 0.0);
    for (int r = 0; r < kN; ++r) {
      double s = 0.0;
      for (int c = 0; c < kK; ++c) s += static_cast<double>(w_deq[static_cast<std::size_t>(r) * kK + c]) * x[c];
      ref[static_cast<std::size_t>(r)] = s;
    }
    std::vector<float> y_scalar(kN), y_simd(kN);
    gemv_q4_0_f32(w_q4.data(), x.data(), y_scalar.data(), kN, kK, nullptr, CpuIsa::Scalar);
    gemv_q4_0_f32(w_q4.data(), x.data(), y_simd.data(), kN, kK, &pool, simd_isa);
    INGINE_CHECK(report("gemv_q4_0_f32 scalar", compare(y_scalar.data(), ref.data(), kN), 1e-4));
    INGINE_CHECK(report("gemv_q4_0_f32 simd+pool", compare(y_simd.data(), ref.data(), kN), 1e-4));
  }

  // Q8_0 GEMV (f32 and q8_0 activations)
  {
    const int blocks = kK / 32;
    std::vector<BlockQ8_0> w_q8(static_cast<std::size_t>(kN) * blocks);
    for (int r = 0; r < kN; ++r) {
      auto v = randn(rng, kK, 0.4);
      quantize_f32_to_q8_0(v.data(), w_q8.data() + static_cast<std::size_t>(r) * blocks, kK);
    }
    auto x = randn(rng, kK, 0.5);
    std::vector<BlockQ8_0> x_q8(static_cast<std::size_t>(blocks));
    quantize_f32_to_q8_0(x.data(), x_q8.data(), kK);

    std::vector<double> ref_f32(kN, 0.0), ref_q8(kN, 0.0);
    for (int r = 0; r < kN; ++r) {
      double sf = 0.0, sq = 0.0;
      for (int b = 0; b < blocks; ++b) {
        const auto& wb = w_q8[static_cast<std::size_t>(r) * blocks + b];
        const double dw = static_cast<double>(fp16_bits_to_float(wb.d_fp16));
        const double dx = static_cast<double>(fp16_bits_to_float(x_q8[static_cast<std::size_t>(b)].d_fp16));
        std::int32_t isum = 0;
        double fdot = 0.0;
        for (int j = 0; j < 32; ++j) {
          fdot += static_cast<double>(wb.qs[j]) * static_cast<double>(x[b * 32 + j]);
          isum += static_cast<std::int32_t>(wb.qs[j]) *
                  static_cast<std::int32_t>(x_q8[static_cast<std::size_t>(b)].qs[j]);
        }
        sf += dw * fdot;
        sq += (dw * dx) * static_cast<double>(isum);
      }
      ref_f32[static_cast<std::size_t>(r)] = sf;
      ref_q8[static_cast<std::size_t>(r)] = sq;
    }

    std::vector<float> y_f32_s(kN), y_f32_v(kN), y_q8_s(kN), y_q8_v(kN);
    gemv_q8_0_f32(w_q8.data(), x.data(), y_f32_s.data(), kN, kK, nullptr, CpuIsa::Scalar);
    gemv_q8_0_f32(w_q8.data(), x.data(), y_f32_v.data(), kN, kK, &pool, simd_isa);
    gemv_q8_0_q8_0(w_q8.data(), x_q8.data(), y_q8_s.data(), kN, kK, nullptr, CpuIsa::Scalar);
    gemv_q8_0_q8_0(w_q8.data(), x_q8.data(), y_q8_v.data(), kN, kK, &pool, simd_isa);
    INGINE_CHECK(report("gemv_q8_0_f32 scalar", compare(y_f32_s.data(), ref_f32.data(), kN), 1e-4));
    INGINE_CHECK(report("gemv_q8_0_f32 simd+pool", compare(y_f32_v.data(), ref_f32.data(), kN), 1e-4));
    INGINE_CHECK(report("gemv_q8_0_q8_0 scalar", compare(y_q8_s.data(), ref_q8.data(), kN), 1e-4));
    INGINE_CHECK(report("gemv_q8_0_q8_0 simd+pool", compare(y_q8_v.data(), ref_q8.data(), kN), 1e-4));
  }

  // FWHT 128 involution + orthogonality
  {
    auto v0 = randn(rng, 128, 1.0);
    std::vector<float> v = v0;
    fwht_128_cpu(v.data(), simd_isa);
    double norm0 = 0.0, norm1 = 0.0;
    for (int i = 0; i < 128; ++i) {
      norm0 += static_cast<double>(v0[static_cast<std::size_t>(i)]) * v0[static_cast<std::size_t>(i)];
      norm1 += static_cast<double>(v[static_cast<std::size_t>(i)]) * v[static_cast<std::size_t>(i)];
    }
    INGINE_CHECK(std::fabs(norm0 - norm1) < 1e-4 * (norm0 + 1.0));
    fwht_128_cpu(v.data(), simd_isa);
    double max_err = 0.0;
    for (int i = 0; i < 128; ++i) max_err = std::max(max_err, std::fabs(double(v[i]) - double(v0[i])));
    INGINE_CHECK(max_err < 1e-5);
  }

  // EXL3 2-bit and 3-bit matrix dequant & GEMV parity
  constexpr int kExRows = 256;
  constexpr int kExCols = 256;
  for (int bits : {2, 3}) {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> rin, rout;
    build_exl3_fixture(rng, kExRows, kExCols, bits, codes, rin, rout);

    std::vector<float> mat(static_cast<std::size_t>(kExRows) * kExCols);
    dequant_exl3_matrix_cpu(codes.data(), rin.data(), rout.data(), bits, kExRows, kExCols, mat.data(), &pool, simd_isa);

    auto x = randn(rng, kExCols, 0.5);
    std::vector<double> ref_y(kExRows, 0.0);
    for (int r = 0; r < kExRows; ++r) {
      double s = 0.0;
      for (int c = 0; c < kExCols; ++c) s += static_cast<double>(mat[static_cast<std::size_t>(r) * kExCols + c]) * x[c];
      ref_y[static_cast<std::size_t>(r)] = s;
    }

    std::vector<float> y_scalar(kExRows), y_simd(kExRows);
    gemv_exl3_f32(codes.data(), rin.data(), rout.data(), bits, x.data(), y_scalar.data(), kExRows, kExCols, nullptr,
                  CpuIsa::Scalar);
    gemv_exl3_f32(codes.data(), rin.data(), rout.data(), bits, x.data(), y_simd.data(), kExRows, kExCols, &pool,
                  simd_isa);

    const std::string tag = "gemv_exl3_" + std::to_string(bits);
    INGINE_CHECK(report((tag + " scalar").c_str(), compare(y_scalar.data(), ref_y.data(), kExRows), 1e-4));
    INGINE_CHECK(report((tag + " simd+pool").c_str(), compare(y_simd.data(), ref_y.data(), kExRows), 1e-4));
  }

  // EXL3 SwiGLU MoE expert MLP vs explicit dequantized reference
  {
    constexpr int kH = 256;
    constexpr int kI = 256;
    std::vector<std::uint8_t> c13, c2;
    std::vector<std::uint16_t> rin13, rout13, rin2, rout2;
    build_exl3_fixture(rng, 2 * kI, kH, 2, c13, rin13, rout13);
    build_exl3_fixture(rng, kH, kI, 3, c2, rin2, rout2);

    std::vector<float> m13(static_cast<std::size_t>(2 * kI) * kH), m2(static_cast<std::size_t>(kH) * kI);
    dequant_exl3_matrix_cpu(c13.data(), rin13.data(), rout13.data(), 2, 2 * kI, kH, m13.data(), &pool, simd_isa);
    dequant_exl3_matrix_cpu(c2.data(), rin2.data(), rout2.data(), 3, kH, kI, m2.data(), &pool, simd_isa);

    auto x = randn(rng, kH, 0.5);
    std::vector<double> gu(2 * kI, 0.0), act_h(kI, 0.0), ref_out(kH, 0.0);
    for (int r = 0; r < 2 * kI; ++r) {
      double s = 0.0;
      for (int c = 0; c < kH; ++c) s += static_cast<double>(m13[static_cast<std::size_t>(r) * kH + c]) * x[c];
      gu[static_cast<std::size_t>(r)] = s;
    }
    for (int i = 0; i < kI; ++i) {
      const double g = gu[static_cast<std::size_t>(i)];
      const double u = gu[static_cast<std::size_t>(kI + i)];
      act_h[static_cast<std::size_t>(i)] = (g / (1.0 + std::exp(-g))) * u;
    }
    for (int r = 0; r < kH; ++r) {
      double s = 0.0;
      for (int c = 0; c < kI; ++c) s += static_cast<double>(m2[static_cast<std::size_t>(r) * kI + c]) * act_h[c];
      ref_out[static_cast<std::size_t>(r)] = s;
    }

    std::vector<float> got(kH);
    moe_expert_mlp_exl3_cpu(c13.data(), rin13.data(), rout13.data(), 2, c2.data(), rin2.data(), rout2.data(), 3,
                            x.data(), got.data(), kH, kI, &pool, simd_isa);
    INGINE_CHECK(report("moe_expert_mlp_exl3_cpu", compare(got.data(), ref_out.data(), kH), 1e-4));
  }
}

// ----------------------------------------------------------------------------------------------- Benchmarks

class MutexBarrierPool {
 public:
  using TaskFn = void (*)(int tid, int nt, void* ctx);
  explicit MutexBarrierPool(int nt) : nt_(nt) {
    for (int i = 1; i < nt_; ++i) {
      workers_.emplace_back([this, i]() { worker(i); });
    }
  }
  ~MutexBarrierPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      ++epoch_;
    }
    cv_work_.notify_all();
    for (auto& t : workers_) t.join();
  }
  void dispatch(TaskFn fn, void* ctx) {
    if (nt_ <= 1) {
      fn(0, 1, ctx);
      return;
    }
    std::uint64_t my_epoch = 0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      fn_ = fn;
      ctx_ = ctx;
      done_ = 0;
      my_epoch = ++epoch_;
    }
    cv_work_.notify_all();
    fn(0, nt_, ctx);
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&]() { return done_ == nt_ - 1; });
    (void)my_epoch;
  }

 private:
  void worker(int tid) {
    std::uint64_t seen = 0;
    for (;;) {
      TaskFn fn = nullptr;
      void* ctx = nullptr;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_work_.wait(lk, [&]() { return epoch_ != seen; });
        seen = epoch_;
        if (stop_) return;
        fn = fn_;
        ctx = ctx_;
      }
      fn(tid, nt_, ctx);
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (++done_ == nt_ - 1) cv_done_.notify_one();
      }
    }
  }
  int nt_;
  std::mutex mu_;
  std::condition_variable cv_work_, cv_done_;
  std::uint64_t epoch_ = 0;
  int done_ = 0;
  bool stop_ = false;
  TaskFn fn_ = nullptr;
  void* ctx_ = nullptr;
  std::vector<std::thread> workers_;
};

double percentile_ns(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  const std::size_t idx = std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size() - 1)));
  return v[idx];
}

double median_d(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void run_benchmarks() {
  using namespace ingine::ops::cpu;
  using namespace ingine::test;
  const CpuIsa isa = detect_cpu_isa();
  const CpuTopologyInfo topo = discover_cpu_topology();
  std::printf("=== bare-metal Phase B CPU benchmarks (ISA=%s, hw_cores=%d, logical=%d, numa_node=%d) ===\n",
              cpu_isa_name(isa), default_cpu_pool_threads(), topo.logical_cpus, topo.caller_numa_node);

  // 1. 512 MiB HugePage + NUMA-local mbind vs 4 KiB TLB page-walk microbenchmark
  {
    constexpr std::size_t kBytes = 512ull << 20;
    constexpr std::size_t kPages = kBytes / 4096ull;
    constexpr std::size_t kSamples = kPages * 2;  // 262,144 scattered 4 KiB-strided loads per pass

    const int pin_cpu = !topo.primary_cpus.empty() ? topo.primary_cpus[0] : topo.caller_cpu;
    ScopedThreadPin pin(pin_cpu);
    const int local_node = ingine::current_numa_node() >= 0 ? ingine::current_numa_node() : 0;

    // Deterministic shuffled 4 KiB page order to force L1/L2 dTLB pressure.
    std::vector<std::uint32_t> perm(kPages);
    std::iota(perm.begin(), perm.end(), 0u);
    Rng prng(987654321ull);
    for (std::size_t i = kPages - 1; i > 0; --i) {
      const std::size_t j = static_cast<std::size_t>(prng.next() % (i + 1));
      std::swap(perm[i], perm[j]);
    }

    auto measure_tlb = [&](bool allow_hp, const char* label) -> double {
      ingine::HugePageMapping buf =
          ingine::alloc_hugepage_anon(kBytes, allow_hp, local_node, ingine::NumaPolicy::Bind);
      auto* words = reinterpret_cast<std::uint64_t*>(buf.ptr);
      // Populate 1 word per 4 KiB page (touches all 512 MiB on local_node).
      for (std::size_t p = 0; p < kPages; ++p) {
        words[p * 512ull] = p * 6364136223846793005ull + 1ull;
      }
      const auto sm = ingine::query_range_smaps_hugepages(buf.ptr, kBytes);
      const auto nm = ingine::query_range_numa_nodes(buf.ptr, kBytes);
      const std::uint64_t local_mb =
          (nm.node_bytes.count(local_node) ? nm.node_bytes.at(local_node) : 0ull) / 1048576ull;

      volatile std::uint64_t sink = 0;
      std::vector<double> passes_ns;
      // 1 warmup + 7 timed sweeps over scattered 4 KiB pages + 1-slot dependent hop
      for (int t = 0; t < 8; ++t) {
        const auto t0 = Clock::now();
        std::uint64_t acc = 0;
        for (std::size_t i = 0; i < kSamples; ++i) {
          const std::size_t p = perm[i % kPages];
          acc += words[p * 512ull + ((acc >> 12) & 7u)];
        }
        const auto t1 = Clock::now();
        sink = acc;
        if (t > 0) {
          passes_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() /
                              static_cast<double>(kSamples));
        }
      }
      (void)sink;
      ingine::free_hugepage_anon(buf);
      const double med_ns = median_d(passes_ns);
      std::printf("  TLB 512MiB %-14s (cpu=%d node=%d): AnonHugePages=%5.1f MiB  Rss=%5.1f MiB  localN%d=%llu MiB  median=%.2f ns/access\n",
                  label, pin_cpu, local_node, sm.anon_huge_pages_bytes / 1048576.0, sm.rss_bytes / 1048576.0,
                  local_node, static_cast<unsigned long long>(local_mb), med_ns);
      return med_ns;
    };

    const double ns_4k = measure_tlb(false, "4KiB-base");
    const double ns_2m = measure_tlb(true, "2MiB-THP");
    std::printf("  TLB 512MiB speedup (4KiB / 2MiB-THP): %.2fx\n", ns_4k / ns_2m);
  }

  // 2. CpuThreadPool vs std::mutex + condition_variable barrier latency
  {
    std::printf("\n[Bench 2] Thread-pool dispatch barrier latency (2000 dispatches after 200 warmup):\n");
    for (int nt : {8, 16}) {
      MutexBarrierPool mpool(nt);
      CpuThreadPool fpool(nt, 4000);
      std::atomic<int> counter{0};
      auto task = [](int tid, int /*nt*/, void* raw) {
        if (tid == 0) static_cast<std::atomic<int>*>(raw)->fetch_add(1, std::memory_order_relaxed);
      };

      for (int i = 0; i < 200; ++i) {
        mpool.dispatch(task, &counter);
        fpool.dispatch(nt, task, &counter);
      }

      std::vector<double> m_ns(2000), f_ns(2000);
      for (int i = 0; i < 2000; ++i) {
        const auto t0 = Clock::now();
        mpool.dispatch(task, &counter);
        const auto t1 = Clock::now();
        m_ns[static_cast<std::size_t>(i)] = std::chrono::duration<double, std::nano>(t1 - t0).count();
      }
      for (int i = 0; i < 2000; ++i) {
        const auto t0 = Clock::now();
        fpool.dispatch(nt, task, &counter);
        const auto t1 = Clock::now();
        f_ns[static_cast<std::size_t>(i)] = std::chrono::duration<double, std::nano>(t1 - t0).count();
      }
      std::printf("  %2dT mutex+cv   : p50=%.2f us  p95=%.2f us  p99=%.2f us\n",
                  nt, percentile_ns(m_ns, 0.50) / 1e3, percentile_ns(m_ns, 0.95) / 1e3,
                  percentile_ns(m_ns, 0.99) / 1e3);
      std::printf("  %2dT futex+spin : p50=%.2f us  p95=%.2f us  p99=%.2f us  (p50 speedup %.2fx, p99 %.2fx)\n",
                  nt, percentile_ns(f_ns, 0.50) / 1e3, percentile_ns(f_ns, 0.95) / 1e3,
                  percentile_ns(f_ns, 0.99) / 1e3,
                  percentile_ns(m_ns, 0.50) / percentile_ns(f_ns, 0.50),
                  percentile_ns(m_ns, 0.99) / percentile_ns(f_ns, 0.99));
    }
  }

  // 3. Microkernel throughput: 1T Scalar vs 1T SIMD+prefetch vs 8T vs 16T Pool+SIMD
  {
    std::printf("\n[Bench 3] Microkernel throughput (median of 7 timed runs):\n");
    Rng rng(7777777ull);
    CpuThreadPool pool8(8), pool16(16);

    auto time_med_ms = [](auto&& fn, int iters = 7) {
      fn();  // warmup
      std::vector<double> ms;
      ms.reserve(static_cast<std::size_t>(iters));
      for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      return median_d(ms);
    };

    // Q4_0 GEMV [4096, 4096]
    {
      constexpr int N = 4096, K = 4096;
      const std::size_t rb = K / 2 + (K / 32) * 2;
      const std::size_t bytes = static_cast<std::size_t>(N) * rb;
      std::vector<std::uint8_t> w(bytes);
      for (std::size_t i = 0; i < bytes; ++i) w[i] = static_cast<std::uint8_t>(rng.next() & 0xffu);
      auto x = randn(rng, K, 0.5);
      std::vector<float> y(N);
      const double t_1s = time_med_ms([&] { gemv_q4_0_f32(w.data(), x.data(), y.data(), N, K, nullptr, CpuIsa::Scalar); });
      const double t_1v = time_med_ms([&] { gemv_q4_0_f32(w.data(), x.data(), y.data(), N, K, nullptr, isa); });
      const double t_8v = time_med_ms([&] { gemv_q4_0_f32(w.data(), x.data(), y.data(), N, K, &pool8, isa); });
      const double t_16v = time_med_ms([&] { gemv_q4_0_f32(w.data(), x.data(), y.data(), N, K, &pool16, isa); });
      const double gb = static_cast<double>(bytes) / 1e9;
      std::printf("  gemv_q4_0_f32  [4096x4096]: 1T-Scalar=%6.3fms (%5.1f GB/s) | 1T-SIMD=%6.3fms (%5.1f GB/s, %.2fx) | 8T=%6.3fms (%5.1f GB/s) | 16T=%6.3fms (%5.1f GB/s, %.2fx vs 1T-Scalar)\n",
                  t_1s, gb / (t_1s * 1e-3), t_1v, gb / (t_1v * 1e-3), t_1s / t_1v,
                  t_8v, gb / (t_8v * 1e-3), t_16v, gb / (t_16v * 1e-3), t_1s / t_16v);
    }

    // Q8_0 x Q8_0 GEMV [4096, 4096]
    {
      constexpr int N = 4096, K = 4096;
      const int blocks = K / 32;
      std::vector<BlockQ8_0> w(static_cast<std::size_t>(N) * blocks);
      for (auto& b : w) {
        b.d_fp16 = 0x3800;
        for (int j = 0; j < 32; ++j) b.qs[j] = static_cast<std::int8_t>((rng.next() & 0x7fu) - 64);
      }
      auto x = randn(rng, K, 0.5);
      std::vector<BlockQ8_0> xq(static_cast<std::size_t>(blocks));
      quantize_f32_to_q8_0(x.data(), xq.data(), K);
      std::vector<float> y(N);
      const double t_1s = time_med_ms([&] { gemv_q8_0_q8_0(w.data(), xq.data(), y.data(), N, K, nullptr, CpuIsa::Scalar); });
      const double t_1v = time_med_ms([&] { gemv_q8_0_q8_0(w.data(), xq.data(), y.data(), N, K, nullptr, isa); });
      const double t_8v = time_med_ms([&] { gemv_q8_0_q8_0(w.data(), xq.data(), y.data(), N, K, &pool8, isa); });
      const double t_16v = time_med_ms([&] { gemv_q8_0_q8_0(w.data(), xq.data(), y.data(), N, K, &pool16, isa); });
      const double gb = static_cast<double>(w.size() * sizeof(BlockQ8_0)) / 1e9;
      std::printf("  gemv_q8_0_q8_0 [4096x4096]: 1T-Scalar=%6.3fms (%5.1f GB/s) | 1T-SIMD=%6.3fms (%5.1f GB/s, %.2fx) | 8T=%6.3fms (%5.1f GB/s) | 16T=%6.3fms (%5.1f GB/s, %.2fx vs 1T-Scalar)\n",
                  t_1s, gb / (t_1s * 1e-3), t_1v, gb / (t_1v * 1e-3), t_1s / t_1v,
                  t_8v, gb / (t_8v * 1e-3), t_16v, gb / (t_16v * 1e-3), t_1s / t_16v);
    }

    // Q8_0 x F32 GEMV [4096, 4096]
    {
      constexpr int N = 4096, K = 4096;
      const int blocks = K / 32;
      std::vector<BlockQ8_0> w(static_cast<std::size_t>(N) * blocks);
      for (auto& b : w) {
        b.d_fp16 = 0x3800;
        for (int j = 0; j < 32; ++j) b.qs[j] = static_cast<std::int8_t>((rng.next() & 0x7fu) - 64);
      }
      auto x = randn(rng, K, 0.5);
      std::vector<float> y(N);
      const double t_1s = time_med_ms([&] { gemv_q8_0_f32(w.data(), x.data(), y.data(), N, K, nullptr, CpuIsa::Scalar); });
      const double t_1v = time_med_ms([&] { gemv_q8_0_f32(w.data(), x.data(), y.data(), N, K, nullptr, isa); });
      const double t_8v = time_med_ms([&] { gemv_q8_0_f32(w.data(), x.data(), y.data(), N, K, &pool8, isa); });
      const double t_16v = time_med_ms([&] { gemv_q8_0_f32(w.data(), x.data(), y.data(), N, K, &pool16, isa); });
      const double gb = static_cast<double>(w.size() * sizeof(BlockQ8_0)) / 1e9;
      std::printf("  gemv_q8_0_f32  [4096x4096]: 1T-Scalar=%6.3fms (%5.1f GB/s) | 1T-SIMD=%6.3fms (%5.1f GB/s, %.2fx) | 8T=%6.3fms (%5.1f GB/s) | 16T=%6.3fms (%5.1f GB/s, %.2fx vs 1T-Scalar)\n",
                  t_1s, gb / (t_1s * 1e-3), t_1v, gb / (t_1v * 1e-3), t_1s / t_1v,
                  t_8v, gb / (t_8v * 1e-3), t_16v, gb / (t_16v * 1e-3), t_1s / t_16v);
    }

    // EXL3 2-bit and 3-bit GEMV [2048, 2048]
    for (int bits : {2, 3}) {
      constexpr int R = 2048, C = 2048;
      std::vector<std::uint8_t> codes;
      std::vector<std::uint16_t> rin, rout;
      build_exl3_fixture(rng, R, C, bits, codes, rin, rout);
      auto x = randn(rng, C, 0.5);
      std::vector<float> y(R);
      const double t_1s = time_med_ms([&] { gemv_exl3_f32(codes.data(), rin.data(), rout.data(), bits, x.data(), y.data(), R, C, nullptr, CpuIsa::Scalar); });
      const double t_1v = time_med_ms([&] { gemv_exl3_f32(codes.data(), rin.data(), rout.data(), bits, x.data(), y.data(), R, C, nullptr, isa); });
      const double t_8v = time_med_ms([&] { gemv_exl3_f32(codes.data(), rin.data(), rout.data(), bits, x.data(), y.data(), R, C, &pool8, isa); });
      const double t_16v = time_med_ms([&] { gemv_exl3_f32(codes.data(), rin.data(), rout.data(), bits, x.data(), y.data(), R, C, &pool16, isa); });
      const double gb = static_cast<double>(codes.size()) / 1e9;
      std::printf("  gemv_exl3_%dbit [2048x2048]: 1T-Scalar=%6.3fms (%5.2f GB/s) | 1T-SIMD=%6.3fms (%5.2f GB/s, %.2fx) | 8T=%6.3fms (%5.2f GB/s) | 16T=%6.3fms (%5.2f GB/s, %.2fx vs 1T-Scalar)\n",
                  bits, t_1s, gb / (t_1s * 1e-3), t_1v, gb / (t_1v * 1e-3), t_1s / t_1v,
                  t_8v, gb / (t_8v * 1e-3), t_16v, gb / (t_16v * 1e-3), t_1s / t_16v);
    }

    // Q4_0 decode_q4_0_rows [1024, 4096]
    {
      constexpr int R = 1024, H = 4096;
      const std::size_t rb = H / 2 + (H / 32) * 2;
      std::vector<std::uint8_t> tbl(static_cast<std::size_t>(R) * rb);
      for (auto& b : tbl) b = static_cast<std::uint8_t>(rng.next() & 0xffu);
      std::vector<float> out(static_cast<std::size_t>(R) * H);
      const double t_1s = time_med_ms([&] { decode_q4_0_rows(tbl.data(), R, H, out.data(), nullptr, CpuIsa::Scalar); });
      const double t_1v = time_med_ms([&] { decode_q4_0_rows(tbl.data(), R, H, out.data(), nullptr, isa); });
      const double t_8v = time_med_ms([&] { decode_q4_0_rows(tbl.data(), R, H, out.data(), &pool8, isa); });
      const double t_16v = time_med_ms([&] { decode_q4_0_rows(tbl.data(), R, H, out.data(), &pool16, isa); });
      std::printf("  decode_q4_0    [1024x4096]: 1T-Scalar=%6.3fms | 1T-SIMD=%6.3fms (%.2fx) | 8T=%6.3fms | 16T=%6.3fms (%.2fx vs 1T-Scalar)\n",
                  t_1s, t_1v, t_1s / t_1v, t_8v, t_16v, t_1s / t_16v);
    }
  }

  // 4. Phase B: Pinned sibling-0 vs unpinned jitter + oversubscribed adaptive spin=0
  {
    std::printf("\n[Bench 4] Phase B affinity & oversubscription benchmarks:\n");
    const int nt = topo.physical_cores >= 16 ? 16 : (topo.physical_cores >= 8 ? 8 : topo.physical_cores);
    if (nt >= 4) {
      CpuThreadPool unpinned(CpuPoolOptions{nt, 4000, /*pin_workers=*/0});
      CpuThreadPool pinned(CpuPoolOptions{nt, 4000, /*pin_workers=*/1});
      std::atomic<int> counter{0};
      auto task = [](int tid, int /*nt*/, void* raw) {
        if (tid == 0) static_cast<std::atomic<int>*>(raw)->fetch_add(1, std::memory_order_relaxed);
      };
      for (int i = 0; i < 200; ++i) {
        unpinned.dispatch(nt, task, &counter);
        pinned.dispatch(nt, task, &counter);
      }
      std::vector<double> u_ns(2000), p_ns(2000);
      for (int i = 0; i < 2000; ++i) {
        const auto t0 = Clock::now();
        unpinned.dispatch(nt, task, &counter);
        const auto t1 = Clock::now();
        u_ns[static_cast<std::size_t>(i)] = std::chrono::duration<double, std::nano>(t1 - t0).count();
      }
      for (int i = 0; i < 2000; ++i) {
        const auto t0 = Clock::now();
        pinned.dispatch(nt, task, &counter);
        const auto t1 = Clock::now();
        p_ns[static_cast<std::size_t>(i)] = std::chrono::duration<double, std::nano>(t1 - t0).count();
      }
      std::printf("  %2dT unpinned barrier : p50=%.2f us  p95=%.2f us  p99=%.2f us\n",
                  nt, percentile_ns(u_ns, 0.50) / 1e3, percentile_ns(u_ns, 0.95) / 1e3,
                  percentile_ns(u_ns, 0.99) / 1e3);
      std::printf("  %2dT sibling0 pinned  : p50=%.2f us  p95=%.2f us  p99=%.2f us\n",
                  nt, percentile_ns(p_ns, 0.50) / 1e3, percentile_ns(p_ns, 0.95) / 1e3,
                  percentile_ns(p_ns, 0.99) / 1e3);

      constexpr int N = 4096, K = 4096;
      const std::size_t rb = K / 2 + (K / 32) * 2;
      Rng qr(44556677ull);
      std::vector<std::uint8_t> w(static_cast<std::size_t>(N) * rb);
      for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<std::uint8_t>(qr.next() & 0xffu);
      auto x = randn(qr, K, 0.5);
      std::vector<float> y(N);
      auto med_gemv = [&](CpuThreadPool& pl) {
        gemv_q4_0_f32(w.data(), x.data(), y.data(), N, K, &pl, isa);
        std::vector<double> ms;
        for (int i = 0; i < 7; ++i) {
          const auto t0 = Clock::now();
          gemv_q4_0_f32(w.data(), x.data(), y.data(), N, K, &pl, isa);
          const auto t1 = Clock::now();
          ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return median_d(ms);
      };
      const double ug = med_gemv(unpinned);
      const double pg = med_gemv(pinned);
      std::printf("  %2dT q4_0 GEMV [4096x4096]: unpinned=%.3f ms | sibling0-pinned=%.3f ms (%.2fx)\n",
                  nt, ug, pg, ug / pg);
    }

#ifdef __linux__
    if (topo.ordered_cpus.size() >= 2) {
      std::thread([&]() {
        const int c0 = topo.ordered_cpus[0];
        const int c1 = topo.ordered_cpus[1];
        const int maxc = std::max(c0, c1) + 1;
        const auto bytes = CPU_ALLOC_SIZE(maxc);
        const auto free_set = [](cpu_set_t* m) { CPU_FREE(m); };
        std::unique_ptr<cpu_set_t, decltype(free_set)> mask(CPU_ALLOC(maxc), free_set);
        CPU_ZERO_S(bytes, mask.get());
        CPU_SET_S(c0, bytes, mask.get());
        CPU_SET_S(c1, bytes, mask.get());
        if (::sched_setaffinity(0, bytes, mask.get()) != 0) return;

        ScopedEnvVar clear_spin("INGINE_CPU_POOL_SPIN", nullptr);
        CpuThreadPool spin_pool(CpuPoolOptions{8, 4000, 0});
        CpuThreadPool adaptive_pool(CpuPoolOptions{8, -1, -1});
        INGINE_CHECK(adaptive_pool.oversubscribed());
        INGINE_CHECK(adaptive_pool.effective_spin_iters() == 0);

        std::atomic<int> counter{0};
        auto task = [](int tid, int /*nt*/, void* raw) {
          if (tid == 0) static_cast<std::atomic<int>*>(raw)->fetch_add(1, std::memory_order_relaxed);
        };
        for (int i = 0; i < 20; ++i) {
          spin_pool.dispatch(8, task, &counter);
          adaptive_pool.dispatch(8, task, &counter);
        }
        std::vector<double> s_ns(500), a_ns(500);
        for (int i = 0; i < 500; ++i) {
          const auto t0 = Clock::now();
          spin_pool.dispatch(8, task, &counter);
          const auto t1 = Clock::now();
          s_ns[static_cast<std::size_t>(i)] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }
        for (int i = 0; i < 500; ++i) {
          const auto t0 = Clock::now();
          adaptive_pool.dispatch(8, task, &counter);
          const auto t1 = Clock::now();
          a_ns[static_cast<std::size_t>(i)] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }
        std::printf("  Oversubscribed 8T on 2 CPUs: spin=4000 p50=%.2f us (p99=%.2f us) | adaptive spin=0 p50=%.2f us (p99=%.2f us, %.2fx p50)\n",
                    percentile_ns(s_ns, 0.50) / 1e3, percentile_ns(s_ns, 0.99) / 1e3,
                    percentile_ns(a_ns, 0.50) / 1e3, percentile_ns(a_ns, 0.99) / 1e3,
                    percentile_ns(s_ns, 0.50) / percentile_ns(a_ns, 0.50));
      }).join();
    }
#endif
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool bench = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--bench") == 0) bench = true;
  }
  test_hugepage_allocations();
  test_cpu_thread_pool();
  test_q4_0_and_ops_parity();
  test_gemv_q4_q8_and_exl3_kernels();
  if (bench) run_benchmarks();
  std::printf("core_baremetal_cpu: ALL PASSED\n");
  return 0;
}
