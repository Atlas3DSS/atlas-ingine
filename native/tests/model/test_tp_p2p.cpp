// Unit tests for multi-GPU TP startup byte-exact P2P probe and fail-closed IOMMU guard.
#include "model/tp_engine.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <iostream>

using namespace ingine::model;

int main() {
  // 1. Deterministic non-zero pattern sanity (CPU-only, always runs).
  const std::uint32_t w00 = tp_p2p_probe_word(0, 0);
  const std::uint32_t w10 = tp_p2p_probe_word(1, 0);
  const std::uint32_t w01 = tp_p2p_probe_word(0, 1);
  assert(w00 != 0 && w00 != 0xDEADBEEFu);
  assert(w00 != w10);
  assert(w00 != w01);

  // 2. tp_size <= 1 is a clean no-op.
  {
    const TpP2pProbeResult r1 = probe_and_enable_tp_p2p(1, false);
    assert(r1.capable_pairs == 0 && r1.verified_pairs == 0 && !r1.p2p_enabled && !r1.corrupted);
  }

  // 3. When >= 2 CUDA GPUs are visible, run live P2P probe + fail-closed synthetic corruption test.
  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count >= 2) {
    const int n = dev_count >= 4 ? 4 : 2;
    const TpP2pProbeResult ok = probe_and_enable_tp_p2p(n, false);
    std::cout << "live P2P probe (" << n << " GPUs): capable=" << ok.capable_pairs
              << " verified=" << ok.verified_pairs << " enabled=" << ok.p2p_enabled
              << " corrupted=" << ok.corrupted << "\n";
    assert(!ok.corrupted);
    assert(ok.verified_pairs == ok.capable_pairs);
    assert(ok.p2p_enabled == (ok.capable_pairs > 0));

    if (ok.capable_pairs > 0) {
      const TpP2pProbeResult bad = probe_and_enable_tp_p2p(n, true);
      std::cout << "forced-fail P2P probe: corrupted=" << bad.corrupted << " enabled=" << bad.p2p_enabled
                << " bad_pair=(" << bad.first_bad_src << "->" << bad.first_bad_dst << ") mismatches="
                << bad.first_bad_mismatches << " detail=" << bad.detail << "\n";
      assert(bad.corrupted);
      assert(!bad.p2p_enabled);
      assert(bad.first_bad_src >= 0 && bad.first_bad_dst >= 0);
      assert(bad.first_bad_mismatches == 1);
    }
  } else {
    std::cout << "CUDA multi-GPU probe skipped (visible GPUs=" << dev_count << ")\n";
  }
  std::cout << "model_tp_p2p: PASS\n";
  return 0;
}
