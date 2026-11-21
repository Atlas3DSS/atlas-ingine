// Counter-based RNG of the draft kernels (host and device): Philox4x32-10 (Salmon et al. 2011).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::draft {

struct Philox {
  static __host__ __device__ __forceinline__ void round(std::uint32_t (&c)[4], const std::uint32_t (&k)[2]) {
    const std::uint64_t p0 = static_cast<std::uint64_t>(0xD2511F53u) * c[0];
    const std::uint64_t p1 = static_cast<std::uint64_t>(0xCD9E8D57u) * c[2];
    const std::uint32_t hi0 = static_cast<std::uint32_t>(p0 >> 32), lo0 = static_cast<std::uint32_t>(p0);
    const std::uint32_t hi1 = static_cast<std::uint32_t>(p1 >> 32), lo1 = static_cast<std::uint32_t>(p1);
    c[0] = hi1 ^ c[1] ^ k[0];
    c[1] = lo1;
    c[2] = hi0 ^ c[3] ^ k[1];
    c[3] = lo0;
  }
  static __host__ __device__ __forceinline__ void generate(std::uint32_t (&c)[4], std::uint64_t seed) {
    std::uint32_t k[2] = {static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)};
#pragma unroll
    for (int r = 0; r < 10; ++r) {
      round(c, k);
      k[0] += 0x9E3779B9u;
      k[1] += 0xBB67AE85u;
    }
  }
};

// Uniform in [0, 1) (24-bit resolution, like torch.rand for FP32) for (seed, stream, a, b).
__host__ __device__ __forceinline__ float uniform01(std::uint64_t seed, std::uint64_t stream, std::uint32_t a,
                                                    std::uint32_t b) {
  std::uint32_t c[4] = {static_cast<std::uint32_t>(stream), static_cast<std::uint32_t>(stream >> 32), a, b};
  Philox::generate(c, seed);
  return static_cast<float>(c[0] >> 8) * (1.0f / 16777216.0f);
}

}  // namespace ingine::kernels::draft
