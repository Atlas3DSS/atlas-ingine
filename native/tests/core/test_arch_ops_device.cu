// Device-compilation & execution parity test for shared portable helpers in kernels/common/arch_ops.cuh
// and kernels/common/warp.cuh. Built with --Werror cross-execution-space-call to catch any host-only call.
#include "common/test_util.h"
#include "core/act.h"
#include "kernels/common/arch_ops.cuh"
#include "kernels/common/warp.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using ingine::test::DeviceArray;
namespace arch = ingine::kernels::arch;
namespace warp = ingine::kernels::warp;

std::uint32_t fbits(float v) noexcept {
  std::uint32_t b = 0;
  std::memcpy(&b, &v, sizeof(b));
  return b;
}

struct Q4ProbeCase {
  std::uint8_t nibble;
  std::uint16_t scale_bits;
  std::uint32_t expect_bits;
};

__global__ void test_arch_ops_kernel(const std::uint32_t* in_a, const std::uint32_t* in_b,
                                     const std::uint32_t* in_c, const Q4ProbeCase* q4_cases, int n_q4,
                                     std::uint32_t* out_alu, float* out_q4) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < 4) {
    const std::uint32_t a = in_a[tid];
    const std::uint32_t b = in_b[tid];
    const std::uint32_t c = in_c[tid];
    out_alu[tid * 8 + 0] = arch::lop3_lut<0x6au>(a, b, c);
    out_alu[tid * 8 + 1] = arch::lop3_lut<0xcau>(a, b, c);
    out_alu[tid * 8 + 2] = arch::lop3_lut<0x96u>(a, b, c);
    out_alu[tid * 8 + 3] = arch::exl3_mcg_mask_xor(a);
    out_alu[tid * 8 + 4] = arch::funnelshift_l(a, b, tid * 11 + 5);
    out_alu[tid * 8 + 5] = static_cast<std::uint32_t>(arch::dp4a_s8(static_cast<std::int32_t>(a),
                                                                    static_cast<std::int32_t>(b),
                                                                    static_cast<std::int32_t>(c)));
    out_alu[tid * 8 + 6] = arch::byte_perm(a, b, 0x7052u);
    out_alu[tid * 8 + 7] = arch::vsub4_u8(a, b);
  }
  for (int i = tid; i < n_q4; i += static_cast<int>(blockDim.x)) {
    out_q4[i] = arch::decode_q4_0_nibble_canonical(q4_cases[i].nibble, q4_cases[i].scale_bits);
  }
}

__global__ void test_warp_kernel(const float* in_vals, float* out_reduce_sum, float* out_all_sum,
                                 float* out_all_max, std::uint32_t* out_ballots,
                                 std::uint32_t* out_half_ballots) {
  const int lane = warp::lane_id_32();
  const float x = in_vals[lane];
  warp::sync_32();
  const float rsum = warp::reduce_sum_32(x);
  const float asum = warp::allreduce_sum_32(x);
  const float amax = warp::allreduce_max_32(x);
  const std::uint32_t b32 = warp::ballot_32((lane % 3) == 0);
  out_reduce_sum[lane] = rsum;
  out_all_sum[lane] = asum;
  out_all_max[lane] = amax;
  out_ballots[lane] = b32;
  out_half_ballots[lane] = warp::logical_32_half_ballot(0x12345678abcdef01ull, (lane & 1) ? 35 : 3);
}

}  // namespace

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;

  const std::vector<std::uint32_t> h_a = {0x12345678u, 0x80808080u, 0xa1b2c3d4u, 0x7f7f7f7fu};
  const std::vector<std::uint32_t> h_b = {0x9abcdef0u, 0x80808080u, 0x0f1e2d3cu, 0x01ff02feu};
  const std::vector<std::uint32_t> h_c = {0x00ff00ffu, 0x00000000u, 0xfffffff6u, 0x7fffffffu};

  std::vector<Q4ProbeCase> q4_cases = {
      {8, 0xbc00u, 0x00000000u},   // -1.0 * (8 - 8) -> +0.0f
      {0, 0xbc00u, 0x41000000u},   // -1.0 * (0 - 8) -> +8.0f
      {15, 0xbc00u, 0xc0e00000u},  // -1.0 * (15 - 8) -> -7.0f
      {8, 0x3c00u, 0x00000000u},   // +1.0 * (8 - 8) -> +0.0f
      {0, 0x8000u, 0x00000000u},   // -0.0 * (0 - 8) -> +0.0f
      {5, 0x8000u, 0x00000000u},   // -0.0 * (5 - 8) -> +0.0f
      {7, 0x0000u, 0x00000000u},   // +0.0 * (7 - 8) -> +0.0f
      {8, 0x8001u, 0x00000000u},   // subnormal scale, nibble 8 -> +0.0f
      {3, 0x7c00u, 0x00000000u},   // +Inf scale -> 0.0f
      {3, 0xfc00u, 0x00000000u},   // -Inf scale -> 0.0f
      {3, 0x7e00u, 0x00000000u},   // NaN scale -> 0.0f
  };
  for (std::uint8_t nib = 0; nib < 16; ++nib) {
    const float ref = arch::decode_q4_0_nibble_canonical(nib, 0x4200u);  // +3.0f
    q4_cases.push_back(Q4ProbeCase{nib, 0x4200u, fbits(ref)});
  }

  DeviceArray<std::uint32_t> d_a(h_a), d_b(h_b), d_c(h_c);
  DeviceArray<Q4ProbeCase> d_q4(q4_cases);
  DeviceArray<std::uint32_t> d_out_alu(4 * 8);
  DeviceArray<float> d_out_q4(q4_cases.size());

  test_arch_ops_kernel<<<1, 64>>>(d_a.get(), d_b.get(), d_c.get(), d_q4.get(),
                                  static_cast<int>(q4_cases.size()), d_out_alu.get(), d_out_q4.get());
  INGINE_TEST_CUDA(cudaGetLastError());
  INGINE_TEST_CUDA(cudaDeviceSynchronize());

  const std::vector<std::uint32_t> got_alu = d_out_alu.download();
  const std::vector<float> got_q4 = d_out_q4.download();
  for (int i = 0; i < 4; ++i) {
    const std::uint32_t a = h_a[i], b = h_b[i], c = h_c[i];
    INGINE_CHECK(got_alu[i * 8 + 0] == arch::lop3_lut<0x6au>(a, b, c));
    INGINE_CHECK(got_alu[i * 8 + 1] == arch::lop3_lut<0xcau>(a, b, c));
    INGINE_CHECK(got_alu[i * 8 + 2] == arch::lop3_lut<0x96u>(a, b, c));
    INGINE_CHECK(got_alu[i * 8 + 3] == arch::exl3_mcg_mask_xor(a));
    INGINE_CHECK(got_alu[i * 8 + 4] == arch::funnelshift_l(a, b, i * 11 + 5));
    INGINE_CHECK(got_alu[i * 8 + 5] == static_cast<std::uint32_t>(arch::dp4a_s8(
                                           static_cast<std::int32_t>(a), static_cast<std::int32_t>(b),
                                           static_cast<std::int32_t>(c))));
    INGINE_CHECK(got_alu[i * 8 + 6] == arch::byte_perm(a, b, 0x7052u));
    INGINE_CHECK(got_alu[i * 8 + 7] == arch::vsub4_u8(a, b));
  }
  for (std::size_t i = 0; i < q4_cases.size(); ++i) {
    const float host_val = arch::decode_q4_0_nibble_canonical(q4_cases[i].nibble, q4_cases[i].scale_bits);
    INGINE_CHECK(fbits(got_q4[i]) == q4_cases[i].expect_bits);
    INGINE_CHECK(fbits(got_q4[i]) == fbits(host_val));
  }

  std::vector<float> h_vals(32, 0.0f);
  float expect_sum = 0.0f;
  float expect_max = -1e9f;
  std::uint32_t expect_ballot = 0u;
  for (int i = 0; i < 32; ++i) {
    h_vals[i] = static_cast<float>(i + 1) * 0.25f;
    expect_sum += h_vals[i];
    if (h_vals[i] > expect_max) expect_max = h_vals[i];
    if ((i % 3) == 0) expect_ballot |= (1u << i);
  }
  DeviceArray<float> d_vals(h_vals), d_rsum(32), d_asum(32), d_amax(32);
  DeviceArray<std::uint32_t> d_bal(32), d_hbal(32);
  test_warp_kernel<<<1, 32>>>(d_vals.get(), d_rsum.get(), d_asum.get(), d_amax.get(), d_bal.get(), d_hbal.get());
  INGINE_TEST_CUDA(cudaGetLastError());
  INGINE_TEST_CUDA(cudaDeviceSynchronize());

  const std::vector<float> got_rsum = d_rsum.download();
  const std::vector<float> got_asum = d_asum.download();
  const std::vector<float> got_amax = d_amax.download();
  const std::vector<std::uint32_t> got_bal = d_bal.download();
  const std::vector<std::uint32_t> got_hbal = d_hbal.download();
  INGINE_CHECK(got_rsum[0] == expect_sum);
  for (int i = 0; i < 32; ++i) {
    INGINE_CHECK(got_asum[i] == expect_sum);
    INGINE_CHECK(got_amax[i] == expect_max);
    INGINE_CHECK(got_bal[i] == expect_ballot);
    INGINE_CHECK(got_hbal[i] == ((i & 1) ? 0x12345678u : 0xabcdef01u));
  }

  std::printf("PASS core_arch_ops_cuda (device arch_ops + decode_q4_0_nibble_canonical + warp primitives)\n");
  return 0;
}
