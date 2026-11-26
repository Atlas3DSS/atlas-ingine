// Device-compilation & execution parity test for shared portable helpers in kernels/common/arch_ops.cuh
// and kernels/common/warp.cuh. Built with --Werror cross-execution-space-call to catch any host-only call.
#include "common/test_util.h"
#include "core/act.h"
#include "core/gpu_compat.h"
#include "kernels/common/arch_ops.cuh"
#include "kernels/common/warp.cuh"

#include <cmath>
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

__global__ void test_e4m3_fp16_sweep_kernel(std::uint8_t* out_portable, std::uint8_t* out_ref) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= 65536) return;
  const std::uint16_t hbits = static_cast<std::uint16_t>(idx);
  const float x = __half2float(*reinterpret_cast<const __half*>(&hbits));
  out_portable[idx] = ingine::gpu::float_to_e4m3fn_sat_portable(x);
  out_ref[idx] = ingine::gpu::float_to_e4m3fn_sat(x);
}

__global__ void test_e4m3_f32_edge_kernel(const float* in_x, int n, std::uint8_t* out_portable,
                                          std::uint8_t* out_ref, std::uint16_t* out_pair_port,
                                          std::uint16_t* out_pair_ref) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    out_portable[i] = ingine::gpu::float_to_e4m3fn_sat_portable(in_x[i]);
    out_ref[i] = ingine::gpu::float_to_e4m3fn_sat(in_x[i]);
  }
  if (i * 2 + 1 < n) {
    const float a = in_x[i * 2];
    const float b = in_x[i * 2 + 1];
    out_pair_port[i] = static_cast<std::uint16_t>(ingine::gpu::float_to_e4m3fn_sat_portable(a)) |
                       static_cast<std::uint16_t>(static_cast<std::uint16_t>(ingine::gpu::float_to_e4m3fn_sat_portable(b)) << 8);
    out_pair_ref[i] = ingine::gpu::float2_to_e4m3fnx2_sat(a, b);
  }
}

__global__ void test_e4m3_decode_256_kernel(std::uint16_t* out_f16_port, std::uint16_t* out_f16_ref,
                                            std::uint32_t* out_f32_port, std::uint32_t* out_f32_ref) {
  const int code = static_cast<int>(threadIdx.x);
  if (code >= 256) return;
  const std::uint8_t c = static_cast<std::uint8_t>(code);
  out_f16_port[code] = ingine::gpu::e4m3fn_to_f16_bits_portable(c);
  out_f32_port[code] = __float_as_uint(ingine::gpu::e4m3fn_to_float_portable(c));
  out_f32_ref[code] = __float_as_uint(ingine::gpu::e4m3fn_to_float(c));
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && defined(INGINE_BACKEND_CUDA) && INGINE_BACKEND_CUDA
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(c, __NV_E4M3);
  out_f16_ref[code] = hr.x;
#else
  out_f16_ref[code] = out_f16_port[code];
#endif
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

  // Bit-exact OCP E4M3FN portable vs native hardware conversion check across all 65,536 FP16 values,
  // all exact round-to-even midpoints, saturations, Infs, NaNs, and all 256 E4M3 decode codes.
  {
    DeviceArray<std::uint8_t> d_e4_port(65536), d_e4_ref(65536);
    test_e4m3_fp16_sweep_kernel<<<256, 256>>>(d_e4_port.get(), d_e4_ref.get());
    INGINE_TEST_CUDA(cudaGetLastError());
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto h_port = d_e4_port.download();
    const auto h_ref = d_e4_ref.download();
    for (int i = 0; i < 65536; ++i) {
      INGINE_CHECK(h_port[i] == h_ref[i]);
    }
  }
  {
    std::vector<float> edges;
    auto push_pm = [&](float v) {
      edges.push_back(v);
      edges.push_back(-v);
    };
    push_pm(0.0f);
    // Exact subnormal midpoints and neighbors: k * 2^-9 + delta
    const float sub_step = 1.0f / 512.0f;
    for (int k = 0; k <= 8; ++k) {
      const float base = float(k) * sub_step;
      push_pm(base);
      push_pm(base + 0.5f * sub_step);
      push_pm(base + 0.5f * sub_step - 1e-7f);
      push_pm(base + 0.5f * sub_step + 1e-7f);
    }
    // Exact normal midpoints across all exponents [-6..8] and mantissas [0..7]
    for (int e = -6; e <= 8; ++e) {
      const float ulp = std::ldexp(1.0f, e - 3);
      for (int m = 0; m < 8; ++m) {
        if (e == 8 && m >= 6) continue;
        const float val = std::ldexp(float(8 + m), e - 3);
        push_pm(val);
        push_pm(val + 0.5f * ulp);
        push_pm(val + 0.5f * ulp - ulp * 1e-4f);
        push_pm(val + 0.5f * ulp + ulp * 1e-4f);
      }
    }
    // Saturation, overflow, Inf, and quiet/signaling NaN bit patterns
    push_pm(448.0f);
    push_pm(448.001f);
    push_pm(464.0f);
    push_pm(1e10f);
    auto from_bits = [](std::uint32_t u) {
      float f = 0.0f;
      std::memcpy(&f, &u, sizeof(f));
      return f;
    };
    edges.push_back(from_bits(0x7f800000u));  // +Inf
    edges.push_back(from_bits(0xff800000u));  // -Inf
    edges.push_back(from_bits(0x7fc00000u));  // +qNaN
    edges.push_back(from_bits(0xffc00000u));  // -qNaN
    edges.push_back(from_bits(0x7f800001u));  // +sNaN
    edges.push_back(from_bits(0xff800001u));  // -sNaN
    if ((edges.size() & 1u) != 0u) edges.push_back(1.0f);

    const int n = static_cast<int>(edges.size());
    DeviceArray<float> d_in(edges);
    DeviceArray<std::uint8_t> d_port(n), d_ref(n);
    DeviceArray<std::uint16_t> d_pp(n / 2), d_pr(n / 2);
    test_e4m3_f32_edge_kernel<<<(n + 255) / 256, 256>>>(d_in.get(), n, d_port.get(), d_ref.get(), d_pp.get(),
                                                        d_pr.get());
    INGINE_TEST_CUDA(cudaGetLastError());
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto h_port = d_port.download();
    const auto h_ref = d_ref.download();
    const auto h_pp = d_pp.download();
    const auto h_pr = d_pr.download();
    for (int i = 0; i < n; ++i) {
      if (h_port[i] != h_ref[i]) {
        std::fprintf(stderr, "E4M3 mismatch idx=%d x=%.9g bits=0x%08x port=0x%02x ref=0x%02x\n", i, edges[i],
                     fbits(edges[i]), unsigned(h_port[i]), unsigned(h_ref[i]));
      }
      INGINE_CHECK(h_port[i] == h_ref[i]);
      INGINE_CHECK(h_port[i] == ingine::gpu::float_to_e4m3fn_sat_portable(edges[i]));
    }
    for (int i = 0; i < n / 2; ++i) {
      INGINE_CHECK(h_pp[i] == h_pr[i]);
    }
  }
  {
    DeviceArray<std::uint16_t> d_f16_p(256), d_f16_r(256);
    DeviceArray<std::uint32_t> d_f32_p(256), d_f32_r(256);
    test_e4m3_decode_256_kernel<<<1, 256>>>(d_f16_p.get(), d_f16_r.get(), d_f32_p.get(), d_f32_r.get());
    INGINE_TEST_CUDA(cudaGetLastError());
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto h_f16_p = d_f16_p.download();
    const auto h_f16_r = d_f16_r.download();
    const auto h_f32_p = d_f32_p.download();
    const auto h_f32_r = d_f32_r.download();
    for (int c = 0; c < 256; ++c) {
      if (h_f16_p[c] != h_f16_r[c]) {
        std::fprintf(stderr, "E4M3 decode mismatch code=0x%02x port=0x%04x ref=0x%04x\n", c,
                     unsigned(h_f16_p[c]), unsigned(h_f16_r[c]));
      }
      INGINE_CHECK(h_f16_p[c] == h_f16_r[c]);
      INGINE_CHECK(h_f16_p[c] == ingine::gpu::e4m3fn_to_f16_bits_portable(static_cast<std::uint8_t>(c)));
      const bool is_nan = ((c & 0x7f) == 0x7f);
      if (is_nan) {
        INGINE_CHECK((h_f32_p[c] & 0x7f800000u) == 0x7f800000u && (h_f32_p[c] & 0x007fffffu) != 0u);
        INGINE_CHECK((h_f32_r[c] & 0x7f800000u) == 0x7f800000u && (h_f32_r[c] & 0x007fffffu) != 0u);
      } else {
        INGINE_CHECK(h_f32_p[c] == h_f32_r[c]);
      }
    }
  }

  std::printf("PASS core_arch_ops_cuda (device arch_ops + Q4 + warp + bit-exact E4M3FN sweep)\n");
  return 0;
}
