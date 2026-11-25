// Pure Metal Shading Language (MSL) compute kernels for atlas-ingine Q4_0 SoA embedding + RMSNorm.
// Canonical rule matches CUDA and CPU oracles bit-for-bit: negative scale * nibble=8 -> +0.0f.
#include <metal_stdlib>
using namespace metal;

constant constexpr int kInklingHidden = 2304;
constant constexpr float kInklingEps = 1e-6f;
constant constexpr float kFp16Max = 65504.0f;

struct EmbedParams {
  int tokens;
  int vocab_size;
  int rounding_mode;  // 0=None, 1=Bf16Rn, 2=Fp16RnSat
  int has_oob_flag;
};

struct DecodeParams {
  int hidden_size;
};

inline float clamp_fp16_finite(float x) {
  return isfinite(x) ? clamp(x, -kFp16Max, kFp16Max) : x;
}

inline float round_act(float x, int rounding_mode) {
  if (rounding_mode == 1) {
    uint w = as_type<uint>(x);
    if ((w & 0x7fffffffu) > 0x7f800000u) {
      return as_type<float>((w & 0xffff0000u) | 0x00400000u);
    }
    uint r = ((w + 0x7fffu + ((w >> 16) & 1u)) >> 16) << 16;
    return as_type<float>(r);
  }
  if (rounding_mode == 2) {
    return float(half(clamp_fp16_finite(x)));
  }
  return x;
}

inline float4 decode_q4_0_x4(ushort b2, ushort scale_bits) {
  float d_raw = float(as_type<half>(scale_bits));
  float d = isfinite(d_raw) ? d_raw : 0.0f;
  float m8d = -8.0f * d;
  return float4(fma(float(b2 & 0xfu), d, m8d), fma(float((b2 >> 4) & 0xfu), d, m8d),
                fma(float((b2 >> 8) & 0xfu), d, m8d), fma(float((b2 >> 12) & 0xfu), d, m8d));
}

kernel void decode_q4_0_msl(device const uchar* table [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant DecodeParams& params [[buffer(2)]],
                            uint row_idx [[threadgroup_position_in_grid]],
                            uint t [[thread_position_in_threadgroup]],
                            uint tg_size [[threads_per_threadgroup]]) {
  const int H = params.hidden_size;
  const int nv = H / 4;
  const ulong nib_bytes = ulong(H) / 2ul;
  const ulong row_bytes = nib_bytes + ulong(H / 32) * 2ul;
  device const uchar* row = table + ulong(row_idx) * row_bytes;
  device const uchar* scales = row + nib_bytes;
  device float* out_row = out + ulong(row_idx) * ulong(H);
  for (int idx = int(t); idx < nv; idx += int(tg_size)) {
    const ushort b2 = ushort(row[2 * idx]) | (ushort(row[2 * idx + 1]) << 8);
    const ushort sb = ushort(scales[2 * (idx >> 3)]) | (ushort(scales[2 * (idx >> 3) + 1]) << 8);
    const float4 v = decode_q4_0_x4(b2, sb);
    out_row[4 * idx + 0] = v.x;
    out_row[4 * idx + 1] = v.y;
    out_row[4 * idx + 2] = v.z;
    out_row[4 * idx + 3] = v.w;
  }
}

kernel void embed_q4_rmsnorm_msl(device const long* token_ids [[buffer(0)]],
                                 device const uchar* table [[buffer(1)]],
                                 device const float* norm_w [[buffer(2)]],
                                 device float* out [[buffer(3)]],
                                 device atomic_uint* oob_flag [[buffer(4)]],
                                 constant EmbedParams& params [[buffer(5)]],
                                 uint m [[threadgroup_position_in_grid]],
                                 uint t [[thread_position_in_threadgroup]],
                                 uint simd_lane [[thread_index_in_simdgroup]],
                                 uint simd_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int BW = 256;
  constexpr int NV = kInklingHidden / 4;
  constexpr int PER = (NV + BW - 1) / BW;
  constexpr ulong NIB_BYTES = kInklingHidden / 2;
  constexpr ulong ROW_BYTES = NIB_BYTES + (kInklingHidden / 32) * 2;
  threadgroup float simd_part[16];
  threadgroup float rinv_s;

  const long tok = token_ids[m];
  if (tok < 0 || tok >= params.vocab_size) {
    if (t == 0 && params.has_oob_flag != 0) {
      atomic_exchange_explicit(oob_flag, 1u, memory_order_relaxed);
    }
    return;
  }

  device const uchar* row = table + ulong(tok) * ROW_BYTES;
  device const uchar* scales = row + NIB_BYTES;
  float4 v[PER];
  float4 g[PER];
  float acc = 0.0f;

  for (int k = 0; k < PER; ++k) {
    const int idx = int(t) + k * BW;
    if (idx < NV) {
      g[k] = reinterpret_cast<device const float4*>(norm_w)[idx];
      const ushort b2 = ushort(row[2 * idx]) | (ushort(row[2 * idx + 1]) << 8);
      const ushort sb = ushort(scales[2 * (idx >> 3)]) | (ushort(scales[2 * (idx >> 3) + 1]) << 8);
      v[k] = decode_q4_0_x4(b2, sb);
      acc += dot(v[k], v[k]);
    }
  }

  const float warp_sum = simd_sum(acc);
  if (simd_lane == 0) simd_part[simd_id] = warp_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (t == 0) {
    float total = 0.0f;
    // 256 threads with 32-wide Apple SIMDgroups = 8 partials.
    for (uint i = 0; i < 8; ++i) total += simd_part[i];
    rinv_s = rsqrt(total * (1.0f / float(kInklingHidden)) + kInklingEps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float rinv = rinv_s;
  device float* o = out + ulong(m) * ulong(kInklingHidden);
  for (int k = 0; k < PER; ++k) {
    const int idx = int(t) + k * BW;
    if (idx < NV) {
      const float4 scaled = v[k] * rinv * g[k];
      reinterpret_cast<device float4*>(o)[idx] =
          float4(round_act(scaled.x, params.rounding_mode), round_act(scaled.y, params.rounding_mode),
                 round_act(scaled.z, params.rounding_mode), round_act(scaled.w, params.rounding_mode));
    }
  }
}
