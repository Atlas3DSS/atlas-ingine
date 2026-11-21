// mix: ablated residual update + attention residual over block states + weighted RMS norm.
//
// Design. Latency-bound (a few 9 KB vectors per token), so one CTA per token with 576 threads, each owning
// 4 consecutive channels (one float4 of every state, query and output: coalesced 16-byte accesses, states
// live in registers for the whole kernel). At most three block reductions, each a warp stage + one shared
// round: (1) r . write for the ablation, (2) for every state i its sum of squares and the raw dots
// q_j . s_i (q_j . n_i = rinv_i * (q_j . s_i), so normalisation, logits and softmax need a single
// reduction; its up to 30 values use the transposing reduce_scatter, 31 shuffles instead of 150), (3) the
// sum of squares of each mixed output for the final RMS norm. The state count (blocks + 1) and query
// count are template parameters, so no work is spent on absent states. Block pointers are read from
// device memory, so the graph can be replayed with other blocks; counts (blocks, queries, which optional
// tensors exist) are fixed per call site. Queries, ablation direction/weight and norm weights are loaded
// before griddepcontrol.wait (pdl=True), the pointer table and every state after it. `embed` (also here)
// reproduces torch's reduction order so that it is bit-exact with model.Model.embed.
// Ported from engine/csrc/dense/mix.cu (kernel bodies unchanged; torch launchers -> raw pointers).
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"

namespace ingine::kernels::dense {
namespace {

constexpr int MIX_T = H / 4;  // 576
constexpr int MIX_W = MIX_T / 32;
constexpr int MAXB = 5, MAXQ = 4;

struct MixArgs {
  const float* const* blocks;  // NS - 1 device pointers, each [M, H]
  const float* base;   // [M, H] or null
  const float* write;  // [M, H] or null
  const float* abl_r;  // [H] or null (no ablation)
  const float* abl_w;  // [1]
  float* partial;      // [M, H] or null
  const float* queries;  // [Q, H]
  const float* norm_w;   // [Q, H] (needed with out_bf16)
  __nv_bfloat16* out_bf16;  // [Q, M, H] or null
  float* out_f32;           // [Q, M, H] or null
  int M;
};

__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ __forceinline__ float4 scale4(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }

__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* p, float4 v) {
  __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
  uint2 u;
  u.x = *reinterpret_cast<uint32_t*>(&a);
  u.y = *reinterpret_cast<uint32_t*>(&b);
  *reinterpret_cast<uint2*>(p) = u;
}

// NS = states in the mix (blocks + new), Q = queries; NS == 1 && Q == 1 is the layer-0 path (no mix).
template <int NS, int Q>
__global__ void __launch_bounds__(MIX_T) mix_kernel(MixArgs a) {
  constexpr bool SKIP = NS == 1 && Q == 1;
  constexpr int NV = NS + Q * NS;  // sums of squares + query dots
  __shared__ float red1[(MIX_W + 1) * block_sum_slots<1>()], red2[(MIX_W + 1) * block_sum_slots<NV>()],
      red3[(MIX_W + 1) * block_sum_slots<Q>()];
  const int m = blockIdx.x, t = threadIdx.x;
  const size_t off = size_t(m) * H + 4 * t;
  auto ld4 = [](const float* p) { return *reinterpret_cast<const float4*>(p); };

  // constants (queries, ablation direction and weight, norm weights) before the PDL wait; states after
  float4 s[NS], q[Q], r4 = make_float4(0.f, 0.f, 0.f, 0.f), g4[Q];
  float aw = 0.f;
  if constexpr (!SKIP) {
#pragma unroll
    for (int j = 0; j < Q; ++j) q[j] = ld4(a.queries + size_t(j) * H + 4 * t);
  }
  if (a.abl_r) {
    r4 = ld4(a.abl_r + 4 * t);
    aw = __ldg(a.abl_w);
  }
  if (a.out_bf16) {
#pragma unroll
    for (int j = 0; j < Q; ++j) g4[j] = ld4(a.norm_w + size_t(j) * H + 4 * t);
  }

  pdl_wait();
  pdl_trigger();
#pragma unroll
  for (int i = 0; i < NS - 1; ++i) s[i] = ld4(a.blocks[i] + off);  // pointer table, then block states

  // new = base + (write - (w * (write . r)) * r)
  float4 nw = a.base ? ld4(a.base + off) : make_float4(0.f, 0.f, 0.f, 0.f);
  if (a.write) {
    float4 y = ld4(a.write + off);
    if (a.abl_r) {
      const float4 r = r4;
      float d[1] = {dot4(y, r)};
      block_sum(d, red1);
      const float c = aw * d[0];
      y = make_float4(y.x - c * r.x, y.y - c * r.y, y.z - c * r.z, y.w - c * r.w);
    }
    if (a.base) nw = make_float4(nw.x + y.x, nw.y + y.y, nw.z + y.z, nw.w + y.w);
    else nw = y;
  }
  if (a.partial) *reinterpret_cast<float4*>(a.partial + off) = nw;

  if constexpr (SKIP) {
    float ss[1] = {dot4(nw, nw)};
    block_sum(ss, red3);
    const float rinv = rsqrtf(ss[0] / H + EPS);
    if (a.out_f32) *reinterpret_cast<float4*>(a.out_f32 + off) = nw;
    if (a.out_bf16) {
      const float4 g = g4[0];
      const float4 n = scale4(nw, rinv);
      store_bf16x4(a.out_bf16 + off, make_float4(n.x * g.x, n.y * g.y, n.z * g.z, n.w * g.w));
    }
    return;
  } else {
    s[NS - 1] = nw;
    float v[NV];
#pragma unroll
    for (int i = 0; i < NS; ++i) {
      v[i] = dot4(s[i], s[i]);
#pragma unroll
      for (int j = 0; j < Q; ++j) v[NS + j * NS + i] = dot4(q[j], s[i]);
    }
    block_sum(v, red2);

    float rinv[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) {
      rinv[i] = rsqrtf(v[i] / H + EPS);
      s[i] = scale4(s[i], rinv[i]);  // n_i
    }
    const float inv_sqrt_h = 1.f / 48.f;  // H ** -0.5
    float4 mix[Q];
    float ms[Q];
#pragma unroll
    for (int j = 0; j < Q; ++j) {
      float lg[NS], mx = -INFINITY;
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        lg[i] = v[NS + j * NS + i] * rinv[i] * inv_sqrt_h;
        mx = fmaxf(mx, lg[i]);
      }
      float den = 0.f;
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        lg[i] = expf(lg[i] - mx);
        den += lg[i];
      }
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        const float p = lg[i] / den;
        o = make_float4(o.x + p * s[i].x, o.y + p * s[i].y, o.z + p * s[i].z, o.w + p * s[i].w);
      }
      mix[j] = o;
      ms[j] = dot4(o, o);
    }
    if (a.out_f32) {
#pragma unroll
      for (int j = 0; j < Q; ++j) *reinterpret_cast<float4*>(a.out_f32 + size_t(j) * a.M * H + off) = mix[j];
    }
    if (!a.out_bf16) return;
    block_sum(ms, red3);
#pragma unroll
    for (int j = 0; j < Q; ++j) {
      const float r = rsqrtf(ms[j] / H + EPS);
      const float4 g = g4[j];
      const float4 n = scale4(mix[j], r);
      store_bf16x4(a.out_bf16 + size_t(j) * a.M * H + off, make_float4(n.x * g.x, n.y * g.y, n.z * g.z, n.w * g.w));
    }
  }
}

template <int NS>
void launch_mix_q(int Q, int M, const MixArgs& a, bool pdl, cudaStream_t stream) {
  switch (Q) {
    case 1: launch_ex(mix_kernel<NS, 1>, dim3(M), MIX_T, 0, 1, pdl, stream, a); break;
    case 2: launch_ex(mix_kernel<NS, 2>, dim3(M), MIX_T, 0, 1, pdl, stream, a); break;
    case 3: launch_ex(mix_kernel<NS, 3>, dim3(M), MIX_T, 0, 1, pdl, stream, a); break;
    case 4: launch_ex(mix_kernel<NS, 4>, dim3(M), MIX_T, 0, 1, pdl, stream, a); break;
  }
}

// embed: out[m] = float(bf16(rms_norm(float(table[token[m]])) * norm_w)), one CTA per token.
// Bit-exact with model.Model.embed: the sum of squares follows torch's CUDA mean reduction for a call
// with M rows (ATen Reduce.cuh: input vectorised by 4, block width BW = 512 / last_pow2(M) threads,
// thread t accumulates vectors t, t + BW, ... in 4 per-lane partials combined in order, then a shared
// tree for offsets BW/2 .. 32 and a shuffle-down tree 16 .. 1), mean = sum * fl(1/2304), rsqrtf as
// torch.rsqrt, and every product rounded separately (no FMA contraction) as torch's separate kernels.
template <int BW>
__global__ void __launch_bounds__(BW) embed_kernel(const int64_t* __restrict__ token,
                                                  const __nv_bfloat16* __restrict__ table,
                                                  const float* __restrict__ norm_w, float* __restrict__ out) {
  constexpr int NV = H / 4;                   // 576 four-element vectors
  constexpr int PER = (NV + BW - 1) / BW;     // vectors per thread (some threads one fewer)
  __shared__ float part[BW];
  __shared__ float rinv_s;
  const int m = blockIdx.x, t = threadIdx.x;
  float4 g[PER];
#pragma unroll
  for (int k = 0; k < PER; ++k)
    if (t + k * BW < NV) g[k] = *reinterpret_cast<const float4*>(norm_w + 4 * (t + k * BW));
  pdl_wait();  // the token id comes from an earlier kernel
  pdl_trigger();
  const __nv_bfloat16* row = table + size_t(token[m]) * H;
  float4 v[PER];
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    if (t + k * BW < NV) {
      const uint2 u = *reinterpret_cast<const uint2*>(row + 4 * (t + k * BW));
      v[k] = make_float4(bf_lo(u.x), bf_hi(u.x), bf_lo(u.y), bf_hi(u.y));
      acc[0] = __fadd_rn(acc[0], __fmul_rn(v[k].x, v[k].x));
      acc[1] = __fadd_rn(acc[1], __fmul_rn(v[k].y, v[k].y));
      acc[2] = __fadd_rn(acc[2], __fmul_rn(v[k].z, v[k].z));
      acc[3] = __fadd_rn(acc[3], __fmul_rn(v[k].w, v[k].w));
    }
  }
  float s = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
  if constexpr (BW > 32) {
    part[t] = s;
#pragma unroll
    for (int o = BW / 2; o >= 32; o >>= 1) {
      __syncthreads();
      if (t < o) {
        s = __fadd_rn(s, part[t + o]);
        part[t] = s;
      }
    }
    __syncthreads();
  }
  if (t < 32) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, o));
    if (t == 0) rinv_s = rsqrtf(__fadd_rn(__fmul_rn(s, 1.f / H), EPS));
  }
  __syncthreads();
  const float rinv = rinv_s;
  float* o = out + size_t(m) * H;
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    if (t + k * BW < NV) {
      const float4 x = v[k], w = g[k];
      *reinterpret_cast<float4*>(o + 4 * (t + k * BW)) =
          make_float4(round_bf16(__fmul_rn(__fmul_rn(x.x, rinv), w.x)), round_bf16(__fmul_rn(__fmul_rn(x.y, rinv), w.y)),
                      round_bf16(__fmul_rn(__fmul_rn(x.z, rinv), w.z)), round_bf16(__fmul_rn(__fmul_rn(x.w, rinv), w.w)));
    }
  }
}

}  // namespace

void mix(const MixParams& p, bool pdl, cudaStream_t stream) {
  require(p.base || p.write, "mix needs base and/or write");
  const int M = p.M;
  require(M >= 1, "mix: M must be >= 1");  // one CTA per row: decode uses 1..8, prefill any chunk length
  const int nb = p.nblocks;
  require(nb >= 0 && nb <= MAXB, "mix: at most 5 blocks");
  require(nb == 0 || p.blocks, "mix: blocks table required");
  const int Q = p.Q;
  require(Q >= 1 && Q <= MAXQ, "mix: 1..4 queries");
  MixArgs a;
  a.blocks = nb > 0 ? p.blocks : nullptr;
  a.base = p.base;
  a.write = p.write;
  a.abl_r = p.abl_r;
  a.abl_w = p.abl_w;
  require(!a.abl_r || a.abl_w, "mix: abl_w is required with abl_r");
  a.partial = p.partial_out;
  const bool skip = nb == 0 && Q == 1;
  a.queries = p.queries;
  require(skip || a.queries, "mix: queries are required unless the mix is skipped");
  a.out_f32 = p.out_f32;
  a.out_bf16 = p.out_bf16;
  a.norm_w = p.out_bf16 ? p.norm_w : nullptr;
  require(!a.out_bf16 || a.norm_w, "mix: norm_w is required with out_bf16");
  require(a.out_f32 || a.out_bf16 || a.partial, "mix has no output");
  a.M = M;
  switch (nb + 1) {
    case 1: launch_mix_q<1>(Q, M, a, pdl, stream); break;
    case 2: launch_mix_q<2>(Q, M, a, pdl, stream); break;
    case 3: launch_mix_q<3>(Q, M, a, pdl, stream); break;
    case 4: launch_mix_q<4>(Q, M, a, pdl, stream); break;
    case 5: launch_mix_q<5>(Q, M, a, pdl, stream); break;
    case 6: launch_mix_q<6>(Q, M, a, pdl, stream); break;
  }
}

void embed(const std::int64_t* token, int M, const __nv_bfloat16* table, const float* norm_w, float* out,
           bool pdl, cudaStream_t stream) {
  require(M >= 1, "embed: M must be >= 1");  // one CTA per row (prefill calls it with M = chunk length)
  require(token && table && norm_w && out, "embed: null argument");
  // torch's reduction block width for an [M, 2304] fp32 mean: 512 / last_pow2(M)
  if (M == 1) launch_ex(embed_kernel<512>, dim3(M), 512, 0, 1, pdl, stream, token, table, norm_w, out);
  else if (M < 4) launch_ex(embed_kernel<256>, dim3(M), 256, 0, 1, pdl, stream, token, table, norm_w, out);
  else if (M < 8) launch_ex(embed_kernel<128>, dim3(M), 128, 0, 1, pdl, stream, token, table, norm_w, out);
  else launch_ex(embed_kernel<64>, dim3(M), 64, 0, 1, pdl, stream, token, table, norm_w, out);
}

}  // namespace ingine::kernels::dense
