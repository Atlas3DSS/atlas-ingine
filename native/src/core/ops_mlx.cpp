// Pure native C++20 + Objective-C++ Apple Metal compute dispatch for Q4_0 SoA embedding + RMSNorm.
// Uses embedded pure Metal Shading Language source matching native/src/kernels/metal/embed_q4.metal.
#include "core/hal.h"
#include "core/ops.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine {
id<MTLBuffer> lookup_metal_buffer_for_ptr(const void* ptr, std::size_t& out_offset);
}

namespace ingine::ops {
namespace {

constexpr const char* kEmbedQ4MslSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct EmbedParams { int tokens; int vocab_size; int rounding_mode; int has_oob_flag; };
struct DecodeParams { int hidden_size; };
inline float round_act(float x, int rounding_mode) {
  if (rounding_mode == 1) {
    uint w = as_type<uint>(x);
    if ((w & 0x7fffffffu) > 0x7f800000u) return as_type<float>((w & 0xffff0000u) | 0x00400000u);
    return as_type<float>(((w + 0x7fffu + ((w >> 16) & 1u)) >> 16) << 16);
  }
  if (rounding_mode == 2) return float(half(isfinite(x) ? clamp(x, -65504.0f, 65504.0f) : x));
  return x;
}
inline float4 decode_q4_0_x4(ushort b2, ushort sb) {
  float d_raw = float(as_type<half>(sb));
  float d = isfinite(d_raw) ? d_raw : 0.0f;
  float m8d = -8.0f * d;
  return float4(fma(float(b2 & 0xfu), d, m8d), fma(float((b2 >> 4) & 0xfu), d, m8d),
                fma(float((b2 >> 8) & 0xfu), d, m8d), fma(float((b2 >> 12) & 0xfu), d, m8d));
}
kernel void decode_q4_0_msl(device const uchar* table [[buffer(0)]], device float* out [[buffer(1)]],
                            constant DecodeParams& params [[buffer(2)]],
                            uint row_idx [[threadgroup_position_in_grid]],
                            uint t [[thread_position_in_threadgroup]],
                            uint tg_size [[threads_per_threadgroup]]) {
  int H = params.hidden_size; int nv = H / 4;
  ulong nib_bytes = ulong(H) / 2ul; ulong row_bytes = nib_bytes + ulong(H / 32) * 2ul;
  device const uchar* row = table + ulong(row_idx) * row_bytes;
  device const uchar* scales = row + nib_bytes;
  device float* out_row = out + ulong(row_idx) * ulong(H);
  for (int idx = int(t); idx < nv; idx += int(tg_size)) {
    ushort b2 = ushort(row[2 * idx]) | (ushort(row[2 * idx + 1]) << 8);
    ushort sb = ushort(scales[2 * (idx >> 3)]) | (ushort(scales[2 * (idx >> 3) + 1]) << 8);
    float4 v = decode_q4_0_x4(b2, sb);
    out_row[4 * idx + 0] = v.x; out_row[4 * idx + 1] = v.y; out_row[4 * idx + 2] = v.z; out_row[4 * idx + 3] = v.w;
  }
}
kernel void embed_q4_rmsnorm_msl(device const long* token_ids [[buffer(0)]], device const uchar* table [[buffer(1)]],
                                 device const float* norm_w [[buffer(2)]], device float* out [[buffer(3)]],
                                 device atomic_uint* oob_flag [[buffer(4)]], constant EmbedParams& params [[buffer(5)]],
                                 uint m [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                                 uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int H = 2304; constexpr int BW = 256; constexpr int NV = H / 4; constexpr int PER = (NV + BW - 1) / BW;
  threadgroup float simd_part[16]; threadgroup float rinv_s;
  long tok = token_ids[m];
  if (tok < 0 || tok >= params.vocab_size) {
    if (t == 0 && params.has_oob_flag != 0) atomic_exchange_explicit(oob_flag, 1u, memory_order_relaxed);
    return;
  }
  device const uchar* row = table + ulong(tok) * 1296ul; device const uchar* scales = row + 1152ul;
  float4 v[PER]; float4 g[PER]; float acc = 0.0f;
  for (int k = 0; k < PER; ++k) {
    int idx = int(t) + k * BW;
    if (idx < NV) {
      g[k] = reinterpret_cast<device const float4*>(norm_w)[idx];
      ushort b2 = ushort(row[2 * idx]) | (ushort(row[2 * idx + 1]) << 8);
      ushort sb = ushort(scales[2 * (idx >> 3)]) | (ushort(scales[2 * (idx >> 3) + 1]) << 8);
      v[k] = decode_q4_0_x4(b2, sb);
      acc += dot(v[k], v[k]);
    }
  }
  float ws = simd_sum(acc); if (simd_lane == 0) simd_part[simd_id] = ws;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float tot = 0.0f; for (uint i = 0; i < 8; ++i) tot += simd_part[i];
    rinv_s = rsqrt(tot * (1.0f / float(H)) + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rinv = rinv_s; device float* o = out + ulong(m) * ulong(H);
  for (int k = 0; k < PER; ++k) {
    int idx = int(t) + k * BW;
    if (idx < NV) {
      float4 s = v[k] * rinv * g[k];
      reinterpret_cast<device float4*>(o)[idx] = float4(round_act(s.x, params.rounding_mode), round_act(s.y, params.rounding_mode),
                                                       round_act(s.z, params.rounding_mode), round_act(s.w, params.rounding_mode));
    }
  }
}
)MSL";

struct MtlPipelines {
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> decode_pso = nil;
  id<MTLComputePipelineState> embed_pso = nil;
};

MtlPipelines& shared_pipelines() {
  static std::once_flag once;
  static MtlPipelines p;
  std::call_once(once, [] {
    @autoreleasepool {
      p.device = MTLCreateSystemDefaultDevice();
      if (!p.device) throw std::runtime_error("ops(MLX/Metal): no Metal device");
      NSError* err = nil;
      NSString* src = [NSString stringWithUTF8String:kEmbedQ4MslSource];
      id<MTLLibrary> lib = [p.device newLibraryWithSource:src options:nil error:&err];
      if (!lib) {
        throw std::runtime_error(std::string("ops(MLX/Metal): Metal shader compile failed: ") +
                                 (err ? err.localizedDescription.UTF8String : "unknown"));
      }
      id<MTLFunction> fn_dec = [lib newFunctionWithName:@"decode_q4_0_msl"];
      id<MTLFunction> fn_emb = [lib newFunctionWithName:@"embed_q4_rmsnorm_msl"];
      p.decode_pso = [p.device newComputePipelineStateWithFunction:fn_dec error:&err];
      if (!p.decode_pso) throw std::runtime_error("ops(MLX/Metal): decode PSO creation failed");
      p.embed_pso = [p.device newComputePipelineStateWithFunction:fn_emb error:&err];
      if (!p.embed_pso) throw std::runtime_error("ops(MLX/Metal): embed PSO creation failed");
    }
  });
  return p;
}

id<MTLCommandQueue> extract_queue(hal::StreamHandle stream) {
  struct MtlStreamState {
    id<MTLCommandQueue> queue;
  };
  auto* st = static_cast<MtlStreamState*>(stream.native);
  if (st && st->queue) return st->queue;
  static id<MTLCommandQueue> default_q = [shared_pipelines().device newCommandQueue];
  return default_q;
}

}  // namespace

OpResolution resolve_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, bool allow_fallback) {
  validate_embed_q4_rmsnorm_desc(desc);
  const bool native_ok = desc.hidden_size == 2304 && desc.out_row_stride == 2304 &&
                         std::fabs(desc.eps - 1e-6f) == 0.f;
  if (native_ok) {
    return OpResolution{true, hal::BackendKind::Mlx, DispatchPath::Native, "mlx native Metal Q4_0 embed + RMSNorm"};
  }
  if (allow_fallback) {
    return OpResolution{true, hal::BackendKind::Mlx, DispatchPath::PortableFallback,
                        "mlx fallback for non-specialized shape, stride, or eps"};
  }
  return OpResolution{false, hal::BackendKind::Mlx, DispatchPath::PortableFallback,
                      "mlx native embed_q4 requires H=2304, out_row_stride=2304, and eps=1e-6"};
}

void decode_q4_0(const void* table_q4, int rows, int hidden_size, float* out_f32, hal::StreamHandle stream) {
  if (!table_q4 || !out_f32) throw std::invalid_argument("decode_q4_0(MLX): null pointer");
  if (rows < 1) throw std::invalid_argument("decode_q4_0(MLX): rows must be >= 1");
  if (hidden_size <= 0 || (hidden_size % 32) != 0) {
    throw std::invalid_argument("decode_q4_0(MLX): hidden_size must be a positive multiple of 32");
  }
  std::size_t tbl_off = 0, out_off = 0;
  id<MTLBuffer> b_tbl = lookup_metal_buffer_for_ptr(table_q4, tbl_off);
  id<MTLBuffer> b_out = lookup_metal_buffer_for_ptr(out_f32, out_off);
  if (!b_tbl || !b_out) {
    throw std::invalid_argument("decode_q4_0(MLX): buffers must be HAL-allocated Metal shared buffers");
  }
  auto& p = shared_pipelines();
  struct DecodeParams {
    int hidden_size;
  } params{hidden_size};
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [extract_queue(stream) commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p.decode_pso];
    [enc setBuffer:b_tbl offset:tbl_off atIndex:0];
    [enc setBuffer:b_out offset:out_off atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
  }
}

DispatchPath run_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                                  hal::StreamHandle stream, bool allow_fallback, bool force_fallback) {
  const OpResolution res = resolve_embed_q4_rmsnorm(desc, allow_fallback);
  if (!res.supported) throw std::invalid_argument(res.reason);
  if (!args.token_ids || !args.table_q4 || !args.norm_weight || !args.out) {
    throw std::invalid_argument("embed_q4_rmsnorm(MLX): null buffer argument");
  }
  std::size_t tok_off = 0, tbl_off = 0, norm_off = 0, out_off = 0, oob_off = 0;
  id<MTLBuffer> b_tok = lookup_metal_buffer_for_ptr(args.token_ids, tok_off);
  id<MTLBuffer> b_tbl = lookup_metal_buffer_for_ptr(args.table_q4, tbl_off);
  id<MTLBuffer> b_norm = lookup_metal_buffer_for_ptr(args.norm_weight, norm_off);
  id<MTLBuffer> b_out = lookup_metal_buffer_for_ptr(args.out, out_off);
  id<MTLBuffer> b_oob = args.invalid_token_flag ? lookup_metal_buffer_for_ptr(args.invalid_token_flag, oob_off) : nil;

  if (!force_fallback && res.path == DispatchPath::Native && b_tok && b_tbl && b_norm && b_out) {
    if (desc.token_bounds == TokenBoundsMode::Validate) {
      hal::stream_synchronize(stream);
      try {
        validate_token_ids(args.token_ids, desc.tokens, desc.vocab_size);
        if (args.invalid_token_flag != nullptr) *args.invalid_token_flag = 0u;
      } catch (const std::out_of_range&) {
        if (args.invalid_token_flag != nullptr) *args.invalid_token_flag = 1u;
        throw;
      }
    } else if (desc.token_bounds == TokenBoundsMode::DeferredFlag) {
      if (args.invalid_token_flag == nullptr) {
        throw std::invalid_argument("embed_q4_rmsnorm(MLX): TokenBoundsMode::DeferredFlag requires invalid_token_flag");
      }
    }

    auto& p = shared_pipelines();
    struct EmbedParams {
      int tokens;
      int vocab_size;
      int rounding_mode;
      int has_oob_flag;
    } params{desc.tokens, desc.vocab_size, static_cast<int>(desc.output_rounding), (b_oob != nil) ? 1 : 0};

    @autoreleasepool {
      id<MTLCommandBuffer> cb = [extract_queue(stream) commandBuffer];
      if (args.invalid_token_flag != nullptr && b_oob) {
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:b_oob range:NSMakeRange(oob_off, sizeof(std::uint32_t)) value:0];
        [blit endEncoding];
      }
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:p.embed_pso];
      [enc setBuffer:b_tok offset:tok_off atIndex:0];
      [enc setBuffer:b_tbl offset:tbl_off atIndex:1];
      [enc setBuffer:b_norm offset:norm_off atIndex:2];
      [enc setBuffer:b_out offset:out_off atIndex:3];
      [enc setBuffer:(b_oob ? b_oob : b_out) offset:(b_oob ? oob_off : out_off) atIndex:4];
      [enc setBytes:&params length:sizeof(params) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(desc.tokens), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc endEncoding];
      [cb commit];
    }
    return DispatchPath::Native;
  }

  if (!allow_fallback) throw std::invalid_argument("embed_q4_rmsnorm(MLX): portable fallback disabled");
  return run_embed_q4_rmsnorm_fallback(desc, args, stream);
}

}  // namespace ingine::ops
