# Multi-Vendor Backend Blueprint (#835): Apple MLX/Metal, AMD ROCm/HIP, Intel SYCL/XPU

Status: Phase 2 First Slice RFC (worktree `feat/multi-vendor-backends` @ `b823de5`)  
Scope: Hardware-backend abstraction, upstream ecosystem lift audit, phased execution plan, and qualification gates for `Atlas3DSS/atlas-ingine`.

---

## 1. Architectural Boundary (Avoiding a Vendor × Model Cross-Product)

`atlas-ingine` separates **hardware execution** from **model-family semantics** so adding `N` hardware backends and `M` model families scales as `O(N + M)` rather than `O(N × M)`:

```text
+-----------------------------------------------------------------------------+
| Serving & Control Plane (100% hardware-agnostic today)                      |
|   ingine_text | ingine_image | ingine_persona | ingine_engine | ingine_serve |
|   apps/serve (OpenAI HTTP/SSE, response_format) | ledger/ (Verified Ledger) |
+--------------------------------------+--------------------------------------+
                                       | ingine::model::Generator
                                       v (native/include/ingine/generator.h)
+-----------------------------------------------------------------------------+
| Model-Family Layer (ingine_model + model-family adapters)                   |
|   First: Inkling-2-Mini (48 visits, 3x L6-17 loop, rel-bias, DSpark, ViT)   |
|   Next:  Qwen3.8 (GDN + QSA v2) | Future context: DeepSeek / GLM            |
|   Owns: topology/visit plan, weight layout & checkpoint codecs,             |
|         tokenizer/chat template, KV/conv cache identity & Reuse semantics,  |
|         ablation/persona hook binding, speculative loop policy              |
+--------------------------------------+--------------------------------------+
                                       | core/hal.h + core/ops.h
                                       v
+-----------------------------------------------------------------------------+
| Hardware-Backend Layer (ingine_hal_<backend> + ingine_ops_<backend>)        |
|   Backends: CUDA (existing default) | HIP | MLX | SYCL | CPU (host/CI)      |
|   Owns: device discovery & placement, memory & arena buffers, pinned/mapped |
|         host transfers & bounce rings, streams/events/timers, linear &      |
|         conditional graph capture, BLAS GEMM, descriptor resolution, and    |
|         supported primitive/fused operations                                |
+-----------------------------------------------------------------------------+
```

### Ownership Split

| Concern | Hardware-Backend Side (`ingine_hal` + `ingine_ops`) | Model-Family Side (`ingine_model`) |
|---|---|---|
| **Device & Placement** | Device enumeration, VRAM/unified-RAM query (`query_memory_info`), stream/queue binding, DMA bounce rings, capability flags (`DeviceCaps`, `SubgroupCaps`) | Tier budgets (`offload_vram_mb`, `kv_vram_mb`, `park_mb`), expert hot/warm/cold placement policy |
| **Memory & Buffers** | `DeviceBuffer`, `DeviceArena`, `PinnedHostBuffer`, `MappedHostBuffer`, `HostRegion`, `DeviceUploader`, `PinnedBounceRing`, `TieredDmaStream` | Tensor shapes/views (`Tensor`), logical representation, `State` rings/full-KV/conv offsets, `Buffers` workspace lifetimes |
| **Execution & Graphs** | `Stream`, `Event`, `DecodeGraphDefinition/Executable`, optional conditional SWITCH node (`CUDART_VERSION >= 12080` + active driver/context probe), explicit capture-vs-switch policy (`require_graph_capture_or_eager`, `require_conditional_switch_or_eager`, `GraphRequirement`) | Visit schedule (`VISITS[48]`), decode graph capture strategy, controlled eager fallback when `DeviceCaps` or runtime SWITCH capture is unavailable |
| **Operation Resolution (`core/ops.h`)** | Resolves descriptors (`EmbedQ4RmsNormDesc`, `DenseGemmDesc`, future family ops) containing dtype/accumulator, packed layout, shape/strides, rounding mode, and unified `TokenBoundsMode`; executes Native kernel or explicit portable fallback | Constructs typed operation descriptors from model weights/state (`TokenBoundsMode::PreValidated` after admission validation) and invokes model-facing `ops::run_*` entry points |
| **Weights & Identity** | Raw device storage, async upload/copy, queue/event completion & host visibility | Safetensors/GGUF checkpoint mapping, `ExpertFormat`, Q4_0 layouts, `model_id()`, tokenizer/template, `Reuse` cache identity |

### Buffer Lifetime, Deferred Execution & Graph/GEMM Contracts

1. **Buffer & View Lifetime Through Completion:** `ingine::Tensor` and operation arguments are non-owning views. Callers must keep backing `DeviceBuffer`, `DeviceArena` scopes, `PinnedHostBuffer`, and `MappedHostBuffer` objects alive until every enqueued asynchronous operation and `hal::Event` referencing them has completed on its `hal::StreamHandle` (via `hal::stream_synchronize` or `hal::Event::synchronize`).
2. **Deferred Execution & Host Visibility:** Under graph capture or lazy-array backends (CUDA graphs, HIP graphs, MLX lazy graph, SYCL `command_graph`), device execution is deferred until graph launch/commit; host readback buffers populated by `memcpy_d2h_async` or mapped exchange structures (`VerifyRecord`, `chain_host_`) are valid to read on the host only after synchronizing the associated `hal::StreamHandle` or `hal::Event`.
3. **GEMM Policy Preservation (`DenseGemmDesc`):** BLAS matrix multiplications preserve caller-supplied row/column leading strides (`lda`, `ldb`, `ldc`), explicit output dtype (`ActT` or `FP32`), FP32 accumulation (`CUBLAS_COMPUTE_32F`), FP32 split-K reduction (`CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE` for `prefill::Gemm`), and workspace bounds (32 MiB prefill / drafter workspace).
4. **Unsupported Graph Capture & Conditional SWITCH Policy:** `hal::GraphRequirement::Capture` (ordinary linear graph capture) and `hal::GraphRequirement::ConditionalSwitch` (conditional SWITCH nodes) are admitted independently. Conditional SWITCH support requires toolkit floor `CUDART_VERSION >= 12080`, active driver/runtime `>= 12080`, compute capability `>= 7.0`, and a live `cudaGraphCondTypeSwitch` probe on the active device/context (`INGINE_DISABLE_CONDITIONAL_GRAPHS=1` disables it dynamically). Backends or contexts lacking the requested graph capability either run through the explicit eager execution path (`allow_eager == true`, or controlled `Runtime::try_capture_switch` fallback) or throw loudly via `require_graph_capture_or_eager` / `require_conditional_switch_or_eager`, never aborting or silently dropping work.

### Why This Prevents `Vendor × Model` Explosion

1. **Shared Primitive Surface Across Model Families:** Inkling-2-Mini, Qwen3.8, and future model families share `Gemm`, `Gemv`, `RmsNorm`, `LayerNorm`, `FlashAttn`, `Sconv`, `MoeUp`/`MoeDown`, `Sampling`, and paged KV primitives implemented once per backend.
2. **Single Parameterized CUDA/HIP Kernel Tree:** CUDA and AMD ROCm/HIP compile the same kernel sources through `kernels/common/arch_ops.cuh` and `kernels/common/warp.cuh`, eliminating duplicated CUDA vs. HIP `.cu` trees.
3. **Descriptor-Resolved Dispatch:** Model code resolves operations against explicit descriptors; backends advertise native support and fall back to portable primitives or composition cleanly when a fused kernel is not present.

---

## 2. Codebase CUDA Surface Audit (`atlas-ingine @ b823de5`)

### 2.1 Already 100% CUDA-Free Components

Verified across `native/CMakeLists.txt`, `native/include/`, `native/src/`, `native/apps/`, and `ledger/`:
- Public API: `native/include/ingine/generator.h`, `native/include/ingine/persona.h`
- Host libraries: `src/text` (`ingine_text`), `src/image` (`ingine_image`), `src/persona` (`ingine_persona`), `src/engine` (`ingine_engine`), `src/serve` (`ingine_serve`)
- Server binary: `apps/serve/main.cpp` (only links `ingine_model` when `TARGET ingine_model` exists; otherwise builds `--fake` cleanly)
- Verified Ledger: `ledger/` (Python CLI/library)
- Core host headers already CUDA-free: `src/core/dtype.{h,cpp}`, `src/core/tensor.{h,cpp}`, `src/core/arena.h`, `src/kernels/common/persona_abi.h`, and host conversion helpers in `src/core/act.h`.

### 2.2 `src/core/`, `tests/`, and `apps/` Coupling Audit

| File | Current Coupling | Extraction Target |
|---|---|---|
| `src/core/dtype.{h,cpp}`, `src/core/tensor.{h,cpp}` | Zero CUDA dependencies | `ingine_core_host` (always built) |
| `src/core/act.h` | Unconditionally includes `<cuda_bf16.h>`, `<cuda_fp16.h>`, `<cuda_runtime.h>`, `<library_types.h>` | Guard CUDA headers/device helpers behind `#if INGINE_BACKEND_CUDA`; provide POD 16-bit `ActT` + host helpers everywhere |
| `src/core/safetensors.{h,cpp}` | `SafetensorsFile`/`SafetensorsModel` are pure POSIX `pread`; `DeviceUploader` uses `cudaHostAlloc`, `cudaEvent_t`, `cudaStream_t`, `cudaMemcpyAsync` | Keep `safetensors_io.cpp` in `ingine_core_host`; hide `DeviceUploader` CUDA/CPU internals behind `core/hal.h` + PIMPL |
| `src/core/device.{h,cu}`, `src/core/arena.{h,cu}` | `cudaMalloc`, `cudaMemGetInfo`, `cudaStream_t`, `cudaEvent_t`, `cudaDeviceProp`, `CUDA_CHECK` | `ingine_hal_<backend>` behind `core/hal.h` |
| `src/core/host_memory.{h,cpp}` | POSIX `mmap`/`MADV_HUGEPAGE` + `cudaHostRegister`/`cudaHostAlloc` | `HostRegion` + `MappedHostBuffer` implemented per HAL (`hal_cpu.cpp` on CPU) |
| `src/core/paged_tier.{h,cpp}` | `MmapFile`/`SwapFile` are pure POSIX; `PinnedBounceRing` & `TieredDmaStream` use `cudaStream_t`/`cudaEvent_t` | Keep POSIX file tiers host-only; abstract DMA ring/stream via HAL |
| `src/core/decode_graph.{h,cpp}` | `cudaGraph_t`, `cudaGraphExec_t` | `hal::GraphDefinition` / `hal::GraphExecutable` (optional capture or explicit eager/reject) |
| `src/core/nvtx.h` | `nvtx3` markers | Gate behind `#if INGINE_BACKEND_CUDA` |
| `tests/common/test_util.h`, `tests/kernels/oracle.h` | Pull `<cuda_bf16.h>`/`<cuda_fp16.h>`/`<cuda_runtime.h>` via `core/act.h` and direct includes | Extract pure ISO C++20 `tests/common/scalar_oracle.h` (zero CUDA headers, zero `core/act.h`) |
| `apps/bench/main.cpp`, `apps/validate/main.cpp`, `apps/persona_probe/main.cpp` | `cudaMemGetInfo`, `cudaMemcpyAsync`, `cudaStreamSynchronize`, `cudaGetDeviceProperties` | Route through `core/hal.h` (CUDA-only model apps skipped when `INGINE_BACKEND=CPU`) |

### 2.3 BLAS (`cublasLt`) Audit

Only **2 files (214 lines total)** use `cublasLt`:
1. `native/src/kernels/prefill/gemm.cpp` (136 lines) — `prefill::Gemm::run`, shared by LM prefill and the ViT vision tower. Row-major `C[M,N] = A[M,K] · B[N,K]^T` with `ActT` (`BF16`/`FP16`) inputs, FP32 compute & split-K reduction (`CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE`), FP32 or `ActT` output, 32 MiB workspace.
2. `native/src/kernels/draft/gemm.cpp` (78 lines) — `draft::Bf16Gemm::run` for DSpark draft projections (`ActT` output, FP32 compute).

These map to `DenseGemmDesc` / `hal::BlasGemm` (`hipBLASLt` / `rocBLAS`, `mx::matmul` / MPS, `oneMKL` / `oneDNN`).

### 2.4 Inline PTX (`asm`) and Existing Portability Fallbacks

Grep across `native/src/kernels/` and `native/src/model/` shows **103 non-comment `asm` lines across 16 files**, each with a portable fallback or exact integer equivalent:

| Category | Exact Files | Current Fallback Status | Non-CUDA Equivalent |
|---|---|---|---|
| **SM90+ PDL, Clusters, DSMEM, L2 Hints** | `common/pdl.cuh`, `common/l2.cuh`, `dense/cluster.cuh`, `dense/dsmem.cuh`, `attn/attn.cu`, `moe/moe.cu` | Guarded by `__CUDA_ARCH__ >= 900` / `INGINE_L2_HINTS` and `current_sm_major() >= 9` | Compile to no-ops; `ld_stream` -> plain read |
| **Async GMEM->SMEM (`cp.async`)** | `common/copy.cuh`, `dense/dense.cuh`, `attn/attn.cu`, `prefill/common.cuh`, `prefill/wgemm.cu` | Falls back to `copy::sync<4\|16>` for `__CUDA_ARCH__ < 800` | Synchronous shared-memory staging + workgroup barrier |
| **Tensor Core MMA & `ldmatrix`** | `common/mma_sm70.cuh`, `prefill/wgemm.cu` | Has pure-SIMT FP32-accumulated fallback (`INGINE_FORCE_SIMT_MMA`) | SIMT fallback for initial parity; vendor WMMA/MFMA/XMX/simdgroup_matrix in optimization pass |
| **FP8/FP4 Unpackers** | `core/act.h`, `dense/dense.cuh`, `draft/draft_device.cuh`, `moe/moe.cu` | Portable branchless ALU converters (`e4m3_to_f16_bits_portable`, `e4m3_to_bf16_bits_portable`, `__byte_perm` E2M1) | Portable ALU bit-ops (replace `__byte_perm` with byte shifts / `__builtin_amdgcn_perm` on HIP) |
| **EXL3 MCG Trellis** | `common/exl3.cuh` (lines 20–21, 40–41, 71, 77, 114, 120) | Uses `lop3.b32 ..., 0x8fff8fff, 0x3b603b60, 0x6a` and `__funnelshift_l(lo, hi, s)` | Exact integer equivalents:<br>• `lop3(..., 0x6a)` = `(x & 0x8fff8fffu) ^ 0x3b603b60u`<br>• NVIDIA `__funnelshift_l(lo, hi, s)` shifts `(hi:lo)` left by `k = s & 31u` returning high 32 bits (specifically returning `hi` when `k == 0u`): `const unsigned k = s & 31u; return k == 0u ? hi : ((hi << k) | (lo >> (32u - k)));` |
| **Coherent Volatile `ld_dep` / `ld_act`** | `dense/dense.cuh`, `moe/moe.cu`, `attn/attn.cu`, `draft/draft_device.cuh`, `batch/batch.cu` | Guard against PTX hoisting past `griddepcontrol.wait` | Standard loads when PDL is off |

### 2.5 Two Structural Hazards & Fixes

1. **Logical 32-Lane Grouping & Subgroup Capabilities (`276` matches across `23` files):**
   - Kernels currently assume 32-lane warp semantics (`threadIdx.x & 31`, `>> 5`, `0xffffffffu`, `__shfl*_sync`, `group_sum<32>`, `reduce_scatter<32, N>`, `fwht_128_warp` in `exl3.cuh`). Note that `moe.cu`'s 18-warp block (`18 × 32 = 576 = 9 × 64`) is already a multiple of 64; the portability issue on AMD CDNA wave64 and Intel Xe subgroup-16/8 is logical 32-lane grouping/masks and matching algorithms to the selected kernel's actual `SubgroupCaps`.
   - **Mitigation (deferred until after Phase 2 First Slice per incremental rollout):** Introduce `kernels/common/warp.cuh` parameterized by queried `SubgroupCaps` / subgroup size (`32` on CUDA/RDNA/Metal, `64` with 32-lane logical sub-warp or LDS/DPP on CDNA, `16`/`8` + SLM pairing on Intel Xe).
2. **CUDA Graph Conditional Switch Nodes & DSpark Speculative Fallback (`SpecLoop`):**
   - `cudaGraphConditionalHandle` / `cudaGraphNodeTypeConditional` (`cudaGraphCondTypeSwitch`) is used in `Runtime::try_capture_switch` (`src/model/runtime.cpp`) for `SpecLoop` (`src/model/spec.cpp`).
   - Previously, `SpecLoop::prepare_accept` unconditionally called `capture_switch` and every speculative round unconditionally launched `accept_`, so `verify_select_eager` alone could not run acceptance, rollback, or `verify_publish` without conditional graphs.
   - **Complete Fix:** Gate `try_capture_switch` behind `Runtime::supports_conditional_graphs()` (12.8+ driver/runtime/context probe, `INGINE_DISABLE_CONDITIONAL_GRAPHS=1` override, and controlled non-aborting fallback on unsupported `SWITCH` node errors). When conditional graphs are unavailable or disabled:
     - `SpecLoop::prepare_accept` skips `capture_switch`.
     - Verify runs `verify_select_eager`, synchronizes the 4-byte `width_ready_` event, reads `selected_m`, calls `rt_.record_save_verify(selected_m, stream)`, and executes either `rt_.run_step_eager(selected_m)` (if `needs_eager_step()`) or the linear captured graph via `rt_.run_step(selected_m)` (including hooked graphs via `ensure_hooked_graphs()` when `hooked_active()`).
     - Acceptance, rollback (`record_rollback`), and publication (`verify_publish`) run directly on `stream` for the exact `selected_m` via a shared `SpecLoop::record_accept_for_width(selected_m, sampling, stream)` helper used by both conditional-graph capture and eager fallback.

---

## 3. Upstream Ecosystem Lift Audit (All 3 Targets)

### 3.1 License & Attribution Compatibility

Every candidate is Apache-2.0 or MIT (compatible with `atlas-ingine`'s Apache-2.0 license). Every lifted component must pin its exact upstream source revision and file license/attribution notices in `NOTICE` and `ACKNOWLEDGEMENTS.md`:

| Upstream / Local Reference | Pinned Source / Revision | License | Liftable Components & Role |
|---|---|---|---|
| **`llama.cpp` / `atlas-engine`** | Local `atlas-engine` (`ggml-cuda/vendors/hip.h`, `ggml-hip`, `ggml-sycl`, `ggml-metal`, `src/models/inkling2{,-loop}.cpp`) | MIT | CMake/HIP/SYCL compat headers, DPCT wrappers, independent GGUF Inkling-2 graph reference |
| **Intel SYCLomatic / DPCT** | Vendored in `ggml-sycl/dpct` | Apache-2.0 w/ LLVM Exception | Sub-group, math, and atomic compatibility wrappers for initial SYCL port |
| **`ml-explore/mlx`** | Upstream `libmlx` C++20 API | MIT | `mlx/mlx.h` arrays, unified memory, `mx::matmul`, `mx::fast::metal_kernel` JIT MSL dispatch |
| **`ml-explore/mlx-lm`** | Open PR `#2086` pinned at `e2b5bbd` (`inkling2_loop`) | MIT | Reference Inkling-2-Mini op decomposition (`loop`, `message_gates`, dynamic K/V conv, `rel_bias`) |
| **`EschaLabs/escha-mlx`** | Upstream repo (already cited in `NOTICE`) | Apache-2.0 | Metal shaders for EXL3 MCG (`0xCBAC1FED`) + 128-pt Sylvester Hadamard W2 |
| **`vLLM` / `hiutaky-vllm`** | Local `atlas-v6-refs/hiutaky-vllm` (`vllm/models/inkling/amd/`, `vllm/platforms/{rocm,xpu}.py`, `mxfp8`/`mxfp4`) | Apache-2.0 (`rel_attention_decode.py` LightSeek Foundation: MIT) | ROCm split-KV decode & sconv layout references; MXFP8/MXFP4 ROCm & XPU tiling patterns (Python/Triton reference only) |
| **`Neroued/ninfer` / `iamwavecut/ninfer-all`** | Upstream repos | Apache-2.0 | Audited: pure NVIDIA CUDA (`sm_70..120a`); zero HIP/SYCL/Metal code to lift |

### 3.2 Quantization Format & Feature Matrix

> **Status note:** All AMD HIP, Apple MLX/Metal, and Intel SYCL/XPU entries below are **Target (unverified on target hardware)** until built and qualified on actual AMD, Apple Silicon, or Intel hardware.

| Format / Feature | CUDA (`main`) | CPU (`INGINE_BACKEND=CPU` First Slice) | AMD ROCm / HIP (**Target**) | Apple MLX / Metal (**Target**) | Intel SYCL / XPU (**Target**) |
|---|---|---|---|---|---|
| **Q4_0 Embed + RMSNorm (`embed_q4`)** | Qualified | Qualified in First Slice (`ops_cpu` + `ops_fallback`) | **Target** (shared `mix.cu`) | **Target** (custom MSL / `mx::dequantize`) | **Target** (SYCL kernel / `ggml-sycl` Q4_0) |
| **BF16 / FP16 Dense & Activations** | Qualified | Portable `host_*` helpers (`act.h`) | **Target** (`hipBLASLt` / `__hip_bfloat16`) | **Target** (`mx::matmul` / Metal `bfloat`/`half`) | **Target** (`oneMKL` / `oneDNN` / `joint_matrix`) |
| **Dense FP8 (`E4M3FN` + row scale)** | Qualified | Portable scalar decoder | **Target** (SW ALU `e4m3` first, then gfx942/950/gfx12 FP8) | **Target** (SW E4M3 ALU in MSL) | **Target** (SW E4M3 ALU in SYCL) |
| **MoE `NVFP4` (L0–19)** | Qualified | Portable reference target | **Target** (`__builtin_amdgcn_perm`/shifts in `moe.cu`) | **Target** (custom MSL `moe_up`/`moe_down`) | **Target** (SYCL E2M1 decode + XMX DPAS later) |
| **MoE `MXFP8` (L20–23)** | Qualified | Portable reference target | **Target** (E4M3 + E8M0 shift) | **Target** (custom MSL; evaluate MLX native MXFP8) | **Target** (E4M3 + E8M0; ref `vllm/mxfp8/xpu.py`) |
| **MoE `ATLAS2` / `ATLAS3`** | Qualified | Portable reference target | **Target** (branchless ALU in `moe.cu`) | **Target** (branchless uint32 ALU in MSL) | **Target** (branchless uint32 ALU in SYCL) |
| **MoE `EXL3_2` / `EXL3_3` + Hadamard** | Qualified | Portable reference target | **Target** (safe funnelshift + `lop3` ALU + wave-aware FWHT) | **Target** (lift `escha-mlx` Metal kernels) | **Target** (integer MCG + subgroup-16 FWHT) |
| **KV `Q4_0` (`--kv-q4`) & `FP8`** | Qualified | Portable reference target | **Target** (shared `attn.cu`) | **Target** (custom MSL attention) | **Target** (SYCL attention) |
| **GGUF Weights** | Not in `atlas-ingine` (safetensors only) | Optional model-family adapter | **Target** (model-family adapter) | **Target** (model-family adapter; `atlas-engine` Metal lacks NVFP4/MXFP8/ATLAS/EXL3/DSpark/ViT) | **Target** (model-family adapter) |

### 3.3 Target-Specific Notes & Op Gaps

1. **AMD ROCm / HIP:** Shared CUDA/HIP kernel tree via `arch_ops.cuh` + `warp.cuh` (`hipBLASLt`/`rocBLAS`). **Critical trap:** do **not** define `#define __CUDA_ARCH__ 1300` as in `ggml-cuda/vendors/hip.h:184`, because `__CUDA_ARCH__` activates NVIDIA inline PTX (`mma.sync`, `griddepcontrol`, `lop3`). Gate NVIDIA PTX behind `defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)`. Land SIMT-MMA parity first, then RDNA WMMA / CDNA MFMA fast paths.
2. **Apple Silicon MLX / Metal:** Link C++20 `libmlx` (`INGINE_BACKEND=MLX`) directly into `ingine-serve`/`ingine-bench`/`ingine-validate` so the entire OpenAI HTTP server, `ChatFormat`, `StreamParser`, `Image`, `Persona`, `Scheduler`, and `VerifiedLedger` run unchanged without Python. Implement custom 32-lane MSL kernels via `mx::fast::metal_kernel` for on-the-fly `rel_logits` attention (avoiding `atlas-engine`'s materialized `kq_b` buffer), `NVFP4`/`MXFP8`/`ATLAS2`/`ATLAS3`/`EXL3` MoE (lifting `escha-mlx`), `mix`, `message_gates`, `proj_conv`, `router`, and `head`; cross-check against `mlx-lm` PR `#2086 @ e2b5bbd`.
3. **Intel oneAPI / SYCL / XPU:** Lift `atlas-engine/ggml-sycl/CMakeLists.txt` (`IntelSYCL`, Level Zero `ze_api.h`/`ze_loader`, `-ze-intel-greater-than-4GB-buffer-required`, `DNNL_GPU_VENDOR=INTEL` check rejecting CPU-only `libdnnl-dev`, `MKL::MKL_SYCL::BLAS`, `spir64_gen` AOT with `sg=8` for `dg2`/`acm` and `sg=16` for `pvc`/`bmg`/`xe2`) + `ggml-sycl/dpct` headers. Port SIMT kernels with explicit subgroup capabilities (`reqd_sub_group_size(16)`/`(8)`) and SLM pairing, eager `sycl::queue` -> `command_graph`, then `joint_matrix` (DPAS) acceleration.

---

## 4. CMake & HAL Abstraction Seam (Zero CUDA Regression)

- **CMake Backend Selection (`native/CMakeLists.txt` & `native/cmake/Dependencies.cmake`):**
  - `project(ingine LANGUAGES C CXX)` with `set(INGINE_BACKEND "CUDA" CACHE STRING "Execution backend: CUDA, HIP, MLX, SYCL, CPU")`.
  - `enable_language(CUDA)` and `find_package(CUDAToolkit REQUIRED)` run **only** when `INGINE_BACKEND STREQUAL "CUDA"`.
  - Existing CUDA presets in `native/CMakePresets.json` (`release`, `dev`, `install`, `multiarch-dev`, `sm70*`, `legacy-fp16`) remain untouched; add `cpu-dev` (`INGINE_BACKEND=CPU`, `INGINE_BUILD_TESTS=ON`, `INGINE_BUILD_APPS=ON`) and future backend presets cleanly.
- **Target Graph:**
  - `ingine_core_host`: `dtype.cpp`, `tensor.cpp`, `safetensors_io.cpp`, `ops_fallback.cpp` — zero GPU SDK headers or linkage.
  - `ingine_hal` (`ingine_hal_cuda` | `ingine_hal_cpu` | future `ingine_hal_{hip,mlx,sycl}`): implements `core/hal.h`, `arena.h`, `host_memory.h`, `DeviceUploader`, and (for GPU backends) `decode_graph.h`/`paged_tier.h`.
  - `ingine_core`: `ingine_core_host` + `ingine_hal` (preserving existing target name for all downstream libraries/tests).
  - `ingine_ops` (`ingine_ops_cuda` | `ingine_ops_cpu` | future `ingine_ops_{hip,mlx,sycl}`): implements descriptor validation/resolution and `ops::run_*` model-facing entry points. On `CUDA`, `ingine_ops_cuda` links `ingine_core` + `ingine_kernels_dense` directly so `ingine_model` dispatches `embed_q4` (and future ops) with zero extra dependency cycles.

---

## 5. Phased Execution Order & Qualification Gates

### 5.1 Phased Execution Order

1. **Phase 2 (Current — First Slice first, then remaining HAL/warp seam):**
   - **First Slice:** Blueprint corrections + `core/hal.h` + `core/ops.h` (`EmbedQ4RmsNormDesc`, `DenseGemmDesc`, portable fallback) + `act.h`/`safetensors.h`/`scalar_oracle.h` CUDA decoupling + `INGINE_BACKEND=CPU` zero-CUDA build + `hal_embed_q4_parity` (CPU & CUDA native + fallback) + complete `SpecLoop` conditional-graph fallback (`spec_conditional_fallback`) + CUDA zero-regression qualification.
   - **Remaining Phase 2:** Expand `arch_ops.cuh` + `warp.cuh` across `src/kernels/*` and move remaining direct `<cuda_runtime.h>` / `cublasLt` calls in `src/model` and `apps/` onto `hal::BlasGemm` & `core/hal.h`.
2. **Phase 3:** AMD ROCm / HIP (`INGINE_BACKEND=HIP`).
3. **Phase 4:** Apple Silicon MLX / Metal (`INGINE_BACKEND=MLX`).
4. **Phase 5:** Intel oneAPI / SYCL / XPU (`INGINE_BACKEND=SYCL`).

### 5.2 First Parity Case & Phase 2 First-Slice Exit Gates

1. **Q4_0 Embedding Lookup + RMSNorm Parity (`hal_embed_q4_parity` on BOTH `CPU` and `CUDA`):**
   - Uses Inkling's Q4_0 embedding layout (`dense.h:125–127`, packed `1,296`-byte rows: `1,152` nibble bytes + `72` FP16 scales, `H = 2304`, `eps = 1e-6`).
   - Fixed token IDs (`M ∈ {1, 3, 5, 8}`), packed 1,296-byte rows, and FP32 norm weights uploaded, executed, synchronized via `hal::Event`/`hal::Stream`, and read back through HAL (`core/hal.h`, `core/arena.h`) via the single model-facing entry point `ingine::ops::run_embed_q4_rmsnorm`.
   - Verifies:
     - Device-emitted and host-decoded Q4_0 lanes match the independent scalar decoder (`fmaf(float(nibble), d, -8.0f * d)`, canonicalizing zeroes to `+0.0f`) bit-for-bit across every lane including signed scales and zero fixtures.
     - All normalized outputs are finite, match the specified activation rounding (`BF16` or `FP16`) bit-for-bit on every lane, pass the independent scalar FP64 reference (`scalar_embed_q4_rmsnorm_fp64` in `tests/common/scalar_oracle.h`, `ORACLE` `-fno-fast-math -ffp-contract=off`, `rel = 1/128`, `abs = 1e-6`, `frac = 0.01`), and match across two runs.
     - Dispatch fallback (`DispatchPath::PortableFallback`, both `force_fallback = true` on packed/padded strides and non-specialized descriptor stride) passes on the exact same inputs.
     - Invalid descriptors and out-of-range token IDs (`TokenBoundsMode::Validate` on host/device/fallback/oracle; `TokenBoundsMode::PreValidated` kernel bounds guard + graph capture) obey the single token-bounds contract.
2. **Zero-CUDA CPU Proof (`INGINE_BACKEND=CPU`):**
   - Preprocessor check (`g++ -E` on `generator.h`, `arena.h`, `hal.h`, `ops.h`, `act.h`, `safetensors.h`, `scalar_oracle.h`) with CUDA headers excluded -> 0 errors, 0 CUDA includes.
   - Clean `cmake --preset cpu-dev` build in a CUDA-free `PATH` (`CMakeCache.txt` has zero CUDA discovery; `ldd` on `ingine-serve` and `hal_embed_q4_parity` shows zero `cuda`/`cublas` libraries).
   - Full `ctest --test-dir native/build-cpu` pass (`hal_embed_q4_parity` + all non-GPU host CTests), plus separate `ingine-serve --fake` HTTP `/health` and `/v1/chat/completions` smoke test.
3. **DSpark Non-Conditional-Graph Fallback (`spec_conditional_fallback` on CUDA GPU):**
   - Compares normal `SpecLoop` vs `INGINE_DISABLE_CONDITIONAL_GRAPHS=1` on real Inkling-2-Mini weights (greedy generation, multi-turn follow-up decode proving rollback + continuation, teacher-forced replay comparing `spec_rounds`/`spec_accepted`/per-round trace, and hooked persona verification), verifying exact equality on token IDs, accepted/rejected counts, rollback state, and following decode.
4. **CUDA Qualification & Bitlock Policy:**
   - Built-in 48 reference hashes in `kernels_bitlock` apply by default to BF16 `sm_120` with CUDA 13+. On non-`sm_120` hardware (e.g., RTX 3090 `sm_86`), record **both** `b823de5` baseline and candidate hashes on the exact same card, toolchain, dtype, and preset (`multiarch-dev`), alongside two-run determinism, FP64 oracle tests (`kernels_{dense,moe,attn,head}`, `prefill_*`, `vision_kernels`, `batch_kernels`, `spec_*`), and stable-case `ingine-validate` goldens (`pieces`, `--generate`, `--generate --spec`). Report exact tests run, passed, and skipped.

### 5.3 Local Verification vs. Target Hardware Qualification Matrix

Current local hosts (`devbox` RTX 3090 / RTX PRO 6000, `monkeypc` RTX 4090, `quincy` 7× V100) are NVIDIA-only without `hipcc`, Apple Metal/macOS, or `icpx`/oneAPI installed:

| Gate | Runs Locally (`devbox` / `quincy`) | Requires Target Hardware (AMD / Apple / Intel) |
|---|---|---|
| Zero-CUDA CPU build, `hal_embed_q4_parity`, host CTests, `ingine-serve --fake` | **Yes** | — |
| CUDA `multiarch-dev` zero-regression (`hal_embed_q4_parity`, `spec_conditional_fallback`, bitlock baseline vs candidate, FP64 oracles, `ingine-validate`) | **Yes (`sb run`)** | — |
| HIP / SYCL / MLX `hal_embed_q4_parity`, kernel FP64 oracles, `ingine-validate`, `model_reuse`, `vision_model`, `batch_parity`, `ingine-bench` | — | **Yes** |
