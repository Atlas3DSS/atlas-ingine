// Python bindings of the `dense` extension (see engine/SPEC.md, area `dense`). Optional tensors are passed
// as empty tensors (e.g. `torch.empty(0)`); every op launches on the current stream and never allocates,
// except the setup helper `fp8_rows`. pdl=True launches with programmatic dependent launch: the kernel
// prefetches weights/constants, then waits for the preceding kernel before reading activations or state
// and lets the next kernel start its prologue.
#include <torch/extension.h>

namespace dense {
void proj(torch::Tensor xin, torch::Tensor w, torch::Tensor w_scale, torch::Tensor segments, torch::Tensor out,
          bool pdl);
void proj_conv(torch::Tensor x, torch::Tensor w, torch::Tensor w_scale, torch::Tensor conv_w, torch::Tensor conv_state,
               torch::Tensor out, torch::Tensor conv_hist, bool pdl);
void head(torch::Tensor x, torch::Tensor w, torch::Tensor w_scale, torch::Tensor out, bool pdl);
std::vector<torch::Tensor> fp8_rows(torch::Tensor w);
void router(torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor idx, torch::Tensor wts, bool pdl);
void mix(torch::Tensor blocks, torch::Tensor base, torch::Tensor write, torch::Tensor abl_r, torch::Tensor abl_w,
         torch::Tensor partial_out, torch::Tensor queries, torch::Tensor norm_w, torch::Tensor out_bf16,
         torch::Tensor out_f32, bool pdl);
void message_gates(torch::Tensor h, torch::Tensor msg, torch::Tensor w_down, torch::Tensor w_forget,
                   torch::Tensor w_gain, torch::Tensor norm_w, torch::Tensor out_bf16, torch::Tensor out_f32,
                   bool pdl);
void embed(torch::Tensor token, torch::Tensor table, torch::Tensor norm_w, torch::Tensor out_f32, bool pdl);
}  // namespace dense

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("proj", &dense::proj, "multi-segment GEMV: out[m, r] = bf16(w[r] . xin[seg(r).input, m])", "xin"_a, "w"_a,
        "w_scale"_a, "segments"_a, "out"_a, "pdl"_a = false);
  m.def("proj_conv", &dense::proj_conv,
        "attention output projection + post-attention conv (updates conv_state); conv_hist (fp32 [M, H] or empty) "
        "receives the BF16-rounded conv inputs y[m] for speculative rollback",
        "x"_a, "w"_a, "w_scale"_a, "conv_w"_a, "conv_state"_a, "out"_a, "conv_hist"_a, "pdl"_a = false);
  m.def("head", &dense::head, "unembedding: out = 30 tanh(bf16(w . x) / 30)", "x"_a, "w"_a, "w_scale"_a, "out"_a,
        "pdl"_a = false);
  m.def("fp8_rows", &dense::fp8_rows, "bf16 [R, K] -> (float8_e4m3fn [R, K], fp32 scale [R] = amax / 448)", "w"_a);
  m.def("router", &dense::router, "router logits, biased top-8, normalised x8 weights", "x"_a, "w"_a, "bias"_a,
        "idx"_a, "wts"_a, "pdl"_a = false);
  m.def("mix", &dense::mix, "ablated residual update + attention residual + RMS norm", "blocks"_a, "base"_a,
        "write"_a, "abl_r"_a, "abl_w"_a, "partial_out"_a, "queries"_a, "norm_w"_a, "out_bf16"_a, "out_f32"_a,
        "pdl"_a = false);
  m.def("message_gates", &dense::message_gates, "loop message gates", "h"_a, "msg"_a, "w_down"_a, "w_forget"_a,
        "w_gain"_a, "norm_w"_a, "out_bf16"_a, "out_f32"_a, "pdl"_a = false);
  m.def("embed", &dense::embed,
        "token embedding: out[m] = float(bf16(rms_norm(float(table[token[m]])) * norm_w)) (model.Model.embed)",
        "token"_a, "table"_a, "norm_w"_a, "out_f32"_a, "pdl"_a = false);
}
