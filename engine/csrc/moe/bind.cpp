#include <torch/extension.h>

#include <optional>

void moe_up(const at::Tensor& x, const at::Tensor& idx, const at::Tensor& w13, const at::Tensor& w13_scale,
            const std::optional<at::Tensor>& w13_factor, int64_t fmt, at::Tensor& out, bool pdl);
void moe_down(const at::Tensor& act, const at::Tensor& idx, const at::Tensor& wts, const at::Tensor& w2,
              const at::Tensor& w2_scale, const std::optional<at::Tensor>& w2_factor, int64_t fmt,
              const at::Tensor& conv_w, at::Tensor& conv_state, at::Tensor& out, const at::Tensor& conv_hist,
              bool pdl);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_up", &moe_up,
          "W13 GEMV of the selected experts + softcapped SwiGLU (SPEC.md, area moe). pdl=True launches with "
          "programmatic dependent launch: the kernel waits for its predecessor before reading anything.",
          py::arg("x"), py::arg("idx"), py::arg("w13"), py::arg("w13_scale"), py::arg("w13_factor"),
          py::arg("fmt"), py::arg("out"), py::arg("pdl") = false);
    m.def("moe_down", &moe_down,
          "W2 GEMV + routing-weighted sum + post-MLP conv with state update (SPEC.md, area moe). pdl=True "
          "launches with programmatic dependent launch: weights are prefetched and idx is read before waiting "
          "for the predecessor (moe_up, which triggers only after its own wait), act/wts/conv state after. "
          "conv_hist (fp32 [M, H], empty tensor = off) receives the BF16-rounded conv inputs s[m] for rollback.",
          py::arg("act"), py::arg("idx"), py::arg("wts"), py::arg("w2"), py::arg("w2_scale"), py::arg("w2_factor"),
          py::arg("fmt"), py::arg("conv_w"), py::arg("conv_state"), py::arg("out"), py::arg("conv_hist"), py::arg("pdl") = false);
}
