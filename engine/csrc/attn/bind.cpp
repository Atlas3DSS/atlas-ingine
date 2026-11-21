#include <torch/extension.h>

void attn_decode(const at::Tensor& proj, const at::Tensor& q_scale, const at::Tensor& r_scale,
                 const at::Tensor& rel_proj, const at::Tensor& k_conv, const at::Tensor& v_conv,
                 at::Tensor& conv_k_state, at::Tensor& conv_v_state, at::Tensor& k_cache, at::Tensor& v_cache,
                 at::Tensor& kpos, const at::Tensor& pos, int64_t window, bool log_scaling, at::Tensor& workspace,
                 at::Tensor& out);
int64_t attn_workspace_bytes(int64_t max_m);
int64_t attn_max_active_clusters(int64_t M, int64_t window);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_decode", &attn_decode,
          "Attention core of one layer visit for M consecutive tokens (SPEC.md, area attn): q/rel post-processing, "
          "key/value convs + state update, KV cache write, split-K attention with relative bias, sink and output "
          "gate. window = 1024 (sliding ring, kpos [1024]) or 0 (full layer, linear cache, kpos ignored).",
          py::arg("proj"), py::arg("q_scale"), py::arg("r_scale"), py::arg("rel_proj"), py::arg("k_conv"),
          py::arg("v_conv"), py::arg("conv_k_state"), py::arg("conv_v_state"), py::arg("k_cache"),
          py::arg("v_cache"), py::arg("kpos"), py::arg("pos"), py::arg("window"), py::arg("log_scaling"),
          py::arg("workspace"), py::arg("out"));
    m.def("attn_workspace_bytes", &attn_workspace_bytes,
          "Bytes of workspace attn_decode needs for up to max_m tokens (about 0.2 MB per token): a counter "
          "(zero the workspace once at allocation; the kernel resets the counter itself) and the split partials. "
          "One workspace serves every layer of a stream-ordered decode step.",
          py::arg("max_m") = 8);
    m.def("attn_max_active_clusters", &attn_max_active_clusters,
          "How many of the kernel's 16-block clusters (one per KV head, 8 per call) fit on the GPU at once.",
          py::arg("M"), py::arg("window"));
}
