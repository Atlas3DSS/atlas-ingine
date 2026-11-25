// Pure native AMD ROCm/HIP ordinary graph capture and executable replay.
#include "core/decode_graph.h"
#include "core/device.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ingine {
namespace {

void log_hip_error(const char* op, hipError_t err) noexcept {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP cleanup failed during %s: %s: %s\n", op, hipGetErrorName(err), hipGetErrorString(err));
  }
}

void destroy_graph_exec(hipGraphExec_t& exec) noexcept {
  if (exec != nullptr) {
    log_hip_error("hipGraphExecDestroy", hipGraphExecDestroy(exec));
    exec = nullptr;
  }
}

void destroy_graph(hipGraph_t& graph) noexcept {
  if (graph != nullptr) {
    log_hip_error("hipGraphDestroy", hipGraphDestroy(graph));
    graph = nullptr;
  }
}

void discard_capture(hipStream_t stream) noexcept {
  hipGraph_t discard = nullptr;
  log_hip_error("hipStreamEndCapture(discard)", hipStreamEndCapture(stream, &discard));
  destroy_graph(discard);
}

}  // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(std::exchange(other.graph_, nullptr)) {}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
  if (this != &other) {
    reset();
    graph_ = std::exchange(other.graph_, nullptr);
  }
  return *this;
}

std::pair<std::size_t, std::uint64_t> DecodeGraphDefinition::topology_signature() const {
  if (graph_ == nullptr) return {0, 0};
  std::size_t count = 0;
  HIP_CHECK(hipGraphGetNodes(graph_, nullptr, &count));
  std::vector<hipGraphNode_t> nodes(count);
  HIP_CHECK(hipGraphGetNodes(graph_, nodes.data(), &count));
  std::uint64_t hash = 1469598103934665603ULL;
  for (hipGraphNode_t node : nodes) {
    hipGraphNodeType type = hipGraphNodeTypeEmpty;
    HIP_CHECK(hipGraphNodeGetType(node, &type));
    std::uint64_t payload = static_cast<std::uint64_t>(type) + 1U;
    if (type == hipGraphNodeTypeKernel) {
      hipKernelNodeParams params{};
      HIP_CHECK(hipGraphKernelNodeGetParams(node, &params));
      payload = (payload * 1000003ULL) ^ reinterpret_cast<std::uintptr_t>(params.func);
    }
    hash = (hash ^ payload) * 1099511628211ULL;
  }
  return {count, hash};
}

void DecodeGraphDefinition::capture(GraphStreamT stream, const std::function<void()>& body) {
  reset();
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  try {
    body();
  } catch (...) {
    discard_capture(stream);
    throw;
  }
  hipGraph_t graph = nullptr;
  const hipError_t err = hipStreamEndCapture(stream, &graph);
  if (err != hipSuccess) {
    destroy_graph(graph);
    HIP_CHECK(err);
  }
  graph_ = graph;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }
void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(std::exchange(other.exec_, nullptr)) {}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
  if (this != &other) {
    reset();
    exec_ = std::exchange(other.exec_, nullptr);
  }
  return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
  if (!definition.ready()) throw std::logic_error("cannot instantiate an empty HIP Graph definition");
  reset();
  hipGraphExec_t exec = nullptr;
  const hipError_t err = hipGraphInstantiate(&exec, definition.graph_, nullptr, nullptr, 0);
  if (err != hipSuccess) {
    destroy_graph_exec(exec);
    HIP_CHECK(err);
  }
  exec_ = exec;
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
  if (!ready() || !definition.ready()) {
    throw std::logic_error("HIP Graph update requires a definition and executable");
  }
  hipGraphExecUpdateResult result{};
  hipGraphNode_t error_node = nullptr;
  const hipError_t err = hipGraphExecUpdate(exec_, definition.graph_, &error_node, &result);
  if (err != hipSuccess || result != hipGraphExecUpdateSuccess) {
    throw std::runtime_error("HIP Graph executable update failed: " + std::string(hipGetErrorName(err)) +
                             " (update result " + std::to_string(static_cast<int>(result)) + ")");
  }
}

void DecodeGraphExecutable::upload(GraphStreamT /*stream*/) {}

void DecodeGraphExecutable::launch(GraphStreamT stream) {
  if (!ready()) throw std::logic_error("cannot launch an uninstantiated HIP Graph executable");
  HIP_CHECK(hipGraphLaunch(exec_, stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }
void DecodeGraphExecutable::reset() noexcept { destroy_graph_exec(exec_); }

}  // namespace ingine
