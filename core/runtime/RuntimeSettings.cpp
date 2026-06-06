#include "core/runtime/RuntimeSettings.h"

#include <array>
#include <cstring>
#include <sstream>
#include <tuple>

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

namespace {

// Reverse-lookup tables for ``int32_t`` strategy values. The indices match the
// nvinfer1 enum integers (validated at the Python boundary; no further check
// here). Out-of-range -> "<unknown>".
constexpr std::array<char const*, 3> kDsStrategyNames = {"lazy", "eager", "none"};
constexpr std::array<char const*, 2> kCgStrategyNames = {"disabled", "whole_graph_capture"};

} // namespace

std::string ds_strategy_name(int32_t v) {
  if (v < 0 || static_cast<size_t>(v) >= kDsStrategyNames.size()) {
    return "<unknown>";
  }
  return kDsStrategyNames[static_cast<size_t>(v)];
}

std::string cg_strategy_name(int32_t v) {
  if (v < 0 || static_cast<size_t>(v) >= kCgStrategyNames.size()) {
    return "<unknown>";
  }
  return kCgStrategyNames[static_cast<size_t>(v)];
}

// ---- RuntimeCacheHandle methods ---------------------------------------------
//
// The ``#ifdef TRT_MAJOR_RTX`` is intentionally confined to this translation
// unit: the public header advertises a uniform interface (always-callable
// methods that simply degrade to no-ops on non-RTX builds), and the JIT-binding
// registration file (``register_jit_hooks.cpp``) calls these as plain member
// references with zero conditional compilation.

at::Tensor RuntimeCacheHandle::serialize() const {
  auto opts = at::TensorOptions().dtype(at::kByte);
#ifdef TRT_MAJOR_RTX
  if (!cache) {
    return at::empty({0}, opts);
  }
  auto host_mem = make_trt(cache->serialize());
  if (!host_mem) {
    return at::empty({0}, opts);
  }
  auto tensor = at::empty({static_cast<int64_t>(host_mem->size())}, opts);
  std::memcpy(tensor.data_ptr(), host_mem->data(), host_mem->size());
  return tensor;
#else
  return at::empty({0}, opts);
#endif
}

void RuntimeCacheHandle::deserialize(TORCHTRT_UNUSED at::Tensor data) {
#ifdef TRT_MAJOR_RTX
  if (data.numel() == 0 || !cache) {
    return;
  }
  auto contig = data.contiguous().to(at::kCPU);
  cache->deserialize(contig.data_ptr(), static_cast<size_t>(contig.numel()));
#endif
}

bool RuntimeCacheHandle::has_cache() const {
#ifdef TRT_MAJOR_RTX
  return cache != nullptr;
#else
  return false;
#endif
}

// ---- RuntimeSettings methods ------------------------------------------------

bool RuntimeSettings::operator==(RuntimeSettings const& other) const noexcept {
  // ``runtime_cache`` compares by pointer identity: passing the same handle
  // twice through ``update_runtime_settings`` is a no-op. Hoisted into locals
  // because ``std::tie`` requires lvalues.
  auto* this_cache = runtime_cache.get();
  auto* other_cache = other.runtime_cache.get();
  return std::tie(dynamic_shapes_kernel_specialization_strategy, cuda_graph_strategy, this_cache) ==
      std::tie(other.dynamic_shapes_kernel_specialization_strategy, other.cuda_graph_strategy, other_cache);
}

std::string RuntimeSettings::to_str() const {
  std::ostringstream os;
  os << "Dynamic Shapes Kernel Strategy: " << ds_strategy_name(dynamic_shapes_kernel_specialization_strategy)
     << std::endl;
  os << "CUDA Graph Strategy: " << cg_strategy_name(cuda_graph_strategy) << std::endl;
  if (runtime_cache) {
    auto const& p = runtime_cache->path;
    os << "Runtime Cache: " << (p.empty() ? "<in-memory shared>" : p) << std::endl;
  } else {
    os << "Runtime Cache: <engine-local, in-memory>" << std::endl;
  }
  return os.str();
}

std::ostream& operator<<(std::ostream& os, RuntimeSettings const& rs) {
  os << rs.to_str();
  return os;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
