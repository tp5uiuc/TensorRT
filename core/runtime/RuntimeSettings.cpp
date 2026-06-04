#include "core/runtime/RuntimeSettings.h"

#include <cstring>
#include <sstream>
#include <tuple>

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

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
  os << "Dynamic Shapes Kernel Strategy: " << dynamic_shapes_kernel_specialization_strategy << std::endl;
  os << "CUDA Graph Strategy: " << cuda_graph_strategy << std::endl;
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
