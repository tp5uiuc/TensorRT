#include "core/runtime/RuntimeSettings.h"

#include <sstream>

namespace torch_tensorrt {
namespace core {
namespace runtime {

bool RuntimeSettings::operator==(RuntimeSettings const& other) const noexcept {
  // Same handle pointer counts as identical cache; passing the same handle twice
  // through update_runtime_settings is a no-op.
  return dynamic_shapes_kernel_specialization_strategy == other.dynamic_shapes_kernel_specialization_strategy &&
      cuda_graph_strategy == other.cuda_graph_strategy && runtime_cache.get() == other.runtime_cache.get();
}

RuntimeSettings RuntimeSettings::merge(RuntimeSettings const& override) const {
  RuntimeSettings result = *this;
  result.dynamic_shapes_kernel_specialization_strategy = override.dynamic_shapes_kernel_specialization_strategy;
  result.cuda_graph_strategy = override.cuda_graph_strategy;
  if (override.runtime_cache) {
    result.runtime_cache = override.runtime_cache;
  }
  return result;
}

std::string RuntimeSettings::to_str() const {
  std::ostringstream os;
  os << "Dynamic Shapes Kernel Strategy: " << dynamic_shapes_kernel_specialization_strategy << std::endl;
  os << "CUDA Graph Strategy: " << cuda_graph_strategy << std::endl;
  if (runtime_cache) {
    auto p = runtime_cache->path();
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
