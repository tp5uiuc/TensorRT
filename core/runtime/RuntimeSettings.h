#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "ATen/core/ivalue.h"
#include "NvInfer.h"
#include "torch/custom_class.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

// A passive wrapper around an `IRuntimeCache`. Registered as a torchbind class so
// it can be passed by `c10::intrusive_ptr` across the Python/C++ boundary; the
// same handle gives both runtimes the same underlying `IRuntimeCache*`.
//
// File I/O lives exclusively on the Python side (filelock + serialize/deserialize
// via `trt.IRuntimeCache`). The C++ class is purely a holder; `path` is
// informational and is not consulted by the C++ runtime.
class RuntimeCacheHandle : public torch::CustomClassHolder {
 public:
  explicit RuntimeCacheHandle(std::string path = "") : path_(std::move(path)) {}

  [[nodiscard]] std::string path() const {
    return path_;
  }
  void set_path(std::string p) {
    path_ = std::move(p);
  }

#ifdef TRT_MAJOR_RTX
  // The actual TensorRT runtime cache. The first engine that attaches this handle
  // materializes it via `IRuntimeConfig::createRuntimeCache()` and writes the
  // shared_ptr here; subsequent engines reuse the same pointer for true sharing.
  std::shared_ptr<nvinfer1::IRuntimeCache> cache;
#endif

 private:
  std::string path_;
};

// Per-engine runtime-only knobs sampled at IExecutionContext creation.
//
// `RuntimeSettings` is a plain struct (not a torchbind class) because we flatten
// it into positional args at the torchbind boundary -- TorchBind can't carry a
// dataclass natively. Equality is value-by-value; the cache field compares
// by pointer identity (same handle -> same cache).
struct RuntimeSettings {
  std::string dynamic_shapes_kernel_specialization_strategy = "lazy";
  std::string cuda_graph_strategy = "disabled";
  c10::intrusive_ptr<RuntimeCacheHandle> runtime_cache = nullptr;

  bool operator==(RuntimeSettings const& other) const noexcept;
  bool operator!=(RuntimeSettings const& other) const noexcept {
    return !(*this == other);
  }

  // Apply `override`'s non-default fields on top of *this*, returning a new value.
  // For non-default detection on the strategy strings we always overlay; the cache
  // pointer is overlaid iff `override.runtime_cache` is non-null.
  RuntimeSettings merge(RuntimeSettings const& override) const;

  [[nodiscard]] std::string to_str() const;
};

std::ostream& operator<<(std::ostream& os, RuntimeSettings const& rs);

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
