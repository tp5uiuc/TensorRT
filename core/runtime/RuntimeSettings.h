#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "ATen/core/Tensor.h"
#include "ATen/core/ivalue.h"
#include "NvInfer.h"
#include "torch/custom_class.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

// A passive wrapper around an ``IRuntimeCache``. Registered as a torchbind class
// so it can be passed by ``c10::intrusive_ptr`` across the Python/C++ boundary;
// the same handle gives both runtimes the same underlying ``IRuntimeCache*``.
//
// File I/O lives on the Python side (filelock + on-disk persistence via
// the ``serialize`` / ``deserialize`` members below). The C++ struct is purely
// a holder; ``path`` is informational and is not consulted by the C++ runtime.
struct RuntimeCacheHandle : public torch::CustomClassHolder {
  std::string path;

#ifdef TRT_MAJOR_RTX
  // The actual TensorRT runtime cache. The first engine that attaches this handle
  // materializes it via ``IRuntimeConfig::createRuntimeCache()`` and writes the
  // shared_ptr here; subsequent engines reuse the same pointer for true sharing.
  std::shared_ptr<nvinfer1::IRuntimeCache> cache;
#endif

  explicit RuntimeCacheHandle(std::string p = "") : path(std::move(p)) {}

  // Expose the underlying ``IRuntimeCache`` bytes for the Python side to persist
  // under filelock. Returns an empty uint8 tensor when no cache is attached, or
  // on non-RTX builds.
  //
  // ``at::Tensor`` is used (rather than ``std::string``) because TorchBind
  // forces ``std::string`` to round-trip through Python ``str`` (UTF-8), and
  // serialized cache bytes are not valid UTF-8.
  [[nodiscard]] at::Tensor serialize() const;

  // Inverse of ``serialize``. Expects a uint8 ``at::Tensor``. No-op for empty
  // input, when the underlying ``IRuntimeCache`` has not been materialized yet,
  // or on non-RTX builds.
  void deserialize(at::Tensor data);

  // True iff an engine has populated the underlying ``IRuntimeCache``.
  // Always false on non-RTX builds.
  [[nodiscard]] bool has_cache() const;
};

// Per-engine runtime-only knobs sampled at IExecutionContext creation.
//
// ``RuntimeSettings`` is a plain struct (not a torchbind class) because we
// flatten it into positional args at the torchbind boundary -- TorchBind can't
// carry a dataclass natively. Equality is value-by-value; the cache field
// compares by pointer identity (same handle -> same cache).
//
// The strategy fields are ``int32_t`` rather than strings: the Python user-
// facing API takes strings (``"lazy" | "eager" | "none"`` etc.) and validates
// them at the Python boundary, then converts to ints before crossing into
// C++. On the C++ side the ints are ABI-compatible with the corresponding
// ``nvinfer1::DynamicShapesKernelSpecializationStrategy`` /
// ``nvinfer1::CudaGraphStrategy`` enum values, so applying them is just a
// ``static_cast`` (see ``TRTRuntimeConfig::ensure_initialized``). For
// human-readable logging use ``ds_strategy_name`` / ``cg_strategy_name``.
struct RuntimeSettings {
  // Mirror of nvinfer1::DynamicShapesKernelSpecializationStrategy on RTX.
  // 0 = lazy (default), 1 = eager, 2 = none.
  int32_t dynamic_shapes_kernel_specialization_strategy = 0;
  // Mirror of nvinfer1::CudaGraphStrategy on RTX.
  // 0 = disabled (default), 1 = whole_graph_capture.
  int32_t cuda_graph_strategy = 0;
  c10::intrusive_ptr<RuntimeCacheHandle> runtime_cache = nullptr;

  bool operator==(RuntimeSettings const& other) const noexcept;
  bool operator!=(RuntimeSettings const& other) const noexcept {
    return !(*this == other);
  }

  [[nodiscard]] std::string to_str() const;
};

// Reverse-lookup helpers used by ``to_str`` and ``operator<<``. Out-of-range
// values render as ``"<unknown>"``. Defined here so other translation units
// (e.g. ``TRTEngine.cpp`` for ``LOG_DEBUG``) can use the same mapping.
[[nodiscard]] std::string ds_strategy_name(int32_t v);
[[nodiscard]] std::string cg_strategy_name(int32_t v);

std::ostream& operator<<(std::ostream& os, RuntimeSettings const& rs);

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
