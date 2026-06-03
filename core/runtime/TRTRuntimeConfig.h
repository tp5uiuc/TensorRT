#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <ostream>
#include <string>

#include "NvInfer.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

struct RuntimeSettings;

// Owns the live `IRuntimeConfig` (where supported) and the engine-local fallback
// `IRuntimeCache` used when no external `RuntimeCacheHandle` is attached via
// `RuntimeSettings`. The settings themselves (strategy strings, runtime_cache
// handle) live on `RuntimeSettings`; this struct applies them to TRT at
// `ensure_initialized` time.
//
// `IRuntimeConfig` and runtime-cache `#ifdef`s are confined to this TU.
struct TRTRuntimeConfig {
  // Lazy-constructed live config. `nullptr` until first `ensure_initialized`.
#ifdef TRT_HAS_IRUNTIME_CONFIG
  std::shared_ptr<nvinfer1::IRuntimeConfig> config;
#endif

  // (Re)build the `IRuntimeConfig` from `rs`. Idempotent only if the previous
  // `rs` was identical. Callers ensure the engine is the same across calls --
  // we don't memoize against `cuda_engine` here.
  void ensure_initialized(nvinfer1::ICudaEngine* cuda_engine, RuntimeSettings const& rs);

  // Force the next `ensure_initialized` to rebuild from scratch. Used when
  // settings change at runtime.
  void reset();

  // Lazy-init + create a fresh `IExecutionContext` honoring `allocation_strategy`.
  // Picks the right `createExecutionContext` overload (IRuntimeConfig* vs
  // ExecutionContextAllocationStrategy) so callers stay free of any
  // `TRT_HAS_IRUNTIME_CONFIG` branching.
  [[nodiscard]] std::shared_ptr<nvinfer1::IExecutionContext> create_execution_context(
      nvinfer1::ICudaEngine* cuda_engine,
      RuntimeSettings const& rs,
      nvinfer1::ExecutionContextAllocationStrategy allocation_strategy);

  // Returns true if TRT-RTX owns capture/replay for the given settings -- caller
  // should then bypass its own `at::cuda::CUDAGraph` capture around enqueueV3.
  // Always false on non-RTX builds.
  [[nodiscard]] static bool uses_internal_capture(RuntimeSettings const& rs, bool cudagraphs_enabled) noexcept;

  // Returns true iff the execution context can be safely included in an outer
  // monolithic capture. Non-RTX builds always return true.
  [[nodiscard]] static bool is_monolithic_capturable(
      RuntimeSettings const& rs,
      bool has_dynamic_inputs,
      nvinfer1::IExecutionContext* exec_ctx,
      cudaStream_t stream) noexcept;
};

std::ostream& operator<<(std::ostream& os, const TRTRuntimeConfig& cfg);

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
