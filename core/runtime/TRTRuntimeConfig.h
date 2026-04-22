#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>

#include "NvInfer.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

// TensorRT-RTX-only configuration for how shape-specialized kernels are compiled.
enum class DynamicShapesKernelStrategy : int32_t {
  kLazy = 0,
  kEager = 1,
  kNone = 2,
};

// TensorRT-RTX-only configuration for how CUDA graph capture/replay is handled.
enum class CudaGraphStrategyOption : int32_t {
  kDisabled = 0,
  kWholeGraphCapture = 1,
};

std::string to_string(DynamicShapesKernelStrategy s);
std::string to_string(CudaGraphStrategyOption s);
DynamicShapesKernelStrategy to_dynamic_shapes_kernel_strategy(int v);
CudaGraphStrategyOption to_cuda_graph_strategy_option(int v);

// Encapsulates the nvinfer1::IRuntimeConfig owned by a TRTEngine along with the
// TensorRT-RTX-specific state (runtime cache, dynamic shapes kernel strategy, native
// CUDA graph strategy). All `#ifdef TRT_MAJOR_RTX` guards live in this file and its
// implementation so callers can treat this struct uniformly between RTX and standard
// TensorRT builds.
struct TRTRuntimeConfig {
  // Settings - typically populated from engine deserialization before `ensure_initialized`.
  std::string runtime_cache_path = "";
  DynamicShapesKernelStrategy dynamic_shapes_kernel_strategy = DynamicShapesKernelStrategy::kLazy;
  CudaGraphStrategyOption cuda_graph_strategy = CudaGraphStrategyOption::kDisabled;

  // One-shot: set to true once an outer stream capture has been detected and the
  // engine-internal CUDA graph strategy has been disabled for the remainder of the
  // owning engine's lifetime.
  bool rtx_native_cudagraphs_disabled = false;

  // Live resources. The IRuntimeConfig is lazy-constructed on first `ensure_initialized`.
  std::shared_ptr<nvinfer1::IRuntimeConfig> config;
#ifdef TRT_MAJOR_RTX
  std::shared_ptr<nvinfer1::IRuntimeCache> runtime_cache;
#endif

  // Construct the IRuntimeConfig once and apply all TRT-RTX-specific settings. Safe to
  // call multiple times; only the first call initializes and applies the RTX-only
  // setters. On subsequent calls this is a no-op.
  void ensure_initialized(nvinfer1::ICudaEngine* cuda_engine);

  // Apply (or re-apply) the execution context allocation strategy on the IRuntimeConfig.
  // Available on both standard TensorRT and TensorRT-RTX via IRuntimeConfig.
  void set_execution_context_allocation_strategy(nvinfer1::ExecutionContextAllocationStrategy strategy);

  // Returns true if the TensorRT-RTX runtime owns capture/replay for this engine so the
  // caller should bypass its own at::cuda::CUDAGraph capture around enqueueV3. Always
  // false on non-RTX builds.
  bool uses_internal_capture(bool cudagraphs_enabled) const;

  // One-shot: disable engine-internal CUDA graph capture. Invoked when an outer stream
  // capture is detected around execute_engine, so the outer capture can contain the
  // kernel launches directly. Saves the runtime cache before recreating the context so
  // compiled kernels from the present run are preserved for future reloads.
  void disable_rtx_native_cudagraphs(const std::string& engine_name) noexcept;

  // Whether the execution context is safe to include in an outer monolithic capture.
  // Non-RTX builds always return true.
  bool is_monolithic_capturable(nvinfer1::IExecutionContext* exec_ctx, cudaStream_t stream) const;

  // Load the runtime cache from disk using std::filesystem. No-throw: errors log and
  // return. Invoked internally from `ensure_initialized` when a cache path is set.
  void load_runtime_cache_nothrow() noexcept;

  // Save the runtime cache to disk using std::filesystem (tmp + rename). No-throw:
  // errors log and return, so it is safe to call from a destructor.
  void save_runtime_cache_nothrow() noexcept;

  // Append a human-readable summary to a TRTEngine::to_str stream.
  void write_to_str(std::ostream& os) const;
};

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
