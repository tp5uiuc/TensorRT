#include "core/runtime/TRTRuntimeConfig.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

std::string to_string(DynamicShapesKernelStrategy s) {
  switch (s) {
    case DynamicShapesKernelStrategy::kLazy:
      return "lazy";
    case DynamicShapesKernelStrategy::kEager:
      return "eager";
    case DynamicShapesKernelStrategy::kNone:
      return "none";
  }
  return "unknown";
}

std::string to_string(CudaGraphStrategyOption s) {
  switch (s) {
    case CudaGraphStrategyOption::kDisabled:
      return "disabled";
    case CudaGraphStrategyOption::kWholeGraphCapture:
      return "whole_graph_capture";
  }
  return "unknown";
}

DynamicShapesKernelStrategy to_dynamic_shapes_kernel_strategy(int v) {
  TORCHTRT_CHECK(
      v >= 0 && v <= 2,
      "Invalid dynamic shapes kernel strategy value: " << v << ". Expected 0 (lazy), 1 (eager), or 2 (none).");
  return static_cast<DynamicShapesKernelStrategy>(v);
}

CudaGraphStrategyOption to_cuda_graph_strategy_option(int v) {
  TORCHTRT_CHECK(
      v >= 0 && v <= 1,
      "Invalid CUDA graph strategy value: " << v << ". Expected 0 (disabled) or 1 (whole_graph_capture).");
  return static_cast<CudaGraphStrategyOption>(v);
}

void TRTRuntimeConfig::ensure_initialized(nvinfer1::ICudaEngine* cuda_engine) {
  if (config) {
    return;
  }
  TORCHTRT_CHECK(cuda_engine != nullptr, "Cannot initialize TRTRuntimeConfig without a live ICudaEngine");
  config = make_trt(cuda_engine->createRuntimeConfig());
  TORCHTRT_CHECK(config.get() != nullptr, "Unable to create TensorRT IRuntimeConfig");

#ifdef TRT_MAJOR_RTX
  // Runtime cache -- TRT-RTX only.
  if (!runtime_cache_path.empty()) {
    runtime_cache = make_trt(config->createRuntimeCache());
    if (runtime_cache.get() == nullptr) {
      LOG_WARNING("Failed to create TensorRT IRuntimeCache; runtime cache will be skipped.");
    } else {
      load_runtime_cache_nothrow();
      bool ok = config->setRuntimeCache(*runtime_cache);
      if (!ok) {
        LOG_WARNING("Failed to attach runtime cache to IRuntimeConfig; cache will be unused.");
        runtime_cache.reset();
      } else {
        LOG_DEBUG("TensorRT-RTX runtime cache configured at " << runtime_cache_path);
      }
    }
  } else {
    LOG_DEBUG("Runtime cache disabled (no path configured).");
  }

  // Dynamic shapes kernel specialization strategy -- TRT-RTX only.
  config->setDynamicShapesKernelSpecializationStrategy(
      static_cast<nvinfer1::DynamicShapesKernelSpecializationStrategy>(dynamic_shapes_kernel_strategy));
  LOG_DEBUG("Dynamic shapes kernel specialization strategy set to " << to_string(dynamic_shapes_kernel_strategy));

  // CUDA graph strategy -- TRT-RTX only.
  bool ok = config->setCudaGraphStrategy(
      cuda_graph_strategy == CudaGraphStrategyOption::kWholeGraphCapture
          ? nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE
          : nvinfer1::CudaGraphStrategy::kDISABLED);
  if (!ok) {
    LOG_WARNING("Failed to set CUDA graph strategy; continuing with default.");
  }
#endif
}

void TRTRuntimeConfig::set_execution_context_allocation_strategy(
    nvinfer1::ExecutionContextAllocationStrategy strategy) {
  TORCHTRT_ASSERT(config, "TRTRuntimeConfig::config must be initialized before setting allocation strategy");
  config->setExecutionContextAllocationStrategy(strategy);
}

bool TRTRuntimeConfig::uses_internal_capture(bool cudagraphs_enabled) const {
#ifdef TRT_MAJOR_RTX
  // On TRT-RTX the internal runtime handles capture/replay whenever a non-disabled
  // strategy is set, or when subgraph cudagraphs are enabled globally. In both cases the
  // caller should skip its manual at::cuda::CUDAGraph wrapper because TRT-RTX's internal
  // capture would collide with it.
  return cuda_graph_strategy != CudaGraphStrategyOption::kDisabled || cudagraphs_enabled;
#else
  (void)cudagraphs_enabled;
  return false;
#endif
}

void TRTRuntimeConfig::disable_rtx_native_cudagraphs(const std::string& engine_name) noexcept {
#ifdef TRT_MAJOR_RTX
  if (rtx_native_cudagraphs_disabled || cuda_graph_strategy == CudaGraphStrategyOption::kDisabled) {
    return;
  }
  LOG_WARNING(
      "Outer CUDA stream capture detected; disabling TensorRT-RTX native CUDA graph strategy on engine "
      << engine_name << " for the remainder of its lifetime.");
  // Persist any kernels the engine-internal capture has compiled so far; the outer
  // capture will run without them otherwise, and we want future reloads to reuse them.
  save_runtime_cache_nothrow();
  cuda_graph_strategy = CudaGraphStrategyOption::kDisabled;
  if (config) {
    bool ok = config->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kDISABLED);
    if (!ok) {
      LOG_WARNING("Failed to update CUDA graph strategy on IRuntimeConfig after disable.");
    }
  }
  rtx_native_cudagraphs_disabled = true;
#else
  (void)engine_name;
#endif
}

bool TRTRuntimeConfig::is_monolithic_capturable(nvinfer1::IExecutionContext* exec_ctx, cudaStream_t stream) const {
#if defined(TRT_MAJOR_RTX) && defined(ENABLE_FEATURE_DISABLE_RUNTIME_ALLOCATION)
  TORCHTRT_ASSERT(exec_ctx != nullptr, "is_monolithic_capturable requires a live IExecutionContext");
  // "lazy" kernel specialization swaps specialized kernels in mid-run, which invalidates
  // captured graphs. Other strategies (eager/none) are safe when the context reports the
  // stream capturable.
  return exec_ctx->isStreamCapturable(stream) && dynamic_shapes_kernel_strategy != DynamicShapesKernelStrategy::kLazy;
#else
  // isStreamCapturable is declared inside `#if ENABLE_FEATURE_DISABLE_RUNTIME_ALLOCATION`
  // in the TensorRT-RTX header; conservatively assume the engine is capturable when that
  // feature flag is not enabled at compile time.
  (void)exec_ctx;
  (void)stream;
  return true;
#endif
}

void TRTRuntimeConfig::load_runtime_cache_nothrow() noexcept {
#ifdef TRT_MAJOR_RTX
  TORCHTRT_ASSERT(runtime_cache, "load_runtime_cache_nothrow requires runtime_cache to be initialized");
  if (runtime_cache_path.empty()) {
    return;
  }
  try {
    if (!std::filesystem::exists(runtime_cache_path)) {
      LOG_DEBUG("No existing runtime cache at " << runtime_cache_path);
      return;
    }
    std::ifstream f(runtime_cache_path, std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.empty()) {
      return;
    }
    bool ok = runtime_cache->deserialize(buf.data(), buf.size());
    if (ok) {
      LOG_INFO("Loaded runtime cache from " << runtime_cache_path << " (" << buf.size() << " bytes)");
    } else {
      LOG_WARNING("runtime_cache->deserialize returned false for " << runtime_cache_path);
    }
  } catch (const std::exception& e) {
    LOG_WARNING("Failed to load runtime cache: " << e.what());
  } catch (...) {
    LOG_WARNING("Failed to load runtime cache (unknown exception).");
  }
#endif
}

void TRTRuntimeConfig::save_runtime_cache_nothrow() noexcept {
#ifdef TRT_MAJOR_RTX
  if (!runtime_cache || runtime_cache_path.empty()) {
    return;
  }
  try {
    auto host_mem = make_trt(runtime_cache->serialize());
    if (!host_mem || host_mem->size() == 0) {
      return;
    }
    std::filesystem::path path(runtime_cache_path);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::filesystem::path tmp_path = path;
    tmp_path += ".tmp";
    {
      std::ofstream out(tmp_path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(host_mem->data()), host_mem->size());
    }
    std::filesystem::rename(tmp_path, path);
    LOG_INFO("Saved runtime cache to " << runtime_cache_path << " (" << host_mem->size() << " bytes)");
  } catch (const std::exception& e) {
    LOG_WARNING("Failed to save runtime cache: " << e.what());
  } catch (...) {
    LOG_WARNING("Failed to save runtime cache (unknown exception).");
  }
#endif
}

void TRTRuntimeConfig::write_to_str(std::ostream& os) const {
  os << "  Runtime Cache Path: " << (runtime_cache_path.empty() ? "<disabled>" : runtime_cache_path) << std::endl;
  os << "  Dynamic Shapes Kernel Strategy: " << to_string(dynamic_shapes_kernel_strategy) << std::endl;
  os << "  CUDA Graph Strategy: " << to_string(cuda_graph_strategy) << std::endl;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
