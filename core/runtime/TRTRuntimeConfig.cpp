#include "core/runtime/TRTRuntimeConfig.h"

#include <sstream>
#include <stdexcept>

#include "core/runtime/RuntimeSettings.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

namespace {

#ifdef TRT_MAJOR_RTX
[[nodiscard]] nvinfer1::DynamicShapesKernelSpecializationStrategy to_trt_ds_strategy(std::string const& s) {
  if (s == "lazy") {
    return nvinfer1::DynamicShapesKernelSpecializationStrategy::kLAZY;
  }
  if (s == "eager") {
    return nvinfer1::DynamicShapesKernelSpecializationStrategy::kEAGER;
  }
  if (s == "none") {
    return nvinfer1::DynamicShapesKernelSpecializationStrategy::kNONE;
  }
  TORCHTRT_CHECK(
      false, "Invalid dynamic_shapes_kernel_specialization_strategy: \"" << s << "\" (expected lazy | eager | none)");
}

[[nodiscard]] nvinfer1::CudaGraphStrategy to_trt_cg_strategy(std::string const& s) {
  if (s == "disabled") {
    return nvinfer1::CudaGraphStrategy::kDISABLED;
  }
  if (s == "whole_graph_capture") {
    return nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE;
  }
  TORCHTRT_CHECK(false, "Invalid cuda_graph_strategy: \"" << s << "\" (expected disabled | whole_graph_capture)");
}
#endif

} // namespace

void TRTRuntimeConfig::ensure_initialized(
    TORCHTRT_UNUSED nvinfer1::ICudaEngine* cuda_engine,
    TORCHTRT_UNUSED RuntimeSettings const& rs) {
#ifdef TRT_HAS_IRUNTIME_CONFIG
  if (!config) {
    TORCHTRT_CHECK(cuda_engine != nullptr, "Cannot initialize TRTRuntimeConfig without a live ICudaEngine");
    config = make_trt(cuda_engine->createRuntimeConfig());
    TORCHTRT_CHECK(config.get() != nullptr, "Unable to create TensorRT IRuntimeConfig");
  }

#ifdef TRT_MAJOR_RTX
  // Runtime cache: ONLY attach when the caller provided an external
  // RuntimeCacheHandle. The Python TRTEngine side creates an implicit
  // handle from a path string and passes it in via the handle; without
  // an explicit user opt-in we leave the IRuntimeConfig cache-less.
  if (rs.runtime_cache) {
    if (!rs.runtime_cache->cache) {
      rs.runtime_cache->cache = make_trt(config->createRuntimeCache());
      TORCHTRT_CHECK(
          rs.runtime_cache->cache.get() != nullptr, "Failed to create IRuntimeCache for shared RuntimeCacheHandle");
    }
    if (config->setRuntimeCache(*rs.runtime_cache->cache)) {
      LOG_DEBUG("Attached external IRuntimeCache to IRuntimeConfig.");
    } else {
      LOG_WARNING("Failed to attach IRuntimeCache to IRuntimeConfig; cache will be unused.");
    }
  } else {
    LOG_DEBUG("Runtime cache disabled (no RuntimeCacheHandle provided).");
  }

  config->setDynamicShapesKernelSpecializationStrategy(
      to_trt_ds_strategy(rs.dynamic_shapes_kernel_specialization_strategy));
  LOG_DEBUG(
      "Dynamic shapes kernel specialization strategy set to " << rs.dynamic_shapes_kernel_specialization_strategy);

  if (!config->setCudaGraphStrategy(to_trt_cg_strategy(rs.cuda_graph_strategy))) {
    LOG_WARNING("Failed to set CUDA graph strategy; continuing with default.");
  }
#endif
#endif // TRT_HAS_IRUNTIME_CONFIG
}

void TRTRuntimeConfig::reset() {
#ifdef TRT_HAS_IRUNTIME_CONFIG
  config.reset();
#endif
}

std::shared_ptr<nvinfer1::IExecutionContext> TRTRuntimeConfig::create_execution_context(
    nvinfer1::ICudaEngine* cuda_engine,
    RuntimeSettings const& rs,
    nvinfer1::ExecutionContextAllocationStrategy allocation_strategy) {
  ensure_initialized(cuda_engine, rs);
#ifdef TRT_HAS_IRUNTIME_CONFIG
  config->setExecutionContextAllocationStrategy(allocation_strategy);
  return make_trt(cuda_engine->createExecutionContext(config.get()));
#else
  // Pre-10.11 TRT (e.g. Jetpack): use the legacy strategy overload directly.
  return make_trt(cuda_engine->createExecutionContext(allocation_strategy));
#endif
}

bool TRTRuntimeConfig::uses_internal_capture(
    TORCHTRT_UNUSED RuntimeSettings const& rs,
    TORCHTRT_UNUSED bool cudagraphs_enabled) noexcept {
#ifdef TRT_MAJOR_RTX
  // On TRT-RTX the internal runtime handles capture/replay whenever a non-disabled
  // strategy is set, or when subgraph cudagraphs are enabled globally. In both
  // cases the caller should skip its manual at::cuda::CUDAGraph wrapper.
  return rs.cuda_graph_strategy != "disabled" || cudagraphs_enabled;
#else
  return false;
#endif
}

bool TRTRuntimeConfig::is_monolithic_capturable(
    TORCHTRT_UNUSED RuntimeSettings const& rs,
    TORCHTRT_UNUSED bool has_dynamic_inputs,
    TORCHTRT_UNUSED nvinfer1::IExecutionContext* exec_ctx,
    TORCHTRT_UNUSED cudaStream_t stream) noexcept {
#ifdef TRT_MAJOR_RTX
  TORCHTRT_ASSERT(exec_ctx != nullptr, "is_monolithic_capturable requires a live IExecutionContext");
  if (!exec_ctx->isStreamCapturable(stream)) {
    return false;
  }
  // "lazy" kernel specialization only swaps specialized kernels mid-run when an
  // input has a dynamic dimension; for static-shape engines the kernels are fixed
  // at setup and the captured graph stays valid. Mirrors the Python check.
  return !(rs.dynamic_shapes_kernel_specialization_strategy == "lazy" && has_dynamic_inputs);
#else
  return true;
#endif
}

std::ostream& operator<<(std::ostream& os, const TRTRuntimeConfig& cfg) {
  os << "TRTRuntimeConfig{";
#ifdef TRT_HAS_IRUNTIME_CONFIG
  os << "config=" << (cfg.config ? "live" : "null");
#endif
  os << "}";
  return os;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
