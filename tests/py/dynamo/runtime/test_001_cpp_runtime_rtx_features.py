"""C++ runtime smoke tests for the three TensorRT-RTX runtime features.

These tests verify the C++ runtime path (``use_python_runtime=False``) wires up
the runtime cache, dynamic-shapes kernel specialization strategy, and native
CUDA graph strategy via the serialized engine info indices. The Python runtime
equivalents live in ``test_000_runtime_cache.py``,
``test_001_dynamic_shapes_kernel_strategy.py``, and
``test_001_cuda_graph_strategy.py`` and assert via Python attributes on
:class:`TRTEngine`; the C++ ``torch.classes.tensorrt.Engine`` does not expose
those attributes to Python, so this file asserts on externally observable
behavior (compilation succeeds, inference returns correct outputs, cache files
appear on disk).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import torch
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


def _compile_cpp(**extra_kwargs):
    """Compile :class:`SimpleModel` against the C++ runtime."""
    model = SimpleModel().eval().cuda()
    inputs = [torch.randn(2, 3).cuda()]
    kwargs = {
        "ir": "dynamo",
        "inputs": inputs,
        "use_python_runtime": False,
        "min_block_size": 1,
    }
    kwargs.update(extra_kwargs)
    compiled = torchtrt.compile(model, **kwargs)
    torch._dynamo.reset()
    return compiled, inputs, model


def _assert_cpp_runtime_used(testcase: TestCase, compiled) -> None:
    """Walk the compiled module and assert at least one C++ engine is present."""
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
    from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

    found_cpp = False
    for _, mod in compiled.named_modules():
        if isinstance(mod, TorchTensorRTModule):
            testcase.assertFalse(
                isinstance(mod.engine, TRTEngine),
                "C++ runtime expected but found Python TRTEngine",
            )
            found_cpp = True
    testcase.assertTrue(found_cpp, "No TorchTensorRTModule found in compiled graph")


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "C++ runtime is not available",
)
@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "RTX-only features require TensorRT-RTX",
)
class TestCppRuntimeStrategyValidation(TestCase):
    """Strategy-name typos must be rejected before engine construction."""

    def test_invalid_dynamic_shapes_strategy_rejected(self):
        with self.assertRaises(ValueError):
            _compile_cpp(dynamic_shapes_kernel_specialization_strategy="invalid")

    def test_invalid_cuda_graph_strategy_rejected(self):
        with self.assertRaises(ValueError):
            _compile_cpp(cuda_graph_strategy="invalid_strategy")


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "C++ runtime is not available",
)
@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "RTX-only features require TensorRT-RTX",
)
class TestCppRuntimeSmoke(TestCase):
    """End-to-end compile + infer + correctness on the C++ runtime."""

    def _run_and_check(self, compiled, inputs, model):
        ref = model(*inputs)
        out = compiled(*[inp.clone() for inp in inputs])
        sim = cosine_similarity(ref, out)
        self.assertGreaterEqual(
            sim,
            COSINE_THRESHOLD,
            f"C++ runtime output diverged from reference (cosine={sim})",
        )

    def test_default_settings(self):
        compiled, inputs, model = _compile_cpp()
        _assert_cpp_runtime_used(self, compiled)
        self._run_and_check(compiled, inputs, model)

    def test_eager_kernel_strategy(self):
        compiled, inputs, model = _compile_cpp(
            dynamic_shapes_kernel_specialization_strategy="eager"
        )
        _assert_cpp_runtime_used(self, compiled)
        self._run_and_check(compiled, inputs, model)

    def test_none_kernel_strategy(self):
        compiled, inputs, model = _compile_cpp(
            dynamic_shapes_kernel_specialization_strategy="none"
        )
        _assert_cpp_runtime_used(self, compiled)
        self._run_and_check(compiled, inputs, model)

    def test_whole_graph_capture(self):
        compiled, inputs, model = _compile_cpp(
            cuda_graph_strategy="whole_graph_capture"
        )
        _assert_cpp_runtime_used(self, compiled)
        self._run_and_check(compiled, inputs, model)


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "C++ runtime is not available",
)
@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "RTX-only features require TensorRT-RTX",
)
class TestCppRuntimeCachePersistence(TestCase):
    """Verify the C++ runtime writes a cache file on engine destruction."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.cache_dir, "runtime_cache.bin")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_cache_file_written_on_destruction(self):
        import gc

        compiled, inputs, model = _compile_cpp(runtime_cache_path=self.cache_path)
        _assert_cpp_runtime_used(self, compiled)
        # Run once so the runtime compiles some kernels.
        compiled(*[inp.clone() for inp in inputs])
        del compiled
        gc.collect()
        torch.cuda.synchronize()
        self.assertTrue(
            os.path.exists(self.cache_path),
            f"Runtime cache file not written to {self.cache_path}",
        )
        self.assertGreater(
            os.path.getsize(self.cache_path),
            0,
            "Runtime cache file is empty",
        )


if __name__ == "__main__":
    run_tests()
