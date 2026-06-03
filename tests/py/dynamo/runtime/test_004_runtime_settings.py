"""Whitebox tests for the RuntimeSettings data model + dispatch."""

import dataclasses
import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.runtime import (
    RuntimeCacheHandle,
    RuntimeSettings,
    runtime_config,
)


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


def _compile_simple(*, runtime_settings=None, use_python_runtime=True):
    model = SimpleModel().eval().cuda()
    inputs = [
        torchtrt.Input(
            min_shape=(1, 3),
            opt_shape=(2, 3),
            max_shape=(4, 3),
            dtype=torch.float32,
        )
    ]
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        use_python_runtime=use_python_runtime,
        min_block_size=1,
        runtime_settings=runtime_settings,
    )
    torch._dynamo.reset()
    return compiled


_RUNTIMES = [("python", True), ("cpp", False)]


def _skip_if_cpp_unavailable(testcase, use_python_runtime):
    if not use_python_runtime and not ENABLED_FEATURES.torch_tensorrt_runtime:
        testcase.skipTest("C++ runtime is not available")


class TestRuntimeSettingsDataModel(TestCase):
    """Pure dataclass behavior; no engine compile required."""

    def test_defaults_are_valid(self):
        rs = RuntimeSettings()
        self.assertEqual(rs.dynamic_shapes_kernel_specialization_strategy, "lazy")
        self.assertEqual(rs.cuda_graph_strategy, "disabled")
        self.assertIsNone(rs.runtime_cache)

    def test_invalid_ds_strategy_raises_at_post_init(self):
        with self.assertRaises(ValueError):
            RuntimeSettings(dynamic_shapes_kernel_specialization_strategy="bogus")

    def test_invalid_cg_strategy_raises_at_post_init(self):
        with self.assertRaises(ValueError):
            RuntimeSettings(cuda_graph_strategy="bogus")

    def test_frozen(self):
        rs = RuntimeSettings()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rs.cuda_graph_strategy = "whole_graph_capture"

    def test_merge_overrides(self):
        rs = RuntimeSettings()
        new = rs.merge(cuda_graph_strategy="whole_graph_capture")
        self.assertEqual(new.cuda_graph_strategy, "whole_graph_capture")
        # Original unchanged (frozen + replace).
        self.assertEqual(rs.cuda_graph_strategy, "disabled")

    def test_merge_unknown_key_raises(self):
        rs = RuntimeSettings()
        with self.assertRaises(TypeError):
            rs.merge(not_a_real_field=True)

    def test_equality_compares_all_fields(self):
        a = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        b = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        c = RuntimeSettings(cuda_graph_strategy="disabled")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_runtime_cache_as_path_string(self):
        rs = RuntimeSettings(runtime_cache="/tmp/whatever.bin")
        self.assertEqual(rs.runtime_cache, "/tmp/whatever.bin")


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "RuntimeSettings dispatch is exercised on TRT-RTX",
)
class TestRuntimeSettingsCompileTimeHint(TestCase):
    """Verify the compile-time hint primes the engine without a CM."""

    def test_compile_hint_sets_engine_settings(self):
        rs = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        compiled = _compile_simple(runtime_settings=rs)
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        for _, mod in compiled.named_modules():
            if isinstance(mod, TorchTensorRTModule):
                self.assertEqual(
                    mod.runtime_settings.cuda_graph_strategy, "whole_graph_capture"
                )

    def test_runtime_config_cm_restores_on_exit(self):
        compiled = _compile_simple()
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        mod = next(
            m for _, m in compiled.named_modules() if isinstance(m, TorchTensorRTModule)
        )
        prior = mod.runtime_settings
        with runtime_config(compiled, cuda_graph_strategy="whole_graph_capture"):
            self.assertEqual(
                mod.runtime_settings.cuda_graph_strategy, "whole_graph_capture"
            )
        self.assertEqual(mod.runtime_settings, prior)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Multi-target tests require TRT-RTX",
)
class TestMultiTargetRuntimeConfig(TestCase):
    """`runtime_config([a, b], ...)` applies to engines under both targets."""

    def test_multi_target_runtime_config(self):
        model_a = _compile_simple()
        model_b = _compile_simple()
        with runtime_config(
            [model_a, model_b], cuda_graph_strategy="whole_graph_capture"
        ) as (m_a, m_b):
            self.assertIs(m_a, model_a)
            self.assertIs(m_b, model_b)
            from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
                TorchTensorRTModule,
            )

            for target in (model_a, model_b):
                for _, mod in target.named_modules():
                    if isinstance(mod, TorchTensorRTModule):
                        self.assertEqual(
                            mod.runtime_settings.cuda_graph_strategy,
                            "whole_graph_capture",
                        )


class TestRuntimeConfigInvalidKey(TestCase):
    """Typo in a CM key should raise at construction, not silently no-op."""

    def test_unknown_kwarg_raises(self):
        # Use a Module that's not a TorchTensorRTModule -- we just need the
        # CM constructor to run; __enter__ won't find any engines.
        target = torch.nn.Linear(3, 3)
        with self.assertRaises(TypeError):
            runtime_config(target, not_a_real_field=True)


if __name__ == "__main__":
    run_tests()
