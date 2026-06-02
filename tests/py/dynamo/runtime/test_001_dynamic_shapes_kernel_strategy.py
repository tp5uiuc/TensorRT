import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings

_STRATEGIES = [("lazy",), ("eager",), ("none",)]


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


class DynamicConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv2(torch.relu(self.conv1(x))))


_RUNTIMES = [("python", True), ("cpp", False)]


def _skip_if_cpp_unavailable(testcase, use_python_runtime):
    if not use_python_runtime and not ENABLED_FEATURES.torch_tensorrt_runtime:
        testcase.skipTest("C++ runtime is not available")


def _compile_dynamic_conv(strategy, *, use_python_runtime):
    """Compile DynamicConvModel through the selected runtime with the given strategy."""
    model = DynamicConvModel().eval().cuda()
    inp = torchtrt.Input(
        min_shape=(1, 3, 16, 16),
        opt_shape=(2, 3, 16, 16),
        max_shape=(4, 3, 16, 16),
        dtype=torch.float32,
    )
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=[inp],
        enabled_precisions={torch.float32},
        use_python_runtime=use_python_runtime,
        min_block_size=1,
        dynamic_shapes_kernel_specialization_strategy=strategy,
    )
    torch._dynamo.reset()
    return compiled


def _compile_simple(**extra_kwargs):
    """Helper: compile SimpleModel with dynamic shapes and Python runtime."""
    model = SimpleModel().eval().cuda()
    inputs = [
        torchtrt.Input(
            min_shape=(1, 3),
            opt_shape=(2, 3),
            max_shape=(4, 3),
            dtype=torch.float32,
        )
    ]
    kwargs = {
        "ir": "dynamo",
        "inputs": inputs,
        "use_python_runtime": True,
        "min_block_size": 1,
    }
    kwargs.update(extra_kwargs)
    compiled = torchtrt.compile(model, **kwargs)
    torch._dynamo.reset()
    return compiled


def _find_python_trt_engine(compiled):
    """Walk the compiled graph module and return the Python ``TRTEngine`` instance.

    The C++ and Python runtimes are now both driven through ``TorchTensorRTModule``;
    ``use_python_runtime=True`` causes ``module.engine`` to be a Python ``TRTEngine``.
    """
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
    from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

    for _, mod in compiled.named_modules():
        if isinstance(mod, TorchTensorRTModule) and isinstance(mod.engine, TRTEngine):
            return mod.engine
    return None


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Dynamic shapes kernel specialization strategy requires TensorRT-RTX",
)
class TestDynamicShapesKernelStrategySetup(TestCase):
    """Tests that the dynamic shapes kernel specialization strategy is correctly applied."""

    def test_default_strategy_is_lazy(self):
        import tensorrt as trt

        compiled = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine, "No Python TRTEngine found")
        self.assertIsNotNone(
            engine.runtime_config, "runtime_config should be set for RTX"
        )
        self.assertEqual(
            engine.runtime_config.dynamic_shapes_kernel_specialization_strategy,
            trt.DynamicShapesKernelSpecializationStrategy.LAZY,
        )

    def test_eager_strategy(self):
        import tensorrt as trt

        compiled = _compile_simple(
            dynamic_shapes_kernel_specialization_strategy="eager"
        )
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        self.assertEqual(
            engine.runtime_config.dynamic_shapes_kernel_specialization_strategy,
            trt.DynamicShapesKernelSpecializationStrategy.EAGER,
        )

    def test_none_strategy(self):
        import tensorrt as trt

        compiled = _compile_simple(dynamic_shapes_kernel_specialization_strategy="none")
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        self.assertEqual(
            engine.runtime_config.dynamic_shapes_kernel_specialization_strategy,
            trt.DynamicShapesKernelSpecializationStrategy.NONE,
        )

    def test_context_created_with_each_strategy(self):
        for strategy in ("lazy", "eager", "none"):
            with self.subTest(strategy=strategy):
                compiled = _compile_simple(
                    dynamic_shapes_kernel_specialization_strategy=strategy
                )
                engine = _find_python_trt_engine(compiled)
                self.assertIsNotNone(
                    engine.context,
                    f"Execution context should be created for {strategy}",
                )
                # Test inference with multiple dynamic batch sizes
                for bs in (1, 2, 4):
                    output = compiled(torch.randn(bs, 3).cuda())
                    self.assertEqual(output.shape, (bs, 3))

    def test_setting_in_compilation_settings(self):
        for strategy in ("lazy", "eager", "none"):
            settings = CompilationSettings(
                dynamic_shapes_kernel_specialization_strategy=strategy
            )
            self.assertEqual(
                settings.dynamic_shapes_kernel_specialization_strategy, strategy
            )

    def test_default_compilation_settings(self):
        settings = CompilationSettings()
        self.assertEqual(settings.dynamic_shapes_kernel_specialization_strategy, "lazy")


@unittest.skipIf(
    ENABLED_FEATURES.tensorrt_rtx,
    "This test verifies standard TRT behavior (non-RTX)",
)
class TestDynamicShapesKernelStrategyNonRTX(TestCase):
    """Tests that the setting is ignored on non-RTX builds."""

    def test_setting_ignored_on_non_rtx(self):
        compiled = _compile_simple(
            dynamic_shapes_kernel_specialization_strategy="eager"
        )
        engine = _find_python_trt_engine(compiled)
        if engine is not None:
            self.assertIsNone(
                engine.runtime_config,
                "runtime_config should be None for standard TRT",
            )
        # Inference should still work
        output = compiled(torch.randn(2, 3).cuda())
        self.assertEqual(output.shape, (2, 3))


_STRATEGY_RUNTIME_MATRIX = [
    (strategy, runtime_name, use_python_runtime)
    for (strategy,) in _STRATEGIES
    for (runtime_name, use_python_runtime) in _RUNTIMES
]


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Dynamic shapes kernel strategy is a TensorRT-RTX feature",
)
class TestDynamicShapesKernelStrategyInference(TestCase):
    """End-to-end: compile + infer with each strategy on both runtime paths."""

    @parameterized.expand(_STRATEGY_RUNTIME_MATRIX)
    def test_strategy_inference(self, strategy, _runtime_name, use_python_runtime):
        _skip_if_cpp_unavailable(self, use_python_runtime)
        compiled = _compile_dynamic_conv(
            strategy, use_python_runtime=use_python_runtime
        )
        x = torch.randn(2, 3, 16, 16, device="cuda")
        y = compiled(x)
        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())

    @parameterized.expand(_RUNTIMES)
    def test_dynamic_shape_with_eager(self, _name, use_python_runtime):
        """Exercise shape changes under eager kernel specialization."""
        _skip_if_cpp_unavailable(self, use_python_runtime)
        compiled = _compile_dynamic_conv("eager", use_python_runtime=use_python_runtime)
        for batch in (1, 2, 3, 4):
            x = torch.randn(batch, 3, 16, 16, device="cuda")
            y = compiled(x)
            self.assertEqual(tuple(y.shape), (batch, 8, 16, 16))


class TestDynamicShapesKernelStrategyInvalidValue(TestCase):
    """Invalid strategy names are rejected at TorchTensorRTModule.__init__ on any backend."""

    @parameterized.expand(_RUNTIMES)
    def test_invalid_strategy_raises(self, _name, use_python_runtime):
        _skip_if_cpp_unavailable(self, use_python_runtime)
        model = DynamicConvModel().eval().cuda()
        inp = torchtrt.Input(
            min_shape=(1, 3, 16, 16),
            opt_shape=(2, 3, 16, 16),
            max_shape=(4, 3, 16, 16),
            dtype=torch.float32,
        )
        with self.assertRaises((ValueError, RuntimeError)):
            torchtrt.compile(
                model,
                ir="dynamo",
                inputs=[inp],
                enabled_precisions={torch.float32},
                use_python_runtime=use_python_runtime,
                min_block_size=1,
                dynamic_shapes_kernel_specialization_strategy="not_a_real_strategy",
            )


if __name__ == "__main__":
    run_tests()
