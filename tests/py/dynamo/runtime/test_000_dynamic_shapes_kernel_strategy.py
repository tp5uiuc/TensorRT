import unittest

import torch
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._defaults import (
    DYNAMIC_SHAPES_KERNEL_SPECIALIZATION_STRATEGY,
)
from torch_tensorrt.dynamo._settings import CompilationSettings


class DynamicConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv2(torch.relu(self.conv1(x))))


def _compile_cpp(strategy):
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
        use_python_runtime=False,
        min_block_size=1,
        dynamic_shapes_kernel_specialization_strategy=strategy,
    )
    torch._dynamo.reset()
    return compiled


class TestDynamicShapesKernelStrategySettings(TestCase):
    """Setting-level validation that runs on every build (RTX and non-RTX)."""

    def test_default_value(self):
        settings = CompilationSettings()
        self.assertEqual(
            settings.dynamic_shapes_kernel_specialization_strategy,
            DYNAMIC_SHAPES_KERNEL_SPECIALIZATION_STRATEGY,
        )

    def test_settable_values(self):
        for value in ("lazy", "eager", "none"):
            settings = CompilationSettings(
                dynamic_shapes_kernel_specialization_strategy=value
            )
            self.assertEqual(
                settings.dynamic_shapes_kernel_specialization_strategy, value
            )


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "C++ runtime is not available",
)
@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Dynamic shapes kernel strategy is a TensorRT-RTX feature",
)
class TestDynamicShapesKernelStrategyCpp(TestCase):
    """End-to-end: compile + infer through the C++ runtime with each strategy."""

    def test_lazy(self):
        compiled = _compile_cpp("lazy")
        x = torch.randn(2, 3, 16, 16, device="cuda")
        y = compiled(x)
        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_eager(self):
        compiled = _compile_cpp("eager")
        x = torch.randn(2, 3, 16, 16, device="cuda")
        y = compiled(x)
        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_none(self):
        compiled = _compile_cpp("none")
        x = torch.randn(2, 3, 16, 16, device="cuda")
        y = compiled(x)
        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_dynamic_shape_with_eager(self):
        """Exercise shape changes under eager kernel specialization."""
        compiled = _compile_cpp("eager")
        for batch in (1, 2, 3, 4):
            x = torch.randn(batch, 3, 16, 16, device="cuda")
            y = compiled(x)
            self.assertEqual(tuple(y.shape), (batch, 8, 16, 16))


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "C++ runtime is not available",
)
class TestDynamicShapesKernelStrategyInvalidValue(TestCase):
    """Invalid strategy names are rejected at engine-packing time."""

    def test_invalid_strategy_raises(self):
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
                use_python_runtime=False,
                min_block_size=1,
                dynamic_shapes_kernel_specialization_strategy="not_a_real_strategy",
            )


if __name__ == "__main__":
    run_tests()
