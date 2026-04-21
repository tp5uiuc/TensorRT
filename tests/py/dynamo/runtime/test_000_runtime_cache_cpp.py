import gc
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
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


def _fresh_model_and_inputs(seed=0):
    """Create a deterministic SimpleModel + input tensor pair."""
    torch.manual_seed(seed)
    return SimpleModel().eval().cuda(), [torch.randn(2, 3, 16, 16).cuda()]


def _compile_cpp(model, inputs, runtime_cache_path=None):
    """Compile the given model through the C++ runtime path."""
    kwargs = {
        "ir": "dynamo",
        "inputs": inputs,
        "enabled_precisions": {torch.float32},
        "use_python_runtime": False,
        "min_block_size": 1,
    }
    if runtime_cache_path is not None:
        kwargs["runtime_cache_path"] = runtime_cache_path
    compiled = torchtrt.compile(model, **kwargs)
    torch._dynamo.reset()
    return compiled


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "C++ runtime is not available",
)
@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
class TestRuntimeCacheCppPersistence(TestCase):
    """Exercise C++-runtime runtime cache load/save against disk."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.cache_dir, "runtime_cache.bin")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_cache_saved_on_del(self):
        model, inputs = _fresh_model_and_inputs()
        compiled = _compile_cpp(model, inputs, runtime_cache_path=self.cache_path)
        _ = compiled(*[inp.clone() for inp in inputs])
        self.assertFalse(
            os.path.isfile(self.cache_path),
            "Cache should not exist before module cleanup",
        )
        del compiled
        gc.collect()
        self.assertTrue(
            os.path.isfile(self.cache_path),
            "Cache file should be created after module cleanup",
        )

    def test_cache_file_nonempty(self):
        model, inputs = _fresh_model_and_inputs()
        compiled = _compile_cpp(model, inputs, runtime_cache_path=self.cache_path)
        _ = compiled(*[inp.clone() for inp in inputs])
        del compiled
        gc.collect()
        self.assertGreater(
            os.path.getsize(self.cache_path),
            0,
            "Cache file should have nonzero size",
        )

    def test_cache_roundtrip(self):
        """Compile, infer, save. Then recompile same model+cache and verify correctness."""
        model, inputs = _fresh_model_and_inputs()
        with torch.no_grad():
            ref_output = model(*inputs)

        compiled1 = _compile_cpp(model, inputs, runtime_cache_path=self.cache_path)
        out1 = compiled1(*[inp.clone() for inp in inputs])
        self.assertGreater(
            cosine_similarity(ref_output, out1),
            COSINE_THRESHOLD,
            "First compiled output should match eager",
        )
        del compiled1
        gc.collect()
        self.assertTrue(os.path.isfile(self.cache_path))

        compiled2 = _compile_cpp(model, inputs, runtime_cache_path=self.cache_path)
        out2 = compiled2(*[inp.clone() for inp in inputs])
        self.assertGreater(
            cosine_similarity(ref_output, out2),
            COSINE_THRESHOLD,
            "Second compiled output (warm cache) should still match eager",
        )

    def test_save_creates_directory(self):
        nested_path = os.path.join(self.cache_dir, "a", "b", "c", "runtime_cache.bin")
        model, inputs = _fresh_model_and_inputs()
        compiled = _compile_cpp(model, inputs, runtime_cache_path=nested_path)
        _ = compiled(*[inp.clone() for inp in inputs])
        del compiled
        gc.collect()
        self.assertTrue(
            os.path.isfile(nested_path),
            "Save should create intermediate directories",
        )


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "C++ runtime is not available",
)
class TestCppSerializationIndices(TestCase):
    """Verify the new C++ serialization indices are registered by the runtime."""

    def test_new_indices_registered(self):
        self.assertEqual(int(torch.ops.tensorrt.ABI_VERSION()), 9)
        self.assertEqual(int(torch.ops.tensorrt.SERIALIZATION_LEN()), 14)
        self.assertEqual(int(torch.ops.tensorrt.RUNTIME_CACHE_PATH_IDX()), 11)
        self.assertEqual(
            int(torch.ops.tensorrt.DYNAMIC_SHAPES_KERNEL_STRATEGY_IDX()), 12
        )
        self.assertEqual(int(torch.ops.tensorrt.CUDA_GRAPH_STRATEGY_IDX()), 13)


if __name__ == "__main__":
    run_tests()
