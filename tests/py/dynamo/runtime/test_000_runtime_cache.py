import gc
import os
import shutil
import tempfile
import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity
from torch_tensorrt.runtime import RuntimeSettings, runtime_cache


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


def _fresh_conv_model_and_inputs(seed=0):
    torch.manual_seed(seed)
    return ConvModel().eval().cuda(), [torch.randn(2, 3, 16, 16).cuda()]


def _compile(model, inputs, *, use_python_runtime, runtime_cache_path=None):
    """Compile ``model`` through either runtime.

    ``runtime_cache_path``, when supplied, is threaded as a compile-time hint via
    ``runtime_settings=RuntimeSettings(runtime_cache=path)`` (per-engine cache).
    """
    rs = (
        RuntimeSettings(runtime_cache=runtime_cache_path)
        if runtime_cache_path is not None
        else None
    )
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        use_python_runtime=use_python_runtime,
        min_block_size=1,
        runtime_settings=rs,
    )
    torch._dynamo.reset()
    return compiled


def _compile_simple(runtime_cache_path=None):
    """Compile SimpleModel on the Python runtime (used by introspection tests)."""
    model = SimpleModel().eval().cuda()
    inputs = [torch.randn(2, 3).cuda()]
    return (
        _compile(
            model,
            inputs,
            use_python_runtime=True,
            runtime_cache_path=runtime_cache_path,
        ),
        inputs,
    )


def _find_python_trt_engine(compiled):
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
    from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

    for _, mod in compiled.named_modules():
        if isinstance(mod, TorchTensorRTModule) and isinstance(mod.engine, TRTEngine):
            return mod.engine
    return None


_RUNTIMES = [("python", True), ("cpp", False)]


def _skip_if_cpp_unavailable(testcase, use_python_runtime):
    if not use_python_runtime and not ENABLED_FEATURES.torch_tensorrt_runtime:
        testcase.skipTest("C++ runtime is not available")


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache is only available with TensorRT-RTX",
)
class TestRuntimeCacheSetup(TestCase):
    """Tests that runtime config and per-engine cache are correctly created for RTX."""

    def test_runtime_config_created(self):
        compiled, _ = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.runtime_config)

    def test_context_created_successfully(self):
        compiled, _ = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine.context)

    def test_no_implicit_cache_handle_by_default(self):
        """Default RuntimeSettings has no disk-backing => no implicit handle."""
        compiled, _ = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertIsNone(engine._implicit_cache_handle)

    def test_implicit_cache_handle_for_path_hint(self):
        """Passing a path string in RuntimeSettings.runtime_cache creates an implicit handle."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rc.bin")
            compiled, _ = _compile_simple(runtime_cache_path=path)
            engine = _find_python_trt_engine(compiled)
            self.assertIsNotNone(engine._implicit_cache_handle)
            self.assertEqual(engine._implicit_cache_handle.path, path)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Runtime cache persistence is RTX-only",
)
class TestRuntimeCachePersistence(TestCase):
    """End-to-end: compile with a cache path, infer, destroy, reload, infer again."""

    @parameterized.expand(_RUNTIMES)
    def test_cache_saved_on_del(self, _name, use_python_runtime):
        _skip_if_cpp_unavailable(self, use_python_runtime)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rc.bin")
            model, inputs = _fresh_conv_model_and_inputs(seed=42)
            compiled = _compile(
                model,
                inputs,
                use_python_runtime=use_python_runtime,
                runtime_cache_path=path,
            )
            _ = compiled(*inputs)
            del compiled
            gc.collect()
            self.assertTrue(
                os.path.exists(path),
                f"Implicit cache handle should have saved to {path} on engine __del__",
            )


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "runtime_cache CM is RTX-only",
)
class TestRuntimeCacheContextManager(TestCase):
    """Tests for the runtime_cache(target, path) shared-cache CM."""

    def test_with_cache_loads_and_saves(self):
        compiled, inputs = _compile_simple()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shared.bin")
            with runtime_cache(compiled, path) as rc:
                self.assertIsNotNone(rc)
                self.assertEqual(rc.path, path)
                _ = compiled(*inputs)
            # autosave on exit
            self.assertTrue(os.path.exists(path))

    def test_with_cache_in_memory_only(self):
        """path='' means in-memory only; no disk artifact after exit."""
        compiled, inputs = _compile_simple()
        with tempfile.TemporaryDirectory() as tmp:
            with runtime_cache(compiled, "") as rc:
                self.assertEqual(rc.path, "")
                _ = compiled(*inputs)
            self.assertFalse(os.listdir(tmp), "No files should be created for path=''")

    def test_shared_cache_pointer_across_modules(self):
        """Two modules sharing one runtime_cache handle reference the same IRuntimeCache."""
        compiled_a, inputs_a = _compile_simple()
        compiled_b, inputs_b = _compile_simple()
        eng_a = _find_python_trt_engine(compiled_a)
        eng_b = _find_python_trt_engine(compiled_b)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shared.bin")
            with runtime_cache([compiled_a, compiled_b], path) as rc:
                self.assertIs(eng_a.runtime_settings.runtime_cache, rc)
                self.assertIs(eng_b.runtime_settings.runtime_cache, rc)
                _ = compiled_a(*inputs_a)
                _ = compiled_b(*inputs_b)
            self.assertTrue(os.path.exists(path))

    def test_runtime_cache_on_empty_target_raises(self):
        """A target with no TRT submodules raises a clear error on enter."""
        empty = torch.nn.Linear(3, 3).cuda()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rc.bin")
            with self.assertRaises(RuntimeError):
                with runtime_cache(empty, path):
                    pass

    def test_cm_does_not_double_save_on_rc_gc(self):
        """CM yields handle with autosave_on_del=False; only one save happens.

        Regression: if the CM-yielded handle had autosave_on_del=True, the
        handle's __del__ would re-save after the CM's __exit__ already wrote
        the file. We disable autosave_on_del on CM-created handles to avoid
        that double-write.
        """
        compiled, inputs = _compile_simple()
        save_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shared.bin")
            with runtime_cache(compiled, path) as rc:
                # CM-created handle must not autosave on del (CM saves explicitly).
                self.assertFalse(rc.autosave_on_del)
                original_save = rc.save

                def _tracking_save(p=None):
                    save_calls.append(p)
                    return original_save(p)

                rc.save = _tracking_save  # type: ignore[method-assign]
                _ = compiled(*inputs)
            # CM.__exit__ saves exactly once; rc going out of scope triggers
            # __del__ but autosave_on_del is False, so no second save.
            del rc
            gc.collect()
            self.assertEqual(len(save_calls), 1, f"Expected one save, got {save_calls}")
            self.assertTrue(os.path.exists(path))


class TestRuntimeCacheHandleAutosave(TestCase):
    """Whitebox tests for RuntimeCacheHandle.autosave_on_del semantics."""

    def test_user_built_handle_no_autosave_by_default(self):
        """Hand-built handle defaults to autosave_on_del=False; nothing on GC."""
        from torch_tensorrt.runtime._runtime_cache import RuntimeCacheHandle

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rc.bin")
            handle = RuntimeCacheHandle(path=path)
            self.assertFalse(handle.autosave_on_del)
            del handle
            gc.collect()
            self.assertFalse(
                os.path.exists(path),
                "User-built handle with autosave_on_del=False should not save on GC",
            )


if __name__ == "__main__":
    run_tests()
