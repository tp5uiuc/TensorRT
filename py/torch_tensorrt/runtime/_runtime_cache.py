"""Runtime cache handle + ``runtime_cache()`` context manager.

The handle wraps a ``trt.IRuntimeCache`` plus optional disk-backing config.
Used by:

* The runtime ``cache()`` CM to attach a SHARED cache across one or more
  modules' engines.
* ``RuntimeSettings(runtime_cache=...)`` for compile-time hints (string path
  ⇒ engine creates an implicit per-engine handle; ``RuntimeCacheHandle`` ⇒
  external shared handle attached directly).

File I/O lives entirely on the Python side under a ``filelock`` (this module).
The C++-side ``torch.classes.tensorrt.RuntimeCacheHandle`` is a passive
shared_ptr wrapper used only to cross the Python/C++ boundary.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Any, Optional, Sequence, Union

import torch
import torch_tensorrt

logger = logging.getLogger(__name__)

_FILELOCK_TIMEOUT_S = 10.0


class RuntimeCacheHandle:
    """Wraps a ``trt.IRuntimeCache`` + optional disk path / autosave config.

    Two ways an instance comes into being:

    1. **Engine-implicit** (compile-time hint): when an engine sees
       ``RuntimeSettings(runtime_cache="/path")``, it materializes a
       handle internally during ``_setup_runtime_config`` -- the engine
       owns the lifecycle and saves on ``__del__``.

    2. **Runtime CM** (shared): the :func:`runtime_cache` CM bootstraps from
       the first engine under target, creates a cache, wraps it here, and
       attaches the handle to all engines for the duration of the ``with``
       block. The CM saves on ``__exit__``.

    Both paths produce the same handle shape; the difference is who owns
    the lifecycle.
    """

    def __init__(
        self,
        cache: Any = None,
        path: str = "",
        autosave: bool = True,
    ) -> None:
        # ``cache`` is a ``trt.IRuntimeCache`` once materialized. May be None
        # at construction if the handle is built before any engine has had a
        # chance to call ``runtime_config.create_runtime_cache()``.
        self._cache = cache
        self.path = path
        self.autosave = autosave
        self._lock = threading.Lock()

    @property
    def cache(self) -> Any:
        """The underlying ``trt.IRuntimeCache``. ``None`` if not yet materialized."""
        return self._cache

    def ensure_cache(self, runtime_config: Any) -> Any:
        """Idempotent. First caller materializes via ``runtime_config.create_runtime_cache()``."""
        with self._lock:
            if self._cache is None:
                self._cache = runtime_config.create_runtime_cache()
            return self._cache

    def load(self, path: Optional[str] = None) -> None:
        """Read bytes from disk and deserialize into ``self._cache``.

        No-op if ``self._cache`` is None, the resolved path is empty, or the
        file doesn't exist (first run). Caller must ensure no enqueue is
        concurrently writing (the CM enforces this by ordering load before
        engine attach; ``ensure_cache`` is called inside the engine setup).
        """
        target = path if path is not None else self.path
        if not target or self._cache is None:
            return
        from filelock import FileLock

        if not os.path.exists(target):
            return  # first run; nothing to load
        with FileLock(target + ".lock").acquire(timeout=_FILELOCK_TIMEOUT_S):
            with open(target, "rb") as f:
                data = f.read()
        if data:
            self._cache.deserialize(data)
            logger.debug(f"Loaded runtime cache from {target} ({len(data)} bytes)")

    def save(self, path: Optional[str] = None) -> None:
        """Serialize ``self._cache`` and write to disk under a filelock.

        No-op if path is empty or cache wasn't materialized. Caller must
        ensure no enqueue is concurrently writing (the CM detaches the cache
        from all engines before calling save in ``__exit__``).
        """
        target = path if path is not None else self.path
        if not target or self._cache is None:
            return
        host_mem = self._cache.serialize()
        if host_mem is None or host_mem.nbytes == 0:
            return
        from filelock import FileLock

        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = target + ".tmp"
        with FileLock(target + ".lock").acquire(timeout=_FILELOCK_TIMEOUT_S):
            with open(tmp, "wb") as f:
                f.write(memoryview(host_mem))
            shutil.move(tmp, target)
        logger.debug(f"Saved runtime cache to {target} ({host_mem.nbytes} bytes)")

    def __eq__(self, other: object) -> bool:
        # Identity equality so passing the same handle twice through
        # update_runtime_settings is a fast-path no-op.
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return (
            f"RuntimeCacheHandle(path={self.path!r}, autosave={self.autosave}, "
            f"materialized={self._cache is not None})"
        )


class _RuntimeCacheContextManager:
    """``with runtime_cache(target, path) as rc:`` -- shared cache CM.

    Bootstraps an ``IRuntimeCache`` from one of the engines under target,
    wraps it in a :class:`RuntimeCacheHandle`, loads from disk, attaches to
    all engines under all listed targets for the duration of the block, and
    saves on exit if ``autosave``.
    """

    def __init__(
        self,
        target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
        path: str = "",
        autosave: bool = True,
    ) -> None:
        if isinstance(target_or_targets, torch.nn.Module):
            self._targets: tuple[torch.nn.Module, ...] = (target_or_targets,)
        else:
            self._targets = tuple(target_or_targets)
        self.path = path
        self.autosave = autosave
        self.handle: Optional[RuntimeCacheHandle] = None
        self._inner_cm: Any = None

    def __enter__(self) -> RuntimeCacheHandle:
        # Defer imports to avoid a circular dependency:
        # _runtime_cache -> _runtime_config -> _TorchTensorRTModule -> (indirect) _runtime_cache.
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )
        from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine
        from torch_tensorrt.runtime._runtime_config import runtime_config

        # 1. Find a bootstrap engine to materialize the cache from. Try all
        # targets in order; first TRT submodule wins.
        bootstrap_engine = None
        for target in self._targets:
            for _, mod in target.named_modules():
                if isinstance(mod, TorchTensorRTModule) and isinstance(
                    mod.engine, TRTEngine
                ):
                    bootstrap_engine = mod.engine
                    break
            if bootstrap_engine is not None:
                break
        if bootstrap_engine is None:
            raise RuntimeError(
                "runtime_cache() requires at least one TorchTensorRTModule under "
                "the target(s) using the Python TRT runtime."
            )

        # 2. Materialize the cache via the bootstrap engine's runtime_config.
        # (The cache returned is free-floating; ownership transfers to the handle.)
        cache_obj = bootstrap_engine.runtime_config.create_runtime_cache()
        self.handle = RuntimeCacheHandle(
            cache=cache_obj, path=self.path, autosave=self.autosave
        )

        # 3. Load from disk if path was given.
        self.handle.load()

        # 4. Apply the handle to ALL engines under target(s) via runtime_config CM.
        self._inner_cm = runtime_config(list(self._targets), runtime_cache=self.handle)
        self._inner_cm.__enter__()
        return self.handle

    def __exit__(self, *args: Any) -> None:
        if self._inner_cm is not None:
            self._inner_cm.__exit__(*args)
        if self.autosave and self.path and self.handle is not None:
            self.handle.save()


def runtime_cache(
    target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
    path: str = "",
    autosave: bool = True,
) -> _RuntimeCacheContextManager:
    """Context manager that attaches a shared runtime cache to all engines
    under ``target_or_targets`` for the duration of the ``with`` block.

    Yields the :class:`RuntimeCacheHandle` for inspection or explicit
    ``handle.save()`` calls (e.g., for mid-block checkpointing -- caller is
    responsible for ``torch.cuda.synchronize()`` first).
    """
    return _RuntimeCacheContextManager(target_or_targets, path, autosave)


# When the C++ Torch-TensorRT runtime is loaded, we ALSO expose
# ``torch.classes.tensorrt.RuntimeCacheHandle`` as the canonical
# cross-language handle. The Python class above is the user-facing API;
# at dispatch time the Python module converts to/from the torchbind class as
# needed (see ``TorchTensorRTModule.set_runtime_settings``).
def _to_torchbind_handle(
    rc: Union[None, str, "RuntimeCacheHandle", Any],
) -> Any:
    """Convert a Python-side ``runtime_cache`` value to a torchbind handle
    suitable for ``torch.classes.tensorrt.Engine.update_runtime_settings(...)``.

    Returns ``None`` if no runtime cache is requested. Raises if the C++
    runtime isn't loaded (caller shouldn't dispatch to a C++ engine in that
    case anyway). Already-torchbind handles (``torch.ScriptObject``) are passed
    through unchanged so callers can pre-stash a handle on the module and
    share it across dispatch calls.
    """
    if rc is None:
        return None
    if not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
        raise RuntimeError(
            "torch_tensorrt C++ runtime is not available; cannot construct "
            "torch.classes.tensorrt.RuntimeCacheHandle"
        )
    if isinstance(rc, torch.ScriptObject):
        return rc
    path = rc if isinstance(rc, str) else rc.path
    return torch.classes.tensorrt.RuntimeCacheHandle(path)
