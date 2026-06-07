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
    """Wraps a ``trt.IRuntimeCache`` (or a torchbind sibling) + optional disk path.

    Three construction patterns differ in *who else holds a reference*, which
    drives the ``autosave_on_del`` flag:

    1. **Engine-implicit** (compile-time hint): when an engine sees
       ``RuntimeSettings(runtime_cache="/path")``, the engine's
       ``TRTRuntimeConfig`` materializes a handle internally with
       ``autosave_on_del=True``. No other Python object holds the handle,
       so ``__del__`` writes the cache to disk on the engine's last release.

    2. **Runtime CM** (shared, multi-engine): the :func:`runtime_cache` CM
       constructs a handle with ``autosave_on_del=False`` and explicitly
       calls ``handle.save()`` on ``__exit__``. The handle's ``__del__``
       no-ops since the CM already saved.

    3. **User-constructed** (advanced): hand-built handles default to
       ``autosave_on_del=False`` so save timing stays under the user's
       control. Opt in with ``RuntimeCacheHandle(path=..., autosave_on_del=True)``
       for with-block-style autosave on hand-built handles.

    Bytes are sourced from whichever of ``_cache`` (Python pybind
    ``trt.IRuntimeCache``) or ``_torchbind`` (TorchBind
    ``RuntimeCacheHandle`` sibling) is populated; the Python runtime path
    populates the former, the C++ runtime path populates the latter.
    """

    def __init__(
        self,
        cache: Any = None,
        path: str = "",
        autosave_on_del: bool = False,
        torchbind_handle: Any = None,
    ) -> None:
        # ``cache`` is a ``trt.IRuntimeCache`` once materialized (Python rt).
        # ``torchbind_handle`` is a ``torch.classes.tensorrt.RuntimeCacheHandle``
        # for the C++ runtime path, exposing serialize/deserialize as tensors.
        # Exactly zero or one is populated for a given handle.
        self._cache = cache
        self._torchbind = torchbind_handle
        self.path = path
        self.autosave_on_del = autosave_on_del
        self._lock = threading.Lock()

    @property
    def cache(self) -> Any:
        """The underlying Python pybind ``trt.IRuntimeCache``. ``None`` if not yet materialized or if backed by a torchbind sibling."""
        return self._cache

    def ensure_cache(self, runtime_config: Any) -> Any:
        """Idempotent. First caller materializes via ``runtime_config.create_runtime_cache()``.

        Only meaningful for the Python-runtime path (``_cache``). The C++
        runtime materializes its cache inside the engine and exposes bytes
        through the torchbind sibling.
        """
        with self._lock:
            if self._cache is None:
                self._cache = runtime_config.create_runtime_cache()
            return self._cache

    def _read_bytes(self) -> Optional[bytes]:
        """Serialize whichever of ``_cache`` or ``_torchbind`` is populated."""
        if self._cache is not None:
            host_mem = self._cache.serialize()
            if host_mem is None or host_mem.nbytes == 0:
                return None
            return bytes(memoryview(host_mem))
        if self._torchbind is not None and self._torchbind.has_cache():
            tensor = self._torchbind.serialize()
            if tensor.numel() == 0:
                return None
            return bytes(tensor.cpu().contiguous().numpy())
        return None

    def _write_bytes(self, data: bytes) -> None:
        """Deserialize ``data`` into whichever of ``_cache`` or ``_torchbind`` is populated."""
        if self._cache is not None:
            self._cache.deserialize(data)
            return
        if self._torchbind is not None and self._torchbind.has_cache():
            tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            self._torchbind.deserialize(tensor)
            return

    def load(self, path: Optional[str] = None) -> None:
        """Read bytes from disk and deserialize into the underlying cache.

        No-op if no cache backing is present, the resolved path is empty, or
        the file doesn't exist (first run). Caller must ensure no enqueue is
        concurrently writing (the CM enforces this by ordering load before
        engine attach; ``ensure_cache`` is called inside engine setup).
        """
        target = path if path is not None else self.path
        if not target:
            return
        if self._cache is None and self._torchbind is None:
            return
        if not os.path.exists(target):
            return  # first run; nothing to load
        from filelock import FileLock

        with FileLock(target + ".lock").acquire(timeout=_FILELOCK_TIMEOUT_S):
            with open(target, "rb") as f:
                data = f.read()
        if data:
            self._write_bytes(data)
            logger.debug(f"Loaded runtime cache from {target} ({len(data)} bytes)")

    def save(self, path: Optional[str] = None) -> None:
        """Serialize the underlying cache and write to disk under a filelock.

        No-op if path is empty or the cache wasn't materialized. Caller must
        ensure no enqueue is concurrently writing (the CM detaches the cache
        from all engines before calling save in ``__exit__``).
        """
        target = path if path is not None else self.path
        if not target:
            return
        data = self._read_bytes()
        if not data:
            return
        from filelock import FileLock

        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = target + ".tmp"
        with FileLock(target + ".lock").acquire(timeout=_FILELOCK_TIMEOUT_S):
            with open(tmp, "wb") as f:
                f.write(data)
            shutil.move(tmp, target)
        logger.debug(f"Saved runtime cache to {target} ({len(data)} bytes)")

    def __del__(self) -> None:
        # Best-effort autosave for engine-implicit handles. The CM disables
        # this (``autosave_on_del=False``) since it saves on ``__exit__``;
        # user-constructed handles default to disabled so save timing stays
        # under the user's control. ``__del__`` can fire during interpreter
        # shutdown when imports/filesystem ops fail unpredictably -- swallow.
        if self.autosave_on_del and self.path:
            try:
                self.save()
            except Exception:
                pass

    def __eq__(self, other: object) -> bool:
        # Identity equality so passing the same handle twice through
        # update_runtime_settings is a fast-path no-op.
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return (
            f"RuntimeCacheHandle(path={self.path!r}, "
            f"autosave_on_del={self.autosave_on_del}, "
            f"materialized={self._cache is not None or self._torchbind is not None})"
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
        # ``autosave_on_del=False`` because the CM saves explicitly on ``__exit__``;
        # letting ``__del__`` also save would double-write when ``rc`` falls out
        # of scope after the with-block.
        cache_obj = bootstrap_engine.runtime_config.create_runtime_cache()
        self.handle = RuntimeCacheHandle(
            cache=cache_obj, path=self.path, autosave_on_del=False
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
# needed (see ``TorchTensorRTModule.runtime_settings`` setter).
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
