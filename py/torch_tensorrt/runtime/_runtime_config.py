"""Per-engine runtime-settings context manager.

``runtime_config(target_or_targets, **kw)`` is the one runtime CM that toggles
``RuntimeSettings`` on every TRT engine reachable under the listed targets.
Other CMs (``runtime_cache``, ``set_cuda_graph_strategy``,
``set_dynamic_shapes_kernel_strategy``) are thin sugar that delegate here.

Walks ``named_modules()`` once on enter, snapshots prior settings per engine,
calls ``mod.set_runtime_settings(merged)`` per engine. Restores on exit using
the same snapshot dict.

Yields the target (or tuple of targets) so users can write
``with runtime_config(model, ...) as m: m(*inputs)``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Sequence, Tuple, Union

import torch
from torch_tensorrt.runtime._runtime_settings import RuntimeSettings


class _RuntimeConfigContextManager:
    def __init__(
        self,
        target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
        **overrides: Any,
    ) -> None:
        # Validate keys against RuntimeSettings field names (typo => raise here,
        # not silently no-op later).
        valid_fields = {f.name for f in dataclasses.fields(RuntimeSettings)}
        unknown = set(overrides) - valid_fields
        if unknown:
            raise TypeError(
                f"Unknown RuntimeSettings field(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(valid_fields)}."
            )

        if isinstance(target_or_targets, torch.nn.Module):
            self._targets: Tuple[torch.nn.Module, ...] = (target_or_targets,)
            self._yield_tuple = False
        else:
            self._targets = tuple(target_or_targets)
            self._yield_tuple = True
        self._overrides = overrides
        # Engine ↔ prior RuntimeSettings snapshot; populated on enter.
        self._saved: Dict[Any, RuntimeSettings] = {}

    def __enter__(self) -> Union[torch.nn.Module, Tuple[torch.nn.Module, ...]]:
        # Deferred import to avoid a circular dependency at module-load time.
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        for target in self._targets:
            for _, mod in target.named_modules():
                if isinstance(mod, TorchTensorRTModule) and mod.engine is not None:
                    current = mod.runtime_settings
                    if mod in self._saved:
                        # The same TRTModule appears under multiple targets in the
                        # list (or the tree contains a cycle). Don't snapshot twice.
                        continue
                    self._saved[mod] = current
                    merged = current.merge(**self._overrides)
                    mod.set_runtime_settings(merged)
        return self._targets if self._yield_tuple else self._targets[0]

    def __exit__(self, *args: Any) -> None:
        for mod, prior in self._saved.items():
            mod.set_runtime_settings(prior)


def runtime_config(
    target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
    **overrides: Any,
) -> _RuntimeConfigContextManager:
    """Context manager that applies ``RuntimeSettings`` overrides to all TRT
    engines under ``target_or_targets`` for the duration of the ``with`` block.

    Accepts the same kwargs as :class:`RuntimeSettings` fields. The pool
    semantics collapse N knob changes into one ``update_runtime_settings`` call
    per engine, which means exactly two ``IExecutionContext`` recreates per
    engine (one on enter, one on exit) regardless of how many overrides are
    passed.

    Yields the target module (single form) or a tuple of targets (list form),
    by-reference -- same object the caller passed in.
    """
    return _RuntimeConfigContextManager(target_or_targets, **overrides)
