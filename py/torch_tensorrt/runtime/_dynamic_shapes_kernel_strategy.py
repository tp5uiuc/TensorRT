"""Sugar over ``runtime_config`` for the dynamic-shapes kernel strategy knob."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

from torch_tensorrt.runtime._runtime_config import (
    _RuntimeConfigContextManager,
    runtime_config,
)

if TYPE_CHECKING:
    import torch


def set_dynamic_shapes_kernel_strategy(
    target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
    strategy: str,
) -> _RuntimeConfigContextManager:
    """Context manager that sets the dynamic-shapes kernel specialization
    strategy on all TRT engines under ``target_or_targets``.

    Accepts ``"lazy"``, ``"eager"``, or ``"none"``. Delegates to
    :func:`runtime_config`.
    """
    return runtime_config(
        target_or_targets, dynamic_shapes_kernel_specialization_strategy=strategy
    )
