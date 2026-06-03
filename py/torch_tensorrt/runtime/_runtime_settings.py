"""User-facing runtime-only knobs for TRT-RTX engines.

A knob belongs in :class:`RuntimeSettings` iff changing it requires recreating
the ``IExecutionContext``. Per-execute flags (``cudagraphs_mode``,
``multi_device_safe_mode``, ``pre_allocated_outputs``) stay as their existing
process-global setters.

Three ways to use:

1. **Compile-time hint** (recommended fast path) -- prime the engine with the
   desired initial values so no CM enter/exit recreate is needed::

       compiled = torchtrt.compile(
           model, ...,
           runtime_settings=RuntimeSettings(cuda_graph_strategy="whole_graph_capture"),
       )

2. **Runtime context manager** -- toggle settings inside a ``with`` block. See
   :func:`torch_tensorrt.runtime.runtime_config`.

3. **Programmatic** -- call ``module.set_runtime_settings(rs)`` directly.

``RuntimeSettings`` is intentionally NOT part of ``CompilationSettings`` and is
NOT serialized into the engine tuple (per GitHub pytorch/TensorRT#4310). It's
purely an in-memory initialization parameter / runtime override state.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from torch_tensorrt.runtime._runtime_cache import RuntimeCacheHandle

# Validation maps used by both the engine setup path and the dataclass post-init.
_DYNAMIC_SHAPES_KERNEL_STRATEGY_MAP: Dict[str, int] = {
    "lazy": 0,
    "eager": 1,
    "none": 2,
}
_CUDA_GRAPH_STRATEGY_MAP: Dict[str, int] = {
    "disabled": 0,
    "whole_graph_capture": 1,
}


@dataclass(frozen=True)
class RuntimeSettings:
    """Per-engine runtime-only knobs sampled at IExecutionContext creation.

    Fields:
        dynamic_shapes_kernel_specialization_strategy: ``"lazy" | "eager" | "none"``.
            TRT-RTX-only; no-op on standard TensorRT.
        cuda_graph_strategy: ``"disabled" | "whole_graph_capture"``. TRT-RTX-only.
        runtime_cache: ``None``, a disk path string, or a
            :class:`RuntimeCacheHandle`. ``None`` ⇒ each engine has an in-memory
            cache local to itself. A string is honored at engine construction
            time and primes a per-engine disk-backed cache (matches today's
            ``runtime_cache_path=`` behavior; saved on engine ``__del__``).
            A handle is the shared-cache form, typically obtained from
            :func:`torch_tensorrt.runtime.runtime_cache` -- multiple engines
            attaching the same handle share one ``IRuntimeCache``.

    Equality compares all fields; for ``runtime_cache``, handle equality is
    by identity (same handle ⇒ same cache).
    """

    dynamic_shapes_kernel_specialization_strategy: str = "lazy"
    cuda_graph_strategy: str = "disabled"
    runtime_cache: Optional[Union[str, "RuntimeCacheHandle"]] = None  # noqa: F821

    def __post_init__(self) -> None:
        if (
            self.dynamic_shapes_kernel_specialization_strategy
            not in _DYNAMIC_SHAPES_KERNEL_STRATEGY_MAP
        ):
            raise ValueError(
                "Invalid dynamic_shapes_kernel_specialization_strategy: "
                f"{self.dynamic_shapes_kernel_specialization_strategy!r}. "
                f"Expected one of {list(_DYNAMIC_SHAPES_KERNEL_STRATEGY_MAP)}."
            )
        if self.cuda_graph_strategy not in _CUDA_GRAPH_STRATEGY_MAP:
            raise ValueError(
                f"Invalid cuda_graph_strategy: {self.cuda_graph_strategy!r}. "
                f"Expected one of {list(_CUDA_GRAPH_STRATEGY_MAP)}."
            )

    def merge(self, **overrides: Any) -> "RuntimeSettings":
        """Return a new ``RuntimeSettings`` with ``overrides`` applied on top of self."""
        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise TypeError(
                f"Unknown RuntimeSettings field(s): {sorted(unknown)}. "
                f"Valid fields: {[f.name for f in dataclasses.fields(self)]}."
            )
        return dataclasses.replace(self, **overrides)
