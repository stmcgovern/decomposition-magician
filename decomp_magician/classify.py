"""Op classification: dispatch type, backends, tags, inductor status."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch._ops import OpOverload


@dataclass
class OpClass:
    decomp_type: str  # "CIA", "table", "both", "leaf"
    has_backend: dict[str, bool] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    is_mutable: bool = False
    has_alias_info: bool = False
    inductor_kept: bool = False
    dtensor_strategy: str | None = None  # "registered", "decomp-fallback", "missing"


# Backend dispatch keys to check
_BACKENDS = {
    "cpu": torch._C.DispatchKey.CPU,
    "cuda": torch._C.DispatchKey.CUDA,
    "meta": torch._C.DispatchKey.Meta,
}

_inductor_kept_cache: set[str] | None = None


def _build_inductor_kept() -> set[str]:
    """Build a set of op names that Inductor preserves without decomposition.

    An op is "inductor-kept" if it has a decomposition in the raw table
    but is absent from the Inductor table — meaning Inductor uses a direct
    lowering instead. Ops that are in decomps_to_exclude but re-added with
    a custom Inductor decomp are NOT kept; they're just decomposed differently.
    """
    global _inductor_kept_cache
    if _inductor_kept_cache is not None:
        return _inductor_kept_cache

    result: set[str] = set()
    try:
        from torch._decomp import decomposition_table
        from torch._inductor.decomposition import select_decomp_table

        inductor_table = select_decomp_table()
        for op in decomposition_table:
            if op not in inductor_table:
                result.add(op.name())
    except ImportError:
        pass
    _inductor_kept_cache = result
    return result


def _get_decomp_type(op: OpOverload) -> str:
    from torch._decomp import decomposition_table

    in_table = op in decomposition_table
    has_cia = op._can_decompose()
    if in_table and has_cia:
        return "both"
    if in_table:
        return "table"
    if has_cia:
        return "CIA"
    return "leaf"


def _get_backends(op: OpOverload) -> dict[str, bool]:
    result = {}
    for name, key in _BACKENDS.items():
        result[name] = torch._C._dispatch_has_kernel_for_dispatch_key(
            op.name(), key
        )
    return result


def _get_tags(op: OpOverload) -> list[str]:
    return [str(t).split(".")[-1] for t in op.tags]


def classify(op: OpOverload, dtensor: bool = False) -> OpClass:
    """Classify an op's decomposition type, backend support, and properties."""
    cls = OpClass(
        decomp_type=_get_decomp_type(op),
        has_backend=_get_backends(op),
        tags=_get_tags(op),
        is_mutable=op._schema.is_mutable,
        has_alias_info=any(
            arg.alias_info is not None for arg in op._schema.arguments
        ),
        inductor_kept=op.name() in _build_inductor_kept(),
    )
    if dtensor:
        cls.dtensor_strategy = _get_dtensor_strategy(op)
    return cls


def _get_dtensor_strategy(op: OpOverload) -> str:
    """Check DTensor sharding strategy registration status."""
    try:
        from torch.distributed.tensor._sharding_prop import ShardingPropagator

        prop = ShardingPropagator()
        if op in prop.op_strategy_funcs:
            return "registered"
        if op in prop.op_single_dim_strategy_funcs:
            return "registered"
        # Check decomp fallback
        from torch.distributed.tensor._decompositions import (
            DecompShardingStrategy,
        )

        if DecompShardingStrategy.has_decomp(op):
            return "decomp-fallback"
        return "missing"
    except ImportError:
        return "missing"
