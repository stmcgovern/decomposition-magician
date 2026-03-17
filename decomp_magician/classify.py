"""Op classification: dispatch type, backends, tags, inductor status."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch._ops import OpOverload


DECOMP_TYPES = frozenset({"CIA", "table", "both", "leaf"})
DTENSOR_STRATEGIES = frozenset({"registered", "decomp-fallback", "missing"})


@dataclass(frozen=True)
class OpClass:
    decomp_type: str  # "CIA", "table", "both", "leaf"
    has_backend: dict[str, bool] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    is_mutable: bool = False
    has_alias_info: bool = False
    inductor_kept: bool = False
    dtensor_strategy: str | None = None  # "registered", "decomp-fallback", "missing"
    autograd_type: str | None = None  # "autograd_kernel", "math_kernel", etc.
    has_adiov: bool | None = None  # non-fallthrough ADInplaceOrView kernel

    def __post_init__(self):
        if self.decomp_type not in DECOMP_TYPES:
            raise ValueError(
                f"Invalid decomp_type {self.decomp_type!r}, "
                f"expected one of {sorted(DECOMP_TYPES)}"
            )
        if self.dtensor_strategy is not None and self.dtensor_strategy not in DTENSOR_STRATEGIES:
            raise ValueError(
                f"Invalid dtensor_strategy {self.dtensor_strategy!r}, "
                f"expected one of {sorted(DTENSOR_STRATEGIES)} or None"
            )


def is_dtensor_intercept(strategy: str | None) -> bool:
    """Whether this DTensor strategy intercepts dispatch (children unreachable).

    Only 'registered' strategies intercept — 'decomp-fallback' traces through
    the decomposition, so children ARE reached and need their own strategies.
    """
    return strategy == "registered"


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
            if not isinstance(op, OpOverload):
                continue
            if op not in inductor_table:
                result.add(op.name())
    except Exception:
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


def _get_tags(op: OpOverload) -> tuple[str, ...]:
    return tuple(str(t).split(".")[-1] for t in op.tags)


_dtensor_propagator = None


def _get_dtensor_propagator():
    """Get DTensor's actual ShardingPropagator with all strategies registered."""
    global _dtensor_propagator
    if _dtensor_propagator is not None:
        return _dtensor_propagator
    try:
        # Import DTensor ops to trigger strategy registration
        import torch.distributed.tensor._ops  # noqa: F401
        from torch.distributed.tensor import DTensor

        _dtensor_propagator = DTensor._op_dispatcher.sharding_propagator
    except (ImportError, AttributeError, RuntimeError):
        # Fallback: empty propagator (no strategies will match)
        from torch.distributed.tensor._sharding_prop import ShardingPropagator
        _dtensor_propagator = ShardingPropagator()
    return _dtensor_propagator


def _get_dtensor_strategy(op: OpOverload) -> str:
    """Check DTensor sharding strategy registration status."""
    try:
        prop = _get_dtensor_propagator()
        if op in prop.op_strategy_funcs:
            return "registered"
        if op in prop.op_single_dim_strategy_funcs:
            return "registered"
        if hasattr(prop, "op_to_rules") and op in prop.op_to_rules:
            return "registered"
        # Check decomp fallback
        from torch.distributed.tensor._decompositions import (
            DecompShardingStrategy,
        )

        if DecompShardingStrategy.has_decomp(op):
            return "decomp-fallback"
        return "missing"
    except Exception:
        return "missing"


def classify(op: OpOverload, dtensor: bool = False, dispatch: bool = False) -> OpClass:
    """Classify an op's decomposition type, backend support, and properties.

    Args:
        op: The operator to classify.
        dtensor: If True, check DTensor sharding strategy.
        dispatch: If True, populate autograd_type and has_adiov fields
                  from the dispatch table (requires _dispatch_dump_table).
    """
    autograd_type = None
    has_adiov = None
    if dispatch:
        from decomp_magician.dispatch import get_dispatch_info
        dinfo = get_dispatch_info(op)
        autograd_type = dinfo.autograd_type
        has_adiov = dinfo.has_adiov

    return OpClass(
        decomp_type=_get_decomp_type(op),
        has_backend=_get_backends(op),
        tags=_get_tags(op),
        is_mutable=op._schema.is_mutable,
        has_alias_info=any(
            arg.alias_info is not None for arg in op._schema.arguments
        ),
        inductor_kept=op.name() in _build_inductor_kept(),
        dtensor_strategy=_get_dtensor_strategy(op) if dtensor else None,
        autograd_type=autograd_type,
        has_adiov=has_adiov,
    )
