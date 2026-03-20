"""Op classification: dispatch type, backends, tags, inductor status."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python < 3.11."""
from typing import NamedTuple

import torch
from torch._ops import OpOverload


class DecompType(StrEnum):
    CIA = "CIA"
    TABLE = "table"
    BOTH = "both"
    LEAF = "leaf"


class DtensorStrategy(StrEnum):
    REGISTERED = "registered"
    DECOMP_FALLBACK = "decomp-fallback"
    MISSING = "missing"
    NOT_APPLICABLE = "not-applicable"


class OpCategory(StrEnum):
    """Computational pattern of an op."""
    POINTWISE = "pointwise"
    REDUCTION = "reduction"
    VIEW = "view"
    FACTORY = "factory"
    SCATTER_GATHER = "scatter_gather"
    LINALG = "linalg"
    NORM = "norm"
    LOSS = "loss"
    FFT = "fft"
    SPATIAL = "spatial"
    SCAN = "scan"
    RANDOM = "random"
    OTHER = "other"


# Keep for backward compatibility with any external code checking membership
DECOMP_TYPES = frozenset(DecompType)
DTENSOR_STRATEGIES = frozenset(DtensorStrategy)
OP_CATEGORIES = frozenset(OpCategory)


@dataclass(frozen=True)
class OpClass:
    decomp_type: DecompType
    has_backend: dict[str, bool] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    is_mutable: bool = False
    has_alias_info: bool = False
    inductor_kept: bool = False
    op_category: OpCategory = OpCategory.OTHER
    dtensor_strategy: DtensorStrategy | None = None
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


def is_dtensor_intercept(strategy: DtensorStrategy | None) -> bool:
    """Whether this DTensor strategy intercepts dispatch (children unreachable).

    Only 'registered' strategies intercept — 'decomp-fallback' traces through
    the decomposition, so children ARE reached and need their own strategies.
    """
    return strategy == DtensorStrategy.REGISTERED


def is_dtensor_gap(strategy: DtensorStrategy | None) -> bool:
    """Whether this DTensor strategy represents a real coverage gap.

    Only 'missing' is a gap — the op has tensor inputs (so DTensor dispatch
    can reach it) but no strategy to handle it.  'not-applicable' and
    'decomp-fallback' are not gaps; 'registered' is actively covered.
    """
    return strategy == DtensorStrategy.MISSING


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
        import warnings
        warnings.warn(
            "Could not load Inductor decomposition table — inductor-kept detection disabled.",
            stacklevel=2,
        )
    _inductor_kept_cache = result
    return result


def _get_decomp_type(op: OpOverload) -> DecompType:
    from torch._decomp import decomposition_table

    in_table = op in decomposition_table
    has_cia = op._can_decompose()
    if in_table and has_cia:
        return DecompType.BOTH
    if in_table:
        return DecompType.TABLE
    if has_cia:
        return DecompType.CIA
    return DecompType.LEAF


def _get_backends(op: OpOverload) -> dict[str, bool]:
    result = {}
    for name, key in _BACKENDS.items():
        result[name] = torch._C._dispatch_has_kernel_for_dispatch_key(
            op.name(), key
        )
    return result


def _get_tags(op: OpOverload) -> tuple[str, ...]:
    return tuple(str(t).split(".")[-1] for t in op.tags)


class _DtensorState(NamedTuple):
    propagator: object  # ShardingPropagator instance
    decomp_strategy_cls: type | None  # DecompShardingStrategy class, if available


_dtensor_state: _DtensorState | None = None


def _init_dtensor() -> _DtensorState:
    """Eagerly initialize DTensor propagator and decomposition strategy.

    Both are resolved here so that a single import-order dependency
    (torch.distributed.tensor._ops) is handled in one place.  If either
    component fails, a warning is emitted so the user knows DTensor
    classification may be incomplete.
    """
    import sys
    import warnings

    global _dtensor_state
    if _dtensor_state is not None:
        return _dtensor_state

    propagator = None
    decomp_strategy_cls = None

    try:
        # Import DTensor ops to trigger strategy registration — this also
        # makes torch.distributed.tensor._decompositions importable.
        import torch.distributed.tensor._ops  # noqa: F401
        from torch.distributed.tensor import DTensor

        propagator = DTensor._op_dispatcher.sharding_propagator
    except (ImportError, AttributeError, RuntimeError):
        from torch.distributed.tensor._sharding_prop import ShardingPropagator
        propagator = ShardingPropagator()
        warnings.warn(
            "Could not load DTensor strategies — registered-strategy detection may be incomplete.",
            stacklevel=2,
        )

    try:
        from torch.distributed.tensor._decompositions import DecompShardingStrategy
        decomp_strategy_cls = DecompShardingStrategy
    except Exception:
        if "torch.distributed.tensor._ops" in sys.modules:
            warnings.warn(
                "DTensor ops loaded but DecompShardingStrategy unavailable "
                "— decomp-fallback detection may be incomplete.",
                stacklevel=2,
            )

    _dtensor_state = _DtensorState(propagator, decomp_strategy_cls)
    return _dtensor_state


def _type_involves_tensor(t) -> bool:
    """Check whether a schema type is or contains a Tensor."""
    kind = t.kind()
    if kind == "TensorType":
        return True
    if kind in ("ListType", "OptionalType"):
        return _type_involves_tensor(t.getElementType())
    return False


def _has_tensor_input(op: OpOverload) -> bool:
    """Check whether any argument in the op's schema involves a Tensor."""
    return any(_type_involves_tensor(arg.type) for arg in op._schema.arguments)


# Name patterns for category detection (compiled once).
_P = re.compile  # shorthand

_CATEGORY_PATTERNS: list[tuple[re.Pattern[str], OpCategory]] = [
    (_P(r"(^|_)(conv|pool|upsample|pad|interpolate|avg_pool|max_pool"
        r"|adaptive_avg|adaptive_max|col2im|im2col|grid_sampler|unfold)"),
     OpCategory.SPATIAL),
    (_P(r"(^|_)(mm|matmul|bmm|addmm|addbmm|baddbmm|dot|mv|linalg"
        r"|svd|eig|cholesky|qr|lu|lstsq|solve|inv|det|slogdet"
        r"|pinverse|triangular_solve|ormqr|geqrf)"),
     OpCategory.LINALG),
    (_P(r"(^|_)(batch_norm|layer_norm|group_norm|instance_norm"
        r"|renorm|weight_norm|rms_norm)"),
     OpCategory.NORM),
    (_P(r"(^|_)(nll_loss|cross_entropy|binary_cross_entropy|mse_loss"
        r"|smooth_l1_loss|huber_loss|kl_div|cosine_embedding_loss"
        r"|ctc_loss|margin_ranking_loss|multi_margin_loss"
        r"|multilabel_margin_loss|triplet_margin"
        r"|hinge_embedding_loss|poisson_nll_loss|soft_margin_loss)"),
     OpCategory.LOSS),
    (_P(r"(^|_)(fft|ifft|rfft|irfft|hfft|ihfft|stft|istft)"),
     OpCategory.FFT),
    (_P(r"(^|_)(scatter|gather|index_put|index_add|index_copy"
        r"|index_fill|index_select|index\.Tensor|masked_fill"
        r"|masked_scatter|masked_select|put|take|embedding|one_hot)"),
     OpCategory.SCATTER_GATHER),
    (_P(r"(^|_)(cumsum|cumprod|cummax|cummin|logcumsumexp"
        r"|scan|associative_scan)"),
     OpCategory.SCAN),
    (_P(r"(^|_)(bernoulli|multinomial|normal|uniform|rand"
        r"|poisson|exponential|geometric|log_normal|cauchy|dropout)"),
     OpCategory.RANDOM),
]


def _is_view_op(op: OpOverload) -> bool:
    """Detect view ops structurally: returns alias input without writing."""
    returns = op._schema.returns
    if not returns:
        return False
    return any(
        r.alias_info is not None and not r.alias_info.is_write
        for r in returns
    )


def _get_op_category(op: OpOverload) -> OpCategory:
    """Classify an op's computational pattern.

    Detection priority:
    1. PyTorch tags (authoritative: pointwise, reduction, view_copy,
       nondeterministic_seeded)
    2. Schema analysis (non-write return alias → view, no tensor inputs → factory)
    3. Op name heuristics (norm, linalg, loss, fft, spatial, etc.)
    4. Fallback → OTHER

    Note: categories mix computational shape (pointwise, reduction, view, factory)
    with ML domain (loss, norm, spatial). This is intentional — the classification
    serves human comprehension, not a formal type system.
    """
    tags = op.tags

    # 1. PyTorch tags — authoritative, structural
    if torch.Tag.pointwise in tags:
        return OpCategory.POINTWISE
    if torch.Tag.reduction in tags:
        return OpCategory.REDUCTION
    if torch.Tag.view_copy in tags:
        return OpCategory.VIEW

    # 2. Schema analysis — structural
    if _is_view_op(op):
        return OpCategory.VIEW
    if not _has_tensor_input(op):
        return OpCategory.FACTORY
    if torch.Tag.nondeterministic_seeded in tags:
        return OpCategory.RANDOM

    # 3. Name heuristics — fragile, but useful for domain categories
    name = op.name().split("::")[1] if "::" in op.name() else op.name()
    base = name.split(".")[0]

    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.search(base):
            return category

    return OpCategory.OTHER


def _get_dtensor_strategy(op: OpOverload) -> DtensorStrategy:
    """Check DTensor sharding strategy registration status."""
    dt = _init_dtensor()
    prop = dt.propagator
    decomp_cls = dt.decomp_strategy_cls

    # Check registered strategies (DTensor has an explicit handler).
    # Attribute names vary across PyTorch versions, so guard each.
    if hasattr(prop, "op_strategy_funcs") and op in prop.op_strategy_funcs:
        return DtensorStrategy.REGISTERED
    if hasattr(prop, "op_single_dim_strategy_funcs") and op in prop.op_single_dim_strategy_funcs:
        return DtensorStrategy.REGISTERED
    if hasattr(prop, "op_to_rules") and op in prop.op_to_rules:
        return DtensorStrategy.REGISTERED

    # Check DTensor's own decomposition awareness.
    if decomp_cls is not None and decomp_cls.has_decomp(op):
        return DtensorStrategy.DECOMP_FALLBACK

    # CIA ops auto-decompose before DTensor dispatch, so DTensor
    # handles the children via fallback — same as table decomps.
    # This check is independent of DTensor imports.
    if op._can_decompose():
        return DtensorStrategy.DECOMP_FALLBACK

    # Factory/allocation ops have no tensor inputs, so DTensor dispatch
    # (which is input-driven via overloaded_args) can never reach them.
    if not _has_tensor_input(op):
        return DtensorStrategy.NOT_APPLICABLE

    return DtensorStrategy.MISSING


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
        op_category=_get_op_category(op),
        dtensor_strategy=_get_dtensor_strategy(op) if dtensor else None,
        autograd_type=autograd_type,
        has_adiov=has_adiov,
    )
