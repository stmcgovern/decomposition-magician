"""Decomposition tree construction via meta tensor tracing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from decomp_magician.classify import OpClass, classify


def op_display_name(op) -> str:
    """Short display name: aten.add.Tensor, always showing overload."""
    name = op.name()
    dotted = name.replace("::", ".")
    if dotted.count(".") < 2:
        dotted += ".default"
    return dotted


@dataclass
class DecompNode:
    op: OpOverload
    children: list[DecompNode] = field(default_factory=list)
    count: int = 1
    classification: OpClass = field(default_factory=lambda: OpClass("leaf"))
    traceable: bool = True
    error: str | None = None


class _RecordingMode(TorchDispatchMode):
    """Records all OpOverload calls during decomposition tracing."""

    def __init__(self):
        super().__init__()
        self.ops: list[OpOverload] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if isinstance(func, OpOverload):
            self.ops.append(func)
        return func(*args, **(kwargs or {}))


def _get_decomp_fn(op: OpOverload):
    """Get the decomposition function for an op, checking table first."""
    from torch._decomp import decomposition_table

    if op in decomposition_table:
        return decomposition_table[op]
    if op._can_decompose():
        return op.decompose
    return None


def _make_meta_args(op: OpOverload) -> tuple[list, dict] | None:
    """Create meta tensor arguments from the op's schema.

    Returns (args, kwargs) or None if creation fails.
    """
    args = []
    for arg in op._schema.arguments:
        if arg.kwarg_only:
            break
        val = _make_arg(arg)
        if val is _SENTINEL:
            return None
        args.append(val)

    kwargs = {}
    for arg in op._schema.arguments:
        if not arg.kwarg_only:
            continue
        val = _make_arg(arg)
        if val is _SENTINEL:
            return None
        kwargs[arg.name] = val

    return args, kwargs


_SENTINEL = object()

# Default shapes by tensor position/name heuristics
_DEFAULT_SHAPE = [2, 3, 4, 4]
_SMALL_SHAPE = [3]


_SMALL_NAMES = ("weight", "bias", "mean", "var", "scale", "zero_point")
_SCALAR_NAMES = ("value", "fill_value", "other")
_INDEX_NAMES = ("indices", "index")


def _make_meta_tensor(name: str) -> torch.Tensor:
    """Create a meta tensor with shape chosen by argument name heuristic."""
    name_lower = name.lower()
    # Scalar tensors (0-D) for value-like args
    if name_lower in _SCALAR_NAMES:
        return torch.empty([], device="meta")
    # Index tensors need int64 dtype
    if name_lower in _INDEX_NAMES:
        return torch.empty(_DEFAULT_SHAPE, device="meta", dtype=torch.int64)
    # 1D for weight/bias/mean/var-like args
    if any(n in name_lower for n in _SMALL_NAMES):
        return torch.empty(_SMALL_SHAPE, device="meta")
    return torch.empty(_DEFAULT_SHAPE, device="meta")


def _make_arg(arg):
    """Create a single argument value from schema info."""
    type_str = str(arg.type)
    kind = arg.type.kind()

    if kind == "TensorType":
        return _make_meta_tensor(arg.name)

    if kind == "OptionalType":
        if "Tensor" in type_str:
            return _make_meta_tensor(arg.name)
        if arg.default_value is not None:
            return arg.default_value
        # Most optional scalars should stay None (their schema default).
        # Some ops (e.g. clamp) need at least one optional to be non-None —
        # those are handled by the retry logic in _trace_decomp.
        return None

    if kind == "BoolType":
        if arg.default_value is not None:
            return arg.default_value
        return True

    if kind == "IntType":
        if arg.default_value is not None:
            return arg.default_value
        name_lower = arg.name.lower()
        if "dim" in name_lower:
            return 1
        return 0

    if kind == "FloatType":
        if arg.default_value is not None:
            return arg.default_value
        return 1e-5

    if kind == "NumberType":
        if arg.default_value is not None:
            return arg.default_value
        return 1.0

    if kind == "ListType":
        if arg.default_value is not None:
            return arg.default_value
        elem_kind = arg.type.getElementType().kind()
        if elem_kind == "IntType":
            return [1]
        if elem_kind == "FloatType":
            return [1.0]
        if elem_kind == "TensorType":
            return [torch.empty(_DEFAULT_SHAPE, device="meta")]
        return []

    if kind == "StringType":
        if arg.default_value is not None:
            return arg.default_value
        return ""

    if kind == "DeviceObjType":
        return torch.device("meta")

    if kind == "ScalarTypeType":
        if arg.default_value is not None:
            return arg.default_value
        return None

    # Unknown type — fail this arg
    return _SENTINEL


_trace_cache: dict[OpOverload, list[OpOverload] | str] = {}


def _trace_decomp(op: OpOverload) -> list[OpOverload] | str:
    """Run the decomposition and record which ops are called.

    Results are cached since the same op always produces the same decomposition.
    Returns a list of ops, or an error string if tracing fails.
    """
    if op in _trace_cache:
        return _trace_cache[op]

    result = _trace_decomp_uncached(op)
    _trace_cache[op] = result
    return result


def _trace_decomp_uncached(op: OpOverload) -> list[OpOverload] | str:
    """Run the decomposition and record which ops are called (no caching)."""
    decomp_fn = _get_decomp_fn(op)
    if decomp_fn is None:
        return "no decomposition"

    meta_result = _make_meta_args(op)
    if meta_result is None:
        return f"cannot create meta args for {op.name()}"

    args, kwargs = meta_result
    result = _try_trace(decomp_fn, args, kwargs)
    if isinstance(result, list):
        return result

    # Retry with booleans flipped (e.g. _softmax's half_to_float=True
    # asserts half input, but our default tensors are float32)
    flipped = _flip_bools(args, kwargs)
    if flipped is not None:
        retry = _try_trace(decomp_fn, *flipped)
        if isinstance(retry, list):
            return retry

    # Retry with optional scalars filled in (e.g. clamp(min=None, max=None)
    # needs at least one non-None)
    filled = _fill_optional_scalars(op, args, kwargs)
    if filled is not None:
        retry = _try_trace(decomp_fn, *filled)
        if isinstance(retry, list):
            return retry

    # Retry with integer dtype for ops that require integral tensors
    if "integral" in result or "int" in result.lower():
        int_args = [
            a.to(torch.int64) if isinstance(a, torch.Tensor) else a for a in args
        ]
        int_kwargs = {
            k: v.to(torch.int64) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        retry = _try_trace(decomp_fn, int_args, int_kwargs)
        if isinstance(retry, list):
            return retry

    return result


def _flip_bools(args: list, kwargs: dict) -> tuple[list, dict] | None:
    """Flip all boolean args/kwargs. Returns (new_args, new_kwargs) or None if no bools."""
    has_bool = False
    new_args = []
    for a in args:
        if isinstance(a, bool):
            new_args.append(not a)
            has_bool = True
        else:
            new_args.append(a)
    new_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, bool):
            new_kwargs[k] = not v
            has_bool = True
        else:
            new_kwargs[k] = v
    return (new_args, new_kwargs) if has_bool else None


def _fill_optional_scalars(op: OpOverload, args: list, kwargs: dict) -> tuple[list, dict] | None:
    """Fill None optional scalars with defaults. Returns (new_args, new_kwargs) or None if nothing changed."""
    changed = False
    new_args = list(args)
    new_kwargs = dict(kwargs)
    pos_idx = 0
    for arg in op._schema.arguments:
        if arg.type.kind() != "OptionalType":
            if not arg.kwarg_only:
                pos_idx += 1
            continue
        if "Tensor" in str(arg.type):
            if not arg.kwarg_only:
                pos_idx += 1
            continue
        elem = str(arg.type)
        fill_val = None
        if "int" in elem:
            fill_val = 0
        elif "float" in elem or "number" in elem:
            fill_val = 1.0
        if fill_val is None:
            if not arg.kwarg_only:
                pos_idx += 1
            continue
        if arg.kwarg_only:
            if new_kwargs.get(arg.name) is None:
                new_kwargs[arg.name] = fill_val
                changed = True
        else:
            if pos_idx < len(new_args) and new_args[pos_idx] is None:
                new_args[pos_idx] = fill_val
                changed = True
            pos_idx += 1
    return (new_args, new_kwargs) if changed else None


def _try_trace(decomp_fn, args, kwargs) -> list[OpOverload] | str:
    """Attempt to trace a decomposition function, returning ops or error."""
    recorder = _RecordingMode()
    try:
        with recorder:
            decomp_fn(*args, **kwargs)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return recorder.ops


def build_tree(
    op: OpOverload,
    depth: int = -1,
    dtensor: bool = False,
    compile: bool = False,
    _ancestors: frozenset[str] | None = None,
) -> DecompNode:
    """Build a decomposition tree for the given op.

    Args:
        op: The operator to decompose.
        depth: Maximum recursion depth. -1 for unlimited.
        dtensor: If True, classify DTensor strategy coverage.
        compile: If True, treat inductor-kept ops as leaves.
    """
    if _ancestors is None:
        _ancestors = frozenset()

    node = DecompNode(
        op=op,
        classification=classify(op, dtensor=dtensor),
    )

    # Leaf or depth exhausted — no children
    if node.classification.decomp_type == "leaf" or depth == 0:
        return node

    # In compile mode, inductor-kept ops are treated as leaves
    if compile and node.classification.inductor_kept:
        return node

    # Cycle detection: if this op is already an ancestor, stop recursion
    op_name = op.name()
    if op_name in _ancestors:
        node.traceable = False
        node.error = "cycle detected"
        return node

    result = _trace_decomp(op)
    if isinstance(result, str):
        node.traceable = False
        node.error = result
        return node

    # Count and deduplicate
    op_counts = Counter(result)
    next_depth = depth - 1 if depth > 0 else -1
    child_ancestors = _ancestors | {op_name}

    for child_op, count in op_counts.items():
        child = build_tree(
            child_op, depth=next_depth, dtensor=dtensor, compile=compile,
            _ancestors=child_ancestors,
        )
        child.count = count
        node.children.append(child)

    return node
