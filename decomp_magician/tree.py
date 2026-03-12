"""Decomposition tree construction via meta tensor tracing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from decomp_magician.classify import OpClass, classify


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


def _make_arg(arg):
    """Create a single argument value from schema info."""
    type_str = str(arg.type)
    kind = arg.type.kind()

    if kind == "TensorType":
        # Heuristic: 1D tensors for weight/bias/mean/var-like args,
        # 4D for input-like args
        name_lower = arg.name.lower()
        if any(
            n in name_lower
            for n in ("weight", "bias", "mean", "var", "scale", "zero_point")
        ):
            return torch.empty(_SMALL_SHAPE, device="meta")
        return torch.empty(_DEFAULT_SHAPE, device="meta")

    if kind == "OptionalType":
        if "Tensor" in type_str:
            name_lower = arg.name.lower()
            if any(
                n in name_lower
                for n in (
                    "weight",
                    "bias",
                    "mean",
                    "var",
                    "scale",
                    "zero_point",
                )
            ):
                return torch.empty(_SMALL_SHAPE, device="meta")
            return torch.empty(_DEFAULT_SHAPE, device="meta")
        if arg.default_value is not None:
            return arg.default_value
        # Provide values for optional scalars instead of None,
        # since decomps often require at least one to be non-None
        elem = str(arg.type)
        if "int" in elem:
            return 0
        if "float" in elem or "number" in elem:
            return 1.0
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


def _trace_decomp(op: OpOverload) -> list[OpOverload] | str:
    """Run the decomposition and record which ops are called.

    Returns a list of ops, or an error string if tracing fails.
    """
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
