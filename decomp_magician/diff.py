"""Diff mode: compare decomposition trees between modes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from torch._ops import OpOverload

from decomp_magician.tree import DecompNode, build_tree, op_display_name


@dataclass(frozen=True)
class DecompDiff:
    """Diff between two decomposition modes for a single op."""
    op: str
    left_mode: str
    right_mode: str
    left_leaves: Counter[str]
    right_leaves: Counter[str]
    added: Counter[str]      # ops in right but not left
    removed: Counter[str]    # ops in left but not right
    changed: list[tuple[str, int, int]]  # (op, left_count, right_count) where counts differ


def compute_diff(
    op: OpOverload,
    depth: int = -1,
) -> DecompDiff:
    """Compare full vs compile decomposition for an op.

    Args:
        op: The operator to compare.
        depth: Maximum decomposition depth (-1 for unlimited).

    Returns:
        DecompDiff showing what changes between full and compile modes.
    """
    full_node = build_tree(op, depth=depth, compile=False)
    compile_node = build_tree(op, depth=depth, compile=True)

    full_leaves = _collect_leaf_counts(full_node)
    compile_leaves = _collect_leaf_counts(compile_node)

    all_ops = set(full_leaves) | set(compile_leaves)

    added: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    changed: list[tuple[str, int, int]] = []

    for name in sorted(all_ops):
        fc = full_leaves.get(name, 0)
        cc = compile_leaves.get(name, 0)
        if fc == 0 and cc > 0:
            added[name] = cc
        elif fc > 0 and cc == 0:
            removed[name] = fc
        elif fc != cc:
            changed.append((name, fc, cc))

    return DecompDiff(
        op=op_display_name(op),
        left_mode="full",
        right_mode="compile",
        left_leaves=full_leaves,
        right_leaves=compile_leaves,
        added=added,
        removed=removed,
        changed=changed,
    )


def _collect_leaf_counts(node: DecompNode) -> Counter[str]:
    """Collect leaf ops with propagated counts."""
    counter: Counter[str] = Counter()

    def walk(n: DecompNode, multiplier: int = 1) -> None:
        if not n.children:
            counter[op_display_name(n.op)] += multiplier
            return
        for c in n.children:
            walk(c, multiplier * c.count)

    walk(node)
    return counter
