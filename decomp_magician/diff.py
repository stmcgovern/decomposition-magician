"""Diff mode: compare decomposition trees between modes or overloads."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from torch._ops import OpOverload

from decomp_magician.tree import build_tree, collect_leaf_counts, op_display_name


@dataclass(frozen=True)
class DecompDiff:
    """Diff between two decomposition trees."""
    left_label: str
    right_label: str
    left_leaves: Counter[str]
    right_leaves: Counter[str]
    added: Counter[str]      # ops in right but not left
    removed: Counter[str]    # ops in left but not right
    changed: list[tuple[str, int, int]]  # (op, left_count, right_count) where counts differ


def compute_diff(
    op: OpOverload,
    depth: int = -1,
) -> DecompDiff:
    """Compare full vs compile decomposition for an op."""
    full_node = build_tree(op, depth=depth, compile=False)
    compile_node = build_tree(op, depth=depth, compile=True)

    return _diff_trees(
        collect_leaf_counts(full_node),
        collect_leaf_counts(compile_node),
        left_label=f"{op_display_name(op)}  (full)",
        right_label=f"{op_display_name(op)}  (compile)",
    )


def compute_diff_ops(
    left: OpOverload,
    right: OpOverload,
    depth: int = -1,
    compile: bool = False,
) -> DecompDiff:
    """Compare decomposition trees of two different ops."""
    left_node = build_tree(left, depth=depth, compile=compile)
    right_node = build_tree(right, depth=depth, compile=compile)

    mode = "compile" if compile else "full"
    return _diff_trees(
        collect_leaf_counts(left_node),
        collect_leaf_counts(right_node),
        left_label=f"{op_display_name(left)}  ({mode})",
        right_label=f"{op_display_name(right)}  ({mode})",
    )


def _diff_trees(
    left_leaves: Counter[str],
    right_leaves: Counter[str],
    left_label: str,
    right_label: str,
) -> DecompDiff:
    """Compute diff between two leaf count sets."""
    all_ops = set(left_leaves) | set(right_leaves)

    added: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    changed: list[tuple[str, int, int]] = []

    for name in sorted(all_ops):
        lc = left_leaves.get(name, 0)
        rc = right_leaves.get(name, 0)
        if lc == 0 and rc > 0:
            added[name] = rc
        elif lc > 0 and rc == 0:
            removed[name] = lc
        elif lc != rc:
            changed.append((name, lc, rc))

    return DecompDiff(
        left_label=left_label,
        right_label=right_label,
        left_leaves=left_leaves,
        right_leaves=right_leaves,
        added=added,
        removed=removed,
        changed=changed,
    )


