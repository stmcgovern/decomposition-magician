"""Diff mode: compare decomposition trees between modes or overloads."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from torch._ops import OpOverload

from decomp_magician.tree import build_tree, collect_leaf_counts, op_display_name


@dataclass(frozen=True)
class DecompDiff:
    """Diff between two decomposition trees.

    Only stores the two leaf counters and labels. The added/removed/changed
    views are derived properties — no redundant state to go stale.
    """
    left_label: str
    right_label: str
    left_leaves: Counter[str]
    right_leaves: Counter[str]

    @property
    def added(self) -> Counter[str]:
        """Ops in right but not left."""
        return Counter({n: c for n, c in self.right_leaves.items()
                        if n not in self.left_leaves})

    @property
    def removed(self) -> Counter[str]:
        """Ops in left but not right."""
        return Counter({n: c for n, c in self.left_leaves.items()
                        if n not in self.right_leaves})

    @property
    def changed(self) -> list[tuple[str, int, int]]:
        """Ops present in both but with different counts."""
        result = []
        for name in sorted(set(self.left_leaves) & set(self.right_leaves)):
            lc = self.left_leaves[name]
            rc = self.right_leaves[name]
            if lc != rc:
                result.append((name, lc, rc))
        return result


def compute_diff(
    op: OpOverload,
    depth: int = -1,
) -> DecompDiff:
    """Compare full vs compile decomposition for an op."""
    full_node = build_tree(op, depth=depth, compile=False)
    compile_node = build_tree(op, depth=depth, compile=True)

    return DecompDiff(
        left_label=f"{op_display_name(op)}  (full)",
        right_label=f"{op_display_name(op)}  (compile)",
        left_leaves=collect_leaf_counts(full_node),
        right_leaves=collect_leaf_counts(compile_node),
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
    return DecompDiff(
        left_label=f"{op_display_name(left)}  ({mode})",
        right_label=f"{op_display_name(right)}  ({mode})",
        left_leaves=collect_leaf_counts(left_node),
        right_leaves=collect_leaf_counts(right_node),
    )
