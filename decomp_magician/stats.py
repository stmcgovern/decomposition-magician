"""Bulk statistics across all decomposable ops."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass

from decomp_magician.classify import classify
from decomp_magician.reverse import _is_out_variant
from decomp_magician.tree import DecompNode, build_tree, op_display_name


@dataclass(frozen=True)
class StatsResult:
    """Statistics across all decomposable ops.

    Invariant: traceable + untraceable + classify_errors == total_non_out.
    This is checked at construction time.
    """
    total: int
    total_non_out: int
    by_type: dict[str, int]
    inductor_kept: int
    traceable: int
    untraceable: int
    classify_errors: int
    leaf_ops: Counter[str]
    deepest: list[tuple[str, int]]

    def __post_init__(self):
        accounted = self.traceable + self.untraceable + self.classify_errors
        if accounted != self.total_non_out:
            raise ValueError(
                f"Accounting invariant violated: "
                f"traceable({self.traceable}) + untraceable({self.untraceable}) "
                f"+ classify_errors({self.classify_errors}) = {accounted} "
                f"!= total_non_out({self.total_non_out})"
            )


def compute_stats(compile: bool = False) -> StatsResult:
    """Compute statistics across all decomposable ops."""
    from torch._decomp import decomposition_table

    all_ops = list(decomposition_table.keys())
    by_type: Counter[str] = Counter()
    inductor_kept = 0
    traceable = 0
    untraceable = 0
    classify_errors = 0
    leaf_ops: Counter[str] = Counter()
    depths: list[tuple[str, int]] = []
    non_out_count = 0

    for op in all_ops:
        name = op_display_name(op)
        if _is_out_variant(name):
            continue

        non_out_count += 1

        try:
            cls = classify(op)
        except (AttributeError, Exception):
            classify_errors += 1
            continue

        by_type[cls.decomp_type] += 1
        if cls.inductor_kept:
            inductor_kept += 1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                node = build_tree(op, compile=compile)
        except Exception:
            untraceable += 1
            continue

        if not node.traceable:
            untraceable += 1
        else:
            traceable += 1
            if node.children:
                depth = _tree_depth(node)
                depths.append((name, depth))
                _collect_leaves(node, leaf_ops)

    depths.sort(key=lambda x: x[1], reverse=True)

    return StatsResult(
        total=len(all_ops),
        total_non_out=non_out_count,
        by_type=dict(by_type),
        inductor_kept=inductor_kept,
        traceable=traceable,
        untraceable=untraceable,
        classify_errors=classify_errors,
        leaf_ops=leaf_ops,
        deepest=depths[:10],
    )


def _tree_depth(node: DecompNode) -> int:
    if not node.children:
        return 0
    return 1 + max(_tree_depth(c) for c in node.children)


def _collect_leaves(node: DecompNode, counter: Counter[str]) -> None:
    """Collect unique leaf op names from a tree (not propagated counts)."""
    if not node.children:
        counter[op_display_name(node.op)] += 1
        return
    for c in node.children:
        _collect_leaves(c, counter)
