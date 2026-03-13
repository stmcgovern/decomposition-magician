"""Bulk statistics across all decomposable ops."""

from __future__ import annotations

import warnings
from collections import Counter

from decomp_magician.classify import classify
from decomp_magician.reverse import _is_out_variant
from decomp_magician.tree import DecompNode, build_tree, op_display_name


def compute_stats(compile: bool = False) -> dict:
    """Compute statistics across all decomposable ops.

    Returns a dict with:
        total: total ops in decomposition table
        by_type: {"table": n, "both": n, ...}
        inductor_kept: count of inductor-kept ops
        traceable: count of ops that trace successfully
        untraceable: count of ops that fail tracing
        leaf_ops: Counter of leaf op appearances across all decompositions
        deepest: list of (op_name, depth) for deepest decomposition chains
    """
    from torch._decomp import decomposition_table

    all_ops = list(decomposition_table.keys())
    by_type: Counter[str] = Counter()
    inductor_kept = 0
    traceable = 0
    untraceable = 0
    classify_errors = 0
    leaf_ops: Counter[str] = Counter()
    depths: list[tuple[str, int]] = []

    for op in all_ops:
        name = op_display_name(op)
        if _is_out_variant(name):
            continue

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

        if node.children:
            traceable += 1
            depth = _tree_depth(node)
            depths.append((name, depth))
            _collect_leaves(node, leaf_ops)
        elif not node.traceable:
            untraceable += 1

    depths.sort(key=lambda x: x[1], reverse=True)

    non_out_total = sum(by_type.values())

    return {
        "total": len(all_ops),
        "total_non_out": non_out_total,
        "by_type": dict(by_type),
        "inductor_kept": inductor_kept,
        "traceable": traceable,
        "untraceable": untraceable,
        "leaf_ops": leaf_ops,
        "deepest": depths[:10],
    }


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
