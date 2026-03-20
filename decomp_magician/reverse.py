"""Reverse lookup: find which ops decompose into a given target op."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from decomp_magician.classify import get_all_decomposable_ops, is_out_variant
from decomp_magician.tree import DecompNode, build_tree, op_display_name


@dataclass(frozen=True)
class ReverseEntry:
    """A single result from reverse_lookup."""
    op: str
    count: int
    target_depth: int


def _search_tree(node: DecompNode, target: str) -> tuple[Counter[str], int | None]:
    """Walk a tree collecting all ops with propagated counts, and find shallowest target depth.

    Returns (all_ops_counter, shallowest_target_depth_or_None).
    """
    ops: Counter[str] = Counter()
    best_depth: int | None = None

    def walk(n: DecompNode, multiplier: int = 1, depth: int = 0) -> None:
        nonlocal best_depth
        name = op_display_name(n.op)
        ops[name] += multiplier
        if name == target and depth > 0:
            if best_depth is None or depth < best_depth:
                best_depth = depth
        for c in n.children:
            walk(c, multiplier * c.count, depth + 1)

    walk(node)
    return ops, best_depth


def reverse_lookup(
    target: str,
    depth: int = -1,
    compile: bool = False,
    include_out: bool = False,
) -> list[ReverseEntry]:
    """Find all ops whose decomposition tree contains the target op.

    Scans the decomposition table (CIA-only ops excluded due to C-level crashes).

    Args:
        target: The op name to search for in decomposition trees.
        depth: Decomposition depth (-1 for unlimited).
        compile: If True, treat inductor-kept ops as leaves.
        include_out: If True, include _out variant ops (usually duplicates).

    Returns a list of ReverseEntry sorted by count descending.
    """
    import warnings

    ops = get_all_decomposable_ops()
    results: list[ReverseEntry] = []

    for op in ops:
        name = op_display_name(op)
        # Skip if the op itself is the target
        if name == target:
            continue
        # Skip _out variants unless requested
        if not include_out and is_out_variant(name):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                node = build_tree(op, depth=depth, compile=compile)
        except Exception:
            continue

        all_ops, target_depth = _search_tree(node, target)
        if target in all_ops:
            # target in all_ops guarantees _search_tree found it at depth > 0
            assert target_depth is not None
            results.append(ReverseEntry(
                op=name,
                count=all_ops[target],
                target_depth=target_depth,
            ))

    results.sort(key=lambda r: r.count, reverse=True)
    return results
