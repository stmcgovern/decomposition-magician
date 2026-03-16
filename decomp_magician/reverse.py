"""Reverse lookup: find which ops decompose into a given target op."""

from __future__ import annotations

from collections import Counter

from torch._ops import OpOverload

from decomp_magician.tree import DecompNode, build_tree, op_display_name


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


def _get_all_decomposable_ops() -> list[OpOverload]:
    """Get all ops from the decomposition table.

    CIA-only ops are excluded because some crash at the C level during
    op.decompose() (SIGFPE/segfault), which cannot be caught in Python.
    """
    from torch._decomp import decomposition_table

    return list(decomposition_table.keys())


def _is_out_variant(name: str) -> bool:
    """Check if an op name is an _out variant."""
    overload = name.rsplit(".", 1)[-1] if "." in name else ""
    return overload == "out" or overload.endswith("_out")


def reverse_lookup(
    target: str,
    depth: int = -1,
    compile: bool = False,
    include_out: bool = False,
) -> list[dict]:
    """Find all ops whose decomposition tree contains the target op.

    Scans the decomposition table (CIA-only ops excluded due to C-level crashes).

    Args:
        target: The op name to search for in decomposition trees.
        depth: Decomposition depth (-1 for unlimited).
        compile: If True, treat inductor-kept ops as leaves.
        include_out: If True, include _out variant ops (usually duplicates).

    Returns a list of dicts sorted by count descending:
        [{"op": name, "count": n, "target_depth": d}, ...]
    """
    import warnings

    ops = _get_all_decomposable_ops()
    results = []

    for op in ops:
        name = op_display_name(op)
        # Skip if the op itself is the target
        if name == target:
            continue
        # Skip _out variants unless requested
        if not include_out and _is_out_variant(name):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                node = build_tree(op, depth=depth, compile=compile)
        except Exception:
            continue

        all_ops, target_depth = _search_tree(node, target)
        if target in all_ops:
            results.append({
                "op": name,
                "count": all_ops[target],
                "target_depth": target_depth or 0,
            })

    results.sort(key=lambda r: r["count"], reverse=True)
    return results
