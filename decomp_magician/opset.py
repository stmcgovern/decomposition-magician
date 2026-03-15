"""Target opset definitions and coverage checking."""

from __future__ import annotations

from dataclasses import dataclass

from torch._ops import OpOverload

from decomp_magician.tree import build_tree, collect_leaf_counts, op_display_name


# Supported target opsets
OPSETS = ("core_aten",)


def _build_core_aten_decomposed() -> set[str]:
    """Return the set of op names that have core ATen decompositions.

    These are NOT core ATen ops — they decompose into core ops.
    An op is "core ATen" if it is NOT in this set.
    """
    from torch._decomp import core_aten_decompositions

    return {op_display_name(op) for op in core_aten_decompositions()}


_core_aten_decomposed: set[str] | None = None


def _get_core_aten_decomposed() -> set[str]:
    global _core_aten_decomposed
    if _core_aten_decomposed is None:
        _core_aten_decomposed = _build_core_aten_decomposed()
    return _core_aten_decomposed


def is_core_aten(op_name: str) -> bool:
    """Check if an op name is a core ATen op (not decomposed further)."""
    return op_name not in _get_core_aten_decomposed()


@dataclass(frozen=True)
class OpsetCoverage:
    """Result of checking an op's decomposition against a target opset.

    Invariant: covered_leaves + sum(non_covered counts) == total_leaves.
    """
    op: str
    opset: str
    covered_leaves: int
    non_covered: tuple[tuple[str, int], ...]  # (op_name, count) for non-covered leaves

    def __post_init__(self):
        nc_total = sum(c for _, c in self.non_covered)
        if self.covered_leaves < 0 or nc_total < 0:
            raise ValueError("Leaf counts must be non-negative")

    @property
    def total_leaves(self) -> int:
        return self.covered_leaves + sum(c for _, c in self.non_covered)

    @property
    def fully_covered(self) -> bool:
        return len(self.non_covered) == 0


def check_opset_coverage(
    op: OpOverload,
    opset: str = "core_aten",
    depth: int = -1,
    compile: bool = False,
) -> OpsetCoverage:
    """Check whether an op decomposes fully to ops in the target opset.

    Args:
        op: The operator to check.
        opset: Target opset name (currently only "core_aten").
        depth: Maximum decomposition depth (-1 for unlimited).
        compile: If True, use compile-mode decomposition.

    Returns:
        OpsetCoverage with details about which leaves are/aren't in the opset.
    """
    if opset not in OPSETS:
        raise ValueError(f"Unknown opset: {opset!r}. Supported: {', '.join(OPSETS)}")

    node = build_tree(op, depth=depth, compile=compile)
    leaf_counts = collect_leaf_counts(node)

    covered = 0
    non_covered: list[tuple[str, int]] = []

    for name, count in leaf_counts.most_common():
        if is_core_aten(name):
            covered += count
        else:
            non_covered.append((name, count))

    return OpsetCoverage(
        op=op_display_name(op),
        opset=opset,
        covered_leaves=covered,
        non_covered=tuple(non_covered),
    )
