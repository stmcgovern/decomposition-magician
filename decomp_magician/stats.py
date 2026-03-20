"""Bulk statistics across all decomposable ops."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass

from decomp_magician.classify import (
    DtensorStrategy, classify, get_all_decomposable_ops,
    is_dtensor_gap, is_dtensor_intercept, is_out_variant,
)
from decomp_magician.tree import DecompNode, build_tree, op_display_name


@dataclass(frozen=True)
class DtensorStats:
    """DTensor coverage statistics across all decomposable ops."""
    registered: int  # ops with a direct DTensor strategy
    decomp_fallback: int  # ops handled via decomposition tracing
    missing: int  # ops with no DTensor strategy at all
    not_applicable: int  # ops with no tensor inputs (unreachable by DTensor dispatch)
    fully_covered: int  # traceable ops where all leaf paths hit a registered ancestor
    has_gaps: int  # traceable ops with at least one uncovered leaf path
    top_uncovered: list[tuple[str, int]]  # most common uncovered leaf ops


@dataclass(frozen=True)
class StatsResult:
    """Statistics across all decomposable ops.

    Invariants (checked at construction time):
    1. traceable + untraceable + classify_errors == total_non_out
    2. sum(by_type.values()) == total_non_out - classify_errors
    3. len(untraceable_ops) == untraceable
    4. dtensor.registered + dtensor.decomp_fallback + dtensor.missing
       + dtensor.not_applicable == total_non_out - classify_errors  (when dtensor is not None)
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
    untraceable_ops: tuple[tuple[str, str], ...] = ()
    dtensor: DtensorStats | None = None

    def __post_init__(self):
        accounted = self.traceable + self.untraceable + self.classify_errors
        if accounted != self.total_non_out:
            raise ValueError(
                f"Accounting invariant violated: "
                f"traceable({self.traceable}) + untraceable({self.untraceable}) "
                f"+ classify_errors({self.classify_errors}) = {accounted} "
                f"!= total_non_out({self.total_non_out})"
            )
        type_sum = sum(self.by_type.values())
        classified = self.total_non_out - self.classify_errors
        if type_sum != classified:
            raise ValueError(
                f"Type accounting invariant violated: "
                f"sum(by_type) = {type_sum} "
                f"!= total_non_out({self.total_non_out}) "
                f"- classify_errors({self.classify_errors}) = {classified}"
            )
        if len(self.untraceable_ops) != self.untraceable:
            raise ValueError(
                f"Untraceable ops list invariant violated: "
                f"len(untraceable_ops)={len(self.untraceable_ops)} "
                f"!= untraceable={self.untraceable}"
            )
        if self.dtensor is not None:
            dt = self.dtensor
            dt_sum = dt.registered + dt.decomp_fallback + dt.missing + dt.not_applicable
            if dt_sum != classified:
                raise ValueError(
                    f"DTensor partition invariant violated: "
                    f"registered({dt.registered}) + decomp_fallback({dt.decomp_fallback}) "
                    f"+ missing({dt.missing}) + not_applicable({dt.not_applicable}) "
                    f"= {dt_sum} != {classified}"
                )


def compute_stats(
    compile: bool = False, dtensor: bool = False,
) -> StatsResult:
    """Compute statistics across all decomposable ops.

    Args:
        compile: If True, treat inductor-kept ops as leaves.
        dtensor: If True, include DTensor coverage analysis in the result.
                 Classification always includes DTensor strategy (it's intrinsic);
                 this flag controls whether to aggregate and report it.
    """
    all_ops = get_all_decomposable_ops()
    by_type: Counter[str] = Counter()
    inductor_kept = 0
    traceable = 0
    untraceable = 0
    classify_errors = 0
    leaf_ops: Counter[str] = Counter()
    depths: list[tuple[str, int]] = []
    untraceable_list: list[tuple[str, str]] = []
    non_out_count = 0

    # DTensor tracking
    dt_by_strategy: Counter[str] = Counter()
    dt_fully_covered = 0
    dt_has_gaps = 0
    dt_uncovered_leaves: Counter[str] = Counter()

    for op in all_ops:
        name = op_display_name(op)
        if is_out_variant(name):
            continue

        non_out_count += 1

        try:
            cls = classify(op)
        except Exception:
            classify_errors += 1
            continue

        by_type[cls.decomp_type] += 1
        if cls.inductor_kept:
            inductor_kept += 1

        if dtensor:
            dt_by_strategy[cls.dtensor_strategy] += 1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                node = build_tree(op, compile=compile)
        except Exception as e:
            untraceable += 1
            untraceable_list.append((name, f"{type(e).__name__}: {e}"))
            continue

        if not node.traceable:
            untraceable += 1
            untraceable_list.append((name, node.error))
        else:
            traceable += 1
            if node.children:
                depth = _tree_depth(node)
                depths.append((name, depth))
                _collect_leaves(node, leaf_ops)

                if dtensor:
                    uncovered = _collect_uncovered_dtensor(node)
                    if uncovered:
                        dt_has_gaps += 1
                        for leaf_name in uncovered:
                            dt_uncovered_leaves[leaf_name] += 1
                    else:
                        dt_fully_covered += 1

    depths.sort(key=lambda x: x[1], reverse=True)

    dtensor_stats = None
    if dtensor:
        dtensor_stats = DtensorStats(
            registered=dt_by_strategy[DtensorStrategy.REGISTERED],
            decomp_fallback=dt_by_strategy[DtensorStrategy.DECOMP_FALLBACK],
            missing=dt_by_strategy[DtensorStrategy.MISSING],
            not_applicable=dt_by_strategy[DtensorStrategy.NOT_APPLICABLE],
            fully_covered=dt_fully_covered,
            has_gaps=dt_has_gaps,
            top_uncovered=dt_uncovered_leaves.most_common(15),
        )

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
        untraceable_ops=tuple(untraceable_list),
        dtensor=dtensor_stats,
    )


def _tree_depth(node: DecompNode) -> int:
    if not node.children:
        return 0
    return 1 + max(_tree_depth(c) for c in node.children)


def _collect_leaves(node: DecompNode, counter: Counter[str]) -> None:
    """Count leaf op appearances across tree branches (not propagated counts)."""
    if not node.children:
        counter[op_display_name(node.op)] += 1
        return
    for c in node.children:
        _collect_leaves(c, counter)


def _collect_uncovered_dtensor(node: DecompNode) -> set[str]:
    """Return leaf op names that have no registered DTensor ancestor on any path."""
    uncovered: set[str] = set()

    def walk(n: DecompNode, ancestor_covered: bool = False) -> None:
        covered = ancestor_covered or is_dtensor_intercept(
            n.classification.dtensor_strategy
        )
        if not n.children:
            if is_dtensor_gap(n.classification.dtensor_strategy) and not covered:
                uncovered.add(op_display_name(n.op))
            return
        for c in n.children:
            walk(c, covered)

    walk(node)
    return uncovered
