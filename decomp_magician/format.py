"""Text rendering for all decomposition-magician output.

Every function: data in, string out. No print(), no module-level mutable state.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from decomp_magician.classify import (
    DECOMP_TYPES,
    DecompType,
    DtensorStrategy,
    get_dtensor_strategy,
    is_dtensor_gap,
    is_dtensor_intercept,
)
from decomp_magician.dispatch import DispatchInfo, get_dispatch_info_cached
from decomp_magician.tree import (
    DecompNode,
    DecompSource,
    PurityResult,
    collect_leaf_frontier,
    collect_untraceable_errors,
    op_display_name,
)

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"


@dataclass(frozen=True)
class FormatConfig:
    color: bool = False
    show_dispatch: bool = False
    show_mode_sensitivity: bool = False
    show_dtensor: bool = False


def should_use_color() -> bool:
    """Auto-detect color support: tty + no NO_COLOR env var."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(cfg: FormatConfig, code: str, text: str) -> str:
    """Apply ANSI color code if color is enabled."""
    if cfg.color:
        return f"{code}{text}{_RESET}"
    return text


# ---------------------------------------------------------------------------
# Tree formatting
# ---------------------------------------------------------------------------

def format_tree(
    node: DecompNode, cfg: FormatConfig,
    prefix: str = "", is_last: bool = True,
    is_root: bool = True, ancestor_has_dtensor: bool = False,
) -> str:
    """Format a DecompNode tree as a string with box-drawing characters."""
    lines = []

    annotation = _format_annotation(node, cfg, ancestor_has_dtensor)

    op_name = op_display_name(node.op)
    if node.classification.decomp_type == DecompType.LEAF:
        op_name = _c(cfg, _DIM, op_name)
    else:
        op_name = _c(cfg, _BOLD, op_name)

    if is_root:
        line = f"{op_name}  {annotation}"
    else:
        connector = "└── " if is_last else "├── "
        count_str = f"  x{node.count}" if node.count > 1 else ""
        line = f"{prefix}{connector}{op_name}  {annotation}{count_str}"

    lines.append(line)

    this_has_dtensor = ancestor_has_dtensor or (
        cfg.show_dtensor and is_dtensor_intercept(get_dtensor_strategy(node.op))
    )

    child_prefix = prefix + ("    " if is_last else "│   ") if not is_root else ""
    for i, child in enumerate(node.children):
        is_last_child = i == len(node.children) - 1
        lines.append(format_tree(
            child, cfg, child_prefix, is_last_child,
            is_root=False, ancestor_has_dtensor=this_has_dtensor,
        ))

    return "\n".join(lines)


def _format_annotation(
    node: DecompNode, cfg: FormatConfig,
    ancestor_has_dtensor: bool = False,
) -> str:
    """Format the bracket annotation for a node."""
    parts = []

    cls = node.classification
    if cls.decomp_type == DecompType.LEAF:
        parts.append(_c(cfg, _DIM, "leaf"))
    else:
        parts.append(cls.decomp_type)

    if cls.inductor_kept:
        parts.append(_c(cfg, _YELLOW, "inductor-kept"))

    if cls.is_mutable:
        parts.append(_c(cfg, _RED, "mutable"))

    if not node.traceable:
        parts.append(_c(cfg, _RED, "untraceable"))

    annotation = f"[{', '.join(parts)}]"

    if not node.traceable and node.classification.decomp_type != DecompType.LEAF:
        annotation += "  " + _c(cfg, _DIM, "(has decomposition, could not trace)")

    if cfg.show_dispatch or cfg.show_mode_sensitivity:
        dinfo = get_dispatch_info_cached(node.op)
        if cfg.show_dispatch:
            annotation += "  " + _c(cfg, _DIM, f"({format_dispatch_short(dinfo)})")
        if cfg.show_mode_sensitivity:
            if dinfo.has_adiov:
                annotation += "  " + _c(cfg, _RED, "ADIOV")
            if dinfo.mode_sensitive:
                annotation += "  " + _c(cfg, _YELLOW, "mode-sensitive")
            else:
                annotation += "  " + _c(cfg, _GREEN, "mode-invariant")

    if cfg.show_dtensor:
        dt_strat = get_dtensor_strategy(node.op)
        if dt_strat == DtensorStrategy.REGISTERED:
            annotation += "  " + _c(cfg, _GREEN, "dtensor: registered")
        elif dt_strat == DtensorStrategy.DECOMP_FALLBACK:
            annotation += "  " + _c(cfg, _DIM, "dtensor: decomp-fallback")
        elif dt_strat == DtensorStrategy.NOT_APPLICABLE:
            annotation += "  " + _c(cfg, _DIM, "dtensor: n/a")
        elif dt_strat == DtensorStrategy.MISSING and ancestor_has_dtensor:
            annotation += "  " + _c(cfg, _DIM, "dtensor: registered ancestor")
        elif dt_strat == DtensorStrategy.MISSING:
            annotation += "  " + _c(cfg, _RED, "dtensor: MISSING")

    return annotation


# ---------------------------------------------------------------------------
# Leaf frontier formatting
# ---------------------------------------------------------------------------

def format_leaves(
    node: DecompNode, cfg: FormatConfig,
    opset_checker: tuple[str, Callable[[str], bool]] | None = None,
) -> str:
    """Format the leaf frontier with propagated counts."""
    root_name = op_display_name(node.op)

    if not node.children:
        return f"{_c(cfg, _DIM, root_name)}  [leaf, no decomposition]"

    lf = collect_leaf_frontier(node, check_dtensor=cfg.show_dtensor)

    leaf_dispatch: dict = {}
    if cfg.show_dispatch or cfg.show_mode_sensitivity:
        def _walk_for_dispatch(n: DecompNode):
            if not n.children:
                name = op_display_name(n.op)
                if name not in leaf_dispatch:
                    leaf_dispatch[name] = get_dispatch_info_cached(n.op)
                return
            for c in n.children:
                _walk_for_dispatch(c)
        _walk_for_dispatch(node)

    lines = []
    name_width = max(len(name) for name in lf.counts)
    for name, count in lf.counts.most_common():
        tags = []
        if name in lf.inductor_kept:
            tags.append(_c(cfg, _YELLOW, "inductor-kept"))
        if name in lf.untraceable:
            tags.append(_c(cfg, _RED, "untraceable"))
        if name in lf.dtensor_uncovered:
            tags.append(_c(cfg, _RED, "dtensor: MISSING"))
        dinfo = leaf_dispatch.get(name)
        if dinfo and cfg.show_dispatch:
            tags.append(_c(cfg, _DIM, format_dispatch_short(dinfo)))
        if dinfo and cfg.show_mode_sensitivity:
            if dinfo.has_adiov:
                tags.append(_c(cfg, _RED, "ADIOV"))
            if dinfo.mode_sensitive:
                tags.append(_c(cfg, _YELLOW, "mode-sensitive"))
        if opset_checker:
            opset_name, checker_fn = opset_checker
            if checker_fn(name):
                tags.append(_c(cfg, _GREEN, opset_name))
            else:
                tags.append(_c(cfg, _RED, f"NOT {opset_name}"))
        tag_str = "  [" + ", ".join(tags) + "]" if tags else ""
        lines.append(f"  {name:<{name_width}}  x{count}{tag_str}")

    total = sum(lf.counts.values())
    header = _c(cfg, _BOLD, root_name) + " decomposes to:"
    footer = f"\n{len(lf.counts)} unique ops, {total} total instances"
    if lf.untraceable:
        n = len(lf.untraceable)
        warning = _c(cfg, _RED, f"  ({n} untraceable — frontier may be incomplete)")
        footer += warning
    return header + "\n" + "\n".join(lines) + footer


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def format_summary(node: DecompNode, cfg: FormatConfig) -> str:
    """One-line summary of the tree's composition."""
    counts = {dt: 0 for dt in DECOMP_TYPES}
    inductor_kept = 0
    dtensor_missing = 0
    untraceable_nodes = 0
    untraceable_names: set[str] = set()

    def walk(n: DecompNode, ancestor_covered: bool = False) -> None:
        nonlocal inductor_kept, dtensor_missing, untraceable_nodes
        dt = n.classification.decomp_type
        counts[dt] = counts.get(dt, 0) + 1
        if n.classification.inductor_kept:
            inductor_kept += 1
        dt_strat = get_dtensor_strategy(n.op) if cfg.show_dtensor else None
        if dt_strat is not None and is_dtensor_gap(dt_strat) and not ancestor_covered:
            dtensor_missing += 1
        if not n.traceable:
            untraceable_nodes += 1
            untraceable_names.add(op_display_name(n.op))
        covered = ancestor_covered or (
            dt_strat is not None and is_dtensor_intercept(dt_strat)
        )
        for c in n.children:
            walk(c, covered)

    walk(node)
    total = sum(counts.values())

    type_parts = []
    for dt in sorted(counts, key=lambda t: counts[t], reverse=True):
        if counts[dt] > 0:
            type_parts.append(f"{counts[dt]} {dt}")
    ops_word = "op" if total == 1 else "ops"
    parts = [f"{total} {ops_word} ({', '.join(type_parts)})"]

    if inductor_kept > 0:
        parts.append(_c(cfg, _YELLOW, f"{inductor_kept} inductor-kept"))
    if untraceable_names:
        n_unique = len(untraceable_names)
        op_word = "op" if n_unique == 1 else "ops"
        if untraceable_nodes > n_unique:
            parts.append(_c(cfg, _RED, f"{n_unique} untraceable {op_word} ({untraceable_nodes} nodes)"))
        else:
            parts.append(_c(cfg, _RED, f"{n_unique} untraceable"))
    if cfg.show_dtensor:
        if dtensor_missing > 0:
            parts.append(_c(cfg, _RED, f"dtensor: {dtensor_missing} missing"))
        else:
            parts.append(_c(cfg, _DIM, "dtensor: no gaps"))

    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Purity formatting
# ---------------------------------------------------------------------------

def format_purity(result: PurityResult, cfg: FormatConfig) -> str:
    """Format purity analysis result."""
    lines = []
    if result.is_pure:
        lines.append(_c(cfg, _GREEN, "PURE") + f"  {result.op}")
        lines.append(f"  All {result.total_leaves} leaf ops are non-mutable with no ADIOV kernel.")
        lines.append("  Behavior under inference_mode vs no_grad: " + _c(cfg, _GREEN, "identical"))
    else:
        lines.append(_c(cfg, _RED, "IMPURE") + f"  {result.op}")
        lines.append(f"  {result.total_leaves} leaf ops total")

        if result.mutable_leaves:
            lines.append("")
            lines.append(_c(cfg, _BOLD, "Mutable leaves") + " (in-place operations):")
            for name, count in result.mutable_leaves:
                has_adiov = any(n == name for n, _ in result.adiov_leaves)
                marker = "  " + _c(cfg, _RED, "[ADIOV]") if has_adiov else ""
                lines.append(f"  {name}  x{count}{marker}")

        if result.adiov_leaves:
            non_mutable_adiov = [(n, c) for n, c in result.adiov_leaves
                                 if not any(mn == n for mn, _ in result.mutable_leaves)]
            if non_mutable_adiov:
                lines.append("")
                lines.append(_c(cfg, _BOLD, "Non-mutable ADIOV leaves") + ":")
                for name, count in non_mutable_adiov:
                    lines.append(f"  {name}  x{count}")

        if result.mode_sensitive_leaves:
            lines.append("")
            lines.append("  Leaves differing under inference_mode vs no_grad: " +
                         _c(cfg, _RED, str(len(result.mode_sensitive_leaves))))
            lines.append("  These leaves have autograd/ADIOV kernels whose dispatch")
            lines.append("  path changes depending on the active gradient mode.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stats formatting
# ---------------------------------------------------------------------------

def format_stats(
    data, cfg: FormatConfig, target_opset: str | None = None, compile: bool = False,
) -> str:
    """Format bulk statistics as a string."""
    trace_pct = data.traceable / data.total_non_out * 100 if data.total_non_out else 0

    mode_suffix = "  (compile decomposition)" if compile else ""
    lines = [
        _c(cfg, _BOLD, "Decomposition table statistics") + mode_suffix,
        "",
        f"  Total ops in table:  {data.total}  ({data.total_non_out} excluding _out variants)",
    ]

    type_parts = []
    for dt in (DecompType.TABLE, DecompType.BOTH, DecompType.CIA, DecompType.LEAF):
        if data.by_type.get(dt, 0) > 0:
            type_parts.append(f"{data.by_type[dt]} {dt}")
    lines.append(f"  By type:             {', '.join(type_parts)}")
    lines.append(f"  Inductor-kept:       {_c(cfg, _YELLOW, str(data.inductor_kept))}")
    lines.append(f"  Traceable:           {_c(cfg, _GREEN, str(data.traceable))} ({trace_pct:.0f}%)")
    lines.append(f"  Untraceable:         {_c(cfg, _RED, str(data.untraceable))}")
    if data.classify_errors > 0:
        lines.append(f"  Classify errors:     {_c(cfg, _DIM, str(data.classify_errors))}")

    lines.append("")
    lines.append(_c(cfg, _BOLD, "Top leaf ops") + "  (most common across all decompositions):")
    top_leaves = data.leaf_ops.most_common(15)
    if top_leaves:
        name_width = max(len(name) for name, _ in top_leaves)
        for name, count in top_leaves:
            bar = "█" * min(count // 10, 40)
            lines.append(f"  {name:<{name_width}}  {count:>4}  {_c(cfg, _DIM, bar)}")

    lines.append("")
    lines.append(_c(cfg, _BOLD, "Deepest decomposition chains") + ":")
    for name, depth in data.deepest:
        lines.append(f"  {name:<50}  depth {depth}")

    if data.untraceable_ops:
        err_categories: Counter[str] = Counter()
        for _, reason in data.untraceable_ops:
            err_type = reason.split(":")[0].strip() if ":" in reason else reason
            err_categories[err_type] += 1
        lines.append("")
        lines.append(_c(cfg, _BOLD, "Untraceable ops by error type") + ":")
        for err_type, count in err_categories.most_common():
            lines.append(f"  {count:>4}  {err_type}")

    if target_opset:
        from decomp_magician.opset import is_core_aten
        covered_leaf_ops = [name for name in data.leaf_ops if is_core_aten(name)]
        non_covered_leaf_ops = [name for name in data.leaf_ops if not is_core_aten(name)]
        total_unique = len(data.leaf_ops)
        covered_unique = len(covered_leaf_ops)
        pct = covered_unique / total_unique * 100 if total_unique else 0

        lines.append("")
        lines.append(
            _c(cfg, _BOLD, "Leaf op coverage") +
            f"  (target: {target_opset})"
        )
        lines.append(
            f"  {_c(cfg, _GREEN, str(covered_unique))}/{total_unique} unique leaf ops are in "
            f"{target_opset} ({pct:.0f}%)"
        )
        if non_covered_leaf_ops:
            lines.append("")
            lines.append(
                _c(cfg, _BOLD, "Leaf ops NOT in " + target_opset) + ":"
            )
            nc_width = max(len(n) for n in non_covered_leaf_ops)
            for name in sorted(non_covered_leaf_ops):
                count = data.leaf_ops[name]
                padded = name.ljust(nc_width)
                lines.append(f"  {_c(cfg, _RED, padded)}  appears in {count} decompositions")

    if data.dtensor:
        dt = data.dtensor
        tensor_bearing = dt.registered + dt.decomp_fallback + dt.missing
        reg_pct = dt.registered / tensor_bearing * 100 if tensor_bearing else 0

        lines.append("")
        lines.append(_c(cfg, _BOLD, "DTensor coverage") + ":")
        lines.append(f"  Registered strategy:   {_c(cfg, _GREEN, str(dt.registered))} ({reg_pct:.0f}%)")
        lines.append(f"  Decomp fallback:       {dt.decomp_fallback}")
        lines.append(f"  No strategy:           {_c(cfg, _RED, str(dt.missing))}")
        if dt.not_applicable > 0:
            lines.append(f"  No tensor inputs:      {_c(cfg, _DIM, str(dt.not_applicable))}")
        lines.append("")

        traceable_with_children = dt.fully_covered + dt.has_gaps
        if traceable_with_children > 0:
            cov_pct = dt.fully_covered / traceable_with_children * 100
            covered_str = _c(cfg, _GREEN, str(dt.fully_covered))
            lines.append(f"  Fully covered trees:   {covered_str}/{traceable_with_children} ({cov_pct:.0f}%)")
            lines.append(f"  Trees with gaps:       {_c(cfg, _RED, str(dt.has_gaps))}")

        if dt.top_uncovered:
            lines.append("")
            lines.append(_c(cfg, _BOLD, "Top uncovered leaf ops") + "  (most common gaps across all trees):")
            uc_width = max(len(name) for name, _ in dt.top_uncovered)
            for name, count in dt.top_uncovered:
                bar = "█" * min(count // 2, 40)
                lines.append(f"  {_c(cfg, _RED, name):<{uc_width + 10}}  {count:>4}  {_c(cfg, _DIM, bar)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reverse lookup formatting
# ---------------------------------------------------------------------------

def format_reverse(
    results, target: str, cfg: FormatConfig,
    compile_mode: bool = False,
) -> str:
    """Format reverse lookup results."""
    name_width = max(len(r.op) for r in results)
    mode = "compile" if compile_mode else "full"
    lines = [_c(cfg, _BOLD, f"{len(results)} ops") + f" decompose into {target} ({mode} decomposition):"]
    for r in results:
        depth_str = _c(cfg, _DIM, f"at depth {r.target_depth}")
        lines.append(f"  {r.op:<{name_width}}  x{r.count:>3}  ({depth_str})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Opset coverage formatting
# ---------------------------------------------------------------------------

def format_opset(cov, cfg: FormatConfig, compile_mode: bool = False) -> str:
    """Format opset coverage result."""
    mode = "compile" if compile_mode else "full"
    lines = [
        _c(cfg, _BOLD, cov.op) + f"  target: {cov.opset}  ({mode} decomposition)",
    ]

    if cov.fully_covered:
        lines.append("")
        lines.append(_c(cfg, _GREEN, "FULLY COVERED") + f"  all {cov.total_leaves} leaf ops are in {cov.opset}")
    else:
        pct = cov.covered_leaves / cov.total_leaves * 100 if cov.total_leaves else 0
        lines.append("")
        lines.append(
            _c(cfg, _RED, "NOT FULLY COVERED") +
            f"  {cov.covered_leaves}/{cov.total_leaves} leaf ops in {cov.opset} ({pct:.0f}%)"
        )
        lines.append("")
        lines.append(_c(cfg, _BOLD, "Non-covered ops") + f"  (not in {cov.opset}):")
        if cov.non_covered:
            name_width = max(len(n) for n, _ in cov.non_covered)
            for name, count in cov.non_covered:
                padded = name.ljust(name_width)
                lines.append(f"  {_c(cfg, _RED, padded)}  x{count}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diff formatting
# ---------------------------------------------------------------------------

def format_diff(diff, cfg: FormatConfig) -> str:
    """Format decomposition diff."""
    has_changes = diff.added or diff.removed or diff.changed

    left_short = diff.left_label.split("  ")[0]
    right_short = diff.right_label.split("  ")[0]

    lines = [
        _c(cfg, _BOLD, diff.left_label) + "  vs  " + _c(cfg, _BOLD, diff.right_label),
    ]

    if not has_changes:
        lines.append("")
        lines.append(_c(cfg, _DIM, "No differences — both produce the same leaf frontier."))
        return "\n".join(lines)

    if diff.removed:
        lines.append("")
        lines.append(_c(cfg, _BOLD, "Removed") + f"  (in {left_short} only):")
        for name, count in diff.removed.most_common():
            lines.append(f"  {_c(cfg, _RED, '-')} {name}  x{count}")

    if diff.added:
        lines.append("")
        lines.append(_c(cfg, _BOLD, "Added") + f"  (in {right_short} only):")
        for name, count in diff.added.most_common():
            lines.append(f"  {_c(cfg, _GREEN, '+')} {name}  x{count}")

    if diff.changed:
        lines.append("")
        lines.append(_c(cfg, _BOLD, "Changed counts") + ":")
        for name, lc, rc in diff.changed:
            delta = rc - lc
            direction = _c(cfg, _GREEN, f"+{delta}") if delta > 0 else _c(cfg, _RED, str(delta))
            lines.append(f"  {_c(cfg, _YELLOW, '~')} {name}  x{lc} -> x{rc}  ({direction})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backward formatting
# ---------------------------------------------------------------------------

def format_backward(name: str, op_counts: Counter[str], cfg: FormatConfig) -> str:
    """Format backward trace results."""
    lines = [_c(cfg, _BOLD, name) + "  backward"]
    if not op_counts:
        lines.append("  (no ops recorded during backward)")
    else:
        name_width = max(len(n) for n in op_counts)
        for child_name, count in op_counts.most_common():
            count_str = f"  x{count}" if count > 1 else ""
            lines.append(f"  {child_name:<{name_width}}{count_str}")
        total = sum(op_counts.values())
        lines.append(f"\n{len(op_counts)} unique ops, {total} total instances")
    return "\n".join(lines)


def format_dispatch_short(info: DispatchInfo) -> str:
    """Short annotation string for dispatch info: AG:type, ADIOV:yes/no."""
    parts = []
    ag_labels = {
        "autograd_kernel": "AG:redispatch",
        "math_kernel": "AG:terminal",
        "fallthrough": "AG:fallthrough",
        "other": "AG:other",
        "none": "AG:none",
    }
    parts.append(ag_labels.get(info.autograd_type, f"AG:{info.autograd_type}"))
    if info.has_adiov:
        parts.append("ADIOV:yes")
    return ", ".join(parts)


def format_dispatch_detail(info: DispatchInfo) -> str:
    """Multi-line dispatch info for verbose output."""
    lines = [f"  dispatch: {info.autograd_type}"]
    if info.autograd_entry:
        lines.append(f"    AutogradCPU: [{info.autograd_entry.tag}]"
                      f"{' (fallthrough)' if info.autograd_entry.is_fallthrough else ''}")
    if info.adiov_entry:
        lines.append(f"    ADInplaceOrView: [{info.adiov_entry.tag}]"
                      f"{' (fallthrough)' if info.adiov_entry.is_fallthrough else ''}")
    if info.dense_entry:
        lines.append(f"    CPU: [{info.dense_entry.tag}]"
                      f"{' (fallthrough)' if info.dense_entry.is_fallthrough else ''}")
    return "\n".join(lines)


def format_source(
    src: DecompSource, cfg: FormatConfig, root_op: str | None = None,
) -> str:
    """Format the decomposition function source code.

    If root_op is set and differs from src.op, indicates the source
    is from a child op (e.g. batch_norm showing native_batch_norm's source).
    """
    header = f"Source: {src.location}"
    if root_op and root_op != src.op:
        header += f"  (via {src.op})"
    lines = [
        _c(cfg, _DIM, header),
        "",
        src.source.rstrip(),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model analysis formatting
# ---------------------------------------------------------------------------

def format_model_dtensor_tag(
    dtensor_info: dict[str, str], name: str, cfg: FormatConfig,
) -> str:
    """Format a DTensor tag for model analysis output."""
    strategy = dtensor_info.get(name)
    if strategy is None:
        return ""
    if strategy == DtensorStrategy.REGISTERED:
        return "  " + _c(cfg, _GREEN, "dtensor: registered")
    if strategy == DtensorStrategy.DECOMP_FALLBACK:
        return "  " + _c(cfg, _DIM, "dtensor: decomp-fallback")
    if strategy == DtensorStrategy.NOT_APPLICABLE:
        return "  " + _c(cfg, _DIM, "dtensor: n/a")
    return "  " + _c(cfg, _RED, "dtensor: MISSING")


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------

def format_verbose(node: DecompNode, cfg: FormatConfig, indent: int = 0) -> str:
    """Format detailed classification for each node."""
    lines = []
    prefix = "  " * indent
    cls = node.classification
    name = op_display_name(node.op)
    lines.append(f"{prefix}{name}:")
    lines.append(f"{prefix}  schema: {node.op._schema}")
    lines.append(f"{prefix}  decomp_type: {cls.decomp_type}")
    backends = ", ".join(k for k, v in cls.has_backend.items() if v)
    no_backends = ", ".join(k for k, v in cls.has_backend.items() if not v)
    if backends:
        lines.append(f"{prefix}  backends: {backends}")
    if no_backends:
        lines.append(f"{prefix}  no backend: {no_backends}")
    if cls.tags:
        lines.append(f"{prefix}  tags: {', '.join(cls.tags)}")
    if cls.is_mutable:
        lines.append(f"{prefix}  mutable: True")
    if cls.has_alias_info:
        lines.append(f"{prefix}  alias_info: True")
    if cls.inductor_kept:
        lines.append(f"{prefix}  inductor_kept: True")
    if node.error:
        lines.append(f"{prefix}  error: {node.error}")
    dinfo = get_dispatch_info_cached(node.op)
    lines.append(f"{prefix}  autograd_type: {dinfo.autograd_type}")
    lines.append(f"{prefix}  has_adiov: {dinfo.has_adiov}")
    lines.append(f"{prefix}  mode_sensitive: {dinfo.mode_sensitive}")
    if cfg.show_dtensor:
        lines.append(f"{prefix}  dtensor_strategy: {get_dtensor_strategy(node.op)}")
    for child in node.children:
        lines.append(format_verbose(child, cfg, indent + 1))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Untraceable warnings
# ---------------------------------------------------------------------------

def format_untraceable_warning(node: DecompNode, cfg: FormatConfig) -> str:
    """Format a warning about untraceable ops, or empty string if none."""
    errors = collect_untraceable_errors(node)
    if not errors:
        return ""

    n = len(errors)
    lines = [
        f"\n{_c(cfg, _YELLOW, 'warning')}: {n} {'op' if n == 1 else 'ops'} "
        f"could not be traced with synthetic inputs — "
        f"{'its subtree is' if n == 1 else 'their subtrees are'} "
        f"incomplete.",
    ]
    for op_name, error in errors:
        lines.append(f"  {op_name}: {error}")
    return "\n".join(lines)


