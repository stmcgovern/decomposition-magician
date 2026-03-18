"""CLI entry point for decomposition-magician."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple

from decomp_magician.diff import compute_diff, compute_diff_ops
from decomp_magician.dispatch import (
    DispatchInfo,
    format_dispatch_short,
    get_dispatch_info_cached,
)
from decomp_magician.graph import format_dot, format_mermaid
from decomp_magician.opset import OPSETS, check_opset_coverage
from decomp_magician.resolve import resolve_op
from decomp_magician.reverse import reverse_lookup
from decomp_magician.stats import compute_stats
from decomp_magician.classify import DECOMP_TYPES, is_dtensor_intercept
from decomp_magician.tree import DecompNode, build_tree, collect_leaf_counts, op_display_name, trace_backward


# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"

# Whether color is currently enabled
_use_color = False

# Whether to show dispatch table info in annotations
_show_dispatch = False
_show_mode_sensitivity = False


def _should_use_color() -> bool:
    """Auto-detect color support: tty + no NO_COLOR env var."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Apply ANSI color code if color is enabled."""
    if _use_color:
        return f"{code}{text}{_RESET}"
    return text


def format_tree(
    node: DecompNode, prefix: str = "", is_last: bool = True,
    is_root: bool = True, ancestor_has_dtensor: bool = False,
) -> str:
    """Format a DecompNode tree as a string with box-drawing characters."""
    lines = []

    # Build the annotation
    annotation = _format_annotation(node, ancestor_has_dtensor)

    # Build the line
    op_name = op_display_name(node.op)
    if node.classification.decomp_type == "leaf":
        op_name = _c(_DIM, op_name)
    else:
        op_name = _c(_BOLD, op_name)

    if is_root:
        line = f"{op_name}  {annotation}"
    else:
        connector = "└── " if is_last else "├── "
        count_str = f"  x{node.count}" if node.count > 1 else ""
        line = f"{prefix}{connector}{op_name}  {annotation}{count_str}"

    lines.append(line)

    this_has_dtensor = ancestor_has_dtensor or is_dtensor_intercept(
        node.classification.dtensor_strategy
    )

    # Recurse into children
    child_prefix = prefix + ("    " if is_last else "│   ") if not is_root else ""
    for i, child in enumerate(node.children):
        is_last_child = i == len(node.children) - 1
        lines.append(format_tree(
            child, child_prefix, is_last_child,
            is_root=False, ancestor_has_dtensor=this_has_dtensor,
        ))

    return "\n".join(lines)


def _format_annotation(node: DecompNode, ancestor_has_dtensor: bool = False) -> str:
    """Format the bracket annotation for a node."""
    parts = []

    # Decomp type
    cls = node.classification
    if cls.decomp_type == "leaf":
        parts.append(_c(_DIM, "leaf"))
    else:
        parts.append(cls.decomp_type)

    # Inductor exclusion
    if cls.inductor_kept:
        parts.append(_c(_YELLOW, "inductor-kept"))

    # Mutable
    if cls.is_mutable:
        parts.append(_c(_RED, "mutable"))

    # Traceability
    if not node.traceable:
        parts.append(_c(_RED, "untraceable"))

    annotation = f"[{', '.join(parts)}]"

    # Show brief explanation for untraceable ops
    if not node.traceable and node.classification.decomp_type != "leaf":
        annotation += "  " + _c(_DIM, "(has decomposition, could not trace)")

    # Dispatch table info
    if _show_dispatch or _show_mode_sensitivity:
        dinfo = get_dispatch_info_cached(node.op)
        if _show_dispatch:
            annotation += "  " + _c(_DIM, f"({format_dispatch_short(dinfo)})")
        if _show_mode_sensitivity:
            if dinfo.has_adiov:
                annotation += "  " + _c(_RED, "ADIOV")
            if dinfo.mode_sensitive:
                annotation += "  " + _c(_YELLOW, "mode-sensitive")
            else:
                annotation += "  " + _c(_GREEN, "mode-invariant")

    # DTensor strategy (outside brackets)
    # A leaf is only a real gap if no ancestor on its path has a registered strategy.
    if cls.dtensor_strategy is not None:
        if cls.dtensor_strategy == "registered":
            annotation += "  " + _c(_GREEN, "dtensor: ok")
        elif cls.dtensor_strategy == "decomp-fallback":
            annotation += "  " + _c(_GREEN, "dtensor: ok (via decomp)")
        elif cls.dtensor_strategy == "missing" and ancestor_has_dtensor:
            annotation += "  " + _c(_DIM, "dtensor: ok (via ancestor)")
        elif cls.dtensor_strategy == "missing":
            annotation += "  " + _c(_RED, "dtensor: MISSING")

    return annotation


_BLUE = "\033[34m"
_MAGENTA = "\033[35m"


@dataclass(frozen=True)
class PurityResult:
    """Purity analysis of a decomposition tree."""
    op: str
    is_pure: bool  # no mutable or ADIOV leaves
    total_leaves: int
    mutable_leaves: tuple[tuple[str, int], ...]  # (op_name, count)
    adiov_leaves: tuple[tuple[str, int], ...]  # (op_name, count) — leaves with ADIOV kernel
    mode_sensitive_leaves: tuple[tuple[str, int], ...]  # leaves with non-FT autograd


def _analyze_purity(node: DecompNode) -> PurityResult:
    """Analyze purity of a decomposition tree."""
    counts = collect_leaf_counts(node)
    mutable_names: set[str] = set()
    adiov_names: set[str] = set()
    mode_sensitive_names: set[str] = set()

    def walk(n: DecompNode) -> None:
        if not n.children:
            name = op_display_name(n.op)
            dinfo = get_dispatch_info_cached(n.op)
            if n.classification.is_mutable:
                mutable_names.add(name)
            if dinfo.has_adiov:
                adiov_names.add(name)
            if dinfo.mode_sensitive:
                mode_sensitive_names.add(name)
            return
        for c in n.children:
            walk(c)

    walk(node)

    mutable = sorted(((n, counts[n]) for n in mutable_names), key=lambda x: -x[1])
    adiov = sorted(((n, counts[n]) for n in adiov_names), key=lambda x: -x[1])
    ms = sorted(((n, counts[n]) for n in mode_sensitive_names), key=lambda x: -x[1])

    return PurityResult(
        op=op_display_name(node.op),
        is_pure=len(mutable) == 0 and len(adiov) == 0,
        total_leaves=len(counts),
        mutable_leaves=tuple(mutable),
        adiov_leaves=tuple(adiov),
        mode_sensitive_leaves=tuple(ms),
    )


def _filter_adiov_paths(node: DecompNode) -> DecompNode | None:
    """Filter tree to only include paths that reach ADIOV-bearing ops.

    Returns None if no path reaches an ADIOV op.
    """
    # Leaf node: keep only if it has ADIOV
    if not node.children:
        dinfo = get_dispatch_info_cached(node.op)
        return node if dinfo.has_adiov else None

    # Internal node: filter children recursively
    kept_children = []
    for child in node.children:
        filtered = _filter_adiov_paths(child)
        if filtered is not None:
            kept_children.append(filtered)

    if not kept_children:
        return None

    return replace(node, children=tuple(kept_children))


def format_purity(result: PurityResult) -> str:
    """Format purity analysis result."""
    lines = []
    if result.is_pure:
        lines.append(_c(_GREEN, "PURE") + f"  {result.op}")
        lines.append(f"  All {result.total_leaves} leaf ops are non-mutable with no ADIOV kernel.")
        lines.append("  Behavior under inference_mode vs no_grad: " + _c(_GREEN, "identical"))
    else:
        lines.append(_c(_RED, "IMPURE") + f"  {result.op}")
        lines.append(f"  {result.total_leaves} leaf ops total")

        if result.mutable_leaves:
            lines.append("")
            lines.append(_c(_BOLD, "Mutable leaves") + " (in-place operations):")
            for name, count in result.mutable_leaves:
                has_adiov = any(n == name for n, _ in result.adiov_leaves)
                marker = "  " + _c(_RED, "[ADIOV]") if has_adiov else ""
                lines.append(f"  {name}  x{count}{marker}")

        if result.adiov_leaves:
            non_mutable_adiov = [(n, c) for n, c in result.adiov_leaves
                                 if not any(mn == n for mn, _ in result.mutable_leaves)]
            if non_mutable_adiov:
                lines.append("")
                lines.append(_c(_BOLD, "Non-mutable ADIOV leaves") + ":")
                for name, count in non_mutable_adiov:
                    lines.append(f"  {name}  x{count}")

        if result.mode_sensitive_leaves:
            lines.append("")
            lines.append("  Leaves differing under inference_mode vs no_grad: " +
                         _c(_RED, str(len(result.mode_sensitive_leaves))))
            lines.append("  These leaves have autograd/ADIOV kernels whose dispatch")
            lines.append("  path changes depending on the active gradient mode.")

    return "\n".join(lines)


def _enrich_leaves_with_dispatch(d: dict, node: DecompNode) -> dict:
    """Add dispatch info to leaves JSON output."""
    # Build a map of leaf names to dispatch info
    leaf_dispatch: dict[str, DispatchInfo] = {}

    def walk(n: DecompNode):
        if not n.children:
            name = op_display_name(n.op)
            if name not in leaf_dispatch:
                leaf_dispatch[name] = get_dispatch_info_cached(n.op)
            return
        for c in n.children:
            walk(c)

    walk(node)

    for leaf in d.get("leaves", []):
        dinfo = leaf_dispatch.get(leaf["op"])
        if dinfo:
            leaf["autograd_type"] = dinfo.autograd_type
            leaf["has_adiov"] = dinfo.has_adiov
            leaf["mode_sensitive"] = dinfo.mode_sensitive
    return d


def _enrich_tree_with_dispatch(d: dict, node: DecompNode) -> None:
    """Add dispatch info to tree JSON output (in-place)."""
    dinfo = get_dispatch_info_cached(node.op)
    d["autograd_type"] = dinfo.autograd_type
    d["has_adiov"] = dinfo.has_adiov
    d["mode_sensitive"] = dinfo.mode_sensitive
    for child_dict, child_node in zip(d.get("children", []), node.children):
        _enrich_tree_with_dispatch(child_dict, child_node)


def main(argv: list[str] | None = None) -> int:
    from importlib.metadata import version as pkg_version

    parser = argparse.ArgumentParser(
        prog="decomp-magician",
        description="Inspect PyTorch operator decomposition trees.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {pkg_version('decomposition-magician')}",
    )
    parser.add_argument(
        "op",
        nargs="?",
        help="Operator name (e.g., 'addcmul', 'aten.addcmul.default', 'batch_norm')",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics across all decomposable ops (no op argument needed)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=-1,
        help="Maximum recursion depth (-1 for unlimited)",
    )
    parser.add_argument(
        "--dtensor",
        action="store_true",
        help="Show DTensor sharding strategy coverage",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Show what torch.compile produces (treat inductor-kept ops as leaves)",
    )
    parser.add_argument(
        "--leaves",
        action="store_true",
        help="Show flat leaf frontier with propagated counts instead of tree",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse lookup: find all ops that decompose into the given op",
    )
    parser.add_argument(
        "--include-out",
        action="store_true",
        help="Include _out variant ops in --reverse results (usually duplicates)",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Output as Mermaid flowchart (renders in GitHub markdown)",
    )
    parser.add_argument(
        "--dot",
        action="store_true",
        help="Output as Graphviz DOT graph",
    )
    parser.add_argument(
        "--target-opset",
        metavar="OPSET",
        help=f"Check if op decomposes fully to target opset ({', '.join(OPSETS)})",
    )
    parser.add_argument(
        "--diff",
        nargs="?",
        const=True,
        default=False,
        metavar="OP2",
        help="Compare decompositions: bare=full vs compile, with OP2=compare two ops",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        help="Analyze an exported model (.pt2 file from torch.export)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full classification details per op",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--dispatch-table",
        action="store_true",
        help="Annotate each op with its dispatch table entries (AutogradCPU, ADInplaceOrView, CPU)",
    )
    parser.add_argument(
        "--mode-sensitivity",
        action="store_true",
        help="Show which ops behave differently under inference_mode vs no_grad",
    )
    parser.add_argument(
        "--adiov",
        action="store_true",
        help="Filter tree to only paths reaching ops with ADInplaceOrView kernels",
    )
    parser.add_argument(
        "--pure",
        action="store_true",
        help="Check if decomposition is pure (no mutable or ADIOV leaves)",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Show ops dispatched during the backward pass (gradient computation)",
    )
    args = parser.parse_args(argv)

    # Initialize color
    global _use_color, _show_dispatch, _show_mode_sensitivity
    if args.no_color or args.json:
        _use_color = False
    else:
        _use_color = _should_use_color()
    _show_dispatch = args.dispatch_table
    _show_mode_sensitivity = args.mode_sensitivity

    # Stats mode — no op argument needed
    if args.stats:
        return _run_stats(args)

    # Model analysis mode — no op argument needed
    if args.model:
        return _run_model(args)

    # All other modes require an op
    if args.op is None or args.op == "":
        print("Usage: decomp-magician <op> [options]", file=sys.stderr)
        print("", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  decomp-magician addcmul                  # decomposition tree", file=sys.stderr)
        print("  decomp-magician batch_norm --compile      # compile-mode tree", file=sys.stderr)
        print("  decomp-magician softmax --diff            # full vs compile diff", file=sys.stderr)
        print("  decomp-magician addcmul --target-opset core_aten", file=sys.stderr)
        print("  decomp-magician --stats                   # bulk statistics", file=sys.stderr)
        print("  decomp-magician --model model.pt2 --target-opset core_aten", file=sys.stderr)
        print("", file=sys.stderr)
        print("Run with --help for all options.", file=sys.stderr)
        return 1

    # Validate flag combinations.
    # These output modes are mutually exclusive, EXCEPT --leaves + --target-opset.
    active_modes = [f for f, v in [
        ("--mermaid", args.mermaid), ("--dot", args.dot),
        ("--leaves", args.leaves), ("--reverse", args.reverse),
        ("--target-opset", bool(args.target_opset)), ("--diff", bool(args.diff)),
    ] if v]
    # Allow the one permitted combination
    if set(active_modes) == {"--leaves", "--target-opset"}:
        pass  # this pair is allowed
    elif len(active_modes) > 1:
        print(f"Conflicting flags: {', '.join(active_modes)} (pick one)", file=sys.stderr)
        return 1
    if args.json and (args.mermaid or args.dot):
        flag = "--mermaid" if args.mermaid else "--dot"
        print(f"Conflicting flags: --json, {flag} (pick one)", file=sys.stderr)
        return 1

    # Resolve the op name
    result = resolve_op(args.op)

    if isinstance(result, list):
        if len(result) == 0:
            print(f"No ops found matching '{args.op}'", file=sys.stderr)
            return 1
        print(f"Ambiguous op name '{args.op}'. Candidates:", file=sys.stderr)
        for candidate in sorted(result):
            print(f"  {candidate}", file=sys.stderr)
        return 1

    op = result

    # Show which overload was selected when it's non-obvious
    resolved_name = op_display_name(op)
    graph_mode = args.mermaid or args.dot
    if not resolved_name.endswith(".default") and not args.json and not graph_mode:
        user_input = args.op.replace("::", ".")
        if user_input.count(".") < 2:
            print(f"(resolved to {resolved_name})", file=sys.stderr)

    # Reverse lookup mode
    if args.reverse:
        return _run_reverse(resolved_name, args)

    # Target opset mode (standalone, not combined with --leaves)
    if args.target_opset and not args.leaves:
        return _run_opset(op, args)

    # Diff mode
    if args.diff:
        return _run_diff(op, args)

    # Backward mode
    if args.backward:
        return _run_backward(op, args)

    # Build and print the tree
    node = build_tree(op, depth=args.depth, dtensor=args.dtensor, compile=args.compile)

    # --pure: purity analysis
    if args.pure:
        purity = _analyze_purity(node)
        if args.json:
            json_data = {
                "op": purity.op,
                "pure": purity.is_pure,
                "total_leaves": purity.total_leaves,
                "mutable_leaves": [{"op": n, "count": c} for n, c in purity.mutable_leaves],
                "adiov_leaves": [{"op": n, "count": c} for n, c in purity.adiov_leaves],
                "mode_sensitive_leaves": [{"op": n, "count": c} for n, c in purity.mode_sensitive_leaves],
            }
            print(json.dumps(json_data, indent=2))
        else:
            print(format_purity(purity))
        return 0

    # --adiov: filter to ADIOV paths only
    if args.adiov:
        filtered = _filter_adiov_paths(node)
        if filtered is None:
            root_name = op_display_name(node.op)
            if args.json:
                print(json.dumps({"op": root_name, "adiov_paths": False, "message": "no ADIOV leaves"}))
            else:
                print(f"{_c(_GREEN, 'NO ADIOV PATHS')}  {root_name}")
                print("  No decomposition path reaches an op with an ADInplaceOrView kernel.")
            return 0
        node = filtered

    if args.json:
        if args.leaves:
            d = _leaves_to_dict(node)
            if _show_dispatch or _show_mode_sensitivity:
                d = _enrich_leaves_with_dispatch(d, node)
            if args.target_opset:
                from decomp_magician.opset import is_core_aten
                for leaf in d.get("leaves", []):
                    leaf["in_opset"] = is_core_aten(leaf["op"])
                d["opset"] = args.target_opset
        else:
            d = tree_to_dict(node)
            if _show_dispatch or _show_mode_sensitivity:
                _enrich_tree_with_dispatch(d, node)
        _add_untraceable_warnings(d, node)
        print(json.dumps(d, indent=2))
        return 0

    if args.mermaid:
        print(format_mermaid(node))
        print("", file=sys.stderr)
        print("Paste into ```mermaid fenced block in GitHub markdown to render.", file=sys.stderr)
        _warn_untraceable(node)
        return 0

    if args.dot:
        print(format_dot(node))
        print("", file=sys.stderr)
        print("Pipe to: dot -Tsvg > graph.svg  or  dot -Tpng > graph.png", file=sys.stderr)
        _warn_untraceable(node)
        return 0

    if args.leaves:
        opset_checker = None
        if args.target_opset:
            from decomp_magician.opset import is_core_aten
            opset_checker = (args.target_opset, is_core_aten)
        print(format_leaves(node, opset_checker=opset_checker))
    else:
        print(format_tree(node))
        print()
        print(format_summary(node))

        if args.verbose:
            print()
            _print_verbose(node)

    _warn_untraceable(node)
    return 0


def _run_reverse(target: str, args) -> int:
    """Run reverse lookup and print results."""
    print(f"Scanning decomposition table for ops that produce {_c(_BOLD, target)}...",
          file=sys.stderr)

    results = reverse_lookup(target, depth=args.depth, compile=args.compile,
                             include_out=args.include_out)

    if args.json:
        print(json.dumps({"target": target, "producers": results}, indent=2))
        return 0

    if not results:
        print(f"No ops decompose into {target}")
        # Hint: maybe the user wanted a different overload
        if target.endswith(".default"):
            base = target.rsplit(".", 1)[0]
            parts = base.split(".")
            if len(parts) == 2:
                try:
                    import torch
                    from torch._ops import OpOverloadPacket
                    packet = getattr(getattr(torch.ops, parts[0]), parts[1])
                    if isinstance(packet, OpOverloadPacket):
                        overloads = [ol for ol in packet.overloads() if ol != "default"]
                        if overloads:
                            alts = [f"{base}.{ol}" for ol in overloads[:5]]
                            print(f"Try a different overload: {', '.join(alts)}",
                                  file=sys.stderr)
                except (AttributeError, RuntimeError):
                    pass
        return 0

    name_width = max(len(r["op"]) for r in results)
    mode = "compile" if args.compile else "full"
    lines = [_c(_BOLD, f"{len(results)} ops") + f" decompose into {target} ({mode} decomposition):"]
    for r in results:
        depth_str = _c(_DIM, f"at depth {r['target_depth']}")
        lines.append(f"  {r['op']:<{name_width}}  x{r['count']:>3}  ({depth_str})")
    print("\n".join(lines))
    return 0


def _run_stats(args) -> int:
    """Run bulk statistics across all decomposable ops."""
    print("Scanning all decomposable ops...", file=sys.stderr)

    data = compute_stats(compile=args.compile, dtensor=args.dtensor)

    if args.json:
        json_data = {
            "total": data.total,
            "total_non_out": data.total_non_out,
            "by_type": data.by_type,
            "inductor_kept": data.inductor_kept,
            "traceable": data.traceable,
            "untraceable": data.untraceable,
            "classify_errors": data.classify_errors,
            "top_leaf_ops": [
                {"op": name, "appearances": count}
                for name, count in data.leaf_ops.most_common(20)
            ],
            "deepest": [
                {"op": name, "depth": depth}
                for name, depth in data.deepest
            ],
        }
        if args.target_opset:
            from decomp_magician.opset import is_core_aten
            covered = [n for n in data.leaf_ops if is_core_aten(n)]
            non_covered = [n for n in data.leaf_ops if not is_core_aten(n)]
            json_data["opset"] = args.target_opset
            json_data["leaf_ops_in_opset"] = len(covered)
            json_data["leaf_ops_not_in_opset"] = sorted(non_covered)
        json_data["untraceable_ops"] = [
            {"op": name, "error": reason}
            for name, reason in data.untraceable_ops
        ]
        if data.dtensor:
            dt = data.dtensor
            json_data["dtensor"] = {
                "registered": dt.registered,
                "decomp_fallback": dt.decomp_fallback,
                "missing": dt.missing,
                "fully_covered": dt.fully_covered,
                "has_gaps": dt.has_gaps,
                "top_uncovered": [
                    {"op": name, "appearances": count}
                    for name, count in dt.top_uncovered
                ],
            }
        print(json.dumps(json_data, indent=2))
        return 0

    mode = "compile" if args.compile else "full"
    trace_pct = data.traceable / data.total_non_out * 100 if data.total_non_out else 0

    lines = [
        _c(_BOLD, "Decomposition table statistics") + f"  ({mode} decomposition)",
        "",
        f"  Total ops in table:  {data.total}  ({data.total_non_out} excluding _out variants)",
    ]

    type_parts = []
    for dt in ("table", "both", "CIA", "leaf"):
        if data.by_type.get(dt, 0) > 0:
            type_parts.append(f"{data.by_type[dt]} {dt}")
    lines.append(f"  By type:             {', '.join(type_parts)}")
    lines.append(f"  Inductor-kept:       {_c(_YELLOW, str(data.inductor_kept))}")
    lines.append(f"  Traceable:           {_c(_GREEN, str(data.traceable))} ({trace_pct:.0f}%)")
    lines.append(f"  Untraceable:         {_c(_RED, str(data.untraceable))}")
    if data.classify_errors > 0:
        lines.append(f"  Classify errors:     {_c(_DIM, str(data.classify_errors))}")

    lines.append("")
    lines.append(_c(_BOLD, "Top leaf ops") + "  (most common across all decompositions):")
    top_leaves = data.leaf_ops.most_common(15)
    if top_leaves:
        name_width = max(len(name) for name, _ in top_leaves)
        for name, count in top_leaves:
            bar = "█" * min(count // 10, 40)
            lines.append(f"  {name:<{name_width}}  {count:>4}  {_c(_DIM, bar)}")

    lines.append("")
    lines.append(_c(_BOLD, "Deepest decomposition chains") + ":")
    for name, depth in data.deepest:
        lines.append(f"  {name:<50}  depth {depth}")

    # Untraceable ops breakdown
    if data.untraceable_ops:
        # Group by error type
        err_categories: Counter[str] = Counter()
        for _, reason in data.untraceable_ops:
            err_type = reason.split(":")[0].strip() if ":" in reason else reason
            err_categories[err_type] += 1
        lines.append("")
        lines.append(_c(_BOLD, "Untraceable ops by error type") + ":")
        for err_type, count in err_categories.most_common():
            lines.append(f"  {count:>4}  {err_type}")

    # Opset coverage analysis
    if args.target_opset:
        from decomp_magician.opset import is_core_aten
        covered_leaf_ops = [name for name in data.leaf_ops if is_core_aten(name)]
        non_covered_leaf_ops = [name for name in data.leaf_ops if not is_core_aten(name)]
        total_unique = len(data.leaf_ops)
        covered_unique = len(covered_leaf_ops)
        pct = covered_unique / total_unique * 100 if total_unique else 0

        lines.append("")
        lines.append(
            _c(_BOLD, "Leaf op coverage") +
            f"  (target: {args.target_opset})"
        )
        lines.append(
            f"  {_c(_GREEN, str(covered_unique))}/{total_unique} unique leaf ops are in "
            f"{args.target_opset} ({pct:.0f}%)"
        )
        if non_covered_leaf_ops:
            lines.append("")
            lines.append(
                _c(_BOLD, "Leaf ops NOT in " + args.target_opset) + ":"
            )
            nc_width = max(len(n) for n in non_covered_leaf_ops)
            for name in sorted(non_covered_leaf_ops):
                count = data.leaf_ops[name]
                padded = name.ljust(nc_width)
                lines.append(f"  {_c(_RED, padded)}  appears in {count} decompositions")

    # DTensor coverage section
    if data.dtensor:
        dt = data.dtensor
        total_classified = dt.registered + dt.decomp_fallback + dt.missing
        reg_pct = dt.registered / total_classified * 100 if total_classified else 0

        lines.append("")
        lines.append(_c(_BOLD, "DTensor coverage") + ":")
        lines.append(f"  Registered strategy:   {_c(_GREEN, str(dt.registered))} ({reg_pct:.0f}%)")
        lines.append(f"  Decomp fallback:       {dt.decomp_fallback}")
        lines.append(f"  No strategy:           {_c(_RED, str(dt.missing))}")
        lines.append("")

        traceable_with_children = dt.fully_covered + dt.has_gaps
        if traceable_with_children > 0:
            cov_pct = dt.fully_covered / traceable_with_children * 100
            covered_str = _c(_GREEN, str(dt.fully_covered))
            lines.append(f"  Fully covered trees:   {covered_str}/{traceable_with_children} ({cov_pct:.0f}%)")
            lines.append(f"  Trees with gaps:       {_c(_RED, str(dt.has_gaps))}")

        if dt.top_uncovered:
            lines.append("")
            lines.append(_c(_BOLD, "Top uncovered leaf ops") + "  (most common gaps across all trees):")
            uc_width = max(len(name) for name, _ in dt.top_uncovered)
            for name, count in dt.top_uncovered:
                bar = "█" * min(count // 2, 40)
                lines.append(f"  {_c(_RED, name):<{uc_width + 10}}  {count:>4}  {_c(_DIM, bar)}")

    print("\n".join(lines))
    return 0


def _run_opset(op, args) -> int:
    """Check if an op decomposes fully to a target opset."""
    try:
        cov = check_opset_coverage(
            op, opset=args.target_opset,
            depth=args.depth, compile=args.compile,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.json:
        json_data = {
            "op": cov.op,
            "opset": cov.opset,
            "fully_covered": cov.fully_covered,
            "total_leaves": cov.total_leaves,
            "covered_leaves": cov.covered_leaves,
            "non_covered": [
                {"op": name, "count": count}
                for name, count in cov.non_covered
            ],
        }
        print(json.dumps(json_data, indent=2))
        return 0

    mode = "compile" if args.compile else "full"
    lines = [
        _c(_BOLD, cov.op) + f"  target: {cov.opset}  ({mode} decomposition)",
    ]

    if cov.fully_covered:
        lines.append("")
        lines.append(_c(_GREEN, "FULLY COVERED") + f"  all {cov.total_leaves} leaf ops are in {cov.opset}")
    else:
        pct = cov.covered_leaves / cov.total_leaves * 100 if cov.total_leaves else 0
        lines.append("")
        lines.append(
            _c(_RED, "NOT FULLY COVERED") +
            f"  {cov.covered_leaves}/{cov.total_leaves} leaf ops in {cov.opset} ({pct:.0f}%)"
        )
        lines.append("")
        lines.append(_c(_BOLD, "Non-covered ops") + f"  (not in {cov.opset}):")
        if cov.non_covered:
            name_width = max(len(n) for n, _ in cov.non_covered)
            for name, count in cov.non_covered:
                padded = name.ljust(name_width)
                lines.append(f"  {_c(_RED, padded)}  x{count}")

    print("\n".join(lines))
    return 0


def _run_diff(op, args) -> int:
    """Show diff between decompositions.

    Two modes:
      --diff           → compare full vs compile for the same op
      --diff <op2>     → compare this op vs op2 (same mode)
    """
    if args.diff is True:
        # No second op: compare full vs compile
        diff = compute_diff(op, depth=args.depth)
    else:
        # Second op provided: compare two ops
        result2 = resolve_op(args.diff)
        if isinstance(result2, list):
            if len(result2) == 0:
                print(f"No ops found matching '{args.diff}'", file=sys.stderr)
                return 1
            print(f"Ambiguous op name '{args.diff}'. Candidates:", file=sys.stderr)
            for c in sorted(result2):
                print(f"  {c}", file=sys.stderr)
            return 1
        diff = compute_diff_ops(op, result2, depth=args.depth, compile=args.compile)

    if args.json:
        json_data = {
            "left": diff.left_label,
            "right": diff.right_label,
            "added": [{"op": n, "count": c} for n, c in diff.added.most_common()],
            "removed": [{"op": n, "count": c} for n, c in diff.removed.most_common()],
            "changed": [
                {"op": n, "left_count": lc, "right_count": rc}
                for n, lc, rc in diff.changed
            ],
        }
        print(json.dumps(json_data, indent=2))
        return 0

    has_changes = diff.added or diff.removed or diff.changed

    left_short = diff.left_label.split("  ")[0]
    right_short = diff.right_label.split("  ")[0]

    lines = [
        _c(_BOLD, diff.left_label) + "  vs  " + _c(_BOLD, diff.right_label),
    ]

    if not has_changes:
        lines.append("")
        lines.append(_c(_DIM, "No differences — both produce the same leaf frontier."))
        print("\n".join(lines))
        return 0

    if diff.removed:
        lines.append("")
        lines.append(_c(_BOLD, "Removed") + f"  (in {left_short} only):")
        for name, count in diff.removed.most_common():
            lines.append(f"  {_c(_RED, '-')} {name}  x{count}")

    if diff.added:
        lines.append("")
        lines.append(_c(_BOLD, "Added") + f"  (in {right_short} only):")
        for name, count in diff.added.most_common():
            lines.append(f"  {_c(_GREEN, '+')} {name}  x{count}")

    if diff.changed:
        lines.append("")
        lines.append(_c(_BOLD, "Changed counts") + ":")
        for name, lc, rc in diff.changed:
            delta = rc - lc
            direction = _c(_GREEN, f"+{delta}") if delta > 0 else _c(_RED, str(delta))
            lines.append(f"  {_c(_YELLOW, '~')} {name}  x{lc} -> x{rc}  ({direction})")

    print("\n".join(lines))
    return 0


def _run_backward(op, args) -> int:
    """Show ops dispatched during gradient computation."""
    from collections import Counter

    result = trace_backward(op)
    name = op_display_name(op)

    if isinstance(result, str):
        if args.json:
            print(json.dumps({"op": name, "backward": None, "error": result}))
        else:
            print(f"{_c(_BOLD, name)}  backward")
            print(f"  {_c(_RED, 'error')}: {result}")
        return 1

    op_counts: Counter[str] = Counter()
    for child_op in result:
        op_counts[op_display_name(child_op)] += 1

    if args.json:
        print(json.dumps({
            "op": name,
            "backward": [
                {"op": n, "count": c} for n, c in op_counts.most_common()
            ],
            "total_instances": len(result),
        }, indent=2))
        return 0

    lines = [_c(_BOLD, name) + "  backward"]
    if not op_counts:
        lines.append("  (no ops recorded during backward)")
    else:
        name_width = max(len(n) for n in op_counts)
        for child_name, count in op_counts.most_common():
            count_str = f"  x{count}" if count > 1 else ""
            lines.append(f"  {child_name:<{name_width}}{count_str}")
        total = sum(op_counts.values())
        lines.append(f"\n{len(op_counts)} unique ops, {total} total instances")

    print("\n".join(lines))
    return 0


def _run_model(args) -> int:
    """Analyze an exported model file."""
    import warnings

    import torch

    path = args.model
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ep = torch.export.load(path)
    except Exception as e:
        print(f"Failed to load model from {path}: {e}", file=sys.stderr)
        return 1

    # If target opset specified, decompose the graph first
    if args.target_opset == "core_aten":
        print("Decomposing model to core ATen ops...", file=sys.stderr)
        from torch._decomp import core_aten_decompositions
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ep = ep.run_decompositions(core_aten_decompositions())
        except Exception as e:
            print(f"Failed to decompose model: {e}", file=sys.stderr)
            return 1

    # Collect all aten ops from the graph
    op_counts: Counter[str] = Counter()
    op_objects: dict[str, torch._ops.OpOverload] = {}
    for node in ep.graph.nodes:
        if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload):
            name = op_display_name(node.target)
            op_counts[name] += 1
            op_objects[name] = node.target

    if not op_counts:
        print("No ATen ops found in the model graph.", file=sys.stderr)
        return 1

    # Classify DTensor coverage if requested
    dtensor_info: dict[str, str] = {}
    if args.dtensor:
        from decomp_magician.classify import classify as classify_op
        for name, op_obj in op_objects.items():
            try:
                cls = classify_op(op_obj, dtensor=True)
                dtensor_info[name] = cls.dtensor_strategy or "missing"
            except Exception:
                dtensor_info[name] = "missing"

    if args.json:
        json_data: dict = {
            "model": path,
            "total_ops": sum(op_counts.values()),
            "unique_ops": len(op_counts),
            "ops": [],
        }
        for name, count in op_counts.most_common():
            entry: dict = {"op": name, "count": count}
            if name in dtensor_info:
                entry["dtensor_strategy"] = dtensor_info[name]
            json_data["ops"].append(entry)
        if args.target_opset:
            json_data["opset"] = args.target_opset
            json_data["decomposed"] = True
        if dtensor_info:
            missing = [n for n, s in dtensor_info.items() if s == "missing"]
            json_data["dtensor_covered"] = len(missing) == 0
            json_data["dtensor_missing_ops"] = sorted(missing)
        print(json.dumps(json_data, indent=2))
        return 0

    total = sum(op_counts.values())
    stage = f"  (after {args.target_opset} decomposition)" if args.target_opset else ""
    lines = [
        _c(_BOLD, "Model analysis") + f"  {path}{stage}",
        "",
        f"  Total op instances:  {total}",
        f"  Unique ops:          {len(op_counts)}",
    ]

    # DTensor summary at top if requested
    if dtensor_info:
        missing = [n for n, s in dtensor_info.items() if s == "missing"]
        if missing:
            lines.append(f"  DTensor:             {_c(_RED, f'{len(missing)} ops missing strategies')}")
        else:
            lines.append(f"  DTensor:             {_c(_GREEN, 'all ops covered')}")

    lines.append("")
    lines.append(_c(_BOLD, "Ops in graph") + ":")

    name_width = max(len(n) for n in op_counts)

    if args.target_opset:
        for name, count in op_counts.most_common():
            dt_tag = _format_model_dtensor_tag(dtensor_info, name)
            lines.append(f"  {name:<{name_width}}  x{count}{dt_tag}")
        lines.append("")
        lines.append(
            _c(_DIM, f"These are the ops a {args.target_opset} backend must implement for this model.")
        )
    else:
        # Show decomposition status per op
        from torch._decomp import decomposition_table
        decomposable = 0
        for name, count in op_counts.most_common():
            op_obj = op_objects.get(name)
            has_decomp = op_obj is not None and op_obj in decomposition_table
            tag = _c(_YELLOW, "[decomposable]") if has_decomp else _c(_DIM, "[leaf]")
            dt_tag = _format_model_dtensor_tag(dtensor_info, name)
            lines.append(f"  {name:<{name_width}}  x{count}  {tag}{dt_tag}")
            if has_decomp:
                decomposable += 1
        lines.append("")
        lines.append(
            f"  {decomposable}/{len(op_counts)} ops have decompositions. "
            f"Use --target-opset core_aten to decompose and show final ops."
        )

    print("\n".join(lines))
    return 0


def _format_model_dtensor_tag(dtensor_info: dict[str, str], name: str) -> str:
    """Format a DTensor tag for model analysis output."""
    strategy = dtensor_info.get(name)
    if strategy is None:
        return ""
    if strategy == "registered":
        return "  " + _c(_GREEN, "dtensor: ok")
    if strategy == "decomp-fallback":
        return "  " + _c(_GREEN, "dtensor: ok (via decomp)")
    return "  " + _c(_RED, "dtensor: MISSING")


class LeafFrontier(NamedTuple):
    counts: Counter[str]
    inductor_kept: set[str]
    untraceable: set[str]
    dtensor_uncovered: set[str]  # leaves with at least one path lacking a registered ancestor


def _collect_leaf_frontier(node: DecompNode) -> LeafFrontier:
    """Walk a tree and collect the leaf frontier with propagated counts.

    Counts come from collect_leaf_counts (single source of multiplicative
    walk logic). A separate walk collects set-membership properties and
    DTensor ancestor coverage.
    """
    counts = collect_leaf_counts(node)
    inductor_kept_ops: set[str] = set()
    untraceable_ops: set[str] = set()
    dtensor_uncovered_ops: set[str] = set()

    def walk(n: DecompNode, ancestor_covered: bool = False) -> None:
        covered = ancestor_covered or is_dtensor_intercept(
            n.classification.dtensor_strategy
        )
        if not n.children:
            name = op_display_name(n.op)
            if n.classification.inductor_kept:
                inductor_kept_ops.add(name)
            if not n.traceable:
                untraceable_ops.add(name)
            if n.classification.dtensor_strategy == "missing" and not covered:
                dtensor_uncovered_ops.add(name)
            return
        for c in n.children:
            walk(c, covered)

    walk(node)
    return LeafFrontier(counts, inductor_kept_ops, untraceable_ops, dtensor_uncovered_ops)


def format_leaves(
    node: DecompNode,
    opset_checker: tuple[str, Callable[[str], bool]] | None = None,
) -> str:
    """Format the leaf frontier with propagated counts."""
    root_name = op_display_name(node.op)

    if not node.children:
        return f"{_c(_DIM, root_name)}  [leaf, no decomposition]"

    lf = _collect_leaf_frontier(node)

    # Collect dispatch info for leaves if needed
    leaf_dispatch: dict[str, DispatchInfo] = {}
    if _show_dispatch or _show_mode_sensitivity:
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
            tags.append(_c(_YELLOW, "inductor-kept"))
        if name in lf.untraceable:
            tags.append(_c(_RED, "untraceable"))
        if name in lf.dtensor_uncovered:
            tags.append(_c(_RED, "dtensor: MISSING"))
        # Dispatch info
        dinfo = leaf_dispatch.get(name)
        if dinfo and _show_dispatch:
            tags.append(_c(_DIM, format_dispatch_short(dinfo)))
        if dinfo and _show_mode_sensitivity:
            if dinfo.has_adiov:
                tags.append(_c(_RED, "ADIOV"))
            if dinfo.mode_sensitive:
                tags.append(_c(_YELLOW, "mode-sensitive"))
        if opset_checker:
            opset_name, checker_fn = opset_checker
            if checker_fn(name):
                tags.append(_c(_GREEN, opset_name))
            else:
                tags.append(_c(_RED, f"NOT {opset_name}"))
        tag_str = "  [" + ", ".join(tags) + "]" if tags else ""
        lines.append(f"  {name:<{name_width}}  x{count}{tag_str}")

    total = sum(lf.counts.values())
    header = _c(_BOLD, root_name) + " decomposes to:"
    footer = f"\n{len(lf.counts)} unique ops, {total} total instances"
    if lf.untraceable:
        n = len(lf.untraceable)
        warning = _c(_RED, f"  ({n} untraceable — frontier may be incomplete)")
        footer += warning
    return header + "\n" + "\n".join(lines) + footer


def _leaves_to_dict(node: DecompNode) -> dict:
    """Convert leaf frontier to a JSON-serializable dict."""
    root_name = op_display_name(node.op)

    if not node.children:
        return {"op": root_name, "decomp_type": "leaf", "leaves": []}

    lf = _collect_leaf_frontier(node)

    leaves = []
    for name, count in lf.counts.most_common():
        entry: dict = {"op": name, "count": count}
        if name in lf.inductor_kept:
            entry["inductor_kept"] = True
        if name in lf.untraceable:
            entry["untraceable"] = True
        if name in lf.dtensor_uncovered:
            entry["dtensor_uncovered"] = True
        leaves.append(entry)

    return {
        "op": root_name,
        "decomp_type": node.classification.decomp_type,
        "leaves": leaves,
        "total_instances": sum(lf.counts.values()),
    }


def format_summary(node: DecompNode) -> str:
    """One-line summary of the tree's composition."""
    counts = {dt: 0 for dt in DECOMP_TYPES}
    inductor_kept = 0
    dtensor_missing = 0
    untraceable = 0

    def walk(n: DecompNode, ancestor_covered: bool = False) -> None:
        nonlocal inductor_kept, dtensor_missing, untraceable
        dt = n.classification.decomp_type
        counts[dt] = counts.get(dt, 0) + 1
        if n.classification.inductor_kept:
            inductor_kept += 1
        if n.classification.dtensor_strategy == "missing" and not ancestor_covered:
            dtensor_missing += 1
        if not n.traceable:
            untraceable += 1
        covered = ancestor_covered or is_dtensor_intercept(
            n.classification.dtensor_strategy
        )
        for c in n.children:
            walk(c, covered)

    walk(node)
    total = sum(counts.values())

    # Build type breakdown (display order: most interesting first)
    type_parts = []
    for dt in sorted(counts, key=lambda t: counts[t], reverse=True):
        if counts[dt] > 0:
            type_parts.append(f"{counts[dt]} {dt}")
    ops_word = "op" if total == 1 else "ops"
    parts = [f"{total} {ops_word} ({', '.join(type_parts)})"]

    if inductor_kept > 0:
        parts.append(_c(_YELLOW, f"{inductor_kept} inductor-kept"))
    if untraceable > 0:
        parts.append(_c(_RED, f"{untraceable} untraceable"))
    has_dtensor = node.classification.dtensor_strategy is not None
    if has_dtensor:
        if dtensor_missing > 0:
            parts.append(_c(_RED, f"dtensor: {dtensor_missing} uncovered"))
        else:
            parts.append(_c(_GREEN, "dtensor: covered"))

    return " · ".join(parts)


def _collect_untraceable_errors(node: DecompNode) -> list[tuple[str, str]]:
    """Collect unique (op_name, error) pairs for untraceable nodes in a tree."""
    seen: set[str] = set()
    errors: list[tuple[str, str]] = []

    def walk(n: DecompNode) -> None:
        if not n.traceable and n.error:
            name = op_display_name(n.op)
            if name not in seen:
                seen.add(name)
                errors.append((name, n.error))
        for c in n.children:
            walk(c)

    walk(node)
    return errors


def _warn_untraceable(node: DecompNode) -> None:
    """Print a stderr warning if the tree contains untraceable ops."""
    errors = _collect_untraceable_errors(node)
    if not errors:
        return

    n = len(errors)
    print(
        f"\n{_c(_YELLOW, 'warning')}: {n} {'op' if n == 1 else 'ops'} "
        f"could not be traced with synthetic inputs — "
        f"{'its subtree is' if n == 1 else 'their subtrees are'} "
        f"incomplete.",
        file=sys.stderr,
    )
    for op_name, error in errors:
        print(f"  {op_name}: {error}", file=sys.stderr)


def _add_untraceable_warnings(d: dict, node: DecompNode) -> None:
    """Add a warnings field to a JSON dict if the tree has untraceable ops."""
    errors = _collect_untraceable_errors(node)
    if errors:
        d["warnings"] = [
            {"op": name, "message": f"could not trace: {error}"}
            for name, error in errors
        ]


def tree_to_dict(node: DecompNode) -> dict:
    """Convert a DecompNode tree to a JSON-serializable dict."""
    cls = node.classification
    d = {
        "op": op_display_name(node.op),
        "schema": str(node.op._schema),
        "decomp_type": cls.decomp_type,
        "count": node.count,
        "inductor_kept": cls.inductor_kept,
        "backends": cls.has_backend,
        "tags": cls.tags,
        "mutable": cls.is_mutable,
        "alias_info": cls.has_alias_info,
        "traceable": node.traceable,
    }
    if node.error:
        d["error"] = node.error
    if cls.dtensor_strategy is not None:
        d["dtensor_strategy"] = cls.dtensor_strategy
    if node.children:
        d["children"] = [tree_to_dict(c) for c in node.children]
    return d


def _print_verbose(node: DecompNode, indent: int = 0) -> None:
    """Print detailed classification for each node."""
    prefix = "  " * indent
    cls = node.classification
    name = op_display_name(node.op)
    print(f"{prefix}{name}:")
    print(f"{prefix}  schema: {node.op._schema}")
    print(f"{prefix}  decomp_type: {cls.decomp_type}")
    backends = ", ".join(k for k, v in cls.has_backend.items() if v)
    no_backends = ", ".join(k for k, v in cls.has_backend.items() if not v)
    if backends:
        print(f"{prefix}  backends: {backends}")
    if no_backends:
        print(f"{prefix}  no backend: {no_backends}")
    if cls.tags:
        print(f"{prefix}  tags: {', '.join(cls.tags)}")
    if cls.is_mutable:
        print(f"{prefix}  mutable: True")
    if cls.has_alias_info:
        print(f"{prefix}  alias_info: True")
    if cls.inductor_kept:
        print(f"{prefix}  inductor_kept: True")
    if node.error:
        print(f"{prefix}  error: {node.error}")
    # Always show dispatch info in verbose mode
    dinfo = get_dispatch_info_cached(node.op)
    print(f"{prefix}  autograd_type: {dinfo.autograd_type}")
    print(f"{prefix}  has_adiov: {dinfo.has_adiov}")
    print(f"{prefix}  mode_sensitive: {dinfo.mode_sensitive}")
    for child in node.children:
        _print_verbose(child, indent + 1)


if __name__ == "__main__":
    sys.exit(main())
