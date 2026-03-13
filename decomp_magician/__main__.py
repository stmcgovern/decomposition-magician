"""CLI entry point for decomposition-magician."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import NamedTuple

from decomp_magician.diff import compute_diff
from decomp_magician.graph import format_dot, format_mermaid
from decomp_magician.opset import OPSETS, check_opset_coverage
from decomp_magician.resolve import resolve_op
from decomp_magician.reverse import reverse_lookup
from decomp_magician.stats import compute_stats
from decomp_magician.tree import DecompNode, build_tree, op_display_name


# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"

# Whether color is currently enabled
_use_color = False


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


def format_tree(node: DecompNode, prefix: str = "", is_last: bool = True, is_root: bool = True) -> str:
    """Format a DecompNode tree as a string with box-drawing characters."""
    lines = []

    # Build the annotation
    annotation = _format_annotation(node)

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

    # Recurse into children
    child_prefix = prefix + ("    " if is_last else "│   ") if not is_root else ""
    for i, child in enumerate(node.children):
        is_last_child = i == len(node.children) - 1
        lines.append(format_tree(child, child_prefix, is_last_child, is_root=False))

    return "\n".join(lines)



def _format_annotation(node: DecompNode) -> str:
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

    # Traceability
    if not node.traceable:
        parts.append(_c(_RED, "untraceable"))

    annotation = f"[{', '.join(parts)}]"

    # Show brief explanation for untraceable ops
    if not node.traceable and node.classification.decomp_type != "leaf":
        annotation += "  " + _c(_DIM, "(has decomposition, could not trace)")

    # DTensor strategy (outside brackets)
    if cls.dtensor_strategy is not None:
        if cls.dtensor_strategy == "registered":
            annotation += "  " + _c(_GREEN, "dtensor: ok")
        elif cls.dtensor_strategy == "decomp-fallback":
            annotation += "  " + _c(_GREEN, "dtensor: ok (via decomp)")
        elif cls.dtensor_strategy == "missing":
            annotation += "  " + _c(_RED, "dtensor: MISSING")

    return annotation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decomp_magician",
        description="Inspect PyTorch operator decomposition trees.",
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
        action="store_true",
        help="Show what changes between full and compile decomposition",
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
    args = parser.parse_args(argv)

    # Initialize color
    global _use_color
    if args.no_color or args.json:
        _use_color = False
    else:
        _use_color = _should_use_color()

    # Stats mode — no op argument needed
    if args.stats:
        return _run_stats(args)

    # Model analysis mode — no op argument needed
    if args.model:
        return _run_model(args)

    # All other modes require an op
    if args.op is None:
        parser.error("the following arguments are required: op")

    # Validate flag combinations
    # --mermaid, --dot, --leaves, --reverse, --target-opset, --diff are mutually exclusive output modes
    # --json combines with --leaves, --reverse, --target-opset, --diff but not --mermaid/--dot
    mode_flags = sum([
        args.mermaid, args.dot, args.leaves, args.reverse,
        bool(args.target_opset), args.diff,
    ])
    if mode_flags > 1:
        conflicting = [f for f, v in [
            ("--mermaid", args.mermaid), ("--dot", args.dot),
            ("--leaves", args.leaves), ("--reverse", args.reverse),
            ("--target-opset", bool(args.target_opset)), ("--diff", args.diff),
        ] if v]
        print(f"Conflicting flags: {', '.join(conflicting)} (pick one)", file=sys.stderr)
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

    # Target opset mode
    if args.target_opset:
        return _run_opset(op, args)

    # Diff mode
    if args.diff:
        return _run_diff(op, args)

    # Build and print the tree
    node = build_tree(op, depth=args.depth, dtensor=args.dtensor, compile=args.compile)

    if args.json:
        if args.leaves:
            print(json.dumps(_leaves_to_dict(node), indent=2))
        else:
            print(json.dumps(tree_to_dict(node), indent=2))
        return 0

    if args.mermaid:
        print(format_mermaid(node))
        print("", file=sys.stderr)
        print("Paste into ```mermaid fenced block in GitHub markdown to render.", file=sys.stderr)
        return 0

    if args.dot:
        print(format_dot(node))
        print("", file=sys.stderr)
        print("Pipe to: dot -Tsvg > graph.svg  or  dot -Tpng > graph.png", file=sys.stderr)
        return 0

    if args.leaves:
        print(format_leaves(node))
    else:
        print(format_tree(node))
        print()
        print(format_summary(node))

        if args.verbose:
            print()
            _print_verbose(node)

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

    data = compute_stats(compile=args.compile)

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
        print(json.dumps(json_data, indent=2))
        return 0

    mode = "compile" if args.compile else "full"
    trace_pct = data.traceable / data.total_non_out * 100 if data.total_non_out else 0

    lines = [
        _c(_BOLD, f"Decomposition table statistics") + f"  ({mode} decomposition)",
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
                lines.append(f"  {_c(_RED, name):<{name_width + len(_RED) + len(_RESET)}}  x{count}")

    print("\n".join(lines))
    return 0


def _run_diff(op, args) -> int:
    """Show diff between full and compile decomposition."""
    diff = compute_diff(op, depth=args.depth)

    if args.json:
        json_data = {
            "op": diff.op,
            "left_mode": diff.left_mode,
            "right_mode": diff.right_mode,
            "added": [{"op": n, "count": c} for n, c in diff.added.most_common()],
            "removed": [{"op": n, "count": c} for n, c in diff.removed.most_common()],
            "changed": [
                {"op": n, "full_count": fc, "compile_count": cc}
                for n, fc, cc in diff.changed
            ],
        }
        print(json.dumps(json_data, indent=2))
        return 0

    has_changes = diff.added or diff.removed or diff.changed

    lines = [
        _c(_BOLD, diff.op) + f"  {diff.left_mode} vs {diff.right_mode} decomposition",
    ]

    if not has_changes:
        lines.append("")
        lines.append(_c(_DIM, "No differences — both modes produce the same leaf frontier."))
        print("\n".join(lines))
        return 0

    if diff.removed:
        lines.append("")
        lines.append(_c(_BOLD, "Removed") + f"  (in {diff.left_mode} but not {diff.right_mode}):")
        for name, count in diff.removed.most_common():
            lines.append(f"  {_c(_RED, '-')} {name}  x{count}")

    if diff.added:
        lines.append("")
        lines.append(_c(_BOLD, "Added") + f"  (in {diff.right_mode} but not {diff.left_mode}):")
        for name, count in diff.added.most_common():
            lines.append(f"  {_c(_GREEN, '+')} {name}  x{count}")

    if diff.changed:
        lines.append("")
        lines.append(_c(_BOLD, "Changed counts") + ":")
        for name, fc, cc in diff.changed:
            direction = _c(_GREEN, "+" + str(cc - fc)) if cc > fc else _c(_RED, str(cc - fc))
            lines.append(f"  {_c(_YELLOW, '~')} {name}  x{fc} -> x{cc}  ({direction})")

    print("\n".join(lines))
    return 0


def _run_model(args) -> int:
    """Analyze an exported model file."""
    import torch

    path = args.model
    try:
        ep = torch.export.load(path)
    except Exception as e:
        print(f"Failed to load model from {path}: {e}", file=sys.stderr)
        return 1

    # Collect all aten ops from the graph
    op_counts: Counter[str] = Counter()
    for node in ep.graph.nodes:
        if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload):
            op_counts[op_display_name(node.target)] += 1

    if not op_counts:
        print("No ATen ops found in the model graph.", file=sys.stderr)
        return 1

    if args.json:
        json_data = {
            "model": path,
            "total_ops": sum(op_counts.values()),
            "unique_ops": len(op_counts),
            "ops": [
                {"op": name, "count": count}
                for name, count in op_counts.most_common()
            ],
        }

        # If target opset specified, add coverage info
        if args.target_opset:
            from decomp_magician.opset import is_core_aten
            covered = []
            non_covered = []
            for name, count in op_counts.most_common():
                if is_core_aten(name):
                    covered.append({"op": name, "count": count})
                else:
                    non_covered.append({"op": name, "count": count})
            json_data["opset"] = args.target_opset
            json_data["covered"] = covered
            json_data["non_covered"] = non_covered

        print(json.dumps(json_data, indent=2))
        return 0

    total = sum(op_counts.values())
    lines = [
        _c(_BOLD, f"Model analysis") + f"  {path}",
        "",
        f"  Total op instances:  {total}",
        f"  Unique ops:          {len(op_counts)}",
        "",
        _c(_BOLD, "Ops in graph") + ":",
    ]

    name_width = max(len(n) for n in op_counts)
    for name, count in op_counts.most_common():
        lines.append(f"  {name:<{name_width}}  x{count}")

    # If target opset specified, show coverage
    if args.target_opset:
        from decomp_magician.opset import is_core_aten
        non_covered = [(n, c) for n, c in op_counts.most_common() if not is_core_aten(n)]
        covered_count = len(op_counts) - len(non_covered)

        lines.append("")
        if not non_covered:
            lines.append(
                _c(_GREEN, "FULLY COVERED") +
                f"  all {len(op_counts)} ops are in {args.target_opset}"
            )
        else:
            pct = covered_count / len(op_counts) * 100 if op_counts else 0
            lines.append(
                _c(_RED, "NOT FULLY COVERED") +
                f"  {covered_count}/{len(op_counts)} ops in {args.target_opset} ({pct:.0f}%)"
            )
            lines.append("")
            lines.append(_c(_BOLD, "Non-covered ops") + f"  (not in {args.target_opset}):")
            nc_width = max(len(n) for n, _ in non_covered)
            for name, count in non_covered:
                lines.append(f"  {_c(_RED, name):<{nc_width + len(_RED) + len(_RESET)}}  x{count}")

    print("\n".join(lines))
    return 0


class LeafFrontier(NamedTuple):
    counts: Counter[str]
    inductor_kept: set[str]
    untraceable: set[str]


def _collect_leaf_frontier(node: DecompNode) -> LeafFrontier:
    """Walk a tree and collect the leaf frontier with propagated counts."""
    frontier: Counter[str] = Counter()
    inductor_kept_ops: set[str] = set()
    untraceable_ops: set[str] = set()

    def walk(n: DecompNode, multiplier: int = 1) -> None:
        if len(n.children) == 0:
            name = op_display_name(n.op)
            frontier[name] += multiplier
            if n.classification.inductor_kept:
                inductor_kept_ops.add(name)
            if not n.traceable:
                untraceable_ops.add(name)
            return
        for c in n.children:
            walk(c, multiplier * c.count)

    walk(node)
    return LeafFrontier(frontier, inductor_kept_ops, untraceable_ops)


def format_leaves(node: DecompNode) -> str:
    """Format the leaf frontier with propagated counts."""
    root_name = op_display_name(node.op)

    if len(node.children) == 0:
        return f"{_c(_DIM, root_name)}  [leaf, no decomposition]"

    lf = _collect_leaf_frontier(node)

    lines = []
    name_width = max(len(name) for name in lf.counts)
    for name, count in lf.counts.most_common():
        tags = []
        if name in lf.inductor_kept:
            tags.append(_c(_YELLOW, "inductor-kept"))
        if name in lf.untraceable:
            tags.append(_c(_RED, "untraceable"))
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

    if len(node.children) == 0:
        return {"op": root_name, "decomp_type": "leaf", "leaves": []}

    lf = _collect_leaf_frontier(node)

    leaves = []
    for name, count in lf.counts.most_common():
        entry: dict = {"op": name, "count": count}
        if name in lf.inductor_kept:
            entry["inductor_kept"] = True
        if name in lf.untraceable:
            entry["untraceable"] = True
        leaves.append(entry)

    return {
        "op": root_name,
        "decomp_type": node.classification.decomp_type,
        "leaves": leaves,
        "total_instances": sum(lf.counts.values()),
    }


def format_summary(node: DecompNode) -> str:
    """One-line summary of the tree's composition."""
    counts = {"table": 0, "CIA": 0, "both": 0, "leaf": 0}
    inductor_kept = 0
    dtensor_missing = 0
    untraceable = 0

    def walk(n: DecompNode) -> None:
        nonlocal inductor_kept, dtensor_missing, untraceable
        dt = n.classification.decomp_type
        counts[dt] = counts.get(dt, 0) + 1
        if n.classification.inductor_kept:
            inductor_kept += 1
        if n.classification.dtensor_strategy == "missing":
            dtensor_missing += 1
        if not n.traceable:
            untraceable += 1
        for c in n.children:
            walk(c)

    walk(node)
    total = sum(counts.values())

    # Build type breakdown
    type_parts = []
    for dt in ("table", "CIA", "both", "leaf"):
        if counts[dt] > 0:
            type_parts.append(f"{counts[dt]} {dt}")
    ops_word = "op" if total == 1 else "ops"
    parts = [f"{total} {ops_word} ({', '.join(type_parts)})"]

    if inductor_kept > 0:
        parts.append(_c(_YELLOW, f"{inductor_kept} inductor-kept"))
    if untraceable > 0:
        parts.append(_c(_RED, f"{untraceable} untraceable"))
    if dtensor_missing > 0:
        parts.append(_c(_RED, f"{dtensor_missing} dtensor missing"))

    return " · ".join(parts)


def tree_to_dict(node: DecompNode) -> dict:
    """Convert a DecompNode tree to a JSON-serializable dict."""
    cls = node.classification
    d = {
        "op": op_display_name(node.op),
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
    for child in node.children:
        _print_verbose(child, indent + 1)


if __name__ == "__main__":
    sys.exit(main())
