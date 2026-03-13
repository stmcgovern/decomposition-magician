"""CLI entry point for decomposition-magician."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from decomp_magician.graph import format_dot, format_mermaid, op_display_name
from decomp_magician.resolve import resolve_op
from decomp_magician.tree import DecompNode, build_tree


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
        help="Operator name (e.g., 'addcmul', 'aten.addcmul.default', 'batch_norm')",
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


def format_leaves(node: DecompNode) -> str:
    """Format the leaf frontier with propagated counts."""
    root_name = op_display_name(node.op)

    if len(node.children) == 0:
        return f"{_c(_DIM, root_name)}  [leaf, no decomposition]"

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

    lines = []
    name_width = max(len(name) for name in frontier)
    for name, count in frontier.most_common():
        tags = []
        if name in inductor_kept_ops:
            tags.append(_c(_YELLOW, "inductor-kept"))
        if name in untraceable_ops:
            tags.append(_c(_RED, "untraceable"))
        tag_str = "  [" + ", ".join(tags) + "]" if tags else ""
        lines.append(f"  {name:<{name_width}}  x{count}{tag_str}")

    total = sum(frontier.values())
    header = _c(_BOLD, root_name) + " decomposes to:"
    footer = f"\n{len(frontier)} unique ops, {total} total instances"
    if untraceable_ops:
        n = len(untraceable_ops)
        warning = _c(_RED, f"  ({n} untraceable — frontier may be incomplete)")
        footer += warning
    return header + "\n" + "\n".join(lines) + footer


def _leaves_to_dict(node: DecompNode) -> dict:
    """Convert leaf frontier to a JSON-serializable dict."""
    root_name = op_display_name(node.op)

    if len(node.children) == 0:
        return {"op": root_name, "decomp_type": "leaf", "leaves": []}

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

    leaves = []
    for name, count in frontier.most_common():
        entry: dict = {"op": name, "count": count}
        if name in inductor_kept_ops:
            entry["inductor_kept"] = True
        if name in untraceable_ops:
            entry["untraceable"] = True
        leaves.append(entry)

    return {
        "op": root_name,
        "decomp_type": node.classification.decomp_type,
        "leaves": leaves,
        "total_instances": sum(frontier.values()),
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
