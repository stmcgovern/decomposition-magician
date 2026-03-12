"""CLI entry point for decomposition-magician."""

from __future__ import annotations

import argparse
import sys

from decomp_magician.resolve import resolve_op
from decomp_magician.tree import DecompNode, build_tree


def format_tree(node: DecompNode, prefix: str = "", is_last: bool = True, is_root: bool = True) -> str:
    """Format a DecompNode tree as a string with box-drawing characters."""
    lines = []

    # Build the annotation
    annotation = _format_annotation(node)

    # Build the line
    if is_root:
        line = f"{_op_display_name(node.op)}  {annotation}"
    else:
        connector = "└── " if is_last else "├── "
        count_str = f"  x{node.count}" if node.count > 1 else ""
        line = f"{prefix}{connector}{_op_display_name(node.op)}  {annotation}{count_str}"

    lines.append(line)

    # Recurse into children
    child_prefix = prefix + ("    " if is_last else "│   ") if not is_root else ""
    for i, child in enumerate(node.children):
        is_last_child = i == len(node.children) - 1
        lines.append(format_tree(child, child_prefix, is_last_child, is_root=False))

    return "\n".join(lines)


def _op_display_name(op) -> str:
    """Short display name: aten.add.Tensor, always showing overload."""
    name = op.name()  # "aten::add.Tensor" or "aten::rsqrt" (default omits overload)
    dotted = name.replace("::", ".")
    # PyTorch omits ".default" from name() — add it back for clarity
    # A name without overload has exactly one dot: "aten.rsqrt"
    if dotted.count(".") < 2:
        dotted += ".default"
    return dotted


def _format_annotation(node: DecompNode) -> str:
    """Format the bracket annotation for a node."""
    parts = []

    # Decomp type
    cls = node.classification
    if cls.decomp_type == "leaf":
        parts.append("leaf")
    else:
        parts.append(cls.decomp_type)

    # Inductor exclusion
    if cls.inductor_excluded:
        parts.append("inductor-kept")

    # Traceability
    if not node.traceable:
        parts.append("untraceable")

    annotation = f"[{', '.join(parts)}]"

    # DTensor strategy (outside brackets)
    if cls.dtensor_strategy is not None:
        if cls.dtensor_strategy == "registered":
            annotation += "  dtensor: ok"
        elif cls.dtensor_strategy == "decomp-fallback":
            annotation += "  dtensor: ok (via decomp)"
        elif cls.dtensor_strategy == "missing":
            annotation += "  dtensor: MISSING"

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
        "--verbose",
        action="store_true",
        help="Show full classification details per op",
    )
    args = parser.parse_args(argv)

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

    # Build and print the tree
    node = build_tree(op, depth=args.depth, dtensor=args.dtensor)

    print(format_tree(node))

    if args.verbose:
        print()
        _print_verbose(node)

    return 0


def _print_verbose(node: DecompNode, indent: int = 0) -> None:
    """Print detailed classification for each node."""
    prefix = "  " * indent
    cls = node.classification
    name = _op_display_name(node.op)
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
    if cls.inductor_excluded:
        print(f"{prefix}  inductor_kept: True")
    if node.error:
        print(f"{prefix}  error: {node.error}")
    for child in node.children:
        _print_verbose(child, indent + 1)


if __name__ == "__main__":
    sys.exit(main())
