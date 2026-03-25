"""Graph export: Mermaid, Graphviz DOT, and JSON dict formats."""

from __future__ import annotations

from decomp_magician.classify import DecompType, get_dtensor_strategy
from decomp_magician.dispatch import DispatchInfo, get_dispatch_info_cached
from decomp_magician.tree import (
    DecompNode,
    collect_leaf_frontier,
    collect_untraceable_errors,
    op_display_name,
)


def _node_id(index: int) -> str:
    return f"n{index}"


def _is_terminal(n: DecompNode) -> bool:
    """A node is terminal if it has no children."""
    return not n.children


def _node_tags(n: DecompNode) -> list[str]:
    """Build annotation tags for a node."""
    tags = []
    if n.classification.inductor_kept and _is_terminal(n):
        tags.append("inductor-kept")
    if n.error == "cycle detected":
        tags.append("cycle")
    elif not n.traceable:
        tags.append("untraceable")
    return tags


def _walk_nodes(node: DecompNode) -> list[tuple[int, DecompNode, int | None, int]]:
    """Flatten tree into (index, node, parent_index, count) tuples."""
    result: list[tuple[int, DecompNode, int | None, int]] = []
    counter = [0]

    def walk(n: DecompNode, parent_idx: int | None, count: int) -> None:
        idx = counter[0]
        counter[0] += 1
        result.append((idx, n, parent_idx, count))
        for child in n.children:
            walk(child, idx, child.count)

    walk(node, None, 1)
    return result


def format_mermaid(node: DecompNode) -> str:
    """Format a DecompNode tree as a Mermaid flowchart."""
    lines = ["graph TD"]
    entries = _walk_nodes(node)

    for idx, n, parent_idx, count in entries:
        nid = _node_id(idx)
        name = op_display_name(n.op)
        tags = _node_tags(n)

        # Build label: name + count + tags, using <br/> for line breaks
        parts = [name]
        if count > 1:
            parts[0] += f" ×{count}"
        if tags:
            parts.append(f"({', '.join(tags)})")
        label = "<br/>".join(parts)

        # Mermaid requires quotes when label contains special chars
        needs_quotes = "<br/>" in label or "×" in label

        # Node shape: terminal=square, untraceable=trapezoid, decomposable=rounded
        if _is_terminal(n) and n.traceable:
            if needs_quotes:
                lines.append(f'    {nid}["{label}"]')
            else:
                lines.append(f"    {nid}[{label}]")
        elif not n.traceable:
            if needs_quotes:
                lines.append(f'    {nid}[/"{label}"\\]')
            else:
                lines.append(f"    {nid}[/{label}\\]")
        else:
            if needs_quotes:
                lines.append(f'    {nid}(["{label}"])')
            else:
                lines.append(f"    {nid}([{label}])")

        # Edge from parent
        if parent_idx is not None:
            pid = _node_id(parent_idx)
            lines.append(f"    {pid} --> {nid}")

    # Style classes — applied in priority order (last wins in Mermaid)
    leaf_ids = [_node_id(idx) for idx, n, _, _ in entries
                if _is_terminal(n) and n.traceable and not n.classification.inductor_kept]
    kept_ids = [_node_id(idx) for idx, n, _, _ in entries
                if n.classification.inductor_kept]
    untraceable_ids = [_node_id(idx) for idx, n, _, _ in entries
                       if not n.traceable and n.error != "cycle detected"]
    cycle_ids = [_node_id(idx) for idx, n, _, _ in entries
                 if n.error == "cycle detected"]
    decomposed_ids = [_node_id(idx) for idx, n, _, _ in entries
                      if not _is_terminal(n) and n.traceable and not n.classification.inductor_kept]

    lines.append("")
    lines.append("    %% Legend: square=terminal, rounded=decomposable, trapezoid=untraceable")
    lines.append("    %% gray=leaf, green=decomposed, yellow=inductor-kept, red=untraceable")
    lines.append("    classDef leaf fill:#f5f5f5,stroke:#999,color:#666")
    lines.append("    classDef decomposed fill:#d4edda,stroke:#28a745,color:#155724")
    lines.append("    classDef kept fill:#fff3cd,stroke:#ffc107,color:#856404")
    lines.append("    classDef untraceable fill:#f8d7da,stroke:#dc3545,color:#721c24")
    lines.append("    classDef cycle fill:#f8d7da,stroke:#dc3545,stroke-dasharray:5,color:#721c24")

    if decomposed_ids:
        lines.append(f"    class {','.join(decomposed_ids)} decomposed")
    if leaf_ids:
        lines.append(f"    class {','.join(leaf_ids)} leaf")
    if kept_ids:
        lines.append(f"    class {','.join(kept_ids)} kept")
    if untraceable_ids:
        lines.append(f"    class {','.join(untraceable_ids)} untraceable")
    if cycle_ids:
        lines.append(f"    class {','.join(cycle_ids)} cycle")

    return "\n".join(lines)


# Shared color constants for DOT
_COLORS = {
    "leaf": {"fill": "#f5f5f5", "font": "#666666"},
    "decomposed": {"fill": "#d4edda", "font": "#155724"},
    "kept": {"fill": "#fff3cd", "font": "#856404"},
    "untraceable": {"fill": "#f8d7da", "font": "#721c24"},
}


def format_dot(node: DecompNode) -> str:
    """Format a DecompNode tree as Graphviz DOT."""
    lines = [
        "digraph decomp {",
        '    rankdir=TB;',
        '    node [fontname="Helvetica", fontsize=11];',
        '    edge [fontname="Helvetica", fontsize=9];',
    ]
    entries = _walk_nodes(node)

    for idx, n, parent_idx, count in entries:
        nid = _node_id(idx)
        name = op_display_name(n.op)
        tags = _node_tags(n)
        cls = n.classification

        # Build label: name + count + tags
        label = name
        if count > 1:
            label += f" ×{count}"
        if tags:
            label += f"\\n({', '.join(tags)})"

        attrs: dict[str, str] = {"label": label, "style": "filled", "shape": "box"}

        if _is_terminal(n) and n.traceable:
            if cls.inductor_kept:
                attrs["fillcolor"] = _COLORS["kept"]["fill"]
                attrs["fontcolor"] = _COLORS["kept"]["font"]
                attrs["penwidth"] = "2"
            else:
                attrs["fillcolor"] = _COLORS["leaf"]["fill"]
                attrs["fontcolor"] = _COLORS["leaf"]["font"]
        elif not n.traceable:
            attrs["fillcolor"] = _COLORS["untraceable"]["fill"]
            attrs["fontcolor"] = _COLORS["untraceable"]["font"]
            attrs["shape"] = "trapezium"
            if n.error == "cycle detected":
                attrs["style"] = "filled,dashed"
        else:
            if cls.inductor_kept:
                attrs["fillcolor"] = _COLORS["kept"]["fill"]
                attrs["fontcolor"] = _COLORS["kept"]["font"]
            else:
                attrs["fillcolor"] = _COLORS["decomposed"]["fill"]
                attrs["fontcolor"] = _COLORS["decomposed"]["font"]
            attrs["penwidth"] = "2"

        attr_str = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
        lines.append(f"    {nid} [{attr_str}];")

        # Edge from parent
        if parent_idx is not None:
            pid = _node_id(parent_idx)
            lines.append(f"    {pid} -> {nid};")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON dict conversions
# ---------------------------------------------------------------------------

def tree_to_dict(node: DecompNode, *, include_dtensor: bool = False) -> dict:
    """Convert a DecompNode tree to a JSON-serializable dict."""
    cls = node.classification
    d: dict = {
        "op": op_display_name(node.op),
        "schema": str(node.op._schema),
        "decomp_type": cls.decomp_type,
        "count": node.count,
        "inductor_kept": cls.inductor_kept,
        "backends": dict(cls.has_backend),
        "tags": cls.tags,
        "mutable": cls.is_mutable,
        "alias_info": cls.has_alias_info,
        "traceable": node.traceable,
    }
    if node.error:
        d["error"] = node.error
    if include_dtensor:
        d["dtensor_strategy"] = get_dtensor_strategy(node.op)
    if node.children:
        d["children"] = [tree_to_dict(c, include_dtensor=include_dtensor) for c in node.children]
    return d


def leaves_to_dict(node: DecompNode) -> dict:
    """Convert leaf frontier to a JSON-serializable dict."""
    root_name = op_display_name(node.op)

    if not node.children:
        return {"op": root_name, "decomp_type": DecompType.LEAF, "leaves": []}

    lf = collect_leaf_frontier(node)

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


def enrich_leaves_with_dispatch(d: dict, node: DecompNode) -> dict:
    """Add dispatch info to leaves JSON output."""
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


def enrich_tree_with_dispatch(d: dict, node: DecompNode) -> None:
    """Add dispatch info to tree JSON output (in-place)."""
    dinfo = get_dispatch_info_cached(node.op)
    d["autograd_type"] = dinfo.autograd_type
    d["has_adiov"] = dinfo.has_adiov
    d["mode_sensitive"] = dinfo.mode_sensitive
    for child_dict, child_node in zip(d.get("children", []), node.children):
        enrich_tree_with_dispatch(child_dict, child_node)


def add_untraceable_warnings(d: dict, node: DecompNode) -> None:
    """Add a warnings field to a JSON dict if the tree has untraceable ops."""
    errors = collect_untraceable_errors(node)
    if errors:
        d["warnings"] = [
            {"op": name, "message": f"could not trace: {error}"}
            for name, error in errors
        ]
