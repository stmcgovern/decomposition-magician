"""Graph export: Mermaid and Graphviz DOT formats."""

from __future__ import annotations

from decomp_magician.tree import DecompNode


def op_display_name(op) -> str:
    """Short display name: aten.add.Tensor, always showing overload."""
    name = op.name()
    dotted = name.replace("::", ".")
    if dotted.count(".") < 2:
        dotted += ".default"
    return dotted


def _node_id(index: int) -> str:
    return f"n{index}"


def _is_terminal(n: DecompNode) -> bool:
    """A node is terminal if it has no children."""
    return len(n.children) == 0


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
