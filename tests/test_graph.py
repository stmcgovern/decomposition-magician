"""Tests for graph export formats (Mermaid and DOT)."""

import torch

from decomp_magician.export import format_mermaid, format_dot, _COLORS
from decomp_magician.tree import build_tree
from decomp_magician.cli import main


class TestMermaid:
    def test_basic_structure(self):
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        output = format_mermaid(node)
        assert output.startswith("graph TD")
        assert "aten.addcmul.default" in output

    def test_edges(self):
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        output = format_mermaid(node)
        assert "-->" in output

    def test_count_in_node_label(self):
        """Counts appear in node labels, not edge labels."""
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        output = format_mermaid(node)
        assert "×2" in output  # mul.Tensor appears x2

    def test_leaf_style(self):
        node = build_tree(torch.ops.aten.mm.default)
        output = format_mermaid(node)
        assert "n0[aten.mm.default]" in output
        assert "classDef leaf" in output

    def test_inductor_kept_class(self, inductor_kept_op):
        node = build_tree(inductor_kept_op, depth=0)
        output = format_mermaid(node)
        assert "classDef kept" in output
        assert "class n0 kept" in output

    def test_cycle_detection(self):
        node = build_tree(torch.ops.aten.roll.default)
        output = format_mermaid(node)
        assert "cycle" in output

    def test_untraceable_shape(self):
        """Untraceable nodes use trapezoid shape."""
        node = build_tree(torch.ops.aten.roll.default)
        output = format_mermaid(node)
        assert "[/" in output
        assert "\\]" in output

    def test_batch_norm_count_in_label(self):
        node = build_tree(torch.ops.aten._native_batch_norm_legit.default, depth=1)
        output = format_mermaid(node)
        assert "squeeze.dims" in output
        assert "×3" in output

    def test_mermaid_uses_br_not_backslash_n(self):
        """Mermaid line breaks must use <br/>, not \\n."""
        node = build_tree(torch.ops.aten._native_batch_norm_legit.default,
                          depth=1, compile=True)
        output = format_mermaid(node)
        # Annotated nodes should use <br/> for line breaks
        assert "\\n" not in output
        if "inductor-kept" in output:
            assert "<br/>" in output

    def test_quoted_labels(self):
        """Labels with special chars must be quoted."""
        node = build_tree(torch.ops.aten._native_batch_norm_legit.default,
                          depth=1, compile=True)
        output = format_mermaid(node)
        # Nodes with <br/> need quotes
        for line in output.split("\n"):
            if "<br/>" in line:
                assert '"' in line

    def test_compile_inductor_kept_terminal(self):
        """In compile mode, inductor-kept ops are terminal with annotation."""
        node = build_tree(torch.ops.aten._native_batch_norm_legit.default,
                          depth=1, compile=True)
        output = format_mermaid(node)
        assert "inductor-kept" in output

    def test_legend_comment(self):
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        output = format_mermaid(node)
        assert "%% Legend:" in output

    def test_decomposed_style(self):
        """Non-terminal decomposed nodes get green styling."""
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        output = format_mermaid(node)
        assert "classDef decomposed" in output


class TestDot:
    def test_basic_structure(self):
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        output = format_dot(node)
        assert output.startswith("digraph decomp {")
        assert output.endswith("}")

    def test_node_labels(self):
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        output = format_dot(node)
        assert "aten.addcmul.default" in output
        assert "aten.mul.Tensor" in output

    def test_edges(self):
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        output = format_dot(node)
        assert "n0 -> n1" in output

    def test_count_in_dot_label(self):
        """Counts appear in DOT node labels."""
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        output = format_dot(node)
        assert "×2" in output

    def test_leaf_style(self):
        node = build_tree(torch.ops.aten.mm.default)
        output = format_dot(node)
        assert 'fillcolor="#f5f5f5"' in output

    def test_inductor_kept_style(self, inductor_kept_op):
        node = build_tree(inductor_kept_op, depth=0)
        output = format_dot(node)
        assert 'fillcolor="#fff3cd"' in output

    def test_cycle_dashed(self):
        node = build_tree(torch.ops.aten.roll.default)
        output = format_dot(node)
        assert "dashed" in output

    def test_compile_inductor_kept_terminal_dot(self):
        node = build_tree(torch.ops.aten._native_batch_norm_legit.default,
                          depth=1, compile=True)
        output = format_dot(node)
        assert "inductor-kept" in output
        assert _COLORS["kept"]["fill"] in output


class TestCliFlags:
    def test_mermaid_flag(self, capsys):
        assert main(["addcmul", "--depth", "1", "--mermaid"]) == 0
        captured = capsys.readouterr()
        assert "graph TD" in captured.out

    def test_dot_flag(self, capsys):
        assert main(["addcmul", "--depth", "1", "--dot"]) == 0
        captured = capsys.readouterr()
        assert "digraph decomp" in captured.out

    def test_mermaid_no_resolve_note(self, capsys):
        """Graph output should not print resolve notes."""
        assert main(["softmax", "--depth", "1", "--mermaid"]) == 0
        captured = capsys.readouterr()
        assert "resolved to" not in captured.err

    def test_compile_mermaid(self, capsys):
        assert main(["_native_batch_norm_legit", "--compile", "--depth", "1", "--mermaid"]) == 0
        captured = capsys.readouterr()
        assert "graph TD" in captured.out

    def test_mermaid_usage_hint(self, capsys):
        """Mermaid output should include usage hint on stderr."""
        main(["addcmul", "--depth", "1", "--mermaid"])
        captured = capsys.readouterr()
        assert "mermaid" in captured.err.lower()
        assert "github" in captured.err.lower()

    def test_dot_usage_hint(self, capsys):
        """DOT output should include usage hint on stderr."""
        main(["addcmul", "--depth", "1", "--dot"])
        captured = capsys.readouterr()
        assert "dot -T" in captured.err
