"""Tests for CLI entry point and output formatting."""

import json

from decomp_magician.__main__ import main, format_tree, format_summary, tree_to_dict
from decomp_magician.tree import build_tree

import torch


class TestMain:
    def test_basic_op(self):
        assert main(["addcmul"]) == 0

    def test_leaf_op(self):
        assert main(["mm"]) == 0

    def test_depth_limit(self):
        assert main(["_native_batch_norm_legit", "--depth", "1"]) == 0

    def test_verbose(self):
        assert main(["addcmul", "--verbose", "--depth", "0"]) == 0

    def test_nonexistent_op(self, capsys):
        assert main(["zzz_nonexistent_zzz"]) == 1
        captured = capsys.readouterr()
        assert "No ops found" in captured.err

    def test_fully_qualified(self):
        assert main(["aten.addcmul.default"]) == 0


class TestFormatTree:
    def test_leaf_format(self):
        op = torch.ops.aten.mm.default
        node = build_tree(op)
        output = format_tree(node)
        assert "aten.mm" in output
        assert "[leaf]" in output

    def test_tree_has_connectors(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op, depth=1)
        output = format_tree(node)
        assert "├──" in output or "└──" in output

    def test_count_shown(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op, depth=1)
        output = format_tree(node)
        assert "x2" in output  # mul.Tensor appears twice

    def test_batch_norm_squeeze_visible(self):
        op = torch.ops.aten._native_batch_norm_legit.default
        node = build_tree(op, depth=1)
        output = format_tree(node)
        assert "squeeze.dims" in output
        assert "x3" in output

    def test_inductor_kept_annotation(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op, depth=0)
        output = format_tree(node)
        assert "inductor-kept" in output


class TestSummary:
    def test_leaf_summary(self):
        node = build_tree(torch.ops.aten.mm.default)
        s = format_summary(node)
        assert "1 op" in s
        assert "1 leaf" in s

    def test_batch_norm_summary(self):
        node = build_tree(torch.ops.aten._native_batch_norm_legit.default, depth=1)
        s = format_summary(node)
        assert "9 ops" in s
        assert "inductor-kept" in s

    def test_summary_in_output(self, capsys):
        main(["addcmul", "--depth", "0"])
        captured = capsys.readouterr()
        assert "1 op" in captured.out


class TestColor:
    def test_no_color_in_pipe(self, capsys):
        """Non-tty output should have no ANSI codes."""
        main(["addcmul", "--depth", "1"])
        captured = capsys.readouterr()
        assert "\033[" not in captured.out

    def test_no_color_flag(self, capsys):
        """--no-color should suppress ANSI codes even if tty."""
        main(["addcmul", "--depth", "1", "--no-color"])
        captured = capsys.readouterr()
        assert "\033[" not in captured.out

    def test_color_rendering(self):
        """When color is on, output should contain ANSI codes."""
        import decomp_magician.__main__ as m
        old = m._use_color
        try:
            m._use_color = True
            node = build_tree(torch.ops.aten.addcmul.default, depth=1)
            output = format_tree(node)
            assert "\033[" in output  # has ANSI codes
            assert "\033[1m" in output  # bold for decomposable ops
            assert "\033[33m" in output  # yellow for inductor-kept
        finally:
            m._use_color = old

    def test_color_leaf_dim(self):
        """Leaf ops should be dim when color is on."""
        import decomp_magician.__main__ as m
        old = m._use_color
        try:
            m._use_color = True
            node = build_tree(torch.ops.aten.mm.default)
            output = format_tree(node)
            assert "\033[2m" in output  # dim for leaf
        finally:
            m._use_color = old


class TestJson:
    def test_json_output(self, capsys):
        assert main(["addcmul", "--depth", "1", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["op"] == "aten.addcmul.default"
        assert data["decomp_type"] == "table"
        assert "children" in data

    def test_json_valid(self, capsys):
        assert main(["_native_batch_norm_legit", "--depth", "1", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["children"]) > 0

    def test_json_leaf_no_children(self, capsys):
        assert main(["mm", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["decomp_type"] == "leaf"
        assert "children" not in data

    def test_tree_to_dict_fields(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op, depth=0)
        d = tree_to_dict(node)
        assert "op" in d
        assert "decomp_type" in d
        assert "inductor_kept" in d
        assert "backends" in d
        assert "tags" in d
        assert "traceable" in d

    def test_tree_to_dict_count(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op, depth=1)
        d = tree_to_dict(node)
        mul_children = [c for c in d["children"] if "mul" in c["op"]]
        assert mul_children[0]["count"] == 2
