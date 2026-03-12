"""Tests for CLI entry point and output formatting."""

from decomp_magician.__main__ import main, format_tree
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
