"""Tests for CLI entry point and output formatting."""

import json

from decomp_magician.__main__ import main, format_tree, tree_to_dict
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
