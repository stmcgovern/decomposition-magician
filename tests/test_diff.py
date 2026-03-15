"""Tests for diff mode."""

import json

import torch

from decomp_magician.diff import compute_diff, compute_diff_ops
from decomp_magician.__main__ import main


class TestComputeDiff:
    def test_addcmul_has_differences(self):
        """addcmul is inductor-kept, so compile mode stops earlier."""
        op = torch.ops.aten.addcmul.default
        diff = compute_diff(op)
        assert "aten.addcmul.default" in diff.left_label
        assert "full" in diff.left_label
        assert "compile" in diff.right_label
        # In compile mode, addcmul itself becomes a leaf (inductor-kept)
        assert len(diff.added) > 0 or len(diff.removed) > 0 or len(diff.changed) > 0

    def test_leaf_op_no_diff(self):
        """A leaf op should have no differences."""
        op = torch.ops.aten.add.Tensor
        diff = compute_diff(op)
        assert not diff.added
        assert not diff.removed
        assert not diff.changed


class TestComputeDiffOps:
    def test_same_op_no_diff(self):
        op = torch.ops.aten.addcmul.default
        diff = compute_diff_ops(op, op)
        assert not diff.added
        assert not diff.removed
        assert not diff.changed

    def test_different_ops(self):
        left = torch.ops.aten.addcmul.default
        right = torch.ops.aten.softmax.int
        diff = compute_diff_ops(left, right)
        assert "aten.addcmul.default" in diff.left_label
        assert "aten.softmax.int" in diff.right_label
        # Different ops have different decompositions
        assert diff.added or diff.removed or diff.changed


class TestDiffCli:
    def test_diff_flag(self, capsys):
        assert main(["addcmul", "--diff", "--no-color"]) == 0
        captured = capsys.readouterr()
        assert "full" in captured.out
        assert "compile" in captured.out

    def test_diff_json(self, capsys):
        assert main(["addcmul", "--diff", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "left" in data
        assert "right" in data
        assert "added" in data
        assert "removed" in data

    def test_diff_two_ops(self, capsys):
        assert main(["addcmul", "--diff", "aten.softmax.int", "--no-color"]) == 0
        captured = capsys.readouterr()
        assert "addcmul" in captured.out
        assert "softmax" in captured.out

    def test_diff_two_ops_json(self, capsys):
        assert main(["addcmul", "--diff", "aten.softmax.int", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "addcmul" in data["left"]
        assert "softmax" in data["right"]
