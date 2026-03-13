"""Tests for diff mode."""

import json

import torch

from decomp_magician.diff import compute_diff
from decomp_magician.__main__ import main


class TestComputeDiff:
    def test_addcmul_has_differences(self):
        """addcmul is inductor-kept, so compile mode stops earlier."""
        op = torch.ops.aten.addcmul.default
        diff = compute_diff(op)
        assert diff.op == "aten.addcmul.default"
        assert diff.left_mode == "full"
        assert diff.right_mode == "compile"
        # In compile mode, addcmul itself becomes a leaf (inductor-kept)
        assert len(diff.added) > 0 or len(diff.removed) > 0 or len(diff.changed) > 0

    def test_leaf_op_no_diff(self):
        """A leaf op should have no differences."""
        op = torch.ops.aten.add.Tensor
        diff = compute_diff(op)
        assert not diff.added
        assert not diff.removed
        assert not diff.changed


class TestDiffCli:
    def test_diff_flag(self, capsys):
        assert main(["addcmul", "--diff", "--no-color"]) == 0
        captured = capsys.readouterr()
        assert "full vs compile" in captured.out

    def test_diff_json(self, capsys):
        assert main(["addcmul", "--diff", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["left_mode"] == "full"
        assert data["right_mode"] == "compile"
        assert "added" in data
        assert "removed" in data
