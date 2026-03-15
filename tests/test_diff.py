"""Tests for diff mode."""

import json
from collections import Counter

import torch

from decomp_magician.diff import DecompDiff, compute_diff, compute_diff_ops
from decomp_magician.__main__ import main


class TestDecompDiffProperties:
    """Test the derived properties of DecompDiff with synthetic data.

    The set-arithmetic in added/removed/changed is the logical core of
    the diff feature. These tests construct known Counters and verify
    exact outputs — no PyTorch dependency, no circularity.
    """

    def test_added_is_right_only(self):
        d = DecompDiff(
            left_label="L", right_label="R",
            left_leaves=Counter({"a": 2, "b": 1}),
            right_leaves=Counter({"a": 2, "c": 3}),
        )
        assert dict(d.added) == {"c": 3}

    def test_removed_is_left_only(self):
        d = DecompDiff(
            left_label="L", right_label="R",
            left_leaves=Counter({"a": 2, "b": 1}),
            right_leaves=Counter({"a": 2, "c": 3}),
        )
        assert dict(d.removed) == {"b": 1}

    def test_changed_is_shared_with_different_counts(self):
        d = DecompDiff(
            left_label="L", right_label="R",
            left_leaves=Counter({"a": 2, "b": 5}),
            right_leaves=Counter({"a": 7, "b": 5}),
        )
        assert d.changed == [("a", 2, 7)]

    def test_same_counters_produce_no_diff(self):
        c = Counter({"x": 3, "y": 1})
        d = DecompDiff(left_label="L", right_label="R",
                       left_leaves=c, right_leaves=c.copy())
        assert not d.added
        assert not d.removed
        assert not d.changed

    def test_empty_counters(self):
        d = DecompDiff(left_label="L", right_label="R",
                       left_leaves=Counter(), right_leaves=Counter())
        assert not d.added
        assert not d.removed
        assert not d.changed

    def test_disjoint_counters(self):
        d = DecompDiff(
            left_label="L", right_label="R",
            left_leaves=Counter({"a": 1, "b": 2}),
            right_leaves=Counter({"c": 3, "d": 4}),
        )
        assert dict(d.removed) == {"a": 1, "b": 2}
        assert dict(d.added) == {"c": 3, "d": 4}
        assert d.changed == []

    def test_changed_excludes_equal_counts(self):
        """Ops present in both with SAME count must NOT appear in changed."""
        d = DecompDiff(
            left_label="L", right_label="R",
            left_leaves=Counter({"a": 5, "b": 3}),
            right_leaves=Counter({"a": 5, "b": 7}),
        )
        assert d.changed == [("b", 3, 7)]  # a excluded because counts equal


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
