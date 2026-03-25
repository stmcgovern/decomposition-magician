"""Tests for backward (gradient) tracing."""

import json
from collections import Counter

import torch

from decomp_magician.cli import main
from decomp_magician.tree import trace_backward, op_display_name


class TestTraceBackward:
    def test_differentiable_op(self):
        """A differentiable op should produce backward ops."""
        result = trace_backward(torch.ops.aten.mul.Tensor)
        assert isinstance(result, tuple)
        assert len(result) > 0

    def test_trivial_backward(self):
        """add's backward is identity — zero dispatched ops is correct."""
        result = trace_backward(torch.ops.aten.add.Tensor)
        assert isinstance(result, tuple)
        assert len(result) == 0

    def test_non_differentiable_output(self):
        """An op returning a non-tensor (e.g. int) should return an error string."""
        result = trace_backward(torch.ops.aten.sym_size.int)
        assert isinstance(result, str)

    def test_result_contains_op_overloads(self):
        """Backward trace should return OpOverload instances."""
        result = trace_backward(torch.ops.aten.mul.Tensor)
        assert isinstance(result, tuple)
        for op in result:
            assert isinstance(op, torch._ops.OpOverload)

    def test_addcmul_backward(self):
        """addcmul has a multi-op backward — should produce several ops."""
        result = trace_backward(torch.ops.aten.addcmul.default)
        assert isinstance(result, tuple)
        assert len(result) >= 3  # at least mul, add, etc.

    def test_duplicate_ops_preserved(self):
        """Raw result should preserve duplicates (Counter can aggregate later)."""
        result = trace_backward(torch.ops.aten.addcmul.default)
        assert isinstance(result, tuple)
        counts = Counter(op_display_name(op) for op in result)
        # addcmul backward involves mul multiple times
        assert any(c > 1 for c in counts.values()), f"expected duplicates: {counts}"


class TestBackwardCli:
    def test_backward_text(self, capsys):
        """--backward should show backward ops in text format."""
        assert main(["mul.Tensor", "--backward"]) == 0
        out = capsys.readouterr().out
        assert "backward" in out
        assert "aten." in out  # should list some aten ops

    def test_backward_json(self, capsys):
        """--backward --json should produce valid JSON with backward field."""
        assert main(["mul.Tensor", "--backward", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert "backward" in data
        assert isinstance(data["backward"], list)
        assert len(data["backward"]) > 0
        assert "op" in data["backward"][0]
        assert "count" in data["backward"][0]
        assert "total_instances" in data

    def test_backward_error_text(self, capsys):
        """--backward on a non-differentiable op should return exit code 1."""
        assert main(["sym_size.int", "--backward"]) == 1
        out = capsys.readouterr().out
        assert "error" in out

    def test_backward_error_json(self, capsys):
        """--backward --json on a failing op should include error field."""
        assert main(["sym_size.int", "--backward", "--json"]) == 1
        data = json.loads(capsys.readouterr().out)
        assert data["backward"] is None
        assert "error" in data

    def test_backward_working_op_text(self, capsys):
        """--backward for a differentiable op with known backward ops."""
        assert main(["softmax.int", "--backward"]) == 0
        out = capsys.readouterr().out
        assert "backward" in out
        assert "unique ops" in out
        assert "total instances" in out

    def test_backward_working_op_json(self, capsys):
        """--backward --json for a differentiable op returns structured data."""
        assert main(["softmax.int", "--backward", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert len(data["backward"]) > 0
        assert data["total_instances"] > 0
        # Every backward entry has required fields
        for entry in data["backward"]:
            assert "op" in entry
            assert "count" in entry
            assert isinstance(entry["count"], int)
            assert entry["count"] >= 1
        # total_instances should equal sum of counts
        assert data["total_instances"] == sum(e["count"] for e in data["backward"])
