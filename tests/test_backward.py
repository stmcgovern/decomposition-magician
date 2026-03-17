"""Tests for backward (gradient) tracing."""

import json

import torch

from decomp_magician.__main__ import main
from decomp_magician.tree import trace_backward


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
