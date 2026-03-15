"""Tests for opset coverage checking."""

import json

import torch

from decomp_magician.opset import OPSETS, check_opset_coverage, is_core_aten
from decomp_magician.__main__ import main


class TestIsCorAten:
    def test_expand_is_core(self):
        assert is_core_aten("aten.expand.default")

    def test_add_tensor_is_core(self):
        assert is_core_aten("aten.add.Tensor")

    def test_addcmul_is_not_core(self):
        assert not is_core_aten("aten.addcmul.default")


class TestCheckOpsetCoverage:
    def test_addcmul_fully_covered(self):
        op = torch.ops.aten.addcmul.default
        cov = check_opset_coverage(op)
        assert cov.fully_covered
        assert cov.total_leaves > 0
        assert cov.covered_leaves == cov.total_leaves
        assert cov.non_covered == ()

    def test_opset_field(self):
        op = torch.ops.aten.addcmul.default
        cov = check_opset_coverage(op, opset="core_aten")
        assert cov.opset == "core_aten"

    def test_unknown_opset_raises(self):
        import pytest
        op = torch.ops.aten.addcmul.default
        with pytest.raises(ValueError, match="Unknown opset"):
            check_opset_coverage(op, opset="bogus")

    def test_opsets_constant(self):
        assert "core_aten" in OPSETS


class TestOpsetCli:
    def test_target_opset_flag(self, capsys):
        assert main(["addcmul", "--target-opset", "core_aten", "--no-color"]) == 0
        captured = capsys.readouterr()
        assert "core_aten" in captured.out
        assert "FULLY COVERED" in captured.out

    def test_target_opset_json(self, capsys):
        assert main(["addcmul", "--target-opset", "core_aten", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["fully_covered"] is True
        assert data["opset"] == "core_aten"

    def test_target_opset_unknown(self, capsys):
        assert main(["addcmul", "--target-opset", "bogus"]) == 1
