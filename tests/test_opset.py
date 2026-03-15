"""Tests for opset coverage checking."""

import json

import pytest
import torch

from decomp_magician.opset import OPSETS, OpsetCoverage, check_opset_coverage, is_core_aten
from decomp_magician.__main__ import main


class TestOpsetCoverageProperties:
    """Test the derived properties of OpsetCoverage with synthetic data.

    These verify the exact logic of fully_covered and total_leaves
    without depending on any PyTorch decomposition state.
    """

    def test_fully_covered_when_no_non_covered(self):
        cov = OpsetCoverage(op="x", opset="test", covered_leaves=5, non_covered=())
        assert cov.fully_covered is True
        assert cov.total_leaves == 5

    def test_not_fully_covered_when_non_covered(self):
        cov = OpsetCoverage(op="x", opset="test", covered_leaves=3,
                            non_covered=(("bad_op", 2),))
        assert cov.fully_covered is False
        assert cov.total_leaves == 5

    def test_total_leaves_sums_correctly(self):
        cov = OpsetCoverage(op="x", opset="test", covered_leaves=10,
                            non_covered=(("a", 3), ("b", 7)))
        assert cov.total_leaves == 20

    def test_zero_covered_all_non_covered(self):
        cov = OpsetCoverage(op="x", opset="test", covered_leaves=0,
                            non_covered=(("a", 1), ("b", 2)))
        assert cov.fully_covered is False
        assert cov.total_leaves == 3

    def test_post_init_rejects_negative_covered(self):
        with pytest.raises(ValueError):
            OpsetCoverage(op="x", opset="test", covered_leaves=-1, non_covered=())


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
        op = torch.ops.aten.addcmul.default
        with pytest.raises(ValueError, match="Unknown opset"):
            check_opset_coverage(op, opset="bogus")

    def test_opsets_constant(self):
        assert "core_aten" in OPSETS


class TestCheckOpsetNotCovered:
    def test_roll_not_fully_covered(self):
        """roll decomposes to itself (cycle), so it has non-core leaves."""
        op = torch.ops.aten.roll.default
        cov = check_opset_coverage(op)
        # roll contains itself as untraceable leaf — not a core op
        assert not cov.fully_covered
        assert cov.non_covered != ()
        assert cov.total_leaves > 0
        assert cov.covered_leaves < cov.total_leaves

    def test_non_covered_invariant(self):
        """covered + non_covered == total for any op."""
        op = torch.ops.aten.roll.default
        cov = check_opset_coverage(op)
        nc_total = sum(c for _, c in cov.non_covered)
        assert cov.covered_leaves + nc_total == cov.total_leaves


class TestOpsetCli:
    def test_target_opset_flag(self, capsys):
        assert main(["addcmul", "--target-opset", "core_aten", "--no-color"]) == 0
        captured = capsys.readouterr()
        assert "core_aten" in captured.out
        assert "FULLY COVERED" in captured.out

    def test_target_opset_not_covered(self, capsys):
        assert main(["roll", "--target-opset", "core_aten", "--no-color"]) == 0
        captured = capsys.readouterr()
        assert "NOT FULLY COVERED" in captured.out

    def test_target_opset_json(self, capsys):
        assert main(["addcmul", "--target-opset", "core_aten", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["fully_covered"] is True
        assert data["opset"] == "core_aten"

    def test_target_opset_json_not_covered(self, capsys):
        assert main(["roll", "--target-opset", "core_aten", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["fully_covered"] is False
        assert len(data["non_covered"]) > 0

    def test_target_opset_unknown(self, capsys):
        assert main(["addcmul", "--target-opset", "bogus"]) == 1

    def test_stats_target_opset(self, capsys):
        """--stats --target-opset should show leaf op coverage."""
        assert main(["--stats", "--target-opset", "core_aten", "--no-color"]) == 0
        captured = capsys.readouterr()
        assert "Leaf op coverage" in captured.out
        assert "core_aten" in captured.out

    def test_stats_target_opset_json(self, capsys):
        assert main(["--stats", "--target-opset", "core_aten", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["opset"] == "core_aten"
        assert "leaf_ops_in_opset" in data

    def test_leaves_target_opset(self, capsys):
        """--leaves --target-opset should annotate each leaf."""
        assert main(["addcmul", "--leaves", "--target-opset", "core_aten", "--no-color"]) == 0
        captured = capsys.readouterr()
        assert "core_aten" in captured.out

    def test_leaves_target_opset_json(self, capsys):
        assert main(["addcmul", "--leaves", "--target-opset", "core_aten", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["opset"] == "core_aten"
        assert all("in_opset" in leaf for leaf in data["leaves"])
