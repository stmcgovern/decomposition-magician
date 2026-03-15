"""Tests for bulk statistics."""

import json
from collections import Counter

from decomp_magician.stats import StatsResult, compute_stats
from decomp_magician.__main__ import main


class TestComputeStats:
    def test_returns_stats_result(self):
        data = compute_stats()
        assert isinstance(data, StatsResult)
        assert data.total > 0
        assert data.traceable >= 0
        assert data.leaf_ops is not None
        assert data.deepest is not None

    def test_total_gt_zero(self):
        data = compute_stats()
        assert data.total > 1000

    def test_traceable_gt_zero(self):
        data = compute_stats()
        assert data.traceable > 0

    def test_leaf_ops_has_entries(self):
        data = compute_stats()
        assert len(data.leaf_ops) > 0
        # At least one prims or aten op should appear as a leaf
        assert any(
            name.startswith(("aten.", "prims.")) for name in data.leaf_ops
        )

    def test_deepest_sorted(self):
        data = compute_stats()
        depths = [d for _, d in data.deepest]
        assert depths == sorted(depths, reverse=True)

    def test_compile_mode_different(self):
        full = compute_stats()
        compiled = compute_stats(compile=True)
        # Compile mode treats inductor-kept ops as leaves, so fewer untraceable
        # failures cascade from deep decompositions
        assert compiled.untraceable <= full.untraceable

    def test_accounting_invariant(self):
        """traceable + untraceable + classify_errors == total_non_out."""
        data = compute_stats()
        # The invariant is enforced by StatsResult.__post_init__,
        # so construction itself would have raised if violated.
        # Verify it explicitly here too.
        assert data.traceable + data.untraceable + data.classify_errors == data.total_non_out

    def test_accounting_invariant_rejects_bad_data(self):
        """StatsResult raises ValueError if invariant is violated."""
        import pytest
        with pytest.raises(ValueError, match="Accounting invariant"):
            StatsResult(
                total=100, total_non_out=50, by_type={}, inductor_kept=0,
                traceable=10, untraceable=10, classify_errors=0,
                leaf_ops=Counter(), deepest=[],
            )


class TestStatsCli:
    def test_stats_flag(self, capsys):
        assert main(["--stats"]) == 0
        captured = capsys.readouterr()
        assert "Decomposition table statistics" in captured.out
        assert "Top leaf ops" in captured.out

    def test_stats_json(self, capsys):
        assert main(["--stats", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "total" in data
        assert "top_leaf_ops" in data
        assert "classify_errors" in data

    def test_stats_compile(self, capsys):
        assert main(["--stats", "--compile"]) == 0
        captured = capsys.readouterr()
        assert "compile" in captured.out

    def test_stats_no_op_required(self, capsys):
        """--stats should work without an op argument."""
        assert main(["--stats"]) == 0
