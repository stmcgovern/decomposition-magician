"""Tests for bulk statistics."""

import json
from collections import Counter

from decomp_magician.stats import DtensorStats, StatsResult, compute_stats
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
        """StatsResult raises ValueError if traceability accounting is wrong."""
        import pytest
        with pytest.raises(ValueError, match="Accounting invariant"):
            StatsResult(
                total=100, total_non_out=50, by_type={"table": 50}, inductor_kept=0,
                traceable=10, untraceable=10, classify_errors=0,
                leaf_ops=Counter(), deepest=[],
            )

    def test_type_accounting_invariant(self):
        """sum(by_type.values()) must equal total_non_out - classify_errors."""
        data = compute_stats()
        assert sum(data.by_type.values()) == data.total_non_out - data.classify_errors

    def test_type_accounting_rejects_bad_data(self):
        """StatsResult raises ValueError if type accounting is wrong."""
        import pytest
        with pytest.raises(ValueError, match="Type accounting invariant"):
            StatsResult(
                total=100, total_non_out=50, by_type={"table": 40}, inductor_kept=0,
                traceable=30, untraceable=20, classify_errors=0,
                leaf_ops=Counter(), deepest=[],
            )


    def test_untraceable_ops_populated(self):
        """untraceable_ops should contain (name, reason) tuples."""
        data = compute_stats()
        assert len(data.untraceable_ops) == data.untraceable
        for name, reason in data.untraceable_ops:
            assert isinstance(name, str)
            assert isinstance(reason, str)
            assert "aten." in name or "prims." in name

    def test_untraceable_ops_list_invariant_rejects_bad_data(self):
        """StatsResult raises ValueError if untraceable_ops count doesn't match."""
        import pytest
        with pytest.raises(ValueError, match="Untraceable ops list invariant"):
            StatsResult(
                total=100, total_non_out=50, by_type={"table": 50}, inductor_kept=0,
                traceable=40, untraceable=10, classify_errors=0,
                leaf_ops=Counter(), deepest=[],
                untraceable_ops=[("aten.foo.default", "error")] * 5,  # 5 != 10
            )

    def test_dtensor_partition_invariant_rejects_bad_data(self):
        """StatsResult raises ValueError if dtensor partition doesn't sum correctly."""
        import pytest
        with pytest.raises(ValueError, match="DTensor partition invariant"):
            StatsResult(
                total=100, total_non_out=50, by_type={"table": 50}, inductor_kept=0,
                traceable=50, untraceable=0, classify_errors=0,
                leaf_ops=Counter(), deepest=[],
                dtensor=DtensorStats(
                    registered=10, decomp_fallback=10, missing=10,  # 30 != 50
                    fully_covered=0, has_gaps=0, top_uncovered=[],
                ),
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
        assert "untraceable_ops" in data
        assert isinstance(data["untraceable_ops"], list)
        if data["untraceable_ops"]:
            assert "op" in data["untraceable_ops"][0]
            assert "error" in data["untraceable_ops"][0]

    def test_stats_text_shows_error_breakdown(self, capsys):
        assert main(["--stats"]) == 0
        out = capsys.readouterr().out
        assert "Untraceable ops by error type" in out

    def test_stats_compile(self, capsys):
        assert main(["--stats", "--compile"]) == 0
        captured = capsys.readouterr()
        assert "compile" in captured.out

    def test_stats_no_op_required(self, capsys):
        """--stats should work without an op argument."""
        assert main(["--stats"]) == 0

    def test_stats_dtensor_text(self, capsys):
        """--stats --dtensor should show DTensor coverage section."""
        assert main(["--stats", "--dtensor"]) == 0
        out = capsys.readouterr().out
        assert "DTensor coverage" in out
        assert "Registered strategy" in out
        assert "Fully covered trees" in out

    def test_stats_dtensor_json(self, capsys):
        """--stats --dtensor --json should include dtensor field."""
        assert main(["--stats", "--dtensor", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert "dtensor" in data
        dt = data["dtensor"]
        assert "registered" in dt
        assert "decomp_fallback" in dt
        assert "fully_covered" in dt
        assert "top_uncovered" in dt

    def test_stats_without_dtensor_no_dtensor_field(self, capsys):
        """--stats without --dtensor should not include dtensor data."""
        assert main(["--stats", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert "dtensor" not in data


class TestDtensorStats:
    def test_dtensor_stats_populated(self):
        data = compute_stats(dtensor=True)
        assert data.dtensor is not None
        assert isinstance(data.dtensor, DtensorStats)

    def test_dtensor_stats_nonzero(self):
        data = compute_stats(dtensor=True)
        dt = data.dtensor
        assert dt.registered > 0
        assert dt.registered + dt.decomp_fallback + dt.missing > 0

    def test_dtensor_coverage_accounting(self):
        """fully_covered + has_gaps should not exceed traceable ops with children."""
        data = compute_stats(dtensor=True)
        dt = data.dtensor
        assert dt.fully_covered + dt.has_gaps <= data.traceable

    def test_no_dtensor_when_not_requested(self):
        data = compute_stats(dtensor=False)
        assert data.dtensor is None
