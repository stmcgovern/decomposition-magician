"""Tests for bulk statistics."""

import json
from collections import Counter

import pytest

from decomp_magician.stats import DtensorStats, StatsResult, compute_stats
from decomp_magician.cli import main


@pytest.fixture(scope="module")
def stats_full():
    """Compute full stats once for the module — expensive (~3s)."""
    return compute_stats()


@pytest.fixture(scope="module")
def stats_compile():
    """Compute compile-mode stats once for the module."""
    return compute_stats(compile=True)


class TestComputeStats:
    def test_returns_stats_result(self, stats_full):
        assert isinstance(stats_full, StatsResult)
        assert stats_full.total > 0
        assert stats_full.traceable >= 0
        assert stats_full.leaf_ops is not None
        assert stats_full.deepest is not None

    def test_total_gt_zero(self, stats_full):
        assert stats_full.total > 1000

    def test_traceable_gt_zero(self, stats_full):
        assert stats_full.traceable > 0

    def test_leaf_ops_has_entries(self, stats_full):
        assert len(stats_full.leaf_ops) > 0
        assert any(
            name.startswith(("aten.", "prims.")) for name in stats_full.leaf_ops
        )

    def test_deepest_sorted(self, stats_full):
        depths = [d for _, d in stats_full.deepest]
        assert depths == sorted(depths, reverse=True)

    def test_compile_mode_different(self, stats_full, stats_compile):
        assert stats_compile.untraceable <= stats_full.untraceable

    def test_accounting_invariant(self, stats_full):
        """traceable + untraceable + classify_errors == total_non_out."""
        data = stats_full
        assert data.traceable + data.untraceable + data.classify_errors == data.total_non_out

    def test_type_accounting_invariant(self, stats_full):
        """sum(by_type.values()) must equal total_non_out - classify_errors."""
        assert sum(stats_full.by_type.values()) == stats_full.total_non_out - stats_full.classify_errors

    def test_untraceable_ops_populated(self, stats_full):
        """untraceable_ops should contain (name, reason) tuples."""
        assert len(stats_full.untraceable_ops) == stats_full.untraceable
        for name, reason in stats_full.untraceable_ops:
            assert isinstance(name, str)
            assert isinstance(reason, str)
            assert "aten." in name or "prims." in name


class TestStatsResultInvariants:
    """Test invariant rejection with synthetic data — no PyTorch tracing needed."""

    def test_accounting_invariant_rejects_bad_data(self):
        """StatsResult raises ValueError if traceability accounting is wrong."""
        with pytest.raises(ValueError, match="Accounting invariant"):
            StatsResult(
                total=100, total_non_out=50, by_type={"table": 50}, inductor_kept=0,
                traceable=10, untraceable=10, classify_errors=0,
                leaf_ops=Counter(), deepest=[],
            )

    def test_type_accounting_rejects_bad_data(self):
        """StatsResult raises ValueError if type accounting is wrong."""
        with pytest.raises(ValueError, match="Type accounting invariant"):
            StatsResult(
                total=100, total_non_out=50, by_type={"table": 40}, inductor_kept=0,
                traceable=30, untraceable=20, classify_errors=0,
                leaf_ops=Counter(), deepest=[],
            )

    def test_untraceable_ops_list_invariant_rejects_bad_data(self):
        """StatsResult raises ValueError if untraceable_ops count doesn't match."""
        with pytest.raises(ValueError, match="Untraceable ops list invariant"):
            StatsResult(
                total=100, total_non_out=50, by_type={"table": 50}, inductor_kept=0,
                traceable=40, untraceable=10, classify_errors=0,
                leaf_ops=Counter(), deepest=[],
                untraceable_ops=(("aten.foo.default", "error"),) * 5,  # 5 != 10
            )

    def test_dtensor_partition_invariant_rejects_bad_data(self):
        """StatsResult raises ValueError if dtensor partition doesn't sum correctly."""
        with pytest.raises(ValueError, match="DTensor partition invariant"):
            StatsResult(
                total=100, total_non_out=50, by_type={"table": 50}, inductor_kept=0,
                traceable=50, untraceable=0, classify_errors=0,
                leaf_ops=Counter(), deepest=[],
                dtensor=DtensorStats(
                    registered=10, decomp_fallback=10, missing=10, not_applicable=0,  # 30 != 50
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


@pytest.fixture(scope="module")
def stats_dtensor():
    """Compute dtensor stats once for the module."""
    return compute_stats(dtensor=True)


class TestDtensorStats:
    def test_dtensor_stats_populated(self, stats_dtensor):
        assert stats_dtensor.dtensor is not None
        assert isinstance(stats_dtensor.dtensor, DtensorStats)

    def test_dtensor_stats_nonzero(self, stats_dtensor):
        dt = stats_dtensor.dtensor
        assert dt.registered > 0
        assert dt.registered + dt.decomp_fallback + dt.missing > 0

    def test_dtensor_coverage_accounting(self, stats_dtensor):
        """fully_covered + has_gaps should not exceed traceable ops with children."""
        dt = stats_dtensor.dtensor
        assert dt.fully_covered + dt.has_gaps <= stats_dtensor.traceable

    def test_no_dtensor_when_not_requested(self, stats_full):
        assert stats_full.dtensor is None
