"""Tests for reverse lookup."""

import json

from decomp_magician.reverse import reverse_lookup
from decomp_magician.__main__ import main


class TestReverseLookup:
    def test_prims_mul_has_results(self):
        results = reverse_lookup("prims.mul.default", depth=1)
        assert len(results) > 0
        # addcmul decomposes into mul.Tensor which decomposes into prims.mul
        # but at depth=1, we only see direct children
        ops = [r["op"] for r in results]
        assert any("mul" in op for op in ops)

    def test_squeeze_dims_found(self):
        results = reverse_lookup("aten.squeeze.dims", depth=1)
        ops = [r["op"] for r in results]
        assert any("batch_norm" in op for op in ops)

    def test_sorted_by_count(self):
        results = reverse_lookup("prims.mul.default", depth=1)
        counts = [r["count"] for r in results]
        assert counts == sorted(counts, reverse=True)

    def test_excludes_self(self):
        """The target op itself should not appear in results."""
        results = reverse_lookup("prims.mul.default")
        ops = [r["op"] for r in results]
        assert "prims.mul.default" not in ops

    def test_nonexistent_target(self):
        results = reverse_lookup("zzz.nonexistent.default")
        assert results == []

    def test_depth_affects_results(self):
        full = reverse_lookup("prims.mul.default")
        shallow = reverse_lookup("prims.mul.default", depth=1)
        # Full decomposition finds more producers than depth=1
        assert len(full) >= len(shallow)

    def test_result_has_target_depth(self):
        results = reverse_lookup("prims.mul.default", depth=1)
        assert all("target_depth" in r for r in results)
        assert all(r["target_depth"] >= 1 for r in results)

    def test_target_depth_is_shallowest(self):
        """target_depth should be the shallowest level at which target appears."""
        # mul.Tensor decomposes directly into prims.mul at depth 1
        results = reverse_lookup("prims.mul.default", depth=1)
        mul_results = [r for r in results if "mul.Tensor" in r["op"]]
        if mul_results:
            assert mul_results[0]["target_depth"] == 1

    def test_scans_table_ops(self):
        """Reverse lookup scans decomposition table ops."""
        # mul.Tensor is in the table and decomposes into prims.mul
        results = reverse_lookup("prims.mul.default", depth=1)
        ops = [r["op"] for r in results]
        assert any("mul.Tensor" in op for op in ops)

    def test_excludes_out_variants_by_default(self):
        """_out variants are filtered by default."""
        results = reverse_lookup("prims.mul.default")
        ops = [r["op"] for r in results]
        assert not any(op.endswith(".out") for op in ops)

    def test_include_out_flag(self):
        """include_out=True brings back _out variants."""
        without = reverse_lookup("prims.mul.default")
        with_out = reverse_lookup("prims.mul.default", include_out=True)
        assert len(with_out) > len(without)


class TestReverseCliFlag:
    def test_reverse_flag(self, capsys):
        assert main(["prims.mul", "--reverse", "--depth", "1"]) == 0
        captured = capsys.readouterr()
        assert "ops" in captured.out
        assert "decompose into" in captured.out

    def test_reverse_no_results(self, capsys):
        # mm is a true primitive — nothing decomposes into it
        assert main(["mm", "--reverse", "--depth", "1"]) == 0
        captured = capsys.readouterr()
        assert "No ops" in captured.out

    def test_reverse_json(self, capsys):
        assert main(["aten.squeeze.dims", "--reverse", "--depth", "1", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "target" in data
        assert "producers" in data
        assert len(data["producers"]) > 0

    def test_reverse_scanning_message(self, capsys):
        """Scanning message should appear on stderr."""
        main(["prims.mul", "--reverse", "--depth", "1"])
        captured = capsys.readouterr()
        assert "Scanning" in captured.err

    def test_reverse_overload_hint(self, capsys):
        """When no results, suggest alternative overloads."""
        main(["mul", "--reverse"])
        captured = capsys.readouterr()
        assert "No ops" in captured.out
        assert "aten.mul.Tensor" in captured.err
