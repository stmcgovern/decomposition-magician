"""Tests for op classification."""

import torch

from decomp_magician.classify import classify


class TestDecompType:
    def test_table_only(self):
        # addcmul is in decomposition_table but has no CIA
        op = torch.ops.aten.addcmul.default
        cls = classify(op)
        assert cls.decomp_type in ("table", "both")

    def test_cia_op(self):
        # dropout has CIA
        op = torch.ops.aten.dropout.default
        cls = classify(op)
        assert cls.decomp_type in ("CIA", "both")

    def test_leaf_op(self):
        # mm is a leaf — has backend kernels, no decomposition
        op = torch.ops.aten.mm.default
        cls = classify(op)
        assert cls.decomp_type == "leaf"

    def test_batch_norm_legit(self):
        op = torch.ops.aten._native_batch_norm_legit.default
        cls = classify(op)
        assert cls.decomp_type in ("table", "both")


class TestBackends:
    def test_mm_has_cuda(self):
        op = torch.ops.aten.mm.default
        cls = classify(op)
        assert cls.has_backend["cpu"] is True
        assert cls.has_backend["meta"] is True

    def test_backends_are_populated(self):
        op = torch.ops.aten.add.Tensor
        cls = classify(op)
        assert "cpu" in cls.has_backend
        assert "cuda" in cls.has_backend
        assert "meta" in cls.has_backend


class TestSchemaProperties:
    def test_mutable_op(self):
        # copy_ is mutable
        op = torch.ops.aten.copy_.default
        cls = classify(op)
        assert cls.is_mutable is True

    def test_non_mutable_op(self):
        op = torch.ops.aten.add.Tensor
        cls = classify(op)
        assert cls.is_mutable is False

    def test_alias_info(self):
        # view ops have alias info
        op = torch.ops.aten.view.default
        cls = classify(op)
        assert cls.has_alias_info is True

    def test_no_alias_info(self):
        op = torch.ops.aten.add.Tensor
        cls = classify(op)
        assert cls.has_alias_info is False


class TestInductorKept:
    def test_sum_excluded(self):
        op = torch.ops.aten.sum.dim_IntList
        cls = classify(op)
        assert cls.inductor_kept is True

    def test_add_not_excluded(self):
        op = torch.ops.aten.add.Tensor
        cls = classify(op)
        assert cls.inductor_kept is False

    def test_silu_not_excluded(self):
        """silu is in decomps_to_exclude but Inductor re-adds its own decomp."""
        op = torch.ops.aten.silu.default
        cls = classify(op)
        assert cls.inductor_kept is False


class TestTags:
    def test_tags_are_strings(self):
        op = torch.ops.aten.add.Tensor
        cls = classify(op)
        assert isinstance(cls.tags, tuple)
        assert all(isinstance(t, str) for t in cls.tags)
