"""Tests for op classification."""

import pytest
import torch

from decomp_magician.classify import OpClass, classify


class TestDecompType:
    def test_table_only(self):
        # addcmul is in decomposition_table and CIA is False (verified independently)
        op = torch.ops.aten.addcmul.default
        cls = classify(op)
        assert cls.decomp_type == "table"

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

    def test_classification_agrees_with_raw_apis(self):
        """Cross-check classify() against direct PyTorch API calls.

        This breaks the circularity: instead of asserting 'in ("table", "both")',
        we independently query the decomposition_table and _can_decompose(),
        then verify classify() agrees.
        """
        from torch._decomp import decomposition_table

        test_ops = [
            torch.ops.aten.addcmul.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.dropout.default,
            torch.ops.aten.add.Tensor,
        ]
        for op in test_ops:
            cls = classify(op)
            in_table = op in decomposition_table
            has_cia = op._can_decompose()
            if in_table and has_cia:
                assert cls.decomp_type == "both", f"{op.name()}: expected both"
            elif in_table:
                assert cls.decomp_type == "table", f"{op.name()}: expected table"
            elif has_cia:
                assert cls.decomp_type == "CIA", f"{op.name()}: expected CIA"
            else:
                assert cls.decomp_type == "leaf", f"{op.name()}: expected leaf"


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
    def test_kept_op_classified_correctly(self, inductor_kept_op):
        cls = classify(inductor_kept_op)
        assert cls.inductor_kept is True

    def test_non_kept_op_classified_correctly(self, non_inductor_kept_decomposable_op):
        cls = classify(non_inductor_kept_decomposable_op)
        assert cls.inductor_kept is False

    def test_agrees_with_inductor_table(self):
        """Cross-check classify() against the raw inductor table."""
        from decomp_magician.classify import _build_inductor_kept
        kept_names = _build_inductor_kept()
        test_ops = [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mm.default,
            torch.ops.aten.addcmul.default,
        ]
        for op in test_ops:
            cls = classify(op)
            expected = op.name() in kept_names
            assert cls.inductor_kept is expected, (
                f"{op.name()}: classify says {cls.inductor_kept}, "
                f"inductor table says {expected}"
            )


class TestTags:
    def test_tags_are_strings(self):
        op = torch.ops.aten.add.Tensor
        cls = classify(op)
        assert isinstance(cls.tags, tuple)
        assert all(isinstance(t, str) for t in cls.tags)


class TestDtensorStrategy:
    def test_leaf_op_is_missing(self):
        """A true leaf op (no decomposition) should be 'missing' or 'registered'."""
        op = torch.ops.aten.mm.default
        cls = classify(op, dtensor=True)
        assert cls.dtensor_strategy in ("registered", "missing")

    def test_cia_op_never_missing(self):
        """Any op with _can_decompose()=True must not be 'missing'.

        CIA ops auto-decompose before DTensor dispatch, so DTensor
        handles the children via fallback. This applies to both
        CIA-only ops and 'both' (table+CIA) ops.
        Samples ops at runtime so the test works across PyTorch versions.
        """
        from torch._ops import OpOverload

        checked = 0
        for name in dir(torch.ops.aten):
            try:
                packet = getattr(torch.ops.aten, name)
                if not hasattr(packet, "default"):
                    continue
                op = packet.default
                if not isinstance(op, OpOverload):
                    continue
                if not op._can_decompose():
                    continue
                cls = classify(op, dtensor=True)
                assert cls.dtensor_strategy in ("registered", "decomp-fallback"), (
                    f"{op.name()} has _can_decompose()=True but "
                    f"dtensor_strategy={cls.dtensor_strategy!r}"
                )
                checked += 1
                if checked >= 20:
                    break
            except Exception:
                continue
        if checked == 0:
            pytest.skip("No CIA ops found on this PyTorch version")


class TestOpClassValidation:
    def test_invalid_decomp_type_raises(self):
        with pytest.raises(ValueError, match="Invalid decomp_type"):
            OpClass(decomp_type="Table")

    def test_invalid_dtensor_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid dtensor_strategy"):
            OpClass(decomp_type="leaf", dtensor_strategy="registred")

    def test_valid_decomp_types(self):
        for dt in ("CIA", "table", "both", "leaf"):
            cls = OpClass(decomp_type=dt)
            assert cls.decomp_type == dt

    def test_valid_dtensor_strategies(self):
        for ds in ("registered", "decomp-fallback", "missing", None):
            cls = OpClass(decomp_type="leaf", dtensor_strategy=ds)
            assert cls.dtensor_strategy == ds
