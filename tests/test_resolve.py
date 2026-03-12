"""Tests for op name resolution."""

import torch
from torch._ops import OpOverload

from decomp_magician.resolve import resolve_op


class TestExactMatch:
    def test_fully_qualified(self):
        result = resolve_op("aten.addcmul.default")
        assert result is torch.ops.aten.addcmul.default

    def test_cpp_format(self):
        """PyTorch error messages use aten::op.overload format."""
        result = resolve_op("aten::add.Tensor")
        assert result is torch.ops.aten.add.Tensor

    def test_cpp_format_default(self):
        result = resolve_op("aten::addcmul.default")
        assert result is torch.ops.aten.addcmul.default

    def test_fully_qualified_nondefault(self):
        result = resolve_op("aten.softmax.int")
        assert result is torch.ops.aten.softmax.int

    def test_nonexistent_exact(self):
        result = resolve_op("aten.nonexistent_op_xyz.default")
        assert isinstance(result, list)


class TestDefaultOverload:
    def test_namespace_dot_opname(self):
        result = resolve_op("aten.addcmul")
        assert isinstance(result, OpOverload)
        assert result is torch.ops.aten.addcmul.default

    def test_no_default_overload(self):
        # softmax has no .default — should pick the primary non-out overload
        result = resolve_op("aten.softmax")
        assert isinstance(result, OpOverload)

    def test_bare_name_no_default(self):
        # "softmax" with no namespace should also resolve
        result = resolve_op("softmax")
        assert isinstance(result, OpOverload)
        assert "softmax" in result.name()


class TestNamespacePrefix:
    def test_bare_name(self):
        result = resolve_op("addcmul")
        assert isinstance(result, OpOverload)
        assert result is torch.ops.aten.addcmul.default

    def test_bare_name_with_underscore(self):
        result = resolve_op("_native_batch_norm_legit")
        assert isinstance(result, OpOverload)
        assert result is torch.ops.aten._native_batch_norm_legit.default


class TestSubstringSearch:
    def test_exact_aten_match_wins_over_substring(self):
        # "batch_norm" resolves exactly to aten.batch_norm.default
        result = resolve_op("batch_norm")
        assert isinstance(result, OpOverload)
        assert result is torch.ops.aten.batch_norm.default

    def test_substring_finds_candidates(self):
        # "batchnrm" doesn't match any op name exactly, falls to substring
        result = resolve_op("native_batch_norm")
        # There are multiple native_batch_norm variants
        assert isinstance(result, (OpOverload, list))
        if isinstance(result, list):
            assert len(result) > 0

    def test_no_matches(self):
        result = resolve_op("zzz_nonexistent_zzz")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_partial_match(self):
        result = resolve_op("addcm")
        assert isinstance(result, list)
        assert any("addcmul" in s for s in result)
