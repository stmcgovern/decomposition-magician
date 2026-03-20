"""Tests for decomposition tree construction."""

import pytest
import torch

from decomp_magician.classify import OpClass
from decomp_magician.tree import (
    DecompNode, build_tree, collect_leaf_counts,
    _trace_decomp, _make_meta_args,
)


class TestDecompNodeInvariants:
    def test_untraceable_with_children_raises(self):
        """Untraceable node cannot have children."""
        op = torch.ops.aten.mul.Tensor
        child = DecompNode(op=op)
        with pytest.raises(ValueError, match="Untraceable node cannot have children"):
            DecompNode(op=op, children=(child,), traceable=False)

    def test_error_with_traceable_raises(self):
        """Node with error must be untraceable."""
        op = torch.ops.aten.mul.Tensor
        with pytest.raises(ValueError, match="Node with error must be untraceable"):
            DecompNode(op=op, traceable=True, error="something failed")

    def test_valid_untraceable_node(self):
        """Untraceable node with no children and an error is valid."""
        op = torch.ops.aten.mul.Tensor
        node = DecompNode(op=op, traceable=False, error="test error")
        assert not node.traceable
        assert node.error == "test error"
        assert node.children == ()

    def test_untraceable_without_error_raises(self):
        """Untraceable node must explain why (traceable ↔ error is None)."""
        op = torch.ops.aten.mul.Tensor
        with pytest.raises(ValueError, match="Untraceable node must have an error reason"):
            DecompNode(op=op, traceable=False)

    def test_count_zero_raises(self):
        """count must be >= 1."""
        op = torch.ops.aten.mul.Tensor
        with pytest.raises(ValueError, match="count must be >= 1"):
            DecompNode(op=op, count=0)

    def test_valid_traceable_with_children(self):
        """Traceable node with children is valid."""
        op = torch.ops.aten.mul.Tensor
        child = DecompNode(op=op)
        parent = DecompNode(op=op, children=(child,), classification=OpClass("table"))
        assert parent.traceable
        assert len(parent.children) == 1


class TestTraceDecomp:
    def test_addcmul(self):
        op = torch.ops.aten.addcmul.default
        result = _trace_decomp(op)
        assert isinstance(result, tuple)
        names = [o.name() for o in result]
        assert "aten::mul.Tensor" in names
        assert "aten::add.Tensor" in names

    def test_batch_norm_legit(self):
        op = torch.ops.aten._native_batch_norm_legit.default
        result = _trace_decomp(op)
        assert isinstance(result, tuple)
        names = [o.name() for o in result]
        assert "aten::squeeze.dims" in names

    def test_leaf_op(self):
        op = torch.ops.aten.mm.default
        result = _trace_decomp(op)
        assert isinstance(result, str)  # error string
        assert "no decomposition" in result

    def test_dropout(self):
        # dropout has CIA
        op = torch.ops.aten.dropout.default
        result = _trace_decomp(op)
        assert isinstance(result, tuple)


class TestMakeMetaArgs:
    def test_simple_op(self):
        op = torch.ops.aten.addcmul.default
        result = _make_meta_args(op)
        assert result is not None
        args, kwargs = result
        assert len(args) >= 3
        assert all(a.device.type == "meta" for a in args if isinstance(a, torch.Tensor))

    def test_batch_norm(self):
        op = torch.ops.aten._native_batch_norm_legit.default
        result = _make_meta_args(op)
        assert result is not None
        args, kwargs = result
        assert len(args) == 8


class TestBuildTree:
    def test_leaf_node(self):
        op = torch.ops.aten.mm.default
        node = build_tree(op)
        assert node.classification.decomp_type == "leaf"
        assert len(node.children) == 0

    def test_addcmul_tree(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op)
        assert node.classification.decomp_type in ("table", "both")
        assert len(node.children) > 0
        child_names = [c.op.name() for c in node.children]
        assert "aten::mul.Tensor" in child_names
        assert "aten::add.Tensor" in child_names

    def test_batch_norm_has_squeeze(self):
        """The motivating example: squeeze.dims appears in batch_norm's tree."""
        op = torch.ops.aten._native_batch_norm_legit.default
        node = build_tree(op, depth=1)
        child_names = [c.op.name() for c in node.children]
        assert "aten::squeeze.dims" in child_names
        # Verify count
        squeeze_nodes = [c for c in node.children if c.op.name() == "aten::squeeze.dims"]
        assert len(squeeze_nodes) == 1
        assert squeeze_nodes[0].count == 3

    def test_depth_limit(self):
        op = torch.ops.aten._native_batch_norm_legit.default
        node = build_tree(op, depth=0)
        # depth=0 means don't expand at all
        assert len(node.children) == 0

    def test_depth_one_no_grandchildren(self):
        op = torch.ops.aten._native_batch_norm_legit.default
        node = build_tree(op, depth=1)
        # depth=1 means expand root but not children
        for child in node.children:
            assert len(child.children) == 0

    def test_count_deduplication(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op, depth=1)
        mul_nodes = [c for c in node.children if c.op.name() == "aten::mul.Tensor"]
        assert len(mul_nodes) == 1
        assert mul_nodes[0].count == 2

    def test_untraceable_node_doesnt_crash(self):
        # Building a tree on a leaf should work fine
        op = torch.ops.aten.mm.default
        node = build_tree(op)
        assert node.traceable is True  # leaf nodes are trivially traceable
        assert node.error is None

    def test_cycle_detected(self):
        """roll decomposes to [view, roll, view] — cycle must not hang."""
        op = torch.ops.aten.roll.default
        node = build_tree(op)
        # The recursive roll child should be marked as a cycle
        roll_children = [c for c in node.children if c.op.name() == "aten::roll"]
        assert len(roll_children) == 1
        assert roll_children[0].traceable is False
        assert roll_children[0].error == "cycle detected"

    def test_softmax_traces(self):
        """softmax was previously untraceable due to half_to_float=True default."""
        op = torch.ops.aten._softmax.default
        node = build_tree(op, depth=1)
        assert node.traceable is True
        child_names = [c.op.name() for c in node.children]
        assert "aten::exp" in child_names
        assert "aten::div.Tensor" in child_names

    def test_to_copy_traces(self):
        """_to_copy was previously untraceable due to memory_format=0."""
        op = torch.ops.aten._to_copy.default
        node = build_tree(op, depth=1)
        assert node.traceable is True
        child_names = [c.op.name() for c in node.children]
        assert "aten::clone" in child_names

    def test_fill_traces(self):
        """fill.Tensor needs a scalar (0-D) value tensor."""
        op = torch.ops.aten.fill.Tensor
        node = build_tree(op, depth=1)
        assert node.traceable is True

    def test_clamp_traces(self):
        """clamp needs at least one of min/max to be non-None."""
        op = torch.ops.aten.clamp.default
        node = build_tree(op, depth=1)
        assert node.traceable is True
        assert len(node.children) > 0

    def test_conv2d_traces(self):
        """conv2d needs N-D weight matching input dimensionality."""
        op = torch.ops.aten.conv2d.default
        node = build_tree(op, depth=1)
        assert node.traceable is True
        child_names = [c.op.name() for c in node.children]
        assert "aten::convolution" in child_names

    def test_conv1d_traces(self):
        """conv1d needs 3D weight."""
        op = torch.ops.aten.conv1d.default
        node = build_tree(op, depth=1)
        assert node.traceable is True

    def test_conv_transpose2d_traces(self):
        """conv_transpose2d needs weight[0]==input[1] (square channels shape)."""
        op = torch.ops.aten.conv_transpose2d.input
        node = build_tree(op, depth=1)
        assert node.traceable is True

    def test_leaf_frontier_correctness(self):
        """Verify the leaf frontier has correct propagated counts for addcmul.

        addcmul(self, t1, t2, value) = self + value * (t1 * t2)
        mul decomposes to prims.mul; add decomposes to prims.mul + prims.add + expand.
        mul appears x2, so prims.mul from mul path = 2, plus 1 from add path = 3.
        """
        from collections import Counter
        op = torch.ops.aten.addcmul.default
        node = build_tree(op)

        frontier: Counter[str] = Counter()
        def walk(n, mult=1):
            if not n.children:
                frontier[n.op.name()] += mult
                return
            for c in n.children:
                walk(c, mult * c.count)
        walk(node)

        assert frontier["prims::mul"] == 3  # 2 from mul.Tensor x2 + 1 from add.Tensor
        assert frontier["prims::add"] == 1

    def test_compile_stops_at_inductor_kept(self):
        """--compile treats inductor-kept ops as leaves."""
        op = torch.ops.aten._native_batch_norm_legit.default
        normal = build_tree(op, depth=2)
        compiled = build_tree(op, depth=2, compile=True)

        def count_nodes(n):
            return 1 + sum(count_nodes(c) for c in n.children)

        # compile mode should have fewer nodes (inductor-kept ops not expanded)
        assert count_nodes(compiled) < count_nodes(normal)

        # inductor-kept ops should have no children in compile mode
        for child in compiled.children:
            if child.classification.inductor_kept:
                assert len(child.children) == 0


class TestCollectLeafCounts:
    def test_leaf_op_returns_self(self):
        op = torch.ops.aten.mm.default
        node = build_tree(op)
        counts = collect_leaf_counts(node)
        assert counts["aten.mm.default"] == 1
        assert len(counts) == 1

    def test_addcmul_prims_mul_count(self):
        """addcmul = self + value * (t1 * t2) → prims.mul should appear 3x."""
        op = torch.ops.aten.addcmul.default
        node = build_tree(op)
        counts = collect_leaf_counts(node)
        assert counts["prims.mul.default"] == 3
        assert counts["prims.add.default"] == 1

    def test_matches_manual_walk(self):
        """collect_leaf_counts should match a manual walk."""
        from collections import Counter
        from decomp_magician.tree import op_display_name

        op = torch.ops.aten._native_batch_norm_legit.default
        node = build_tree(op, depth=1)

        # Manual walk
        manual: Counter[str] = Counter()
        for c in node.children:
            manual[op_display_name(c.op)] += c.count

        auto = collect_leaf_counts(node)
        assert auto == manual


class TestClassifyCache:
    def test_classify_is_cached(self):
        """classify() returns the same object on repeated calls."""
        from decomp_magician.classify import classify
        op = torch.ops.aten.addcmul.default
        cls1 = classify(op)
        cls2 = classify(op)
        assert cls1 is cls2

    def test_classify_always_has_dtensor_strategy(self):
        """classify() always populates dtensor_strategy (no None)."""
        from decomp_magician.classify import classify
        op = torch.ops.aten.addcmul.default
        cls = classify(op)
        assert cls.dtensor_strategy is not None

    def test_classify_returns_opclass(self):
        from decomp_magician.classify import classify
        op = torch.ops.aten.addcmul.default
        cls = classify(op)
        assert isinstance(cls, OpClass)
        assert cls.decomp_type in ("table", "both")
