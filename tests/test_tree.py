"""Tests for decomposition tree construction."""

import torch

from decomp_magician.tree import build_tree, _trace_decomp, _make_meta_args


class TestTraceDecomp:
    def test_addcmul(self):
        op = torch.ops.aten.addcmul.default
        result = _trace_decomp(op)
        assert isinstance(result, list)
        names = [o.name() for o in result]
        assert "aten::mul.Tensor" in names
        assert "aten::add.Tensor" in names

    def test_batch_norm_legit(self):
        op = torch.ops.aten._native_batch_norm_legit.default
        result = _trace_decomp(op)
        assert isinstance(result, list)
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
        assert isinstance(result, list)


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
