"""Tests for CLI entry point and output formatting."""

import json

import pytest
import torch

from decomp_magician.cli import main
from decomp_magician.export import add_untraceable_warnings, leaves_to_dict, tree_to_dict
from decomp_magician.format import (
    FormatConfig,
    format_leaves,
    format_summary,
    format_tree,
    format_untraceable_warning,
)
from decomp_magician.tree import (
    DecompNode,
    build_tree,
    collect_untraceable_errors,
)
from decomp_magician.classify import OpClass

# No-color config for tests (no tty)
_CFG = FormatConfig()
_COLOR_CFG = FormatConfig(color=True)


class TestMain:
    def test_basic_op(self, capsys):
        assert main(["addcmul"]) == 0
        out = capsys.readouterr().out
        assert "aten.addcmul.default" in out
        assert "[table" in out

    def test_leaf_op(self, capsys):
        assert main(["mm"]) == 0
        out = capsys.readouterr().out
        assert "aten.mm.default" in out
        assert "[leaf" in out

    def test_depth_limit(self, capsys):
        assert main(["_native_batch_norm_legit", "--depth", "1"]) == 0
        out = capsys.readouterr().out
        assert "squeeze.dims" in out
        assert "x3" in out

    def test_verbose_shows_schema(self, capsys):
        assert main(["addcmul", "--verbose", "--depth", "0"]) == 0
        out = capsys.readouterr().out
        assert "schema:" in out
        assert "decomp_type:" in out
        assert "autograd_type:" in out

    def test_nonexistent_op(self, capsys):
        assert main(["zzz_nonexistent_zzz"]) == 1
        captured = capsys.readouterr()
        assert "No ops found" in captured.err

    def test_fully_qualified(self, capsys):
        assert main(["aten.addcmul.default"]) == 0
        out = capsys.readouterr().out
        assert "addcmul" in out

    def test_compile_flag(self, capsys):
        assert main(["_native_batch_norm_legit", "--compile"]) == 0
        out = capsys.readouterr().out
        assert "inductor-kept" in out

    def test_conflicting_flags(self, capsys):
        assert main(["addcmul", "--mermaid", "--leaves"]) == 1
        captured = capsys.readouterr()
        assert "Conflicting" in captured.err

    def test_conflicting_reverse_mermaid(self, capsys):
        assert main(["addcmul", "--reverse", "--dot"]) == 1
        captured = capsys.readouterr()
        assert "Conflicting" in captured.err

    def test_json_mermaid_conflict(self, capsys):
        assert main(["addcmul", "--json", "--mermaid"]) == 1
        captured = capsys.readouterr()
        assert "Conflicting" in captured.err

    def test_json_leaves_allowed(self, capsys):
        assert main(["addcmul", "--json", "--leaves"]) == 0

    def test_no_args_shows_usage(self, capsys):
        assert main([]) == 1
        captured = capsys.readouterr()
        assert "Usage:" in captured.err
        assert "Examples:" in captured.err
        assert "--help" in captured.err


class TestDispatchFlags:
    def test_dispatch_table_flag(self, capsys):
        assert main(["addcmul", "--dispatch-table", "--depth", "0"]) == 0
        out = capsys.readouterr().out
        assert "AG:" in out

    def test_mode_sensitivity_flag(self, capsys):
        assert main(["addcmul", "--mode-sensitivity", "--depth", "0"]) == 0
        out = capsys.readouterr().out
        assert "mode-sensitive" in out or "mode-invariant" in out

    def test_dispatch_table_leaves_json(self, capsys):
        assert main(["addcmul", "--dispatch-table", "--leaves", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        for leaf in data["leaves"]:
            assert "has_adiov" in leaf
            assert "mode_sensitive" in leaf


class TestPurityFlag:
    def test_pure_flag_text(self, capsys):
        assert main(["addcmul", "--pure", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "PURE" in out or "IMPURE" in out
        assert "leaf" in out.lower()

    def test_pure_json_structure(self, capsys):
        assert main(["addcmul", "--pure", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data["pure"], bool)
        assert isinstance(data["total_leaves"], int)
        assert isinstance(data["mutable_leaves"], list)
        assert isinstance(data["adiov_leaves"], list)
        assert isinstance(data["mode_sensitive_leaves"], list)

    def test_mutable_op_is_impure(self, capsys):
        assert main(["_native_batch_norm_legit", "--pure", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["pure"] is False
        assert len(data["mutable_leaves"]) > 0
        mutable_names = [m["op"] for m in data["mutable_leaves"]]
        assert any("copy_" in n for n in mutable_names)


class TestADIOVFlag:
    def test_adiov_has_paths(self, capsys):
        assert main(["_native_batch_norm_legit", "--adiov", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "copy_" in out

    def test_adiov_no_paths(self, capsys):
        assert main(["sigmoid", "--adiov", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "NO ADIOV PATHS" in out

    def test_adiov_no_paths_json(self, capsys):
        assert main(["sigmoid", "--adiov", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["adiov_paths"] is False
        assert "no ADIOV" in data["message"]

    def test_adiov_has_paths_json(self, capsys):
        assert main(["_native_batch_norm_legit", "--adiov", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert "op" in data
        assert "children" in data


class TestModelFlag:
    @pytest.fixture
    def tiny_model_path(self, tmp_path):
        import warnings

        class TinyModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x) + 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ep = torch.export.export(TinyModel(), (torch.randn(2, 3),))
            path = tmp_path / "tiny.pt2"
            torch.export.save(ep, str(path))
        return str(path)

    def test_model_text_output(self, tiny_model_path, capsys):
        assert main(["--model", tiny_model_path, "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "Model analysis" in out
        assert "Unique ops:" in out
        assert "relu" in out
        assert "add" in out

    def test_model_json(self, tiny_model_path, capsys):
        assert main(["--model", tiny_model_path, "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["total_ops"] >= 2
        assert data["unique_ops"] >= 2
        op_names = [o["op"] for o in data["ops"]]
        assert any("relu" in n for n in op_names)
        assert any("add" in n for n in op_names)

    def test_model_target_opset(self, tiny_model_path, capsys):
        assert main(["--model", tiny_model_path, "--target-opset", "core_aten", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "core_aten" in out
        assert "must implement" in out

    def test_model_target_opset_json(self, tiny_model_path, capsys):
        assert main(["--model", tiny_model_path, "--target-opset", "core_aten", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["opset"] == "core_aten"
        assert data["decomposed"] is True

    def test_model_nonexistent_file(self, capsys):
        assert main(["--model", "/nonexistent/path.pt2"]) == 1
        assert "not found" in capsys.readouterr().err.lower()

    def test_model_without_opset_shows_decomposable(self, tiny_model_path, capsys):
        assert main(["--model", tiny_model_path, "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "[decomposable]" in out or "[leaf]" in out

    def test_model_dtensor_text(self, tiny_model_path, capsys):
        assert main(["--model", tiny_model_path, "--dtensor", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "dtensor:" in out
        assert "all ops covered" in out

    def test_model_dtensor_json(self, tiny_model_path, capsys):
        assert main(["--model", tiny_model_path, "--dtensor", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert "dtensor_covered" in data
        assert data["dtensor_covered"] is True
        assert data["dtensor_missing_ops"] == []
        for op in data["ops"]:
            assert "dtensor_strategy" in op


class TestFormatTree:
    def test_leaf_format(self):
        op = torch.ops.aten.mm.default
        node = build_tree(op)
        output = format_tree(node, _CFG)
        assert "aten.mm" in output
        assert "[leaf]" in output

    def test_tree_has_connectors(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op, depth=1)
        output = format_tree(node, _CFG)
        assert "├──" in output or "└──" in output

    def test_count_shown(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op, depth=1)
        output = format_tree(node, _CFG)
        assert "x2" in output

    def test_batch_norm_squeeze_visible(self):
        op = torch.ops.aten._native_batch_norm_legit.default
        node = build_tree(op, depth=1)
        output = format_tree(node, _CFG)
        assert "squeeze.dims" in output
        assert "x3" in output

    def test_inductor_kept_annotation(self, inductor_kept_op):
        node = build_tree(inductor_kept_op, depth=0)
        output = format_tree(node, _CFG)
        assert "inductor-kept" in output


class TestSummary:
    def test_leaf_summary(self):
        node = build_tree(torch.ops.aten.mm.default)
        s = format_summary(node, _CFG)
        assert "1 op" in s
        assert "1 leaf" in s

    def test_batch_norm_summary(self):
        node = build_tree(torch.ops.aten._native_batch_norm_legit.default, depth=1)
        s = format_summary(node, _CFG)
        assert "9 ops" in s
        assert "inductor-kept" in s

    def test_summary_in_output(self, capsys):
        main(["addcmul", "--depth", "0"])
        captured = capsys.readouterr()
        assert "1 op" in captured.out

    def test_untraceable_unique_count(self):
        op = torch.ops.aten.mul.Tensor
        cls_table = OpClass(decomp_type="table")
        cls_leaf = OpClass(decomp_type="leaf")
        bad = DecompNode(op=op, classification=cls_leaf, traceable=False, error="fail")
        root = DecompNode(op=op, children=(bad, bad, bad), classification=cls_table)
        s = format_summary(root, _CFG)
        assert "1 untraceable op (3 nodes)" in s

    def test_untraceable_no_parens_when_no_duplication(self):
        op1 = torch.ops.aten.mul.Tensor
        op2 = torch.ops.aten.add.Tensor
        cls_table = OpClass(decomp_type="table")
        cls_leaf = OpClass(decomp_type="leaf")
        bad1 = DecompNode(op=op1, classification=cls_leaf, traceable=False, error="fail")
        bad2 = DecompNode(op=op2, classification=cls_leaf, traceable=False, error="fail")
        root = DecompNode(op=op1, children=(bad1, bad2), classification=cls_table)
        s = format_summary(root, _CFG)
        assert "2 untraceable" in s
        assert "nodes" not in s

    def test_no_untraceable_in_summary(self):
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        s = format_summary(node, _CFG)
        assert "untraceable" not in s


class TestUntraceableWarnings:
    def _make_tree_with_untraceable(self):
        op = torch.ops.aten.mul.Tensor
        cls_table = OpClass(decomp_type="table")
        cls_leaf = OpClass(decomp_type="leaf")
        bad = DecompNode(op=op, classification=cls_leaf, traceable=False, error="test error")
        good = DecompNode(op=torch.ops.aten.add.Tensor, classification=cls_leaf)
        return DecompNode(op=op, children=(bad, good), classification=cls_table)

    def _make_clean_tree(self):
        op = torch.ops.aten.mul.Tensor
        cls_table = OpClass(decomp_type="table")
        cls_leaf = OpClass(decomp_type="leaf")
        child = DecompNode(op=op, classification=cls_leaf)
        return DecompNode(op=op, children=(child,), classification=cls_table)

    def test_collect_errors_finds_untraceable(self):
        errors = collect_untraceable_errors(self._make_tree_with_untraceable())
        assert len(errors) == 1
        assert errors[0][1] == "test error"

    def test_collect_errors_empty_for_clean_tree(self):
        errors = collect_untraceable_errors(self._make_clean_tree())
        assert len(errors) == 0

    def test_collect_errors_deduplicates(self):
        op = torch.ops.aten.mul.Tensor
        cls_table = OpClass(decomp_type="table")
        cls_leaf = OpClass(decomp_type="leaf")
        bad1 = DecompNode(op=op, classification=cls_leaf, traceable=False, error="err")
        bad2 = DecompNode(op=op, classification=cls_leaf, traceable=False, error="err")
        root = DecompNode(op=op, children=(bad1, bad2), classification=cls_table)
        errors = collect_untraceable_errors(root)
        assert len(errors) == 1

    def test_warn_untraceable_returns_warning(self):
        warning = format_untraceable_warning(self._make_tree_with_untraceable(), _CFG)
        assert "warning" in warning
        assert "1 op" in warning
        assert "test error" in warning

    def test_warn_untraceable_empty_for_clean_tree(self):
        warning = format_untraceable_warning(self._make_clean_tree(), _CFG)
        assert warning == ""

    def test_add_warnings_to_json(self):
        d = {}
        add_untraceable_warnings(d, self._make_tree_with_untraceable())
        assert "warnings" in d
        assert len(d["warnings"]) == 1
        assert "could not trace" in d["warnings"][0]["message"]

    def test_no_warnings_in_json_for_clean_tree(self):
        d = {}
        add_untraceable_warnings(d, self._make_clean_tree())
        assert "warnings" not in d


class TestLeaves:
    def test_leaves_output(self, capsys):
        assert main(["_native_batch_norm_legit", "--leaves"]) == 0
        captured = capsys.readouterr()
        assert "decomposes to:" in captured.out
        assert "unique ops" in captured.out
        assert "total instances" in captured.out

    def test_leaves_propagated_counts(self):
        node = build_tree(torch.ops.aten.addcmul.default)
        output = format_leaves(node, _CFG)
        assert "prims.mul.default" in output
        assert "x3" in output or "x4" in output or "x5" in output

    def test_leaves_with_compile(self, capsys):
        assert main(["_native_batch_norm_legit", "--compile", "--leaves"]) == 0
        captured = capsys.readouterr()
        assert "inductor-kept" in captured.out

    def test_leaf_op_leaves(self):
        node = build_tree(torch.ops.aten.mm.default)
        output = format_leaves(node, _CFG)
        assert "aten.mm.default" in output
        assert "no decomposition" in output

    def test_leaves_untraceable_warning(self):
        node = build_tree(torch.ops.aten.roll.default)
        output = format_leaves(node, _CFG)
        assert "untraceable" in output
        assert "incomplete" in output

    def test_json_leaves(self, capsys):
        assert main(["addcmul", "--json", "--leaves"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "leaves" in data
        assert "total_instances" in data
        assert any("mul" in leaf["op"] for leaf in data["leaves"])


class TestColor:
    def test_no_color_in_pipe(self, capsys):
        main(["addcmul", "--depth", "1"])
        captured = capsys.readouterr()
        assert "\033[" not in captured.out

    def test_no_color_flag(self, capsys):
        main(["addcmul", "--depth", "1", "--no-color"])
        captured = capsys.readouterr()
        assert "\033[" not in captured.out

    def test_color_rendering(self):
        node = build_tree(torch.ops.aten.addcmul.default, depth=1)
        output = format_tree(node, _COLOR_CFG)
        assert "\033[" in output
        assert "\033[1m" in output  # bold
        assert "\033[33m" in output  # yellow for inductor-kept

    def test_color_leaf_dim(self):
        node = build_tree(torch.ops.aten.mm.default)
        output = format_tree(node, _COLOR_CFG)
        assert "\033[2m" in output  # dim for leaf


class TestJson:
    def test_json_output(self, capsys):
        assert main(["addcmul", "--depth", "1", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["op"] == "aten.addcmul.default"
        assert data["decomp_type"] == "table"
        assert "children" in data

    def test_json_valid(self, capsys):
        assert main(["_native_batch_norm_legit", "--depth", "1", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["children"]) > 0

    def test_json_leaf_no_children(self, capsys):
        assert main(["mm", "--json"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["decomp_type"] == "leaf"
        assert "children" not in data

    def test_tree_to_dict_fields(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op, depth=0)
        d = tree_to_dict(node)
        assert "op" in d
        assert "decomp_type" in d
        assert "inductor_kept" in d
        assert "backends" in d
        assert "tags" in d
        assert "traceable" in d

    def test_tree_to_dict_count(self):
        op = torch.ops.aten.addcmul.default
        node = build_tree(op, depth=1)
        d = tree_to_dict(node)
        mul_children = [c for c in d["children"] if "mul" in c["op"]]
        assert mul_children[0]["count"] == 2


class TestSourceFlag:
    def test_source_text(self, capsys):
        assert main(["addcmul", "--source", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "Source:" in out
        assert "def addcmul" in out

    def test_source_json(self, capsys):
        assert main(["addcmul", "--source", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert "source" in data
        assert "file" in data["source"]
        assert "line" in data["source"]
        assert "def addcmul" in data["source"]["code"]

    def test_source_leaf_op(self, capsys):
        """Leaf ops have no decomposition source."""
        assert main(["mm", "--source", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "No decomposition source" in out

    def test_source_json_leaf_no_source(self, capsys):
        """JSON output for leaf ops should not include source field."""
        assert main(["mm", "--source", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert "source" not in data

    def test_source_with_leaves(self, capsys):
        """--source should work with --leaves."""
        assert main(["addcmul", "--leaves", "--source", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "Source:" in out

    def test_source_cia_shows_child(self, capsys):
        """CIA ops with C++ kernels should show first child's source."""
        assert main(["batch_norm", "--source", "--no-color"]) == 0
        out = capsys.readouterr().out
        assert "Source:" in out
        assert "via" in out


class TestDtensorAncestorCoverage:
    _DT_CFG = FormatConfig(show_dtensor=True)

    def _make_node(self, strategy, children=()):
        op = torch.ops.aten.mul.Tensor
        cls = OpClass(
            decomp_type="leaf" if not children else "table",
            dtensor_strategy=strategy,
        )
        return DecompNode(
            op=op, children=tuple(children), count=1,
            classification=cls, traceable=True, error=None,
        )

    def test_missing_suppressed_when_ancestor_registered(self):
        leaf = self._make_node("missing")
        parent = self._make_node("registered", children=[leaf])
        output = format_tree(parent, self._DT_CFG)
        assert "MISSING" not in output
        assert "via ancestor" in output

    def test_missing_shown_when_no_ancestor_registered(self):
        leaf = self._make_node("missing")
        parent = self._make_node("decomp-fallback", children=[leaf])
        output = format_tree(parent, self._DT_CFG)
        assert "MISSING" in output

    def test_decomp_fallback_does_not_cover(self):
        leaf = self._make_node("missing")
        mid = self._make_node("decomp-fallback", children=[leaf])
        root = self._make_node("decomp-fallback", children=[mid])
        output = format_tree(root, self._DT_CFG)
        assert "MISSING" in output

    def test_registered_covers_deep_descendants(self):
        deep_leaf = self._make_node("missing")
        mid = self._make_node("missing", children=[deep_leaf])
        root = self._make_node("registered", children=[mid])
        output = format_tree(root, self._DT_CFG)
        assert "MISSING" not in output

    def test_summary_covered_verdict(self):
        leaf = self._make_node("missing")
        parent = self._make_node("registered", children=[leaf])
        summary = format_summary(parent, self._DT_CFG)
        assert "dtensor: covered" in summary

    def test_summary_uncovered_verdict(self):
        leaf = self._make_node("missing")
        parent = self._make_node("decomp-fallback", children=[leaf])
        summary = format_summary(parent, self._DT_CFG)
        assert "1 uncovered" in summary

    def test_summary_no_verdict_without_dtensor(self):
        leaf = self._make_node("missing")
        parent = self._make_node("missing", children=[leaf])
        summary = format_summary(parent, _CFG)  # show_dtensor=False
        assert "dtensor" not in summary

    def test_leaves_shows_uncovered(self):
        leaf = self._make_node("missing")
        parent = self._make_node("decomp-fallback", children=[leaf])
        output = format_leaves(parent, _CFG)
        assert "MISSING" in output

    def test_leaves_hides_covered(self):
        leaf = self._make_node("missing")
        parent = self._make_node("registered", children=[leaf])
        output = format_leaves(parent, _CFG)
        assert "MISSING" not in output

    def test_json_leaves_uncovered_flag(self):
        leaf = self._make_node("missing")
        parent = self._make_node("decomp-fallback", children=[leaf])
        d = leaves_to_dict(parent)
        assert any(entry.get("dtensor_uncovered") for entry in d["leaves"])

    def test_json_leaves_no_flag_when_covered(self):
        leaf = self._make_node("missing")
        parent = self._make_node("registered", children=[leaf])
        d = leaves_to_dict(parent)
        assert not any(entry.get("dtensor_uncovered") for entry in d["leaves"])

    def test_cli_dtensor_decomposable_root_not_missing(self, capsys):
        assert main(["softmax", "--dtensor", "--no-color"]) == 0
        out = capsys.readouterr().out
        first_line = out.strip().split("\n")[0]
        assert "MISSING" not in first_line
