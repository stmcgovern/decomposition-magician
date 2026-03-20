"""CLI entry point for decomposition-magician.

Thin dispatch layer: parse args → compute → format → print.
No ANSI codes, no data computation beyond calling library functions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from decomp_magician.classify import DtensorStrategy, is_dtensor_gap
from decomp_magician.diff import compute_diff, compute_diff_ops
from decomp_magician.export import (
    add_untraceable_warnings,
    enrich_leaves_with_dispatch,
    enrich_tree_with_dispatch,
    format_dot,
    format_mermaid,
    leaves_to_dict,
    tree_to_dict,
)
from decomp_magician.format import (
    FormatConfig,
    format_backward,
    format_diff,
    format_leaves,
    format_model_dtensor_tag,
    format_opset,
    format_purity,
    format_reverse,
    format_stats,
    format_summary,
    format_tree,
    format_verbose,
    should_use_color,
    format_untraceable_warning,
)
from decomp_magician.opset import OPSETS, check_opset_coverage
from decomp_magician.resolve import resolve_op
from decomp_magician.reverse import reverse_lookup
from decomp_magician.stats import compute_stats
from decomp_magician.tree import (
    analyze_purity,
    build_tree,
    filter_adiov_paths,
    op_display_name,
    trace_backward,
)


def main(argv: list[str] | None = None) -> int:
    from importlib.metadata import version as pkg_version

    parser = _build_parser(pkg_version)
    args = parser.parse_args(argv)

    cfg = FormatConfig(
        color=(not args.no_color and not args.json and should_use_color()),
        show_dispatch=args.dispatch_table,
        show_mode_sensitivity=args.mode_sensitivity,
    )

    # Stats mode
    if args.stats:
        return _run_stats(args, cfg)

    # Model analysis mode
    if args.model:
        return _run_model(args, cfg)

    # All other modes require an op
    if args.op is None or args.op == "":
        _print_usage()
        return 1

    # Validate flag combinations
    err = _validate_flags(args)
    if err:
        print(err, file=sys.stderr)
        return 1

    # Resolve the op name
    result = resolve_op(args.op)

    if isinstance(result, list):
        if len(result) == 0:
            print(f"No ops found matching '{args.op}'", file=sys.stderr)
            return 1
        print(f"Ambiguous op name '{args.op}'. Candidates:", file=sys.stderr)
        for candidate in sorted(result):
            print(f"  {candidate}", file=sys.stderr)
        return 1

    op = result
    resolved_name = op_display_name(op)
    graph_mode = args.mermaid or args.dot
    if not resolved_name.endswith(".default") and not args.json and not graph_mode:
        user_input = args.op.replace("::", ".")
        if user_input.count(".") < 2:
            print(f"(resolved to {resolved_name})", file=sys.stderr)

    if args.reverse:
        return _run_reverse(resolved_name, args, cfg)
    if args.target_opset and not args.leaves:
        return _run_opset(op, args, cfg)
    if args.diff:
        return _run_diff(op, args, cfg)
    if args.backward:
        return _run_backward(op, args, cfg)

    return _run_tree(op, resolved_name, args, cfg)


def _build_parser(pkg_version):
    parser = argparse.ArgumentParser(
        prog="decomp-magician",
        description="Inspect PyTorch operator decomposition trees.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {pkg_version('decomposition-magician')}",
    )
    parser.add_argument(
        "op", nargs="?",
        help="Operator name (e.g., 'addcmul', 'aten.addcmul.default', 'batch_norm')",
    )
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics across all decomposable ops")
    parser.add_argument("--depth", type=int, default=-1,
                        help="Maximum recursion depth (-1 for unlimited)")
    parser.add_argument("--dtensor", action="store_true",
                        help="Show DTensor sharding strategy coverage")
    parser.add_argument("--compile", action="store_true",
                        help="Show what torch.compile produces")
    parser.add_argument("--leaves", action="store_true",
                        help="Show flat leaf frontier with propagated counts")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse lookup: find all ops that decompose into the given op")
    parser.add_argument("--include-out", action="store_true",
                        help="Include _out variant ops in --reverse results")
    parser.add_argument("--mermaid", action="store_true",
                        help="Output as Mermaid flowchart")
    parser.add_argument("--dot", action="store_true",
                        help="Output as Graphviz DOT graph")
    parser.add_argument("--target-opset", metavar="OPSET",
                        help=f"Check if op decomposes fully to target opset ({', '.join(OPSETS)})")
    parser.add_argument("--diff", nargs="?", const=True, default=False, metavar="OP2",
                        help="Compare decompositions: bare=full vs compile, with OP2=compare two ops")
    parser.add_argument("--model", metavar="PATH",
                        help="Analyze an exported model (.pt2 file from torch.export)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full classification details per op")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--dispatch-table", action="store_true",
                        help="Annotate each op with its dispatch table entries")
    parser.add_argument("--mode-sensitivity", action="store_true",
                        help="Show which ops behave differently under inference_mode vs no_grad")
    parser.add_argument("--adiov", action="store_true",
                        help="Filter tree to only paths reaching ops with ADInplaceOrView kernels")
    parser.add_argument("--pure", action="store_true",
                        help="Check if decomposition is pure (no mutable or ADIOV leaves)")
    parser.add_argument("--backward", action="store_true",
                        help="Show ops dispatched during the backward pass")
    return parser


def _print_usage():
    print("Usage: decomp-magician <op> [options]", file=sys.stderr)
    print("", file=sys.stderr)
    print("Examples:", file=sys.stderr)
    print("  decomp-magician addcmul                  # decomposition tree", file=sys.stderr)
    print("  decomp-magician batch_norm --compile      # compile-mode tree", file=sys.stderr)
    print("  decomp-magician softmax --diff            # full vs compile diff", file=sys.stderr)
    print("  decomp-magician addcmul --target-opset core_aten", file=sys.stderr)
    print("  decomp-magician --stats                   # bulk statistics", file=sys.stderr)
    print("  decomp-magician --model model.pt2 --target-opset core_aten", file=sys.stderr)
    print("", file=sys.stderr)
    print("Run with --help for all options.", file=sys.stderr)


def _validate_flags(args) -> str | None:
    active_modes = [f for f, v in [
        ("--mermaid", args.mermaid), ("--dot", args.dot),
        ("--leaves", args.leaves), ("--reverse", args.reverse),
        ("--target-opset", bool(args.target_opset)), ("--diff", bool(args.diff)),
    ] if v]
    if set(active_modes) == {"--leaves", "--target-opset"}:
        pass
    elif len(active_modes) > 1:
        return f"Conflicting flags: {', '.join(active_modes)} (pick one)"
    if args.json and (args.mermaid or args.dot):
        flag = "--mermaid" if args.mermaid else "--dot"
        return f"Conflicting flags: --json, {flag} (pick one)"
    return None


def _warn_untraceable(node, cfg: FormatConfig) -> None:
    """Print untraceable warning to stderr if any."""
    warning = format_untraceable_warning(node, cfg)
    if warning:
        print(warning, file=sys.stderr)


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def _run_tree(op, resolved_name: str, args, cfg: FormatConfig) -> int:
    node = build_tree(op, depth=args.depth, dtensor=args.dtensor, compile=args.compile)

    if args.pure:
        purity = analyze_purity(node)
        if args.json:
            json_data = {
                "op": purity.op,
                "pure": purity.is_pure,
                "total_leaves": purity.total_leaves,
                "mutable_leaves": [{"op": n, "count": c} for n, c in purity.mutable_leaves],
                "adiov_leaves": [{"op": n, "count": c} for n, c in purity.adiov_leaves],
                "mode_sensitive_leaves": [{"op": n, "count": c} for n, c in purity.mode_sensitive_leaves],
            }
            print(json.dumps(json_data, indent=2))
        else:
            print(format_purity(purity, cfg))
        return 0

    if args.adiov:
        filtered = filter_adiov_paths(node)
        if filtered is None:
            root_name = op_display_name(node.op)
            if args.json:
                print(json.dumps({"op": root_name, "adiov_paths": False, "message": "no ADIOV leaves"}))
            else:
                print(f"NO ADIOV PATHS  {root_name}")
                print("  No decomposition path reaches an op with an ADInplaceOrView kernel.")
            return 0
        node = filtered

    if args.json:
        if args.leaves:
            d = leaves_to_dict(node)
            if cfg.show_dispatch or cfg.show_mode_sensitivity:
                d = enrich_leaves_with_dispatch(d, node)
            if args.target_opset:
                from decomp_magician.opset import is_core_aten
                for leaf in d.get("leaves", []):
                    leaf["in_opset"] = is_core_aten(leaf["op"])
                d["opset"] = args.target_opset
        else:
            d = tree_to_dict(node)
            if cfg.show_dispatch or cfg.show_mode_sensitivity:
                enrich_tree_with_dispatch(d, node)
        add_untraceable_warnings(d, node)
        print(json.dumps(d, indent=2))
        return 0

    if args.mermaid:
        print(format_mermaid(node))
        print("", file=sys.stderr)
        print("Paste into ```mermaid fenced block in GitHub markdown to render.", file=sys.stderr)
        _warn_untraceable(node, cfg)
        return 0

    if args.dot:
        print(format_dot(node))
        print("", file=sys.stderr)
        print("Pipe to: dot -Tsvg > graph.svg  or  dot -Tpng > graph.png", file=sys.stderr)
        _warn_untraceable(node, cfg)
        return 0

    if args.leaves:
        opset_checker = None
        if args.target_opset:
            from decomp_magician.opset import is_core_aten
            opset_checker = (args.target_opset, is_core_aten)
        print(format_leaves(node, cfg, opset_checker=opset_checker))
    else:
        print(format_tree(node, cfg))
        print()
        print(format_summary(node, cfg))

        if args.verbose:
            print()
            print(format_verbose(node, cfg))

    _warn_untraceable(node, cfg)
    return 0


def _run_reverse(target: str, args, cfg: FormatConfig) -> int:
    print(f"Scanning decomposition table for ops that produce {target}...",
          file=sys.stderr)

    results = reverse_lookup(target, depth=args.depth, compile=args.compile,
                             include_out=args.include_out)

    if args.json:
        producers = [{"op": r.op, "count": r.count, "target_depth": r.target_depth}
                     for r in results]
        print(json.dumps({"target": target, "producers": producers}, indent=2))
        return 0

    if not results:
        print(f"No ops decompose into {target}")
        if target.endswith(".default"):
            base = target.rsplit(".", 1)[0]
            parts = base.split(".")
            if len(parts) == 2:
                try:
                    import torch
                    from torch._ops import OpOverloadPacket
                    packet = getattr(getattr(torch.ops, parts[0]), parts[1])
                    if isinstance(packet, OpOverloadPacket):
                        overloads = [ol for ol in packet.overloads() if ol != "default"]
                        if overloads:
                            alts = [f"{base}.{ol}" for ol in overloads[:5]]
                            print(f"Try a different overload: {', '.join(alts)}",
                                  file=sys.stderr)
                except (AttributeError, RuntimeError):
                    pass
        return 0

    print(format_reverse(results, target, cfg, compile_mode=args.compile))
    return 0


def _run_stats(args, cfg: FormatConfig) -> int:
    print("Scanning all decomposable ops...", file=sys.stderr)

    data = compute_stats(compile=args.compile, dtensor=args.dtensor)

    if args.json:
        json_data = {
            "total": data.total,
            "total_non_out": data.total_non_out,
            "by_type": data.by_type,
            "inductor_kept": data.inductor_kept,
            "traceable": data.traceable,
            "untraceable": data.untraceable,
            "classify_errors": data.classify_errors,
            "top_leaf_ops": [
                {"op": name, "appearances": count}
                for name, count in data.leaf_ops.most_common(20)
            ],
            "deepest": [
                {"op": name, "depth": depth}
                for name, depth in data.deepest
            ],
        }
        if args.target_opset:
            from decomp_magician.opset import is_core_aten
            covered = [n for n in data.leaf_ops if is_core_aten(n)]
            non_covered = [n for n in data.leaf_ops if not is_core_aten(n)]
            json_data["opset"] = args.target_opset
            json_data["leaf_ops_in_opset"] = len(covered)
            json_data["leaf_ops_not_in_opset"] = sorted(non_covered)
        json_data["untraceable_ops"] = [
            {"op": name, "error": reason}
            for name, reason in data.untraceable_ops
        ]
        if data.dtensor:
            dt = data.dtensor
            json_data["dtensor"] = {
                "registered": dt.registered,
                "decomp_fallback": dt.decomp_fallback,
                "missing": dt.missing,
                "not_applicable": dt.not_applicable,
                "fully_covered": dt.fully_covered,
                "has_gaps": dt.has_gaps,
                "top_uncovered": [
                    {"op": name, "appearances": count}
                    for name, count in dt.top_uncovered
                ],
            }
        print(json.dumps(json_data, indent=2))
        return 0

    print(format_stats(data, cfg, target_opset=args.target_opset, compile=args.compile))
    return 0


def _run_opset(op, args, cfg: FormatConfig) -> int:
    try:
        cov = check_opset_coverage(
            op, opset=args.target_opset,
            depth=args.depth, compile=args.compile,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.json:
        json_data = {
            "op": cov.op,
            "opset": cov.opset,
            "fully_covered": cov.fully_covered,
            "total_leaves": cov.total_leaves,
            "covered_leaves": cov.covered_leaves,
            "non_covered": [
                {"op": name, "count": count}
                for name, count in cov.non_covered
            ],
        }
        print(json.dumps(json_data, indent=2))
        return 0

    print(format_opset(cov, cfg, compile_mode=args.compile))
    return 0


def _run_diff(op, args, cfg: FormatConfig) -> int:
    if args.diff is True:
        diff = compute_diff(op, depth=args.depth)
    else:
        result2 = resolve_op(args.diff)
        if isinstance(result2, list):
            if len(result2) == 0:
                print(f"No ops found matching '{args.diff}'", file=sys.stderr)
                return 1
            print(f"Ambiguous op name '{args.diff}'. Candidates:", file=sys.stderr)
            for c in sorted(result2):
                print(f"  {c}", file=sys.stderr)
            return 1
        diff = compute_diff_ops(op, result2, depth=args.depth, compile=args.compile)

    if args.json:
        json_data = {
            "left": diff.left_label,
            "right": diff.right_label,
            "added": [{"op": n, "count": c} for n, c in diff.added.most_common()],
            "removed": [{"op": n, "count": c} for n, c in diff.removed.most_common()],
            "changed": [
                {"op": n, "left_count": lc, "right_count": rc}
                for n, lc, rc in diff.changed
            ],
        }
        print(json.dumps(json_data, indent=2))
        return 0

    print(format_diff(diff, cfg))
    return 0


def _run_backward(op, args, cfg: FormatConfig) -> int:
    result = trace_backward(op)
    name = op_display_name(op)

    if isinstance(result, str):
        if args.json:
            print(json.dumps({"op": name, "backward": None, "error": result}))
        else:
            print(f"{name}  backward")
            print(f"  error: {result}")
        return 1

    op_counts: Counter[str] = Counter()
    for child_op in result:
        op_counts[op_display_name(child_op)] += 1

    if args.json:
        print(json.dumps({
            "op": name,
            "backward": [
                {"op": n, "count": c} for n, c in op_counts.most_common()
            ],
            "total_instances": len(result),
        }, indent=2))
        return 0

    print(format_backward(name, op_counts, cfg))
    return 0


def _run_model(args, cfg: FormatConfig) -> int:
    import warnings

    import torch

    path = args.model
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ep = torch.export.load(path)
    except Exception as e:
        print(f"Failed to load model from {path}: {e}", file=sys.stderr)
        return 1

    if args.target_opset == "core_aten":
        print("Decomposing model to core ATen ops...", file=sys.stderr)
        from torch._decomp import core_aten_decompositions
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ep = ep.run_decompositions(core_aten_decompositions())
        except Exception as e:
            print(f"Failed to decompose model: {e}", file=sys.stderr)
            return 1

    op_counts: Counter[str] = Counter()
    op_objects: dict[str, torch._ops.OpOverload] = {}
    for node in ep.graph.nodes:
        if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload):
            name = op_display_name(node.target)
            op_counts[name] += 1
            op_objects[name] = node.target

    if not op_counts:
        print("No ATen ops found in the model graph.", file=sys.stderr)
        return 1

    dtensor_info: dict[str, str] = {}
    if args.dtensor:
        from decomp_magician.classify import classify as classify_op
        for name, op_obj in op_objects.items():
            try:
                cls = classify_op(op_obj, dtensor=True)
                dtensor_info[name] = cls.dtensor_strategy or DtensorStrategy.MISSING
            except Exception:
                dtensor_info[name] = DtensorStrategy.MISSING

    if args.json:
        json_data: dict = {
            "model": path,
            "total_ops": sum(op_counts.values()),
            "unique_ops": len(op_counts),
            "ops": [],
        }
        for name, count in op_counts.most_common():
            entry: dict = {"op": name, "count": count}
            if name in dtensor_info:
                entry["dtensor_strategy"] = dtensor_info[name]
            json_data["ops"].append(entry)
        if args.target_opset:
            json_data["opset"] = args.target_opset
            json_data["decomposed"] = True
        if dtensor_info:
            gaps = [n for n, s in dtensor_info.items() if is_dtensor_gap(s)]
            json_data["dtensor_covered"] = len(gaps) == 0
            json_data["dtensor_missing_ops"] = sorted(gaps)
        print(json.dumps(json_data, indent=2))
        return 0

    # Text output for model analysis
    total = sum(op_counts.values())
    stage = f"  (after {args.target_opset} decomposition)" if args.target_opset else ""
    lines = [
        f"Model analysis  {path}{stage}",
        "",
        f"  Total op instances:  {total}",
        f"  Unique ops:          {len(op_counts)}",
    ]

    if dtensor_info:
        missing = [n for n, s in dtensor_info.items() if is_dtensor_gap(s)]
        if missing:
            lines.append(f"  DTensor:             {len(missing)} ops missing strategies")
        else:
            lines.append("  DTensor:             all ops covered")

    lines.append("")
    lines.append("Ops in graph:")

    name_width = max(len(n) for n in op_counts)

    if args.target_opset:
        for name, count in op_counts.most_common():
            dt_tag = format_model_dtensor_tag(dtensor_info, name, cfg)
            lines.append(f"  {name:<{name_width}}  x{count}{dt_tag}")
        lines.append("")
        lines.append(
            f"These are the ops a {args.target_opset} backend must implement for this model."
        )
    else:
        from torch._decomp import decomposition_table
        decomposable = 0
        for name, count in op_counts.most_common():
            op_obj = op_objects.get(name)
            has_decomp = op_obj is not None and op_obj in decomposition_table
            tag = "[decomposable]" if has_decomp else "[leaf]"
            dt_tag = format_model_dtensor_tag(dtensor_info, name, cfg)
            lines.append(f"  {name:<{name_width}}  x{count}  {tag}{dt_tag}")
            if has_decomp:
                decomposable += 1
        lines.append("")
        lines.append(
            f"  {decomposable}/{len(op_counts)} ops have decompositions. "
            f"Use --target-opset core_aten to decompose and show final ops."
        )

    print("\n".join(lines))
    return 0
