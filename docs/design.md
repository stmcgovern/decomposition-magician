# decomposition-magician: Design Document

## Problem

PyTorch has ~1300 decomposable operators spread across two distinct mechanisms
(CIA kernels and explicit decomposition tables), but no tool to inspect what
any given op decomposes into. Answering "what ops are in the decomposition
chain of batch_norm?" requires reading 8 source files across 4 subsystems.

Questions that arise in practice:

- What does op X decompose into? (The full tree, not just one level.)
- Is the decomposition via CIA or the explicit table? Does Inductor keep it?
- If I'm writing DTensor sharding strategies, which ops in the decomposition
  chain are missing strategies? (The batch_norm/squeeze.dims problem.)
- Can I trace this decomposition chain on meta tensors, or will it fail
  with unbacked symbolic shapes?

No existing PyTorch tool answers these. The building blocks exist
(`decomposition_table`, `_can_decompose()`, `decompose()`) but there is
no introspection — no way to ask "show me the tree".

## Motivating Example

`_native_batch_norm_legit` has an explicit table decomposition that calls
`aten.squeeze.dims` three times. Before a sharding strategy was registered for
`squeeze.dims`, `DecompShardingStrategy` would fail when tracing this
decomposition — it couldn't propagate placements through the squeeze ops.

The fix was registering an `aten.squeeze.dims` strategy. But discovering that
`squeeze.dims` was in the decomposition chain required manual source reading.
decomposition-magician should make this relationship immediately visible.

A separate failure remains on the unbacked symbolic shape test path:
`DecompShardingStrategy.propagate_strategy` traces the decomposition on meta
tensors, and when those tensors carry unbacked symbolic shapes, the tracer
hits `GuardOnDataDependentSymNode` and bails out. This is orthogonal to the
squeeze strategy fix — it's a limitation of meta-tensor tracing under symbolic
shapes, not a missing sharding strategy. decomposition-magician should surface
this distinction: "missing strategy in decomposition chain" vs. "decomposition
chain untraceable under symbolic constraints".

## Two Decomposition Mechanisms

Most decompositions are **not** CIA. The numbers from the actual codebase:

| Mechanism | Count |
|-----------|-------|
| Explicit table only (`decomposition_table`) | 968 |
| CIA only (`_can_decompose()`) | 180 |
| Both | 159 |

The explicit decomposition table (`torch._decomp.decomposition_table`) has
6x more entries than CIA. Key ops like `addcmul`, `_native_batch_norm_legit`,
and most of the core ATen set decompose via the table, not CIA. DTensor's
`DecompShardingStrategy` checks both:

```python
def has_decomp(op):
    return op in decomposition_table or op._can_decompose()
```

The tool must do the same.

## What Exists vs. What's Missing

| API | What it does | What's missing |
|-----|-------------|----------------|
| `decomposition_table` | Dict of 1127 decomposition functions | No tree structure, no introspection |
| `op._can_decompose()` | Checks CIA kernel existence | Doesn't show what it decomposes into |
| `op.decompose(*args)` | Runs CIA on tensors | Requires valid args, no tree output |
| `get_decompositions(ops)` | Returns subset of table | Just a dict |

**The gap**: no tool produces a decomposition tree, cross-references
with DTensor strategy coverage, or distinguishes "missing strategy"
from "untraceable decomposition".

## Deliverable

One CLI command. One output: a decomposition tree printed to stdout with
per-op annotations. No TUI framework, no web server, no dependencies beyond
PyTorch.

```
$ decomp-magician batch_norm_legit --dtensor

aten._native_batch_norm_legit.default  [table]
├── aten.var_mean.correction           [table]
├── aten.add.Tensor                    [table]  x4  dtensor: ok
├── aten.rsqrt.default                 [table]       dtensor: ok
├── aten.sub.Tensor                    [table]       dtensor: ok
├── aten.mul.Tensor                    [table]  x7  dtensor: ok
├── aten.squeeze.dims                  [table]  x3  dtensor: MISSING
├── aten.copy_.default                 [leaf]   x2  dtensor: MISSING
└── aten.unsqueeze.default             [table]  x4  dtensor: ok
```

The user types `batch_norm_legit`, not the full qualified name. The tool
resolves it. The output shows exactly which ops in the chain are missing
DTensor strategies.

Optional flags:
- `--dtensor`: cross-reference with DTensor sharding strategy coverage
- `--mode {eager,compile,export}`: annotate decomposition behavior per regime
- `--depth N`: cap recursion depth (default: unlimited)
- `--verbose`: full classification per op (dispatch keys, backends, tags)

### Design Principles

- **Use PyTorch's own machinery.** Don't reimplement dispatch. Call
  `_can_decompose()`, `decompose()`, `has_kernel_for_dispatch_key()`.
  The tool is a lens, not a reimplementation.

- **Meta tensors for tracing.** Build trees by running `op.decompose()`
  on meta tensors under a recording `TorchDispatchMode`. Same approach
  as `DecompShardingStrategy._propagate_through_decomp`.

- **Fail gracefully.** If meta tensor creation fails for an op, report
  it as "untraceable" — don't crash. Distinguish "missing strategy" from
  "untraceable under symbolic constraints" (the unbacked shape problem).

- **No network, no state, no config files.** Pure local, read-only.

### Non-Goals

- Modifying decompositions or dispatch tables.
- Replacing `torch.compile` debugging.
- Supporting custom ops or higher-order operators.

> Note: Graph visualization (`--mermaid`, `--dot`) was added in v0.3.0,
> reversing the original non-goal. Reverse lookup (`--reverse`) and bulk
> statistics (`--stats`) were also added.

## Roadmap (v0.4.0)

### Subcommands

The flat flag namespace has grown to ~15 flags. Several of these are
mutually-exclusive output modes (`--stats`, `--reverse`, `--diff`,
`--leaves`, `--mermaid`, `--dot`, `--target-opset`, `--model`) that
would be clearer as subcommands:

```
decomp tree   addcmul              # default tree view (current bare invocation)
decomp model  model.pt2 --opset core_aten   # model-level analysis
decomp stats  --compile            # bulk statistics
decomp diff   softmax              # full vs compile comparison
decomp reverse prims.mul.default   # reverse lookup
```

This removes the need for mutually-exclusive flag validation, makes
`--help` per-subcommand instead of a wall of text, and lets each
subcommand define only the flags it accepts (e.g. `--include-out` only
on `reverse`). The bare `decomp-magician <op>` invocation should
remain as a shortcut for `decomp tree <op>` for backwards
compatibility.

### Additional opsets

Only `core_aten` is supported today. Potential additions:
- `inductor` — the ops torch.compile/inductor lowers directly
- `onnx` — the ONNX export opset
- `xla` — ops with XLA lowerings

### Diff between overloads

`--diff` currently compares full vs compile mode for a single op.
A natural extension: compare decomposition trees of two different
overloads, e.g. `decomp diff logsumexp.default logsumexp.dim_IntList`.

### Autograd backward pass

Currently shows only the forward decomposition tree. The backward
pass has its own decomposition chain (e.g. `logsumexp_backward` →
`unsqueeze_multiple` → `sub` → `exp` → `mul`). Surfacing both
forward and backward trees would help autograd debugging.

### Git-diff mode

Compare decomposition trees before/after a code change:
`decomp-magician logsumexp --git-diff HEAD~1`. Useful for validating
PyTorch PRs that modify decomposition tables or schemas.
