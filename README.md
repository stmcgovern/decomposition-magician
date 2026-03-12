# decomposition-magician

Inspect PyTorch operator decomposition trees from the command line.

PyTorch has ~1300 decomposable operators spread across two mechanisms (CompositeImplicitAutograd kernels and explicit decomposition tables), but no built-in way to see what a given op decomposes into. This tool fills that gap.

## Install

```
pip install .
```

## Quick start

```
$ decomp-magician addcmul

aten.addcmul.default  [table, inductor-kept]
├── aten.mul.Tensor  [table, inductor-kept]  x2
│   └── prims.mul.default  [leaf]
└── aten.add.Tensor  [table]
    ├── prims.mul.default  [leaf]
    └── prims.add.default  [leaf]

6 ops (3 table, 3 leaf) · 2 inductor-kept
```

Op names are resolved flexibly — you don't need the full qualified name:

```
decomp-magician addcmul               # bare name
decomp-magician aten.addcmul.default  # fully qualified
decomp-magician "aten::addcmul"       # C++ format (from error messages)
decomp-magician batch_norm            # substring match
```

## What does torch.compile actually run?

The most common question: "if I compile this op, what hits the backend?" Use `--compile --leaves`:

```
$ decomp-magician _native_batch_norm_legit --compile --leaves

aten._native_batch_norm_legit.default decomposes to:
  aten.mul.Tensor           x7  [inductor-kept]
  prims.mul.default         x4
  prims.add.default         x4
  aten.unsqueeze.default    x4  [inductor-kept]
  aten.squeeze.dims         x3  [inductor-kept]
  aten.copy_.default        x2
  aten.var_mean.correction  x1  [inductor-kept]
  aten.rsqrt.default        x1  [inductor-kept]
  aten.sub.Tensor           x1  [inductor-kept]
9 unique ops, 27 total instances
```

`--compile` treats inductor-kept ops as leaves (Inductor has direct lowerings for these, so it doesn't decompose them further). `--leaves` shows the flat frontier with propagated counts instead of the tree.

Note: the tool traces decompositions using PyTorch's raw `decomposition_table`. The leaf frontier under `--compile` is correct, but the intermediate decomposition path may differ for ops where Inductor uses custom decompositions.

## Understanding the tree

To see *why* an op decomposes the way it does, use the tree view:

```
$ decomp-magician _native_batch_norm_legit --depth 1

aten._native_batch_norm_legit.default  [table]
├── aten.var_mean.correction  [table, inductor-kept]
├── aten.add.Tensor           [table]  x4
├── aten.rsqrt.default        [table, inductor-kept]
├── aten.sub.Tensor           [table, inductor-kept]
├── aten.mul.Tensor           [table, inductor-kept]  x7
├── aten.squeeze.dims         [table, inductor-kept]  x3
├── aten.copy_.default        [leaf]  x2
└── aten.unsqueeze.default    [table, inductor-kept]  x4

9 ops (8 table, 1 leaf) · 6 inductor-kept
```

### Annotations

Each op is classified:

- `[table]` — explicit decomposition in `torch._decomp.decomposition_table`
- `[CIA]` — CompositeImplicitAutograd kernel
- `[both]` — has both table and CIA decompositions (table takes precedence)
- `[leaf]` — no decomposition; runs as a fused kernel
- `inductor-kept` — Inductor skips this decomposition and uses a direct lowering
- `untraceable` — decomposition exists but could not be traced on meta tensors

## Flags

| Flag | Description |
|------|-------------|
| `--depth N` | Maximum recursion depth (-1 for unlimited, default) |
| `--compile` | Treat inductor-kept ops as leaves |
| `--leaves` | Show flat leaf frontier with propagated counts |
| `--dtensor` | Show DTensor sharding strategy coverage per op |
| `--json` | Output as JSON for scripting and CI (respects `--leaves`) |
| `--verbose` | Show full classification details (backends, tags, schema) |
| `--no-color` | Disable colored output (auto-detected; respects `NO_COLOR`) |

## How it works

A decomposition maps an operator to a multiset of operators. Iterating this map produces a tree whose leaves are the primitive ops that actually execute. The tool computes this tree by:

1. **Resolving** the user-provided name to an `OpOverload`
2. **Tracing** the decomposition on meta tensors under a `TorchDispatchMode` that records all op calls (no real computation occurs)
3. **Recursing** into each child op to build the full tree
4. **Classifying** each node (decomposition source, Inductor status, backend support)

## Limitations

- ~28% of decomposable ops cannot be traced on meta tensors (shape mismatches, missing defaults, data-dependent control flow). These are marked `[untraceable]` and `--leaves` warns when the frontier is incomplete.
- The tool uses the raw `decomposition_table`, not Inductor's table. Inductor has custom decompositions for ~111 ops that may produce different intermediate ops. The `--compile` flag correctly identifies which ops are terminal for Inductor, but the path to get there may differ.
- Only the `aten` namespace is searched for substring matches. Ops in other namespaces (`prims`, `quantized`) require exact names.
