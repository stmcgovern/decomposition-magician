# decomposition-magician

Inspect PyTorch operator decomposition trees from the command line.

PyTorch has ~1300 decomposable operators spread across two mechanisms (CompositeImplicitAutograd kernels and explicit decomposition tables), but no built-in way to see what a given op decomposes into. This tool fills that gap.

## Install

```
pip install .
```

## Usage

```
python -m decomp_magician <op_name> [--depth N] [--dtensor] [--json] [--verbose] [--no-color]
```

Op names are resolved flexibly — you don't need the full qualified name:

```
python -m decomp_magician addcmul               # bare name
python -m decomp_magician aten.addcmul          # namespace.op
python -m decomp_magician aten.addcmul.default  # fully qualified
python -m decomp_magician "aten::addcmul"       # C++ format (from error messages)
python -m decomp_magician batch_norm            # substring match
```

### Example

```
$ python -m decomp_magician _native_batch_norm_legit --depth 1

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

- `[table]` — has an explicit decomposition in `torch._decomp.decomposition_table`
- `[CIA]` — has a CompositeImplicitAutograd kernel
- `[both]` — has both table and CIA decompositions
- `[leaf]` — no decomposition; runs as a fused kernel
- `inductor-kept` — Inductor skips this decomposition and uses a direct lowering
- `untraceable` — decomposition exists but could not be traced on meta tensors

### Flags

- `--depth N` — Maximum recursion depth (-1 for unlimited, default)
- `--dtensor` — Show DTensor sharding strategy coverage per op
- `--json` — Output as JSON for scripting and CI integration
- `--verbose` — Show full classification details (backends, tags, schema properties)
- `--no-color` — Disable colored output (auto-detected; respects `NO_COLOR` env var)

### DTensor coverage

With `--dtensor`, each op in the tree is annotated with its sharding strategy status:

```
$ python -m decomp_magician _native_batch_norm_legit --depth 1 --dtensor

aten._native_batch_norm_legit.default  [table]  dtensor: ok (via decomp)
├── ...
├── aten.copy_.default        [leaf]  dtensor: MISSING  x2
└── ...
```

## How it works

1. **Resolve** the user-provided name to an `OpOverload` (exact match, default overload, namespace prefix, or substring search)
2. **Classify** the op (CIA / explicit table / both / leaf, backend support, Inductor exclusion status)
3. **Trace** the decomposition by running it on meta tensors under a `TorchDispatchMode` that records all op calls
4. **Recurse** into each child op to build the full tree
