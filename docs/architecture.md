# Architecture: decomposition-magician

## Scope

One CLI command. No TUI, no dependencies beyond PyTorch.

```
decomp-magician <op_name> [--depth N] [--compile] [--leaves] [--reverse] [--mermaid] [--dot] [--dtensor] [--json] [--verbose]
decomp-magician --stats [--compile] [--json]
```

Prints a decomposition tree to stdout with per-op annotations.

```
$ decomp-magician aten._native_batch_norm_legit

aten._native_batch_norm_legit.default  [table]
├── aten.var_mean.correction  [table, inductor-kept]
├── aten.add.Tensor           [table]  x4
├── aten.rsqrt.default        [table, inductor-kept]
├── aten.sub.Tensor           [table, inductor-kept]
├── aten.mul.Tensor           [table, inductor-kept]  x7
├── aten.squeeze.dims         [table, inductor-kept]  x3
├── aten.copy_.default        [leaf]  x2
└── aten.unsqueeze.default    [table, inductor-kept]  x4
```

With `--dtensor`:
```
$ decomp-magician aten._native_batch_norm_legit --dtensor

aten._native_batch_norm_legit.default  [table]  dtensor: ok (via decomp)
├── ...
├── aten.copy_.default        [leaf]  dtensor: MISSING  x2
└── ...
```

---

## Two Decomposition Mechanisms

A critical finding from the investigation: **most decompositions come from
the explicit table, not CIA**. The numbers:

| Mechanism | Op count |
|-----------|----------|
| Explicit table only (no CIA) | 968 |
| CIA only (not in table) | 180 |
| Both table and CIA | 159 |
| **Total decomposable** | **1307** |

The tool must handle both. This matches how DTensor's
`DecompShardingStrategy.has_decomp` works:

```python
def has_decomp(op):
    return op in decomposition_table or op._can_decompose()
```

When building a tree, the tool tries the explicit table first (it has more
entries and is what `torch.compile` uses), then falls back to CIA.

---

## Op Name Resolution

Users shouldn't have to type `aten._native_batch_norm_legit.default`.
The tool resolves names progressively:

1. **Exact match**: `aten.addcmul.default` → direct lookup
2. **Default overload**: `aten.addcmul` → try `.default`, then first overload
   (some ops like `aten.softmax` have no `.default` — their primary overload
   is `.int`)
3. **Namespace prefix**: `addcmul` → try `aten.addcmul`
4. **Substring search**: `batch_norm` → list all ops containing the substring

If resolution is ambiguous, print the candidates and exit. Don't guess.
If an op has multiple overloads and none was specified, list them.

---

## Module Structure

```
decomp_magician/
├── __init__.py          # empty
├── __main__.py          # CLI entry point, argument parsing, output formatting
├── tree.py              # decomposition tree construction via RecordingMode (with trace cache)
├── classify.py          # op classification (CIA/CEA/table/leaf, backends, tags)
├── resolve.py           # op name resolution from user input to OpOverload
├── graph.py             # Mermaid and Graphviz DOT export
├── reverse.py           # reverse lookup: find ops that decompose into a target
└── stats.py             # bulk statistics across all decomposable ops
```

Seven files. ~1200 LOC total.

### `resolve.py` — Op Name Resolution

```python
def resolve_op(name: str) -> OpOverload | list[str]:
    """
    Resolve a user-provided name to an OpOverload.
    Returns the op, or a list of candidate names if ambiguous.
    """
```

Walks the `torch.ops.aten` namespace. Tries exact match, then default
overload, then substring search. Returns candidates on ambiguity.

### `tree.py` — Decomposition Tree Construction

Core data structure:

```python
@dataclass
class DecompNode:
    op: OpOverload
    children: list[DecompNode]
    count: int                 # times this op appears in parent's decomposition
    classification: OpClass    # from classify.py
    traceable: bool            # False if meta tensor tracing failed
    error: str | None          # exception message if untraceable
```

Core function:

```python
def build_tree(op: OpOverload, depth: int = -1) -> DecompNode
```

**How it works** (verified against real PyTorch — see below):

1. Look up the decomposition function: check `decomposition_table` first,
   then `op.decompose` for CIA.
2. Create meta tensors from `op._schema.arguments` — parse types, use
   default shapes (`[2, 3, 4, 4]` for 4D, `[3]` for 1D, etc.).
3. Run the decomposition function under `RecordingMode` — a 15-line
   `TorchDispatchMode` that logs every `OpOverload` call.
4. Count occurrences of each op. For each unique op, recursively build
   subtrees if it has a decomposition and depth allows.
5. If meta tensor creation or tracing fails, mark the node as
   `traceable=False` with the error message.

**RecordingMode** — verified working:

```python
class RecordingMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: list[OpOverload] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if isinstance(func, OpOverload):
            self.ops.append(func)
        return func(*args, **(kwargs or {}))
```

**Empirical validation**: This approach was tested on the actual PyTorch
codebase. For `aten._native_batch_norm_legit.default`, the RecordingMode
correctly captures:

```
aten.var_mean.correction  x1    aten.squeeze.dims     x3
aten.add.Tensor           x4    aten.copy_.default    x2
aten.rsqrt.default        x1    aten.unsqueeze.default x4
aten.sub.Tensor           x1    aten.mul.Tensor       x7
```

This matches manual source reading and confirms `squeeze.dims` x3.

**Meta tensor creation** — the hardest part:

Parse `op._schema.arguments`. For each argument:
- `Tensor` → `torch.empty(default_shape, device="meta")`
- `Optional[Tensor]` → same, or `None` if optional
- `bool`, `int`, `float` → reasonable defaults (`True`, `0`, `1e-5`)
- `int[]` → `[0]` or `[1]`

Default shape selection by arg name heuristic:
- Most tensors: `[2, 3, 4, 4]` (4D, batch-like).
- Args named weight/bias/mean/var/scale/zero_point: `[3]` (1D, channel-like).
- Shape can affect which branches are taken in data-dependent control flow
  (rare in decompositions). Optional scalars get reasonable defaults
  (0 for int, 1.0 for float) to avoid None-related failures.

If creation fails, the node is marked untraceable. No crash.

### `classify.py` — Op Classification

```python
@dataclass
class OpClass:
    decomp_type: str              # "CIA", "table", "both", "leaf"
    has_backend: dict[str, bool]  # {"cpu": True, "cuda": True, ...}
    tags: list[str]               # ["maybe_aliasing_or_mutating", ...]
    is_mutable: bool
    has_alias_info: bool
    inductor_excluded: bool
    dtensor_strategy: str | None  # "registered", "decomp-fallback", "missing", None
```

| Field | How to compute |
|-------|----------------|
| `decomp_type` | Check `decomposition_table` and `op._can_decompose()` |
| `has_backend` | `torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), key)` for each backend |
| `tags` | `op.tags` |
| `is_mutable` | `op._schema.is_mutable` |
| `has_alias_info` | `any(arg.alias_info for arg in op._schema.arguments)` |
| `inductor_excluded` | In raw table but absent from Inductor's `select_decomp_table()` |
| `dtensor_strategy` | Check `ShardingPropagator` dicts (lazy import, only with `--dtensor`) |

### `graph.py` — Graph Export

Mermaid flowchart and Graphviz DOT export. Color-coded by classification
(gray=leaf, green=decomposed, yellow=inductor-kept, red=untraceable).
Shape encodes status (square=terminal, rounded=decomposable, trapezoid=untraceable).

### `reverse.py` — Reverse Lookup

Scans all ~730 non-out table ops, building each tree and searching for a
target op. Returns producers sorted by count with shallowest target depth.
CIA-only ops excluded due to C-level SIGFPE crashes in `op.decompose()`.

### `stats.py` — Bulk Statistics

Iterates all decomposable ops, classifying and tracing each. Reports
traceable/untraceable counts, leaf op frequency, and deepest chains.
Invariant: `traceable + untraceable == total_non_out` (no gaps).

### `__main__.py` — CLI and Output

`argparse` with multiple output modes: tree (default), `--leaves`, `--reverse`,
`--stats`, `--mermaid`, `--dot`, `--json`. ANSI color auto-detected (respects
`NO_COLOR` env var and `--no-color` flag). Mutually exclusive output modes
are validated with clear error messages.

---

## What We Don't Build

- **No op schema inference engine.** Default shapes work for ~74% of ops.
  The rest are marked untraceable. Users can file issues for specific ops.
- **No config files.** CLI args are the interface.
- **No DTensor strategy validation.** We report existence, not correctness.

---

## Implementation Order

1. **`resolve.py`** — op name lookup. Immediately useful standalone.
2. **`classify.py`** — pure lookups, no tracing. Testable against known ops.
3. **`tree.py`** — the core. Meta tensor creation + RecordingMode.
   Test with `_native_batch_norm_legit` (motivating example),
   `addcmul` (simple), `dropout` (has CIA).
4. **`__main__.py`** — tree formatting and CLI.
5. **DTensor integration** — `--dtensor` flag with lazy imports.

---

## Packaging

```
decomposition-magician/
├── decomp_magician/        # Python package (import decomp_magician)
├── tests/                  # pytest suite (136 tests)
├── docs/                   # design docs and investigation
├── pyproject.toml           # minimal: name, version, requires pytorch
└── README.md                # value prop, install, usage examples
```

Install: `pip install .` or `pip install decomposition-magician`
Run: `decomp-magician batch_norm`

`pyproject.toml` declares PyTorch as the only dependency. DTensor is
optional — the tool works without it; `--dtensor` requires
`torch.distributed` to be available.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Meta tensor creation fails for complex ops | Mark as "untraceable", don't crash. `--verbose` shows the exception. |
| CIA decomposition has side effects | Meta device: no real computation, no GPU allocation. |
| PyTorch internal APIs change | Minimal API surface (~10 calls). 4 files, ~500 LOC. Easy to update. |
| Recursive decomposition is infinite | Cycle detection via ancestor tracking. `--depth` provides explicit cap. |
| DTensor not installed | `--dtensor` is opt-in. Without it, zero DTensor imports. |
| Op name resolution is wrong | Show candidates on ambiguity. Never guess silently. |
