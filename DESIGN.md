# Design: decomposition-magician v1.0

## Core Insight

Every feature in this tool is a query over the **decomposition relation**: a function `D: Op → {child ops}` that is fixed for a given PyTorch installation. The relation has two independent aspects:

1. **Edges** — which ops does this op trace to? (cached in `_trace_cache`)
2. **Node properties** — what type is this op? DTensor strategy? Dispatch behavior? (computed by `classify()`)

These are currently split across two modules with independent caches. `compute_stats` classifies each op *twice* — once for its own counting, once inside `build_tree`. `reverse_lookup` knows the forward edges exist (in `_trace_cache`) but can't query them backward — it builds 700 trees and searches each one. The information is there; the code can't express the query.

**The decomposition graph is the right central object** — not for performance, but because it makes the code say what it means. But it must be *minimal*: a thin object that co-locates the edge cache and classification cache, nothing more. No premature methods for queries that don't exist yet.

## Design Principles

1. **The graph is a concept, not a framework.** ~25 lines: `children()`, `error()`, `classify()`. No `populate()`, `parents()`, or `all_ops` until a query needs them. The simplest object that expresses "these two caches are the same thing."

2. **Don't add abstraction ahead of need.** `compile` + `depth` compose as two parameters. Don't build a `View` protocol for two booleans. Don't build `parents()` until reverse lookup is rewritten. Don't build `TraceError` enum for heterogeneous exception strings.

3. **Measure before optimizing.** Reverse takes 0.15s. Stats take 0.85s. The graph isn't about making these faster — it's about making the code express the right concept. Performance gains are incidental.

4. **The real value is honest reporting and a clean library API.** A PyTorch developer wants `from decomp_magician import build_tree, compute_stats` with typed results. They want stats that say "87% covered *of the 74% we can trace*." They don't want to parse CLI text.

5. **Things that belong together stay together.** `DecompGraph` lives in `tree.py` alongside `DecompNode` and `build_tree`. No `node.py`, no `types.py`, no `view.py`.

## Final File Structure

```
decomp_magician/
    __init__.py           # Public API exports
    __main__.py           # Entry point: from .cli import main; main()

    # --- Core ---
    tree.py               # DecompGraph, DecompNode, build_tree (the central module)
    classify.py           # OpClass, DecompType, DtensorStrategy
    dispatch.py           # DispatchInfo, AutogradType, DispatchEntry
    resolve.py            # resolve_op

    # --- Queries ---
    stats.py              # compute_stats, StatsResult, confidence qualifiers
    diff.py               # compute_diff
    reverse.py            # reverse_lookup, ReverseEntry
    opset.py              # check_opset_coverage

    # --- Presentation ---
    format.py             # ALL text rendering (extracted from __main__.py)
    export.py             # Mermaid, DOT, JSON (renamed from graph.py)
    cli.py                # Argparse + dispatch (thin)
```

---

## Phase 1: Type Safety + Minimal Graph

**Goal**: Finish StrEnum work, introduce the minimal `DecompGraph`, type the `reverse_lookup` return.

### DecompGraph (~25 lines, in tree.py)

```python
class DecompGraph:
    """The decomposition relation for the current PyTorch installation.

    Co-locates edge tracing and classification in a single cache.
    Lazily computed per op. Once resolved, results don't change.
    """

    def __init__(self, *, dtensor: bool = False):
        self._dtensor = dtensor
        self._edges: dict[OpOverload, tuple[tuple[OpOverload, int], ...]] = {}
        self._errors: dict[OpOverload, str] = {}
        self._classes: dict[OpOverload, OpClass] = {}

    def children(self, op: OpOverload) -> tuple[tuple[OpOverload, int], ...]:
        """Direct decomposition children with multiplicity."""
        self._ensure(op)
        return self._edges.get(op, ())

    def error(self, op: OpOverload) -> str | None:
        """Trace error, or None if traceable."""
        self._ensure(op)
        return self._errors.get(op)

    def classify(self, op: OpOverload) -> OpClass:
        """Op classification (cached)."""
        if op not in self._classes:
            self._classes[op] = _classify_op(op, dtensor=self._dtensor)
        return self._classes[op]

    def _ensure(self, op: OpOverload) -> None:
        """Trace op if not already cached."""
        if op in self._edges or op in self._errors:
            return
        result = _trace_decomp_uncached(op)
        if isinstance(result, str):
            self._errors[op] = result
        else:
            counts = Counter(result)
            self._edges[op] = tuple(counts.items())
```

The existing `_trace_cache` and module-level `_trace_decomp()` are replaced by this. The tracing machinery (`_make_meta_args`, `_try_trace`, `_trace_decomp_uncached`, `_RecordingMode`, etc.) stays as module-level functions in `tree.py` — unchanged.

### Updated build_tree

```python
def build_tree(
    op: OpOverload, *,
    compile: bool = False, depth: int = -1, dtensor: bool = False,
    graph: DecompGraph | None = None,
    _ancestors: frozenset[str] | None = None,
) -> DecompNode:
    if graph is None:
        graph = DecompGraph(dtensor=dtensor)
    if _ancestors is None:
        _ancestors = frozenset()

    cls = graph.classify(op)

    if cls.decomp_type == DecompType.LEAF or depth == 0:
        return DecompNode(op=op, classification=cls)
    if compile and cls.inductor_kept:
        return DecompNode(op=op, classification=cls)

    op_name = op.name()
    if op_name in _ancestors:
        return DecompNode(op=op, classification=cls,
                          traceable=False, error="cycle detected")

    error = graph.error(op)
    if error is not None:
        return DecompNode(op=op, classification=cls,
                          traceable=False, error=error)

    next_depth = depth - 1 if depth > 0 else -1
    children = tuple(
        replace(
            build_tree(child_op, compile=compile, depth=next_depth,
                       graph=graph, _ancestors=_ancestors | {op_name}),
            count=count,
        )
        for child_op, count in graph.children(op)
    )
    return DecompNode(op=op, children=children, classification=cls)
```

Default `graph=None` means every existing call site works unchanged. But callers that build multiple trees (stats, reverse) can share a graph — one classification per op, not two.

### AutogradType StrEnum (in dispatch.py)

```python
class AutogradType(StrEnum):
    AUTOGRAD_KERNEL = "autograd_kernel"
    MATH_KERNEL = "math_kernel"
    FALLTHROUGH = "fallthrough"
    OTHER = "other"
    NONE = "none"
```

### ReverseEntry (in reverse.py)

```python
@dataclass(frozen=True)
class ReverseEntry:
    op: str
    count: int
    depth: int

def reverse_lookup(...) -> tuple[ReverseEntry, ...]:
```

### Public is_out_variant (in reverse.py)

Rename `_is_out_variant` → `is_out_variant`. Already imported by `stats.py` — the underscore is a lie.

### Invariant

After this phase: `_trace_cache` is gone. `DecompGraph` owns all edge and classification caches. `build_tree` reads from graph, never calls `classify()` directly. Every StrEnum-able value uses an enum.

### Tests

- `DecompGraph` unit tests: children, error, classify
- `build_tree(graph=graph)` produces identical trees to current `build_tree()`
- Existing tests pass (default `graph=None` preserves behavior)
- `ReverseEntry` replaces dict access in reverse tests

---

## Phase 2: Presentation Split

**Goal**: `__main__.py` → `cli.py` + `format.py` + `export.py`. Makes the tool a library.

### format.py

```python
@dataclass(frozen=True)
class FormatConfig:
    color: bool = False
    show_dispatch: bool = False
    show_mode_sensitivity: bool = False

def format_tree(node: DecompNode, cfg: FormatConfig, ...) -> str: ...
def format_leaves(node: DecompNode, cfg: FormatConfig, ...) -> str: ...
def format_summary(node: DecompNode, cfg: FormatConfig) -> str: ...
def format_stats(stats: StatsResult, cfg: FormatConfig) -> str: ...
def format_reverse(entries: tuple[ReverseEntry, ...], target: str, cfg: FormatConfig) -> str: ...
def format_diff(diff: DecompDiff, cfg: FormatConfig) -> str: ...
```

Every function: data in, string out. No `print()`, no globals.

### export.py (renamed from graph.py)

```python
def to_mermaid(node: DecompNode) -> str: ...
def to_dot(node: DecompNode) -> str: ...
def tree_to_dict(node: DecompNode) -> dict: ...
def leaves_to_dict(node: DecompNode) -> dict: ...
def stats_to_dict(stats: StatsResult) -> dict: ...
```

### cli.py

```python
def main() -> int:
    args = _parse_args()
    cfg = FormatConfig(
        color=_should_use_color() and not args.no_color,
        show_dispatch=args.dispatch_table,
        show_mode_sensitivity=args.mode_sensitivity,
    )
    graph = DecompGraph(dtensor=args.dtensor)
    # graph is now shared across all _run_* calls
    ...
```

Each `_run_*` function: compute → format → print. The graph is created once in `main()` and passed through.

### __init__.py

```python
from decomp_magician.tree import DecompGraph, DecompNode, build_tree, collect_leaf_counts, op_display_name
from decomp_magician.classify import OpClass, DecompType, DtensorStrategy, classify
from decomp_magician.dispatch import DispatchInfo, AutogradType, get_dispatch_info
from decomp_magician.stats import StatsResult, DtensorStats, compute_stats
from decomp_magician.diff import DecompDiff, compute_diff, compute_diff_ops
from decomp_magician.reverse import ReverseEntry, reverse_lookup
from decomp_magician.opset import OpsetCoverage, check_opset_coverage
from decomp_magician.resolve import resolve_op
from decomp_magician.export import to_mermaid, to_dot
```

### Invariant

- `format.py` never calls `print()`, has zero imports from `cli.py`
- `cli.py` has zero ANSI codes, zero data computation
- No module-level mutable state anywhere except `DecompGraph` instances (which are referentially transparent)
- `from decomp_magician import build_tree, compute_stats` works

### Tests

- Existing CLI tests pass unchanged
- New format tests with `FormatConfig(color=False)`
- Import smoke test for all public names

---

## Phase 3: Confidence and Honest Reporting

**Goal**: Every aggregate statistic is qualified by its confidence. This is the tool's competitive advantage.

### StatsResult additions

```python
@dataclass(frozen=True)
class StatsResult:
    ...  # existing fields

    @property
    def traceability_rate(self) -> float:
        denom = self.traceable + self.untraceable + self.classify_errors
        return self.traceable / denom if denom > 0 else 0.0

    @property
    def leaf_count_is_lower_bound(self) -> bool:
        return self.untraceable > 0
```

### Updated stats display

```
  Traceable:           539/733 (74%)
  Untraceable:         194 (leaf counts are lower bounds)

  Top leaf ops  (across 539 traceable ops):
    aten.expand.default   422 ██████████...

  DTensor coverage:    489/539 traceable trees fully covered (87%)
                       194 ops could not be verified
```

### CompatibilityReport

```python
@dataclass(frozen=True)
class CompatibilityReport:
    pytorch_version: str
    inductor_table_available: bool
    dtensor_available: bool
    warnings: tuple[str, ...]
```

Included in `--json` output. Built cheaply (just import checks).

### Update compute_stats to use graph

```python
def compute_stats(compile=False, dtensor=False, graph: DecompGraph | None = None) -> StatsResult:
    if graph is None:
        graph = DecompGraph(dtensor=dtensor)
    for op in all_ops:
        cls = graph.classify(op)         # classified once
        ...
        node = build_tree(op, compile=compile, graph=graph)  # reuses classification
```

This is where the graph pays off concretely: no double-classification.

### Invariant

Every number in stats output has a visible denominator. JSON includes `traceability_rate` and `compatibility`.

---

## Phase 4: Library Polish

**Goal**: Docstrings, README "Library usage" section, clean public API.

### Changes

- Docstrings on every public function
- README section showing programmatic use
- Version bump to 1.0

---

## What We Explicitly Defer

| Idea | Why defer | When to revisit |
|------|-----------|-----------------|
| `View` protocol | Two parameters compose fine | When a third stopping condition is needed |
| `graph.parents()` / reverse transpose | reverse_lookup takes 0.15s | When someone needs fast reverse queries in a library context |
| `graph.populate()` / `all_ops` | Not needed until stats is rewritten as a graph walk | When stats performance matters |
| `TraceError` enum | Free-form strings preserve actual PyTorch errors | When downstream code needs to branch on error category |
| `TraceProvenance` | No user has reported shape-dependent confusion | When tracing confidence needs to be per-op |
| Disk serialization | Tool starts in ~2s (PyTorch import dominates) | When startup time matters for a specific workflow |

These are natural extensions of the graph — they have a *place* to go when needed. The graph prepared the structure without building the machinery.

---

## Implementation Order

```
Phase 1: Type Safety + Minimal Graph     1-2 sessions    tree.py + dispatch.py + reverse.py
Phase 2: Presentation Split              2-3 sessions    cli.py + format.py + export.py
Phase 3: Confidence + Honest Reporting   1 session       stats.py + format.py
Phase 4: Library Polish                  1 session       __init__.py + docstrings + README
```

## What This Enables

```python
from decomp_magician import (
    DecompGraph, build_tree, compute_stats, resolve_op,
    to_mermaid, reverse_lookup,
)

# Shared graph across queries — classify once, query many times
graph = DecompGraph(dtensor=True)
op = resolve_op("softmax")
tree = build_tree(op, compile=True, graph=graph)
print(to_mermaid(tree))

# Honest stats
stats = compute_stats(dtensor=True, graph=graph)
print(f"{stats.traceability_rate:.0%} of ops traceable")

# Typed results
for entry in reverse_lookup("aten.mul.Tensor"):
    print(f"{entry.op}: {entry.count} instances at depth {entry.depth}")
```

The graph is the central object — not because it's fast, but because it makes the code say what it means.
