# Roadmap

## v0.2.0 — Visualization

### Terminal color

ANSI colors with no new dependencies. Auto-detect tty and respect `NO_COLOR` env var.

| Element | Color |
|---------|-------|
| Op names | bold |
| `[leaf]` | dim gray |
| `inductor-kept` | yellow |
| `dtensor: MISSING` | red |
| `dtensor: ok` | green |
| `[untraceable]` | red |

Flags: `--no-color` to force plain output.

### Summary line

Print a one-line summary after the tree:

```
14 ops (8 table, 1 CIA, 2 leaf) · 7 inductor-kept · 2 dtensor missing
```

### `--json` output

Machine-readable tree for scripting and CI integration.

```
python -m decomp_magician batch_norm --depth 1 --json | jq '.children[] | select(.inductor_kept)'
```

---

## v0.3.0 — Export formats

### `--format mermaid`

Mermaid graph that renders in GitHub markdown. Useful for documentation and PRs.

### `--format dot`

Graphviz DOT output for high-quality rendered diagrams.

---

## Future considerations

### `--table {raw,inductor,export}`

Show decomposition trees under different table contexts. The raw table (v0.1.0 default) shows maximal decomposition. The Inductor table shows what `torch.compile` actually does. The export table shows the core ATen opset.

Blocked by: Inductor's `select_decomp_table()` contains type-conditional decomps that return `NotImplemented` and some entries that segfault on meta tensors. Requires safe iteration with per-op error handling.

### DAG mode

Deduplicate repeated subtrees. Currently ~1.1x duplication for typical ops, so low priority unless deeper decompositions become common.

### `--diff`

Compare decomposition trees between PyTorch versions or between table contexts. Useful for tracking decomposition changes across releases.
