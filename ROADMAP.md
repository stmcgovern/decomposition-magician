# Roadmap

## v0.2.0 — Visualization (shipped)

- Terminal color (bold ops, dim leaves, yellow inductor-kept, red/green dtensor)
- Summary line after tree output
- `--json` output for scripting and CI
- `--no-color` flag with `NO_COLOR` env var support

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
