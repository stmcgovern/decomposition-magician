# Roadmap

## Shipped

### v0.1.0 — Core tree inspection
- Decomposition tree with recursive tracing
- Fuzzy op name resolution
- `--compile`, `--leaves`, `--depth`
- `--reverse` lookup
- `--stats` bulk statistics

### v0.2.0 — Visualization
- Terminal color (bold ops, dim leaves, yellow inductor-kept, red untraceable)
- Summary line after tree output
- `--json` output for scripting and CI
- `--no-color` flag with `NO_COLOR` env var support

### v0.3.0 — Analysis & export
- `--mermaid` and `--dot` graph export
- `--diff` between full/compile modes or between ops
- `--target-opset core_aten` coverage checking
- `--model` exported model analysis
- `--dispatch-table`, `--mode-sensitivity`, `--adiov` dispatch introspection
- `--pure` purity analysis
- `--backward` gradient op tracing

---

## Future considerations

### `--table {raw,inductor,export}`

Show decomposition trees under different table contexts. The raw table (current default) shows maximal decomposition. The Inductor table shows what `torch.compile` actually does. The export table shows the core ATen opset.

Blocked by: Inductor's `select_decomp_table()` contains type-conditional decomps that return `NotImplemented` and some entries that segfault on meta tensors.

### DAG mode

Deduplicate repeated subtrees. Currently ~1.1x duplication for typical ops, so low priority unless deeper decompositions become common.
