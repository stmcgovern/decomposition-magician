# Decompositions in PyTorch: A Structural Analysis

## 0. Setup and Definitions

Let $\mathcal{O}$ denote the set of ATen operators. A **decomposition** of an operator $f \in \mathcal{O}$ is a rewriting $D(f) = (g_1, g_2, \ldots, g_k)$ where each $g_i \in \mathcal{O}$ and the composition is semantically equivalent to $f$ on the forward pass. Decomposition is **lossy**: the map $D$ has no left inverse. Given $D(f)$, one cannot recover $f$, because many distinct operators may share the same expansion into primitives.

There are exactly two mechanisms by which decompositions enter the system:

1. **CompositeImplicitAutograd (CIA)** — a kernel registered at the `CompositeImplicitAutograd` dispatch key. The name "implicit" refers to the fact that no explicit backward formula is written; autograd is derived by tracing through the decomposition. This is a property of the *operator itself*, declared in `native_functions.yaml` or via `py_impl`.

2. **Explicit decomposition tables** — Python functions registered via `@register_decomposition` into named tables (`pre_autograd`, `post_autograd`, `meta`), consulted during `torch.compile` and export tracing. These are an *external overlay* applied by the compilation pipeline.

The distinction matters: CIA decompositions are structural (they affect eager dispatch), while table decompositions are opportunistic (they only affect tracing). A third category — `CompositeExplicitAutograd` (CEA) — is sometimes conflated with decomposition but is distinct: CEA provides a *shared implementation* across backends with a hand-written backward, not a decomposition into primitives. The kernel runs as a unit; autograd does not trace through it.

> `torch/_decomp/__init__.py:37-43` — the three global tables.
> `torch/_ops.py:901-918` — `OpOverload.decompose()` checks for CIA kernel.

---

## 1. The Dispatch Resolution Order

The function `resolve_key` (`torch/_ops.py:211-261`) determines which kernel runs for a given operator and dispatch key. The resolution is a priority chain, but — and this is important to state precisely — every step has guard conditions. The chain is:

```
(i)    Direct registration for key k
(ii)   CompositeExplicitAutogradNonFunctional   [if k ∈ alias set or k = Undefined]
(iii)  CompositeExplicitAutograd                 [if k ∈ alias set or k = Undefined]
(iv)   CompositeImplicitAutogradNestedTensor     [if k ∈ alias set AND k ≠ Undefined AND ¬has_backend]
(v)    CompositeImplicitAutograd                 [if k ∈ alias set or k = Undefined AND ¬has_backend]
(vi)   Autograd
(vii)  FuncTorchBatchedDecomposition
(viii) Backend fallback
```

Step (v) is the decisive one for decompositions. The condition has two parts:

1. **Alias membership**: the dispatch key `k` must be `Undefined` or included in the CIA alias set (backend keys ∪ autograd dispatch keys). This is a structural prerequisite — CIA only *covers* certain keys.

2. **No backend kernel**: define

$$\text{has\_backend}(f, k) \;:=\; f \text{ has a kernel for any backend key derived from } k,\;\text{OR } f \text{ has a CEA kernel}$$

CIA fires only if $\lnot\,\text{has\_backend}(f, k)$.

Note that `has_backend` includes CEA registrations (`_ops.py:229`). This means a CEA kernel *also* blocks CIA from firing — CEA wins at step (iii) anyway, but the `has_backend` computation makes this explicit for the CIA gate.

There is also a safety guard: if `k = AutogradOther` and the op has kernels for any `autogradother_backends`, an ambiguity error is raised rather than falling through to CIA (`_ops.py:242-245`). This prevents silent dispatch ambiguity for operators that partially cover the `AutogradOther` umbrella.

This makes decomposition **conditional on context**: the same operator may decompose on one device and run as a fused kernel on another.

> `torch/_ops.py:227-247` — the `has_backend_kernel` check and CIA gate.

### Formal status

The Coq formalization at `/workspaces/torch_formal` models the alias expansion sets and precedence order:

```coq
(* theories/Registration/AliasKeys.v:52-62 *)
Definition alias_expansion (ak : AliasKey) : FKeySet :=
  match ak with
  | CompositeImplicitAutograd =>
    fun k => is_backend_key k || is_autograd_dispatch_key k
  | CompositeExplicitAutograd =>
    fun k => is_backend_key k
  ...
```

Theorem `alias_key_precedence` (AliasKeys.v:159) proves that direct registrations always override alias-expanded ones. Example `direct_overrides_cia` (AliasKeys.v:184) gives a concrete witness: a `Dense` kernel blocks CIA expansion for `Dense`.

**Gap**: The formalization does not yet encode the `has_backend_kernel` gate as a standalone predicate. Closing this gap would yield a theorem connecting the alias expansion model to `resolve_key`'s actual dispatch semantics. A candidate statement:

```coq
Theorem cia_fires_iff :
  forall (op : OpId) (k : FunctionalityKey) (reg : OpRegistration),
    cia_fires reg k = true <->
    alias_reg reg CompositeImplicitAutograd <> None /\
    direct_reg reg k = None /\
    (k = Undefined \/ alias_expansion CompositeImplicitAutograd k = true) /\
    has_backend_kernel reg k = false.
```

---

## 2. Why the Autograd Key Is Decisive

The autograd dispatch key sits above backend keys in the dispatch stack. When autograd is active, it is the key where CIA decompositions naturally fire — because the autograd key's resolution falls through to step (v) above, which either runs the CIA kernel (no backend) or skips to step (vi) and thence to the backend kernel.

This creates a **fidelity invariant** that `torch.compile` must replicate:

> **Fidelity Invariant**: The set of operators appearing in a compiled graph must match what eager execution would have produced after the autograd dispatch key resolves.

The function `autograd_would_have_decomposed` (`torch/utils/_python_dispatch.py:811-866`) encodes this test. For the first tensor argument, it checks whether a backend kernel exists for that tensor's device. If not, autograd would have decomposed — and so must the tracer.

Two failure modes justify why this invariant is necessary:

**Failure mode A (under-decomposition)**: In `inference_mode`, the autograd key is skipped entirely. If the functionalization layer does not compensate by decomposing, a high-level CIA op leaks into the graph. Inductor may have no lowering for it. The graph becomes untraceable or produces wrong results at runtime.

**Failure mode B (over-decomposition)**: If a fused CUDA kernel exists and the tracer decomposes anyway, the compiled graph contains primitives that are individually slower than the fused kernel. Compile-vs-eager equivalence breaks, and the user loses performance without recourse.

The implementation is deliberately minimal — check one device key, break on first tensor — reflecting that the predicate is structurally simple even though its consequences are pervasive. An important limitation: it only checks the *first* tensor's device. For multi-device scenarios (rare in practice), this could produce incorrect results.

> `torch/utils/_python_dispatch.py:845-866` — the implementation.
> `torch/_subclasses/functional_tensor.py:462-468` — the call site during tracing.

### Formal status

The formal model proves a sharp characterization of when `inference_mode` and `no_grad` produce different dispatch traces, which is the setting where failure mode A bites:

```coq
(* theories/Mode/ModeEquivalence.v:69-105 *)
Theorem mode_equivalence_iff :
  dispatch_trace (active_keys no_grad_included) table =
  dispatch_trace (active_keys inference_mode_included) table
  <-> effectively_mode_sensitive table = false.
```

The integration theorem `mode_determines_version` (EndToEnd.v:158-185) proves the consequence: when `ADInplaceOrView` fires under `no_grad` but is silent under `inference_mode`, inplace ops bump the version counter in one mode but not the other. This is the formal proof of the dispatch divergence that motivates `autograd_would_have_decomposed`.

---

## 3. The Full Decision Cascade During Tracing

During `torch.compile`, `FunctionalTensorMode.__torch_dispatch__` calls an internal `_can_decompose()` (`functional_tensor.py:423-468`). The decision is a cascade, applied in priority order:

| Priority | Condition | Decision | Rationale |
|----------|-----------|----------|-----------|
| 0 | Op tagged `maybe_aliasing_or_mutating` | Decompose | Aliasing must be statically known for functionalization |
| 1 | Schema has `alias_info` or `is_mutable` | Decompose | Same: functionalization requires static aliasing knowledge |
| 2a | Export, pre-dispatch | Preserve | Highest-level IR for portability |
| 2b | Export, post-dispatch | Decompose | Reduce to core ATen opset |
| 3 | Normal `torch.compile` | `autograd_would_have_decomposed()` | Fidelity invariant with eager |

Priorities 0 and 1 can force decomposition *even when autograd would not have decomposed*. This happens for ops whose CIA decomposition involves aliasing or mutation that cannot be determined statically from the schema alone. The tag `maybe_aliasing_or_mutating` marks exactly these ops — the decomposition is needed for functionalization correctness, not eager fidelity.

After functionalization, the proxy tensor layer (`proxy_tensor.py:2912-2924`) applies explicit decomposition tables — a second, independent pass. This is where `post_autograd` decompositions registered via `@register_decomposition` fire. The two-pass structure is load-bearing: functionalization handles CIA decompositions that involve mutation (they must decompose *before* functionalization removes mutation semantics), while proxy tensor handles the user/compiler-specified table on already-functionalized ops.

> `torch/_decomp/__init__.py:46-61` — `_should_decompose_because_unsafe_op`.
> `torch/_subclasses/functional_tensor.py:435-441` — aliasing/mutation check.

### Existing user-facing API

PyTorch exposes decomposition tables through:

- `torch._decomp.get_decompositions(ops, type="post_autograd")` — returns a dict of decomposition functions for requested ops. The docstring notes this API is experimental.
- `torch._decomp.core_aten_decompositions()` — returns the full core ATen decomposition table (delegates to `torch.export.default_decompositions()`).
- `OpOverload._can_decompose()` — checks if an op has a CIA kernel.
- `OpOverload.decompose(*args, **kwargs)` — explicitly runs the CIA decomposition.

There is no existing tool that answers "what does op X decompose into?" as a tree, or "what dispatch path is taken under mode Y?", or "what sharding strategies are needed to support this decomposition chain?". This is the gap `decomposition-magician` aims to fill.

---

## 4. When Decomposition Loses Semantic Content

Decomposition is an irreversible projection from a richer operator space to a sparser one. What is lost falls into distinct categories, and it is worth being precise about each because they have different practical consequences.

### 4.1 Fusion opportunity

An operator $f$ implemented as a single fused kernel performs $n$ FLOPs with $m$ bytes of memory traffic. Its decomposition $D(f) = (g_1, \ldots, g_k)$ performs the same $n$ FLOPs but requires $\sum_{i=1}^{k} m_i$ bytes, with $\sum m_i \gg m$ in general, because intermediate results are materialized to global memory between kernel launches.

For memory-bandwidth-bound operations — which dominate modern GPU workloads — this is the primary cost. Not extra FLOPs, but extra memory round-trips.

Inductor's exclusion list (`torch/_inductor/decomposition.py:110-141`) is a direct enumeration of cases where the fused form is superior. The list groups into four categories:

- **FMA ops** (`aten.addcmul`, `aten.addcdiv`): Decomposing into separate `mul`, `mul`, `add` loses the fused multiply-accumulate.
- **Direct lowerings** (`aten.split`, `aten.sum`, `aten.unbind`, `aten.squeeze`, `aten.glu`, `aten.silu`): Inductor has handwritten lowerings that outperform the decomposed form.
- **Numerical precision** (`aten.baddbmm`): The decomposition upcasts to fp32, introducing a performance regression the fused kernel avoids.
- **Graph structure** (`aten.select_scatter`, `aten.slice_scatter`): These must remain as named ops for the re-inplacing pass to recognize and optimize them. Decomposition destroys the pattern the pass matches against.

This list is empirically maintained, not derived from first principles. Each entry is a discovered regression. The absence of a principled criterion is itself informative — it means the decomposition-vs-preservation decision cannot be made locally by the op author; it depends on the downstream compiler's capabilities.

### 4.2 Sharding decisions (DTensor)

DTensor's `ShardingPropagator` (`torch/distributed/tensor/_sharding_prop.py:650-716`) implements a two-tier decision for each op:

1. **Registered strategy**: If a hand-written strategy function exists (in `op_strategy_funcs` or `op_single_dim_strategy_funcs`), use it. This provides a holistic, globally-informed sharding decision for the op.

2. **Decomposition fallback**: If no registered strategy exists, `DecompShardingStrategy` (`torch/distributed/tensor/_decompositions.py:152-331`) derives one by running the decomposition on meta tensors under `PlacementTrackingMode`. Each candidate input placement is propagated through the decomposed ops (which *do* have strategies), and the output placement is read off the result.

This fallback is clever — it means any CIA op automatically gets a sharding strategy for free — but it has structural limitations:

- Each decomposed op triggers an independent placement decision. If intermediate placements don't align, the propagator may insert redistributions (collective communications) between steps that a holistic strategy would avoid.
- The candidate set is the Cartesian product of per-input placements (`_get_candidate_placements`), which grows combinatorially. Errors during propagation (RuntimeError, KeyError, IndexError) silently skip candidates, potentially missing valid strategies.
- The decomposition must succeed on meta tensors. Operations with data-dependent control flow (`GuardOnDataDependentSymNode`) or unbacked symbolic shapes cause `propagate_strategy` to return `None`, forcing a hard failure. This is an orthogonal failure mode from missing strategies: the batch_norm unbacked test fails not because any op in the chain lacks a strategy, but because the tracer cannot propagate through the decomposition with unbacked symbols at all. A tool must distinguish "missing strategy in decomposition chain" from "decomposition chain untraceable under symbolic constraints".

The design doc's batch_norm example is the canonical case: `_native_batch_norm_legit` decomposes into ops including `aten.squeeze.dims`. If `squeeze.dims` lacks a registered sharding strategy, `DecompShardingStrategy` fails for the entire chain — one missing strategy in the decomposition tree blocks the whole op. This is the fragility that motivates understanding decomposition trees.

### 4.3 Autograd formula quality

CIA's defining feature is that autograd is derived by tracing through the decomposition. This is convenient but produces a mechanical backward that may be suboptimal. A hand-written backward via CEA can:

- **Reuse forward intermediates** that the traced backward would recompute.
- **Exploit mathematical identities**. For example, $\frac{\partial}{\partial x_i}\text{softmax}(x)_j = \text{softmax}(x)_j(\delta_{ij} - \text{softmax}(x)_i)$ has a compact form that element-wise decomposition into `exp`, `sum`, `div` does not reveal.
- **Control numerical stability**. Decomposition may introduce intermediate values that overflow or underflow; a fused backward can use log-space or other stable formulations.

The tradeoff: CIA costs nothing to author but produces a mechanical backward. CEA requires explicit engineering but can be made arbitrarily efficient and stable.

### 4.4 Graph legibility

A graph of $N$ high-level ops decomposes into $O(kN)$ primitive ops. For debugging, profiling, and human reasoning, the high-level graph is strictly more informative. This is not a theoretical concern — it directly affects the experience of anyone who calls `print(fx_graph)` or inspects a trace in a profiler. Pre-dispatch export preserves high-level ops precisely for this reason.

---

## 5. The Three Regimes

The system operates in three distinct regimes. Understanding which regime you are in determines which decomposition logic applies.

### Regime I: Eager execution

Decomposition is determined solely by `resolve_key`. If a backend kernel exists, the fused kernel runs; if not, CIA fires at the autograd key. The user has no control over this short of registering custom kernels.

The autograd key is either active (normal execution, `no_grad`) or excluded (`inference_mode`). When excluded, CIA decompositions that would have fired are simply skipped — the backend fallback or a lower-priority kernel runs instead. This is correct in eager mode but creates problems for tracing (Section 2, failure mode A).

### Regime II: `torch.compile`

Two-pass decomposition:
1. **Functionalization pass**: Replicates the autograd CIA decision via `autograd_would_have_decomposed`. Also unconditionally decomposes aliasing/mutating ops. This pass ensures the graph is functional and matches eager semantics.
2. **Proxy tensor pass**: Applies the explicit decomposition table (`post_autograd`). This is the user-facing knob — the `decomposition_table` argument.

After tracing, Inductor further adjusts: it starts from the core ATen decomposition set, then *removes* decompositions for ops it can lower directly (the exclusion list), and adds Inductor-specific decompositions. The final set is: `core_aten_decompositions() - inductor_exclusions + inductor_custom`.

### Regime III: Export

- **Pre-dispatch**: Preserves all functional CIA ops. Highest-level IR, suitable for model inspection and serialization.
- **Post-dispatch**: Decomposes to the core ATen opset. Maximum portability, minimum op surface, at the cost of semantic content (Section 4).

---

## 6. Connection to the Formal Model

The Coq formalization at `/workspaces/torch_formal` provides machine-checked proofs for several pieces of this story. The alignment is strongest for dispatch mechanics and weakest for the compilation pipeline.

### What is formalized

| Aspect | Location | Status |
|--------|----------|--------|
| CIA expansion set (which keys it covers) | `AliasKeys.v:52-62` | Complete |
| Precedence: direct registration overrides CIA | `AliasKeys.v:159` (T7) | Complete |
| Precedence ordering among alias keys | `AliasKeys.v:108-112` | Complete |
| Inference mode excludes autograd keys | `InferenceMode.v:74-82` | Complete |
| Sharp mode equivalence criterion | `ModeEquivalence.v:69-105` | Complete |
| Version counter divergence under mode sensitivity | `EndToEnd.v:158-185` (T19) | Complete |
| Sensitivity implies trace divergence | `EndToEnd.v:237-275` | Complete |

### What is not formalized (gaps)

| Gap | Why it matters |
|-----|----------------|
| `has_backend_kernel` gate (including CEA in the check) | The single condition determining whether CIA fires. Without it, "CIA fires iff no backend kernel" cannot be stated as a theorem. |
| `autograd_would_have_decomposed` | Fidelity invariant's implementation. Would connect the dispatch model to the tracing model. |
| Functionalization decision cascade (Section 3) | Four-priority decomposition decision during tracing is entirely unformalized. |
| Explicit decomposition tables | The second decomposition mechanism (table-driven) is absent from the formal model. |
| `CompositeImplicitAutogradNestedTensor` | A variant of CIA with stricter conditions (`k ≠ Undefined`) that the current model omits. |

---

## 7. Key Source Locations

| Concept | File | Lines |
|---------|------|-------|
| Global decomposition tables | `torch/_decomp/__init__.py` | 37-43 |
| `get_decompositions` API | `torch/_decomp/__init__.py` | 231-246 |
| `core_aten_decompositions` | `torch/_decomp/__init__.py` | 291-294 |
| Unsafe op check | `torch/_decomp/__init__.py` | 46-61 |
| Dispatch resolution order | `torch/_ops.py` | 211-261 |
| `has_backend_kernel` (includes CEA) | `torch/_ops.py` | 227-229 |
| `AutogradOther` ambiguity guard | `torch/_ops.py` | 242-245 |
| `OpOverload._can_decompose` | `torch/_ops.py` | 901-905 |
| `OpOverload.decompose` | `torch/_ops.py` | 907-918 |
| `autograd_would_have_decomposed` | `torch/utils/_python_dispatch.py` | 811-866 |
| Functionalization decision cascade | `torch/_subclasses/functional_tensor.py` | 423-468 |
| Proxy tensor decomposition | `torch/fx/experimental/proxy_tensor.py` | 2912-2924 |
| AOT autograd graph capture | `torch/_functorch/_aot_autograd/graph_capture.py` | 130-136 |
| Inductor exclusion list | `torch/_inductor/decomposition.py` | 110-141 |
| Export preservation check | `torch/_export/utils.py` | 1313-1336 |
| DTensor sharding propagator | `torch/distributed/tensor/_sharding_prop.py` | 650-716 |
| `DecompShardingStrategy` | `torch/distributed/tensor/_decompositions.py` | 152-331 |
| Formal: CIA expansion | `theories/Registration/AliasKeys.v` | 52-62 |
| Formal: alias precedence (T7) | `theories/Registration/AliasKeys.v` | 159-166 |
| Formal: mode equivalence | `theories/Mode/ModeEquivalence.v` | 69-105 |
| Formal: version divergence (T19) | `theories/Integration/EndToEnd.v` | 158-185 |
