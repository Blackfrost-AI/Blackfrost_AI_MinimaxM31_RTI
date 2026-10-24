# METHOD — Runtime search, bake once

**Applies to:** any DWM arm on any model. This is the general procedure; the
MiniMax-M3.1-Flash specifics live in `RUNBOOK-next-experiment.md`.

---

## The shift

Baking was being used as the *experiment*. It should be the *release step*.

```
BEFORE   capture → bake α=1.0 → eval → bake 1.4 → eval → bake 1.5 → eval → …
         cost: one full checkpoint (≈250 GB) and a reload per α

AFTER    capture → serve stock + RTI → sweep α live → pick keeper → bake ONCE
         cost: 0 GB, ~1 s per α
```

Search in activation space, where it is free. Materialize a checkpoint only for
the winner you intend to ship.

---

## Why it is valid

For a linear layer `y = Wx`, a rank-1 weight edit and an activation projection
are the same map:

```
W' = W − α·d·dᵀ·W              (bake)
y' = y − α·(yᵀd)·d             (runtime, d unit-norm)

W'x = Wx − α·d·dᵀ·Wx = y − α·(yᵀd)·d     ✓
```

Exact in exact arithmetic. Measured 7.15e-07 reconstruction error; TP-linear at
4.77e-07, so per-rank application needs no extra collective.

**Corollary — RTI is more faithful than the bake, not less.** Baking into a
quantized checkpoint rounds the edit through block scales. RTI never touches
`W`, so it carries no requantization error. Close enough behaviourally that
refusal counts match exactly; not bit-identical.

---

## What it buys

| | Bake | RTI |
|---|---|---|
| Disk per α | ~250 GB | 0 |
| New α | ~10 min reload | ~1 s |
| A/B in one process | no | yes |
| Requantization error | yes | none |
| Reversible | no | yes |
| Distributable artifact | **yes** | no |

---

## What it does not replace

- **One direction per layer.** Bank is `(n_layers, hidden)` — a single row
  lookup. Multi-direction edits need a shape and math change.
- **Linear layers only.** Fine for `o_proj` / `down_proj` / `qkv`. Nothing that
  is not `y = Wx`.
- **Not a deliverable.** A bake is a checkpoint you hand someone; RTI is a
  plugin plus a bank. The consumer has to wire it.
- **Throughput is unmeasured.** Small in FLOPs, but it can block kernel fusion.
  Measure before calling it production-grade.

---

## Two traps, both already paid for

### 1. A Python scalar in a CUDA-graphed hook is captured, not live

An α passed as a Python float became a kernel *argument* baked into the graph.
Graph replay never re-ran the hook body, so α=0.0 and α=1.4 produced
byte-identical output — with no error, no warning, and a config that read back
correctly.

**Rule:** any value that affects output and must change at runtime lives in
**device memory**, and a poller `fill_()`s it. Verify by observing output
divergence, never by reading the config back.

### 2. `d` is unit-norm, so α is scale-free

It is tempting to scale α by a per-layer direction norm. Don't. The norm is a
*detectability* measure (how cleanly classes separate before normalisation),
not a strength knob.

The real depth effect is different: the residual stream grows with depth, so a
constant α produces a **larger absolute perturbation in later layers**. Per-layer
α is a depth-compensation knob. That is the correct justification.

Worth knowing: the separation ratio (`per_layer_norm / within_layer_std`) rises
from **0.02 at tap 0 to 115 at tap 58**. Early-layer directions are near-noise
while the edit magnitude there is small — so the two effects cut in opposite
directions. That is exactly why a uniform α is a compromise, and why band-level
search is the right first move. See the runbook §6.

---

## Statistical floor

n=50 gives a 95% CI of **±13.8 points** on a 24/50 rate (true rate in [17, 31]
of 50). The unpaired two-arm difference CI is ±19.6 points, so **arms must
differ by ~10 refusals before the gap is real** rather than noise.

- A 2–5 refusal "win" on n=50 is a **candidate**, not a result.
- 59 free parameters against 50 items is a guaranteed overfit. Search bands
  (4–6 groups) before searching layers.
- Any keeper claim needs the full corpus to tighten the CI before release.

---

## The pipeline, stated once

```
capture (hash-bound) → serve stock + RTI → sweep α (global → band → layer)
  → pick keeper (state its CI) → measure throughput (test-only vs serve)
  → bake ONCE → verify bake ≈ RTI within CI → red-line holdout → release
```

RTI is a **test instrument**. The bake is the **product**. Keeping those roles
separate is what makes the search cheap without making the release sloppy.