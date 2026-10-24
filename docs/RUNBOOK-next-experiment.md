# RUNBOOK — Next RTI experiment

**Status:** not started · **Owner:** operator · **Written:** 2026-10-23
**Read first:** `README.md` (mechanism), `AGENTS.md` (invariants), `evidence/POC-RESULT.md` (what was proven)

This is the procedure for the next experiment. It exists so the work is
resumable by someone who was not in the session that produced the POC.

---

## 1. What is already settled

Do not re-litigate these. They are measured, not assumed.

| Claim | Evidence | Confidence |
|---|---|---|
| RTI α=1.4 reproduces baked DWM α=1.4 | 24/50 vs 24/50 `refused_answer` | high (n=50, exact match) |
| Weight edit ≡ activation projection | 7.15e-07 reconstruction error | high (analytic + measured) |
| TP-linear, no extra collective | 4.77e-07, tp=4 | high |
| Runtime toggle works under CUDA graph | output diverges off→on, both TP ranks log | high |
| Stock weights stay byte-identical | no writes in 90 min, 49 shards / 252,418,287,306 B | high |

**Not settled — do not claim these:**
red-line parity (holdout never run) · n>50 behaviour · tp≠4 · vLLM path (untested) · throughput cost · α=1.2 or any intermediate point.

---

## 2. The open question worth spending a cycle on

The POC proved *parity*. It did not answer whether runtime search finds a
**better α than the bake ever could**, which is the actual commercial argument
for RTI.

```
old pipeline:  capture → bake α=1.0 → eval → bake 1.4 → eval → bake 1.5 → …
               (252 GB and a ~10 min reload per point)

new pipeline:  capture → serve stock + RTI → sweep α live → pick keeper → bake ONCE
               (0 GB, ~1 s per point)
```

The deliverable of the next experiment is **a defensible keeper α plus a single
bake**, not a new model.

---

## 3. Pre-flight gates

All four must pass before any sweep data is trusted.

```
[ ] 1. Bank integrity
        shasum -a 256 banks/directions.npy
        == 3b84104286223d99e6e385c1a010be9082adb5392edd8540b53998e48306dff3

[ ] 2. Tests
        pip install -r requirements.txt && cd tests && python test_rt_intervention.py   # 8/8

[ ] 3. Hooks bound  ← silent-failure gate
        grep "INSTALLED on 58 layers" serve.log

[ ] 4. Toggle fires  ← silent-failure gate
        α=0.0 and α=1.4 produce DIFFERENT output at temp=0, max_tokens>=2500
```

Gate 3 and 4 are the two that fail without erroring. See `AGENTS.md` §I3.

---

## 4. Decode-path decision — settle this BEFORE collecting data

Speculative decoding changes the token distribution. Mixing decode paths
mid-sweep makes arms non-comparable.

| path | use for | why |
|---|---|---|
| **no speculation** | the α sweep | matches the POC baseline exactly (24/50 @ 1.4) |
| DSpark / DFlash2 | production serving | ~2× decode, but re-baseline required |

This checkpoint ships a DSpark draft at `MiniMax-M3.1-Flash/dspark/` (2.2 GB)
and `RadixArk/Minimax-M3.1-Flash-Dflash2` is public (4.1 GB). Both are wired in
the installed sglang build (`srt/speculative/dflash_worker_v2.py`,
`model_overrides/minimax_m3.py:59`). **Neither is enabled today.**

> If you enable speculation, the POC's 24/50 reference is void. Re-run the
> α=1.4 arm under the new decode path before comparing anything.

---

## 5. Sweep design

### Phase A — global α (1D, cheap)

Coarse then fine. Do not go straight to per-layer.

| α | purpose |
|---|---|
| 0.0 | control — must reproduce stock 46/50 |
| 0.8 | floor of interest |
| **1.0** | projection fully removed; natural reference point |
| 1.2 | **unmeasured gap** — interpolating 1.0→1.4 is a guess |
| 1.4 | POC keeper — must reproduce 24/50 or the rig is wrong |
| 1.5 / 1.6 / 1.7 | characterised as degrading; re-confirm |

Per point: `echo '{"enabled":true,"alpha":X}' > rt/toggle.json`, sleep 2, run
smoke-50, record `refused_answer` + `empty` counts.

**Stop rule:** if α=1.4 does not reproduce 24/50 ±3 on the same harness, stop
and fix the rig. Do not sweep on a broken baseline.

### Phase A2 — layer-band check (one extra arm, high information)

The ratio `per_layer_norm / within_layer_std` is **0.04 at tap 2** but **115 at
tap 58** (§6). If that measure is meaningful, the low-tap directions are close
to noise.

Run **one** arm at the chosen keeper α with `layers` restricted to a late band
(e.g. `20–59`) and compare:

| outcome | read |
|---|---|
| same refusals, less capability damage | early hooks never earned their keep — drop them |
| worse refusals | early layers matter; the ratio is misleading |

This is one arm and it directly tests the most likely waste in the current
config (`layers: 2–59`). Do it before per-layer search — it may collapse the
search space on its own.

### Phase B — per-layer α (only if Phase A shows a usable plateau)

See §6. Needs a code change first.

---

## 6. Known gap — α is global, not per-layer

Directions **are** per-layer (bank is `(59, 6144)`, hook registered per layer
via `_make_hook(int(lid))`). The **scalar is not**:

```python
# rt/rt_intervention.py
def _alpha_for(y):
    key = (y.device.index or 0, y.dtype)     # ← no layer index
```

One α per GPU, shared by all 58 hooked layers.

### Why per-layer α is the right next capability

The edit is `y ← y − α·(y·d)·d` with `d` **unit-norm**. Two consequences:

1. **α is scale-free w.r.t. `per_layer_norm`.** That field is the magnitude of
   the mean-difference direction *before* normalisation — a **detectability**
   measure (how cleanly the classes separate), not a strength knob. Do not
   divide α by it.
2. **The residual stream grows with depth**, so `(y·d)` grows, so a constant α
   produces a **larger absolute perturbation in later layers**. That is the real
   argument for per-layer α — it is a depth-compensation knob, not a
   signal-quality knob.

Measured `per_layer_norm` spans **0.56 → 3213.37 (5,749×)** across taps 0–58,
rising monotonically. Signal quality is wildly uneven by depth, and a uniform α
spends magnitude where it is least needed.

**Verified derivation:** `per_layer_norm` is the norm of the mean-difference
direction *before* normalisation — `mean(per_layer_norm) = 1112.552` matches
`manifest.json`'s `sanity_mean_layer_delta = 1112.552` exactly. It is a
**separation/magnitude** measure, not an edit-strength measure.

### Early layers look near-noise — check this before hooking 2–59

Taking `per_layer_norm / sanity_within_layer_std` as a rough separation ratio
(**not** a formal SNR — it is a crude indicator, treat it as such):

| tap | ratio | read |
|---|---|---|
| 0 | 0.02 | essentially no separation |
| 2 | 0.04 | essentially no separation |
| 5 | 0.16 | weak |
| 10 | 0.55 | weak |
| 20 | 8.34 | usable |
| 30 | 34.99 | strong |
| 40 | 67.37 | strong |
| 58 | 115.16 | strongest |

`rt_config.json` currently hooks **layers 2–59**. If the ratio is meaningful,
the low-tap directions are close to noise and the edit there is mostly
perturbation. That is a concrete, cheap hypothesis for Phase A: **compare
2–59 against a late-only band (e.g. 20–59) at the same α.** If late-only matches
on refusals with less capability damage, the early hooks were never earning
their keep.

Do not assume this is true — it is an inference from a crude ratio, and the
uniform-α result (24/50) did not obviously suffer for it. Test it, don't assert
it.

### Change required

```python
_ALPHA_BUF[key] = torch.zeros(len(layers), device=..., dtype=...)   # was zeros(1)
# toggle: {"alpha": [1.4, 1.4, 1.2, ...]}   # 58 floats
```

The buffer trick is unchanged — only wider. Bake equivalence still holds
per-layer: `W_o ← W_o − α_l · d_l · d_lᵀ · W_o`.

**Search method matters more than the code.** 59 dimensions at 1M context gets
expensive fast. Do band-level first (group layers, one α per band), then refine
inside a band. Do not coordinate-descent 59 free parameters against n=50 — you
will overfit the smoke set. See §7.

---

## 7. Statistical honesty

n=50 gives a 95% CI of **±13.8 points** on a 24/50 rate — i.e. the true rate
lies in **[17, 31] of 50**. Comparing two independent arms is worse: the
unpaired difference CI is ±19.6 points, so **two arms must differ by ~10
refusals before the difference is distinguishable from noise** at this n.

Consequences:

- A per-layer α that "wins" by 2–5 refusals on n=50 has proven nothing. It has
  found a candidate.
- 59 free parameters against 50 items is a guaranteed overfit. Band-level
  granularity (4–6 bands) is the most the data can support.
- Any keeper claim needs the **443-item run** to tighten the CI before it is
  called a release.

---

## 8. Throughput gate — unmeasured, and it may bite

The hook adds a dot product and a rank-1 update per layer per token, 58 layers.
FLOPs are trivial. The risk is that it **blocks kernel fusion** and the cost is
superlinear in context length.

Not measured. Do not call RTI production-grade until it is.

```bash
# same server, same prompt, toggle off then on
# record: TTFT, decode tok/s, at 1K / 32K / 256K / 1M
```

If decode cost is material at 1M, RTI is a **test instrument only** and the bake
stays the serving artifact. That was the operator's stated intent and this gate
decides whether it stays true.

---

## 9. Bake-once procedure

Only after a keeper α is chosen.

1. Re-derive `W_o ← W_o − α_l·d_l·d_lᵀ·W_o` per layer from the **same bank**
   (hash-bound — never re-derive directions against a different capture).
2. Requantize to NVFP4/MXFP4 through the normal block-scale path. **This
   introduces rounding the runtime path does not have** — the bake is not
   bit-identical to RTI, only behaviourally close.
3. Verify: baked α=keeper reproduces the RTI refusal count within the §7 CI.
4. Red-line holdout (9-row `SR_RELEASE_REDLINE`) — **still never run for this
   model family.** This blocks any release claim.
5. Package + hash + manifest.

---

## 10. What NOT to carry forward

- **The POC's successor list is partly stale.** It proposed a "0.8/1.0/1.2/1.4
  sweep" as if unmeasured — 1.5/1.6/1.7 were later characterised as degrading,
  and 1.2 remains the real gap. Use §5, not that list.
- **Do not treat `per_layer_norm` as an α scale factor.** See §6.
- **Do not reuse a sibling model's α.** Refusal geometry is checkpoint-specific;
  Muse-Spark and Bolt numbers do not transfer.

---

## 11. Resumption checklist

```
[ ] read AGENTS.md invariants (I1–I7)
[ ] confirm decode path (§4) and record it in every result
[ ] pre-flight gates 1–4 (§3)
[ ] Phase A sweep, α=1.4 reproduces 24/50 ±3
[ ] throughput gate (§8) — decides test-only vs serve
[ ] pick keeper; state its CI honestly (§7)
[ ] bake once (§9); verify vs RTI
[ ] red-line holdout — or explicitly mark the release claim BLOCKED
```

Nothing in this runbook authorizes a release. It authorizes a search.