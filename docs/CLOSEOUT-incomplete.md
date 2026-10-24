# CLOSEOUT — MiniMax-M3.1-Flash RTI · INCOMPLETE

**Closed:** 2026-10-23 · **Status:** `INCOMPLETE — HELD` · **Release:** none
**Operator decision:** hold release to test the runtime-search thesis further.

This record exists so a future session can resume without re-deriving anything.
It is deliberately blunt about what is *not* known.

---

## Why this closed as incomplete

Not a failure and not a bug. The POC answered its question — runtime
intervention reproduces the baked weight edit — and the operator chose to hold
the release in order to test the broader thesis (search α in activation space,
bake once) before shipping anything.

**No release was made. No model was published. Nothing is pending upload.**

---

## What was proven

| Claim | Evidence | Confidence |
|---|---|---|
| RTI α=1.4 ≡ baked DWM α=1.4 | 24/50 vs 24/50 `refused_answer`, exact match | high (n=50) |
| Weight edit ≡ activation projection | 7.15e-07 reconstruction error | high (analytic + measured) |
| TP-linear, no extra collective needed | 4.77e-07 at tp=4 | high |
| Runtime toggle works under CUDA graph | output diverges off→on, all 4 TP ranks log | high |
| Stock weights untouched | 90 min serve, no writes | high |

Bank hash (verified at close, matches `MANIFEST.sha256`):

```
3b84104286223d99e6e385c1a010be9082adb5392edd8540b53998e48306dff3  banks/directions.npy
```

---

## What was NOT proven — do not claim these

- **Red-line holdout never run.** The 9-row `SR_RELEASE_REDLINE` has not been
  evaluated for this model family. Any release claim is **BLOCKED** on this.
- **n=50 only.** CI is ±13.8 points; the 443-item run was never done on RTI.
- **α = 1.2 unmeasured.** Interpolating 1.0→1.4 is a guess.
- **Throughput unmeasured.** Hook cost at 1M context is unknown. This decides
  whether RTI is a test instrument only or a viable serving path.
- **vLLM path untested.** Only sglang tp=4 was ever run.
- **Per-layer α not implemented.** α is global (§ below).
- **tp≠4 untested.**

---

## Final environment state

| item | state |
|---|---|
| GPUs 0–3 | **freed** — 0 MiB each |
| GPUs 4–7 | **untouched** — DeepSeek `ds1017-fp8`, ~261 GB each, up 11 h |
| container `m31-rt` | **stopped** |
| port 30000 | free |
| checkpoint | 277,662,370,860 B / 48 shards — byte-identical to stock |
| toggle at close | `{"enabled": false, "alpha": 0.0}` |
| run artifacts | `/workspace/rti-pkg` — 2.0 MB, no weights |
| canonical package | GitHub `Blackfrost-AI/Blackfrost_AI_MinimaxM31_RTI` (private) |

Restart is one command, no rebuild:

```bash
ssh b300-test 'cd /workspace/rti-pkg && ./rt/serve_container.sh'
```

---

## The open question this was held for

```
old:  capture → bake α=1.0 → eval → bake 1.4 → eval → bake 1.5 → …   (252 GB each)
new:  capture → serve stock + RTI → sweep α live → pick keeper → bake ONCE
```

RTI is a **test instrument**. The bake is the **product**. Whether runtime search
actually finds a better α than the bake pipeline would have is **untested** —
that is the thesis being held open.

---

## Known defect carried forward

**α is global, not per-layer.**

```python
def _alpha_for(y):
    key = (y.device.index or 0, y.dtype)     # ← no layer index
```

Directions *are* per-layer (bank `(59, 6144)`, hook per layer). The scalar is
shared across all 58 hooked layers. Per-layer α requires widening `_ALPHA_BUF`
to a vector and extending the toggle format — the graph-safety trick is
unchanged.

Also carried forward: `rt_config.json` hooks **layers 2–59**, but the separation
ratio (`per_layer_norm / within_layer_std`) is only **0.02–0.16 below tap 10**
versus **115 at tap 58**. One late-band arm would test whether the early hooks
earn their keep. See `RUNBOOK-next-experiment.md` §6 / Phase A2.

---

## Resume here

1. `docs/RUNBOOK-next-experiment.md` — pre-flight gates, sweep design, stop rules
2. `docs/METHOD-runtime-search.md` — the general procedure
3. `evidence/POC-RESULT.md` — what was measured

**Before any sweep:** fix the decode path first. Enabling DFlash2 or DSpark
changes the token distribution and voids the 24/50 baseline. Neither is enabled
today.

---

## Closeout checklist

```
[x] GPUs 0-3 freed (0 MiB), 4-7 untouched
[x] container m31-rt stopped, port 30000 free
[x] checkpoint unmodified (277,662,370,860 B / 48 shards)
[x] bank hash verified against MANIFEST.sha256
[x] no weights in repo (2.0 MB artifacts, 1.4 MB bank is largest file)
[x] package pushed, clean-clone verified, 8/8 tests
[x] runbook + method note written
[ ] RED-LINE HOLDOUT — still open, blocks release
[ ] throughput gate — still open
```