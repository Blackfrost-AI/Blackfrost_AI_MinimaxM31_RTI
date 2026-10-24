# Runtime Intervention POC — MiniMax-M3.1-Flash

**Date:** 2026-10-23
**Operator directive:** *"could we create a runtime intervention POC… I want to start
making both weight edit and runtime, this way we can use clean weights and have
control of when it runs and doesn't — you have the vllm hooks already so let's see
if we can do this with the clean version."*

**Verdict: POC SUCCEEDS.** The α=1.4 DWM is reproduced as a runtime activation
hook on **byte-identical stock weights**, toggleable without a restart.

---

## Headline result

Judgment-free, from the harness's own `refused_answer` flag (no human in the loop):

| Arm | refused /50 | empty /50 |
|---|---|---|
| **RT α=1.4** (stock weights) | **24** | 12 |
| Baked α=1.4 (weight edit) | **24** | 11 |
| Stock base | 46 | 2 |

Exact match on refusal count. 22 points below stock.

---

## Artifacts

### Shippable package
`the_method/DWM_models/MiniMax/MiniMax-3.1-Flash/runtime/`

| File | sha256 (short) | Role |
|---|---|---|
| `rt_intervention.py` | see `MANIFEST.sha256` | Hook installer |
| `sitecustomize.py` | see `MANIFEST.sha256` | PYTHONPATH bootstrap |
| `serve_rt.sh` | see `MANIFEST.sha256` | Launch script |
| `test_rt_intervention.py` | see `MANIFEST.sha256` | 7 CPU tests |
| `smoke50_rt.py` | see `MANIFEST.sha256` | Standalone smoke-50 |
| `toggle_probe.py` | see `MANIFEST.sha256` | Live A/B toggle proof |
| `README.md` | — | Full report |

### Evidence
`work/runs/20261023-minimax-m3p1-flash-s0-intake-v1/rt-poc/`

- `smoke50_rt_canonical.jsonl` / `smoke50_rt_final.jsonl` — two smoke-50 runs
- `judged_verdicts_rt_a14.json` / `judged_verdicts_rt_final.json` — manual verdicts
- `toggle_probe_final_a0.0.json` / `_a1.4.json` / `_disabled.json` — toggle proof
- `smoke50_canonical.log` / `smoke50_final.log` — harness logs

---

## Defects found (all by running it, none visible in CPU tests)

1. **`o_proj` returns a tuple.** `RowParallelLinear.forward` → `(output, output_bias)`.
   Returning a bare tensor crashed the serve. Fixed + regression test.
2. **`.to()` allocates during CUDA graph capture.** Died with "cannot copy between
   CPU and CUDA tensors during CUDA graph capture". Fixed by pre-staging 696
   direction tensors (58 × 3 dtypes × 4 GPUs) at install.
3. **Python-float α is baked into the CUDA graph.** Toggle silently did nothing —
   α=0.0 and α=1.4 gave byte-identical output. Fixed by putting α in device
   memory (`_ALPHA_BUF`) with a poller thread; hook now dereferences live memory
   at replay. This is the defect that would have made the whole POC meaningless.

---

## Honest limitations

- **Manual rubric is noisy.** Two identical-config RT runs disagreed on 13/50
  items (intra-rater drift on the R/C boundary + temp-1.0 decode variance).
  The mechanical refused-count (24 vs 24) is the reliable number; the rubric
  characterises *what kind* of answer appears, not a tight point estimate.
- **n=50.** Adequate for a POC, not for a release gate. A 443-item run would be
  needed to tighten the CI.
- **Single checkpoint, single tp=4 topology.** TP-linearity is proven on CPU;
  other topologies (tp=8, EP-only) are untested.
- **No red-line gate run.** This POC tested refusal behaviour only. The 9-row
  SR_RELEASE_REDLINE holdout was not exercised.
- **`toggle_probe.py`'s regex screener is crude** (fires on any "I can't" in the
  first 500 chars). Character divergence is the evidence; the flag is noisy.

---

## Scope compliance

- Weights under `/workspace/models/MiniMax-M3.1-Flash` **never modified**
  (verified: no writes in 90 min; 49 shards / 252,418,287,306 B).
- **GPUs 4–7 untouched** throughout (`ds1017-fp8` undisturbed).
- Hook dormant unless `BLACKFROST_RT_INTERVENTION` is set.
- No files >1 GB stored locally; no weights stored locally.

---

## Next step (operator decision)

The POC answers the question it was asked. Natural follow-ups, in order of value:

1. **443-item run** on RT α=1.4 to put a tight CI on parity with the baked arm.
2. **Red-line gate** — run the 9-row holdout against RT α=1.4 to confirm the
   hook does not open anything the weight edit closed.
3. **α-ramp sweep** (0.8 / 1.0 / 1.2 / 1.4) — the runtime toggle makes this
   nearly free; the baked path needed a separate 252 GB build per point.
4. **Package as a serve flag** — collapse `sitecustomize.py` + config into a
   single documented `--rt-intervention` style invocation.

None of these are started. Awaiting operator direction.