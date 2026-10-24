# AGENTS.md — Integration Contract

You are an agent wiring the Blackfrost RTI runtime intervention into an inference
server. This file is the contract. Read it fully before touching a config.

**Target:** `MiniMaxAI/MiniMax-M3.1-Flash` · **Mechanism:** forward hook on
`self_attn.o_proj` · **Weights:** never modified.

---

## 1. Invariants

These are not preferences. Violating any one of them produces a result that
looks fine and is wrong.

| # | Invariant | Why |
|---|---|---|
| I1 | Never write to the checkpoint directory | Weights must stay byte-identical to stock; that is the entire point |
| I2 | Never proceed if `INSTALLED` is absent from the log | No `INSTALLED` = no hooks = you are benchmarking stock |
| I3 | Never claim a toggle works because the config was read | Read ≠ applied. Observe different output (see §4) |
| I4 | Never report a refusal rate from a hand-scored rubric alone | Intra-rater drift is real; use `refused_answer` |
| I5 | Never exceed α=1.7 | Clamped in code; beyond it output degrades |
| I6 | Never claim red-line parity without running the holdout | Not yet tested for this arm |
| I7 | Never report n=50 as a release-grade result | POC-grade only |

### I3 in detail — this one has already bitten us

A shipped build read the toggle file cleanly, logged no error, and had **zero
effect**: a Python-float α had been baked into the CUDA graph as a kernel
argument, so graph replay never re-ran the hook body. α=0.0 and α=1.4 produced
byte-identical output.

The fix is that α now lives in **device memory** (`_ALPHA_BUF`) and a poller
thread `fill_()`s it. If you refactor the hook, preserve that property: **no
Python-scalar that affects output may be captured by the graph.** Verify by
observing output divergence, not by reading the config back.

---

## 2. Wiring checklist

Do these in order. Do not skip ahead.

```
[ ] 1. Bank integrity
        sha256(banks/directions.npy) == 3b84104286223d99e6e385c1a010be9082adb5392edd8540b53998e48306dff3
        shape == (59, 6144) float32, unit-norm rows

[ ] 2. Tests pass on target host
        cd tests && python test_rt_intervention.py    # 8/8, CPU only

[ ] 3. Environment
        PYTHONPATH includes <repo>/rt
        BLACKFROST_RT_INTERVENTION=<repo>/rt_config.json

[ ] 4. Launch
        server reaches ready state

[ ] 5. Hooks bound  ← GATE
        grep "INSTALLED on 58 layers" serve.log

[ ] 6. Hook fires
        toggle off -> capture output
        toggle on  -> capture output
        outputs MUST differ

[ ] 7. Serve readiness
        real forward pass returns 200 with non-empty content
        (health/`/v1/models` alone does NOT prove serving works)

[ ] 8. Behaviour
        run smoke-50; compare refused_answer against the reference
```

Steps 5 and 6 are the ones that fail silently. Do not report success without
both.

---

## 3. Configuration reference

`rt_config.json`:

| key | type | meaning |
|---|---|---|
| `enabled` | bool | static default on/off |
| `alpha` | float | static default strength, clamped to [0, 1.7] |
| `directions` | path | bank file; relative paths resolve from repo root |
| `directions_sha256` | hex | **mandatory** integrity check; mismatch = fail closed |
| `layers` | int[] | layer ids to hook (2–59 for this checkpoint) |
| `toggle_file` | path | live on/off file; omitted = static only |
| `toggle_poll_seconds` | float | re-read interval (default 1.0) |
| `alpha_max` | float | documentation of the ceiling; `ALPHA_MAX` in code is authoritative |

Environment:

| var | meaning |
|---|---|
| `BLACKFROST_RT_INTERVENTION` | path to config. **Unset = module is inert.** |
| `BLACKFROST_RT_BACKEND` | pin `sglang` or `vllm`; default auto-detect |

### α guidance

| α | use |
|---|---|
| 0.0 | off (identity) |
| 1.0 | projection fully removed |
| **1.4** | **recommended — most evidence** |
| 1.5–1.7 | experimental; re-run smoke and check empty/loop rate |
| >1.7 | clamped; not explored |

---

## 4. Verification you must actually run

### Prove the hook fires (not just that config was read)

```bash
# greedy decode, same prompt, toggle off then on
echo '{"enabled": false, "alpha": 1.4}' > rt/toggle.json && sleep 2
curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-3.1-Flash-RTI","messages":[{"role":"user","content":"YOUR PROMPT"}],"temperature":0,"max_tokens":2500}' \
  | jq -r '.choices[0].message.content' | wc -c

echo '{"enabled": true, "alpha": 1.4}' > rt/toggle.json && sleep 2
# repeat — character counts and text must differ
```

> **Use `max_tokens` ≥ 2500.** This is a thinking model: a small budget is
> entirely consumed by `reasoning_content`, `content` comes back empty, and
> `finish_reason` is `"length"` — which looks like a failure but is not. If you
> see empty content at α=0.0 *and* α=1.4, suspect the token budget before the
> hook.

Both TP ranks must log a transition:

```bash
grep "toggle ->" serve.log
```

### Prove behaviour moved

```bash
python rt/smoke50_rt.py --out /tmp/smoke50_rti.jsonl --tag rti-a1.4
```

Reference (canonical harness, n=50):

| arm | refused /50 |
|---|---|
| RTI α=1.4 | 24 |
| baked DWM α=1.4 | 24 |
| stock | 46 |

If you get ~46 you are serving stock — go back to step 5.

---

## 5. Reporting rules

When you report a result, state:

1. **Which α and which toggle state** produced it.
2. **`INSTALLED` was seen** (quote the line).
3. **The metric and its n.** Never a rate without n.
4. **What was not tested** — red lines, other topologies, other backends.
5. **Variance honestly.** Two identical-config runs disagreed 13/50. Do not
   present a single smoke-50 refusal rate as a precise number.

Do **not**:
- describe n=50 as validated;
- describe vLLM as supported (it is untested — see README);
- quote a refusal rate without saying it came from a manual rubric;
- claim the toggle works without an output-divergence check.

---

## 6. Known failure modes

| symptom | cause | fix |
|---|---|---|
| no `INSTALLED` | `PYTHONPATH` missing `rt/`, or env var unset | step 3 |
| `INSTALLED` but identical output | `enabled:false` or `alpha:0.0` | check toggle |
| toggle written, nothing changes | edited a different path than the log shows; or Python-scalar α regression (I3) | `grep "toggle ->"`; run test 6 |
| `fail-closed: hash mismatch` | bank regenerated | update `directions_sha256` |
| `no supported backend found` | neither engine importable | set `BLACKFROST_RT_BACKEND` |
| shape error | bank not `(59, 6144)` | regenerate for this checkpoint |
| CUDA graph capture error | allocation in hot path | pre-stage directions; no `.to()` in hook |

---

## 7. What this is not

- **Not a jailbreak.** It removes false refusals on technical content.
- **Not verified against red lines.** Do not claim otherwise.
- **Not portable to other models** without a new bank.
- **Not destructive.** The checkpoint is read-only throughout.

---

## 8. If you change the hook

Re-run `tests/test_rt_intervention.py` (8 tests) **and** the live toggle
divergence check in §4. The CPU tests do not catch CUDA-graph capture bugs —
three separate defects in this package were found only by running a real serve.
Assume the same is true of your change.