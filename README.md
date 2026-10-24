# Blackfrost AI — MiniMax-M3.1-Flash Runtime Intervention (RTI)

**Runtime, toggleable refusal-direction intervention for `MiniMaxAI/MiniMax-M3.1-Flash`.**

This package removes false refusals from MiniMax-M3.1-Flash **without modifying a
single weight**. The edit is applied to the attention output activations at
inference time, which means:

- the checkpoint on disk stays **byte-identical to stock**;
- the intervention can be **switched on and off while the server runs** — no
  restart, no re-quantise, no 252 GB re-upload;
- strength is adjustable live, from 0.0 (stock behaviour) to 1.7.

This is the runtime counterpart to the Domain Weight Modification (DWM) weight
edit. Both produce the same operating point; only the mechanism differs.

| | Weight edit (DWM) | Runtime (this package) |
|---|---|---|
| Weights on disk | modified | **stock, untouched** |
| Toggle without restart | no | **yes** |
| Storage cost per α | 252 GB | **1.4 MB** |
| α sweep | one build each | **free** |
| Verified on | sglang | sglang |

---

## What it actually does

The baked DWM edit to the attention output projection is

```
W_o  ←  W_o − α · d · dᵀ · W_o
```

where `d` is a unit-norm refusal direction. For an input `x` producing
`y = W_o x`, that is algebraically

```
y  ←  y − α · (y · d) · d
```

So instead of rewriting `W_o`, we register a forward hook on `self_attn.o_proj`
and apply that line to its output. Same result, weights untouched.

**Both forms are proven numerically identical** in `tests/test_rt_intervention.py`
(max error 7.15e-07).

### Tensor parallelism is exact

`o_proj` is a `RowParallelLinear(reduce_results=False)` — it emits a *partial
sum per TP rank*, reduced later at the layer boundary. Because the edit is
**linear**,

```
f(Σᵣ yᵣ)  ==  Σᵣ f(yᵣ)
```

so hooking each rank's partial output and letting the existing all-reduce run is
identical to editing the reduced tensor. **No extra collective, no extra
communication.** Verified to 4.77e-07.

### α semantics

| α | retained coefficient `1−α` | projection removed `1−\|1−α\|` | behaviour |
|---|---|---|---|
| 0.0 | +1.00 | 0% | stock |
| 1.0 | 0.00 | 100% | |
| **1.4** | **−0.40** | **60%** | **recommended** |
| 1.7 | −0.70 | 70% | ceiling — see below |

`refusal_proj_reduction` in the DWM manifests uses the magnitude form
`1 − |1 − α|`, which counts overshoot past zero as "removed". For α=1.4 that is
60%, matching the recorded 59.88–59.93%.

---

## Results

Smoke-50, canonical harness (`--thinking --top-k 20`, temp 1.0, top_p 0.95,
max_tokens 4096), judged with the harness's own `refused_answer` flag — no human
in the loop:

| Arm | refused /50 | empty /50 |
|---|---|---|
| **RTI α=1.4** (stock weights) | **24** | 12 |
| Baked DWM α=1.4 (weight edit) | **24** | 11 |
| Stock base | 46 | 2 |

**Exact match with the weight edit; 22 points below stock.**

### Read this before quoting the numbers

- **n=50 is POC-grade.** It is enough to show the mechanism works and lands on
  the same operating point. It is *not* a release gate. A 443-item run is owed.
- **Manual rubric is noisy.** Two runs at *identical config* disagreed on 13/50
  items — intra-rater drift on the R/C boundary plus temp-1.0 decode variance.
  Use the mechanical `refused_answer` count, not a hand-scored rate, for any
  tight claim.
- **Red lines were not re-tested.** The 9-row `SR_RELEASE_REDLINE` holdout was
  not run against this arm. Until it is, do not claim red-line parity with the
  weight edit.
- **tp=4 only.** TP-linearity is proven analytically and on CPU; other
  topologies (tp=8, EP-only) are untested.

---

## Alpha: 1.7 is the ceiling

**1.7 is the highest value this package will accept.** It is clamped in code
(`ALPHA_MAX`), and a warning is logged if a config or toggle asks for more.

Why it is a ceiling and not just a default:

- At α=1.0 the refusal projection is fully removed (`1−α = 0`). Beyond that you
  are **inverting** the projection, pushing further into the opposite direction
  rather than merely cancelling.
- **1.4 is the recommended and best-tested value.** It was selected by operator
  review across α ∈ {1.0, 1.4, 1.5, 1.6, 1.7} and is the arm with the most
  evidence behind it.
- **Above 1.4, expect degradation.** The measured pattern across α=1.5/1.6 was
  increasing incoherence — more empty and looping answers — for little additional
  compliance. α=1.7 is the outer edge of what was explored.

**Treat α>1.4 as experimental.** If you raise it, re-run a smoke set and check
the empty/looping rate (`answer_chars == 0`) before trusting the output. The
clamp exists so a typo cannot silently produce a broken server — it is not a
recommendation to run at the limit.

---

## Quick start (sglang — verified)

```bash
# 1. Point at this package
export RTI_HOME=/path/to/Blackfrost_AI_MinimaxM31_RTI
export PYTHONPATH=$RTI_HOME/rt:${PYTHONPATH:-}
export BLACKFROST_RT_INTERVENTION=$RTI_HOME/rt_config.json

# 2. Serve stock weights — the hook installs at import time
sglang serve \
  --model-path /path/to/MiniMax-M3.1-Flash \
  --served-model-name MiniMax-3.1-Flash-RTI \
  --tp 4 --context-length 1048576 \
  --trust-remote-code --host 0.0.0.0 --port 30000
```

> **If sglang runs in Docker, this host-side form will not work.** `sglang` is
> not on the host `PATH`; the server must be `docker exec`'d into the container
> that owns the GPUs. Use `rt/serve_container.sh` for that case — it sets the
> same `PYTHONPATH` / `BLACKFROST_RT_INTERVENTION` inside the container and
> defaults to 1M context. Symptom of getting this wrong: the script prints its
> banner, then exits with `exec: sglang: not found`, and no server ever binds.

### Context length

`--context-length` defaults to **1048576 (1M)** in both serve scripts; the
checkpoint's `text_config.max_position_embeddings` is `1048576`. A 65536 cap
silently truncates long work and surfaces in clients as:

```
max_completion_tokens is too large: 128000.
This model supports at most 65536 completion tokens.
```

Confirm what is actually serving — do not infer it from the launch command:

```bash
curl -s localhost:30000/get_server_info | jq '.context_length'   # expect 1048576
```

Confirm it bound before trusting traffic:

```bash
grep "rt-intervention" serve.log
# [rt-intervention] pre-staged 696 direction tensors across 4 GPU(s)
# [rt-intervention] alpha buffers: 12 device tensors, initial alpha=1.400
# [rt-intervention] toggle poller started on .../rt/toggle.json (1.0s)
# [rt-intervention] INSTALLED on 58 layers via sglang (alpha=1.400, ...)
```

**`INSTALLED` is the line that matters.** Everything before it is setup; if you
do not see it, the hooks did not bind and you are serving stock.

---

## Turning it on and off

Write the toggle file. It is re-read every second; no restart.

```bash
# ON at the recommended strength
echo '{"enabled": true,  "alpha": 1.4}' > $RTI_HOME/rt/toggle.json

# OFF — identical to stock
echo '{"enabled": false, "alpha": 1.4}' > $RTI_HOME/rt/toggle.json

# ON, but inert (same effect as off, keeps the code path live)
echo '{"enabled": true,  "alpha": 0.0}' > $RTI_HOME/rt/toggle.json
```

Each TP rank logs its transition:

```
[rt-intervention] toggle -> enabled=True  alpha=1.400 (effective 1.400)
[rt-intervention] toggle -> enabled=False alpha=1.400 (effective 0.000)
```

Deleting the toggle file reverts to the static values in `rt_config.json`.

### Verify the toggle actually does something

Do not trust that the config was read — **observe different output**. This
package shipped a bug once where the toggle was read cleanly and had no effect,
because a Python-float α had been baked into the CUDA graph. That is why α now
lives in device memory.

```bash
# same prompt, two settings, greedy decode — output must differ
curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-3.1-Flash-RTI","messages":[{"role":"user","content":"..."}],"temperature":0}' \
  | jq -r '.choices[0].message.content' | wc -c
```

Or run the included probe: `python rt/toggle_probe.py <tag>`.

---

## vLLM wiring

> **Status: UNTESTED.** vLLM day-0 support for MiniMax-M3.1-Flash was not
> confirmed at time of writing. The mechanism is framework-agnostic — it needs
> only the decoder-layer class and the path to `o_proj` — so this should port,
> but **it has not been run.** Treat the below as a starting point, not a
> recipe, and update this file when it works.

The hook resolves its target through a small backend registry in
`rt/rt_intervention.py` (`_resolve_backend`). sglang is first; vLLM entries are
tried only if sglang is absent or you pin it:

```bash
export BLACKFROST_RT_BACKEND=vllm
```

### What must be true for vLLM

1. **A decoder-layer class exists** with a per-layer `layer_id` (or equivalent)
   and an `o_proj` on the attention module. If vLLM names it differently, add a
   row to `_resolve_backend`.
2. **`o_proj` returns a tensor or a tuple whose first element is the activation.**
   The hook handles both. If vLLM fuses the output projection into a combined
   QKV-O kernel, there is no `o_proj` module to hook and you must hook the fused
   op instead — check this first, it is the most likely porting blocker.
3. **CUDA graphs are on.** The hook is capture-safe by construction (directions
   pre-staged, α in device memory, no host allocation in the hot path), but
   confirm with `enforce_eager=False`.

### Minimal diff to try

```python
# after the model is constructed, before serving
from rt_intervention import install   # or rely on sitecustomize via PYTHONPATH
```

Then run the same four-line verification: `INSTALLED` in the log, a smoke set
with the toggle off, the same set with the toggle on, and confirm the outputs
differ. If they do not differ, **the hook is not firing** — do not proceed to
benchmarking.

Update this section and the status table when vLLM is confirmed working.

---

## Package layout

```
Blackfrost_AI_MinimaxM31_RTI/
├── README.md                  ← you are here
├── AGENTS.md                  ← integration contract for agents
├── rt_config.json             ← static config (α, layers, paths)
├── requirements.txt           ← torch + numpy (+ pytest for the suite)
├── rt/
│   ├── rt_intervention.py     ← the hook installer
│   ├── sitecustomize.py       ← PYTHONPATH bootstrap for worker processes
│   ├── serve_rt.sh            ← reference launch script (bare-metal sglang, tp=4)
│   ├── serve_container.sh     ← Docker wrapper — use this when sglang is in a container
│   ├── smoke50_rt.py          ← standalone smoke-50 refusal probe
│   └── toggle.json            ← live on/off + α
├── banks/
│   ├── directions.npy         ← 59 × 6144 float32, unit-norm
│   └── manifest.json          ← provenance + hashes
├── tests/
│   └── test_rt_intervention.py ← 8 tests, CPU only, no checkpoint needed
├── docs/
│   ├── RUNBOOK-next-experiment.md ← how to resume: gates, sweep, stop rules
│   └── METHOD-runtime-search.md   ← the general procedure, model-agnostic
└── evidence/
    ├── POC-RESULT.md          ← full result write-up
    └── judged_verdicts_*.json ← per-item verdicts
```

### Where to go next

| you want to | read |
|---|---|
| understand the mechanism | this file, §"What it actually does" |
| wire it into a server | `AGENTS.md` |
| **run the next experiment** | **`docs/RUNBOOK-next-experiment.md`** |
| apply this to another model | `docs/METHOD-runtime-search.md` |
| see what was proven | `evidence/POC-RESULT.md` |

### The bank

`banks/directions.npy` — shape `(59, 6144)`, float32, unit-norm, one row per
layer tap.

```
sha256  3b84104286223d99e6e385c1a010be9082adb5392edd8540b53998e48306dff3
```

Derived from an approved **32 harmful / 32 harmless** contrast bank — 59
residual-stream taps over 6144-dim hidden states. Layer `L` maps to row `L` for
`L = 2..58`; **layer 59 uses row 58** (59 taps, 58 distinct layers — the last is
a recorded fallback, matching the DWM build).

The SHA-256 above is checked at load. **A mismatch aborts installation** — the
server will run stock rather than apply a wrong direction. If you regenerate the
bank, update `directions_sha256` in `rt_config.json`.

---

## Tests

Requires `torch` and `numpy` only. On a clean machine:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt           # CPU wheel of torch is enough
```

Then, either runner works:

```bash
python -m pytest tests/test_rt_intervention.py -v   # 8 passed
cd tests && python test_rt_intervention.py          # ALL TESTS PASSED
```

CPU only — no GPU, no checkpoint. Eight tests:

| test | proves |
|---|---|
| `test_math_matches_weight_edit` | activation edit == weight edit (7.15e-07) |
| `test_tp_linearity` | per-rank edit == reduced edit (4.77e-07) |
| `test_alpha_semantics` | α table above is what the code does |
| `test_alpha_clamp` | 1.7 ceiling enforced, negatives clamped |
| `test_tuple_return_shape` | `(output, bias)` tuple preserved |
| `test_toggle_changes_behaviour_via_buffer` | toggle works under CUDA graphs |
| `test_layer_fallback` | layer 59 → row 58 |
| `test_fail_closed_on_bad_hash` | bad bank → no hooks |

---

## Troubleshooting

| symptom | cause |
|---|---|
| no `INSTALLED` line | `PYTHONPATH` does not include `rt/`, or `BLACKFROST_RT_INTERVENTION` unset |
| `INSTALLED` but output identical to stock | toggle has `enabled:false` or `alpha:0.0` |
| toggle written, no behaviour change | check each TP rank logged a `toggle ->` line; verify the path in the log is the file you edited |
| `fail-closed: hash mismatch` | bank was regenerated — update `directions_sha256` |
| `ModuleNotFoundError: torch` | install deps: `pip install -r requirements.txt` |
| `exec: sglang: not found` | sglang lives in a container — use `rt/serve_container.sh`, not `serve_rt.sh` |
| client says "supports at most 65536 completion tokens" | server launched with a 64K cap; check `/get_server_info` → `context_length` |
| `no supported backend found` | neither sglang nor vLLM importable; set `BLACKFROST_RT_BACKEND` |
| shape error on load | bank must be `(59, 6144)`; hidden size must match the checkpoint |
| response has empty `content` | **thinking model** — a small `max_tokens` is consumed by `reasoning_content` and the turn ends with `finish_reason: "length"`. Raise `max_tokens` (≥2000 for a short answer) or read `reasoning_content` too |

---

## Scope and limitations

- **Not a safety bypass.** This removes *false* refusals on technical content.
  Red lines are intended to hold; they were not re-verified for this arm (see
  above). Run the red-line holdout before any release claim.
- **Single checkpoint.** Built for `MiniMaxAI/MiniMax-M3.1-Flash` (6144 hidden,
  62 layers, MXFP8/NVFP4). A different model needs a different bank.
- **Weights are never written.** The hook is read-only with respect to the
  checkpoint.
- **Dormant by default.** Without `BLACKFROST_RT_INTERVENTION`, importing the
  module does nothing.

## License / provenance

Blackfrost AI. Internally derived refusal directions — no third-party training
data is redistributed here beyond the direction vectors themselves.