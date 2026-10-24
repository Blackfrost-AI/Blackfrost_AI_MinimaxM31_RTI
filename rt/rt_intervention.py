"""Blackfrost runtime intervention — MiniMax-M3.1-Flash.

Activation-space equivalent of the Domain Weight Modification (DWM) edit.

WHY THIS IS THE SAME MATH
-------------------------
The baked DWM edit is:   W_o <- W_o - alpha * d d^T W_o
so for an input x:       y = W_o x  ->  y - alpha (y.d) d

This module applies exactly that to the activation AFTER o_proj, leaving the
weights byte-identical to stock. Same alpha, same directions, same layers.

TP CORRECTNESS
--------------
`o_proj` is RowParallelLinear(reduce_results=False) — it emits a PARTIAL SUM
per TP rank, reduced later at the layer boundary. Because the edit is linear,

    f(sum_r y_r) == sum_r f(y_r)

so hooking the partial output on every rank and letting the boundary all-reduce
is mathematically identical to editing the reduced tensor. No extra collective.

CONTROL
-------
Intervention can be toggled at runtime without a restart by writing the toggle
file (JSON: {"enabled": true|false, "alpha": <float>}). It is re-read at most
once every `toggle_poll_seconds`. Deleting it reverts to the static config.

FAIL-CLOSED
-----------
If the config is missing, malformed, or the direction file fails its SHA-256
check, the hooks are NOT installed and a warning is logged. It never silently
serves un-intervened traffic while claiming to intervene.

BACKENDS
--------
The hook is framework-agnostic: it needs only the decoder-layer class and the
attribute path to the attention output projection. Backends are resolved in
order and the first that matches wins.

  sglang  VERIFIED  — 24/50 refused at alpha=1.4, matching the baked weight edit.
  vLLM    UNTESTED  — same mechanism, no published MiniMax-M3.1-Flash support
                      confirmed at time of writing. See README "vLLM wiring".

Set BLACKFROST_RT_BACKEND=sglang|vllm to pin a backend instead of auto-detecting.

Installed via PYTHONPATH so it reaches spawned worker processes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time

import numpy as np
import torch

logger = logging.getLogger("blackfrost.rt_intervention")

_ENV = "BLACKFROST_RT_INTERVENTION"
# Operator-specified ceiling. alpha above this is observed to degrade output
# quality (see README "Alpha"). Not a hard failure — clamped with a warning so a
# typo cannot silently produce a broken server.
ALPHA_MAX = 1.7
_STATE = {
    "installed": False,
    "alpha": 0.0,
    "enabled": False,
    "n_layers": 0,
    "calls": 0,
    "toggle_path": None,
}
_LOCK = threading.Lock()
_DIRS: dict[int, torch.Tensor] = {}
# Pre-staged on-device copies, keyed by (layer_id, device_index, dtype).
# CUDA graph capture forbids CPU->CUDA copies mid-capture ("cannot copy between
# CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is
# pinned"), so the hook must never call .to(). Everything is staged up front.
_DIRS_STAGED: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}
# alpha lives in DEVICE MEMORY, one 1-element tensor per (device, dtype).
#
# Why: sglang runs with CUDA graphs. A Python float is baked into the captured
# graph as a kernel argument, so on replay the hook body never runs and changing
# the toggle file has NO effect (observed: alpha=0.0 and alpha=1.4 produced
# byte-identical outputs). Reading alpha from a device tensor makes the captured
# kernels dereference current memory at replay time, so the toggle works without
# a recapture or restart.
_ALPHA_BUF: dict[int, torch.Tensor] = {}

# Per-layer alpha. Same capture-safety property as _ALPHA_BUF, but one scalar
# per edited layer instead of one per device. Populated only when the config (or
# the toggle file) supplies a list; otherwise the global path is used and this
# stays empty.
_LAYER_ALPHA_BUF: dict[int, torch.Tensor] = {}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_root() -> str:
    """Directory containing rt/ and banks/ (this file's parent's parent)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str) -> str:
    """Resolve a config path: absolute stays, relative goes from the repo root.

    Servers are commonly launched from an arbitrary cwd, so a relative path in
    the config would otherwise silently fail to resolve.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    cand = os.path.join(_repo_root(), path)
    return cand if os.path.exists(cand) else os.path.abspath(path)


def _load_config() -> dict | None:
    cfg_path = os.environ.get(_ENV)
    if not cfg_path:
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as e:
        logger.warning("[rt-intervention] cannot read config %s: %s", cfg_path, e)
        return None
    if not cfg.get("enabled", True):
        logger.info("[rt-intervention] config present but disabled")
        return None
    return cfg


def _load_directions(cfg: dict) -> dict[int, torch.Tensor] | None:
    path = _resolve(cfg["directions"])
    if not os.path.exists(path):
        logger.warning("[rt-intervention] directions missing: %s", path)
        return None

    want = cfg.get("directions_sha256")
    if want:
        got = _sha256(path)
        if got != want:
            logger.warning(
                "[rt-intervention] SHA-256 MISMATCH %s\n  want %s\n  got  %s",
                path, want, got,
            )
            return None
        logger.info("[rt-intervention] directions sha256 verified %s", got[:16])

    arr = np.load(path)
    if arr.ndim != 2:
        logger.warning("[rt-intervention] directions must be 2-D, got %s", arr.shape)
        return None

    # Unit-normalise defensively; the DWM edit assumed unit directions.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        logger.warning("[rt-intervention] degenerate direction rows detected")
        return None
    arr = arr / norms

    out = {}
    for layer in cfg.get("layers", []):
        idx = min(int(layer), arr.shape[0] - 1)  # L=59 falls back to row 58
        out[int(layer)] = torch.from_numpy(arr[idx].astype(np.float32))
    return out


def _clamp_alpha(alpha: float, where: str) -> float:
    """Clamp alpha to [0, ALPHA_MAX] with a warning.

    Above 1.7 the edit overshoots the refusal direction and output quality
    degrades (see README). Clamping rather than raising keeps a bad config from
    taking down a live server; the warning makes it visible.
    """
    if alpha < 0.0:
        logger.warning("[rt-intervention] %s: alpha %.3f < 0, clamped to 0.0", where, alpha)
        return 0.0
    if alpha > ALPHA_MAX:
        logger.warning(
            "[rt-intervention] %s: alpha %.3f exceeds tested ceiling %.1f — "
            "clamped. Values above %.1f may degrade output quality.",
            where, alpha, ALPHA_MAX, ALPHA_MAX,
        )
        return ALPHA_MAX
    return alpha


def _stage(d_ref: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the direction pre-staged on y's device/dtype.

    Must not allocate: this runs inside CUDA graph capture. The first call for a
    given (device, dtype) happens during warmup, so capture and replay both hit
    the cache.
    """
    key = (id(d_ref), y.device.index or 0, y.dtype)
    hit = _DIRS_STAGED.get(key)
    if hit is not None:
        return hit
    staged = d_ref.to(device=y.device, dtype=y.dtype).contiguous()
    _DIRS_STAGED[key] = staged
    return staged


def _alpha_for(y: torch.Tensor) -> torch.Tensor:
    """Current alpha as a device-resident [1] scalar matching y's dtype.

    Read from `_ALPHA_BUF` so CUDA-graph replay sees the live value. The cast is
    a device-side op (no CPU sync, no host allocation), so it is capture-safe.
    """
    key = (y.device.index or 0, y.dtype)
    buf = _ALPHA_BUF.get(key)
    if buf is None:
        buf = torch.zeros(1, device=y.device, dtype=y.dtype)
        # Seed from the current state so CPU/uncaptured use is correct too.
        eff = _STATE["alpha"] if _STATE.get("enabled", True) else 0.0
        buf.fill_(eff)
        _ALPHA_BUF[key] = buf
    return buf


def _layer_alpha_for(y: torch.Tensor, layer_id: int) -> torch.Tensor:
    """Per-layer alpha as a device-resident [1] scalar.

    Same capture-safety property as `_alpha_for`: the value lives in device
    memory and is refreshed by the toggle poller, so CUDA-graph replay always
    observes the live value rather than a Python scalar baked into the graph.

    Falls back to the global alpha when no per-layer vector is configured, so
    existing configs keep working unchanged.
    """
    key = (y.device.index or 0, y.dtype)

    # Global mode: no per-layer vector configured. Delegate so the global
    # buffer stays the single source of truth (unchanged behaviour).
    if not _STATE.get("per_layer"):
        return _alpha_for(y)

    buf = _LAYER_ALPHA_BUF.get(key)
    if buf is None:
        n = _STATE.get("n_layers") or 64
        vec = torch.zeros(n, device=y.device, dtype=y.dtype)
        pl = _STATE.get("per_layer")
        if pl:
            t = torch.as_tensor(pl, dtype=y.dtype, device=y.device)
            m = min(n, t.numel())
            vec[:m] = t[:m]
        else:
            vec.fill_(_STATE["alpha"] if _STATE.get("enabled", True) else 0.0)
        _LAYER_ALPHA_BUF[key] = vec
        buf = vec
    if layer_id < buf.numel():
        return buf[layer_id : layer_id + 1]
    return _alpha_for(y)


def _push_alpha(alpha: float) -> None:
    """Write a global alpha into every device buffer (outside capture).

    Also updates `_STATE` so a buffer created *after* this call seeds correctly.
    Buffers are created lazily on first forward, so a push that only touched
    existing buffers would silently no-op on a fresh process.
    """
    _STATE["alpha"] = float(alpha)
    _STATE["per_layer"] = None
    for buf in list(_ALPHA_BUF.values()):
        buf.fill_(float(alpha))
    for vec in list(_LAYER_ALPHA_BUF.values()):
        vec.fill_(float(alpha))


def _push_layer_alpha(vec_in) -> None:
    """Write a per-layer alpha vector into every device buffer (outside capture).

    Same lazy-creation caveat as `_push_alpha`: stash the vector in `_STATE` so
    buffers created later seed from it rather than from a stale global.
    """
    t = torch.as_tensor(vec_in, dtype=torch.float32).flatten()
    if not _STATE.get("enabled", True):
        t = torch.zeros_like(t)
    _STATE["per_layer"] = [float(v) for v in t.tolist()]
    for key, vec in list(_LAYER_ALPHA_BUF.items()):
        n = min(vec.numel(), t.numel())
        if n:
            vec[:n].copy_(t[:n].to(device=vec.device, dtype=vec.dtype))


def _edit(y: torch.Tensor, d_ref: torch.Tensor, layer_id: int) -> torch.Tensor:
    """y <- y - alpha_l * (y . d) d   (d unit-norm, alpha from device memory).

    Unconditional: alpha == 0 is the identity, so there is no Python-level
    branch for the graph to bake in.
    """
    d = _stage(d_ref, y)
    a = _layer_alpha_for(y, layer_id)
    proj = torch.matmul(y, d).unsqueeze(-1)
    return y - a * proj * d


def _make_hook(layer_id: int):
    d_ref = _DIRS[layer_id]

    def hook(module, inputs, output):
        with _LOCK:
            _STATE["calls"] += 1

        # sglang's RowParallelLinear.forward returns (output, output_bias).
        # Hooking the module means we must preserve that exact tuple shape or
        # the caller explodes unpacking it. Handle both forms defensively.
        if isinstance(output, tuple):
            if not output or not isinstance(output[0], torch.Tensor):
                return output
            return (_edit(output[0], d_ref, layer_id),) + tuple(output[1:])
        if isinstance(output, torch.Tensor):
            return _edit(output, d_ref, layer_id)
        return output

    return hook


def _toggle_poller() -> None:
    """Background thread: mirror the toggle file into device memory.

    Runs outside any capture context, so it is free to stat/read the file and
    fill_() the alpha buffers. This is what makes the runtime toggle work under
    CUDA graphs.
    """
    path = _STATE["toggle_path"]
    base_on = _STATE["_static_enabled"]
    base_alpha = _STATE["_static_alpha"]
    interval = _STATE.get("_poll_interval", 1.0)
    last_mtime = None
    while True:
        time.sleep(interval)
        try:
            mtime = os.stat(path).st_mtime_ns
        except OSError:
            on, alpha = base_on, base_alpha
            mtime = None
        else:
            if mtime == last_mtime:
                continue
            last_mtime = mtime
            try:
                with open(path) as f:
                    t = json.load(f)
                on = bool(t.get("enabled", base_on))
                raw = t.get("alpha", base_alpha)
                if isinstance(raw, (list, tuple)):
                    # Per-layer vector: clamp elementwise, keep the scalar view
                    # for logging as the mean.
                    vec = [
                        _clamp_alpha(float(v), "toggle") for v in raw
                    ]
                    alpha = sum(vec) / len(vec) if vec else 0.0
                    per_layer = vec
                else:
                    alpha = _clamp_alpha(float(raw), "toggle")
                    per_layer = None
            except Exception:
                on, alpha, per_layer = base_on, base_alpha, None

        eff = alpha if on else 0.0
        if eff != _STATE["alpha"] or on != _STATE["enabled"] or per_layer is not None:
            _STATE["alpha"] = eff
            _STATE["enabled"] = on
            if per_layer is None:
                _push_alpha(eff)
            else:
                _push_layer_alpha([v if on else 0.0 for v in per_layer])
            logger.warning(
                "[rt-intervention] toggle -> enabled=%s alpha=%.3f (effective %.3f)%s",
                on, alpha, eff, " per-layer" if per_layer is not None else "",
            )


# Surfaces to edit, as (attention_attr, projection_attr).
#
# Hybrid architectures split attention across two module types. On
# Qwen3.5-MoE / Sage-A1.5 the real index carries BOTH:
#     self_attn.o_proj     x 11
#     linear_attn.out_proj x 30   <- 73% of surfaces
# A hook that only knows `self_attn.o_proj` binds 11 of 41 matrices and reports
# success. That is the failure this list exists to prevent.
_DEFAULT_SURFACES = (
    ("self_attn", "o_proj"),
    ("linear_attn", "out_proj"),
)

# Known (module, class) targets per engine. `surfaces` is per-entry because a
# dense model simply has no linear_attn to bind.
_KNOWN_LAYER_CLASSES = (
    # sglang — MiniMax-M3.1 verified
    ("sglang", "sglang.srt.models.minimax_m3", "MiniMaxM3DecoderLayer", (("self_attn", "o_proj"),)),
    # sglang — Qwen3.5-MoE / Sage-A1.5 (hybrid: dense + linear attention)
    ("sglang", "sglang.srt.models.qwen3_5_moe", "Qwen3_5MoeDecoderLayer", _DEFAULT_SURFACES),
    ("sglang", "sglang.srt.models.qwen3_5", "Qwen3_5DecoderLayer", _DEFAULT_SURFACES),
    # vLLM — UNTESTED
    ("vllm", "vllm.model_executor.models.minimax_m3", "MiniMaxM3DecoderLayer", (("self_attn", "o_proj"),)),
    ("vllm", "vllm.model_executor.models.minimax_vl", "MiniMaxTextDecoderLayer", (("self_attn", "o_proj"),)),
)


def _import_layer_class(spec: str):
    """Resolve 'module:ClassName' to a class, or None."""
    if ":" not in spec:
        return None
    mod_path, cls_name = spec.split(":", 1)
    try:
        import importlib

        mod = importlib.import_module(mod_path.strip())
    except Exception as e:
        logger.warning("[rt-intervention] layer_class import failed (%s): %s", spec, e)
        return None
    return getattr(mod, cls_name.strip(), None)


def _resolve_backend(cfg=None):
    """Resolve (engine_name, decoder_layer_cls, surfaces) for the active model.

    `surfaces` is a tuple of (attn_attr, proj_attr). Returns None if nothing
    importable matched.

    Order:
      1. explicit `layer_class` in config or BLACKFROST_RT_LAYER_CLASS env
      2. known per-engine classes
    """
    cfg = cfg or {}
    pin = os.environ.get("BLACKFROST_RT_BACKEND", "").strip().lower()
    surfaces = tuple(
        tuple(s) for s in cfg.get("surfaces", _DEFAULT_SURFACES)
    )

    # 1. explicit override — the escape hatch for a new architecture.
    explicit = (os.environ.get("BLACKFROST_RT_LAYER_CLASS", "") or cfg.get("layer_class", "")).strip()
    if explicit:
        cls = _import_layer_class(explicit)
        if cls is not None:
            return (pin or "explicit"), cls, surfaces
        logger.warning(
            "[rt-intervention] explicit layer_class %r did not import; falling back",
            explicit,
        )

    # 2. known classes
    candidates = list(_KNOWN_LAYER_CLASSES)
    if pin:
        candidates = [c for c in candidates if c[0] == pin] or candidates

    for name, mod_path, cls_name, entry_surfaces in candidates:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        # An explicit `surfaces` in config wins over the per-entry default.
        return name, cls, (surfaces if cfg.get("surfaces") else entry_surfaces)
    return None


def _install() -> None:
    cfg = _load_config()
    if cfg is None:
        return

    dirs = _load_directions(cfg)
    if dirs is None:
        logger.warning("[rt-intervention] NOT installing hooks (direction load failed)")
        return

    _DIRS.update(dirs)

    # Pre-stage every direction onto every visible GPU in the activation dtypes
    # BEFORE any capture happens. This keeps the hook allocation-free, which is
    # required inside CUDA graph capture.
    if torch.cuda.is_available():
        try:
            n_dev = torch.cuda.device_count()
            for dev in range(n_dev):
                for dt in (torch.bfloat16, torch.float16, torch.float32):
                    for lid, d_ref in dirs.items():
                        _DIRS_STAGED[(id(d_ref), dev, dt)] = d_ref.to(
                            device=f"cuda:{dev}", dtype=dt
                        ).contiguous()
            logger.warning(
                "[rt-intervention] pre-staged %d direction tensors across %d GPU(s)",
                len(_DIRS_STAGED), n_dev,
            )
        except Exception as e:
            logger.warning("[rt-intervention] pre-stage failed (will stage lazily): %s", e)
    _STATE.update(
        {
            "_static_enabled": bool(cfg.get("enabled", True)),
            "_static_alpha": _clamp_alpha(float(cfg.get("alpha", 1.4)), "config"),
            "alpha": float(cfg.get("alpha", 1.4)),
            "enabled": bool(cfg.get("enabled", True)),
            "toggle_path": _resolve(cfg["toggle_file"]) if cfg.get("toggle_file") else None,
            "_poll_interval": float(cfg.get("toggle_poll_seconds", 1.0)),
            "n_layers": len(dirs),
        }
    )

    # Allocate the device alpha buffers and seed them with the configured value.
    if torch.cuda.is_available():
        eff0 = _STATE["_static_alpha"] if _STATE["_static_enabled"] else 0.0
        for dev in range(torch.cuda.device_count()):
            for dt in (torch.bfloat16, torch.float16, torch.float32):
                buf = torch.zeros(1, device=f"cuda:{dev}", dtype=dt)
                buf.fill_(eff0)
                _ALPHA_BUF[(dev, dt)] = buf
        logger.warning(
            "[rt-intervention] alpha buffers: %d device tensors, initial alpha=%.3f",
            len(_ALPHA_BUF), eff0,
        )
        if _STATE["toggle_path"]:
            threading.Thread(
                target=_toggle_poller, name="rt-toggle-poller", daemon=True
            ).start()
            logger.warning(
                "[rt-intervention] toggle poller started on %s (%.1fs)",
                _STATE["toggle_path"], _STATE["_poll_interval"],
            )

    # Patch the decoder-layer constructor so we catch the model whenever it is
    # actually built, regardless of import order.
    backend = _resolve_backend(cfg)
    if backend is None:
        logger.warning(
            "[rt-intervention] no supported backend found (tried sglang, vllm). "
            "Hooks NOT installed."
        )
        return
    backend_name, layer_cls, surfaces = backend

    orig_init = layer_cls.__init__
    if getattr(orig_init, "_blackfrost_patched", False):
        return

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        lid = getattr(self, "layer_id", None)
        if lid is None:
            return
        if int(lid) not in _DIRS:
            return
        hit = False
        for attn_attr, proj_attr in surfaces:
            attn = getattr(self, attn_attr, None)
            if attn is None:
                continue
            proj = getattr(attn, proj_attr, None)
            if proj is None:
                continue
            proj.register_forward_hook(_make_hook(int(lid)))
            hit = True
        if not hit:
            # A decoder layer with no matching projection means the surface list
            # is wrong for this architecture. Warn once per layer id so a
            # partially-bound install cannot pass silently.
            logger.warning(
                "[rt-intervention] layer %s bound 0 surfaces (looked for %s)",
                lid, [f"{a}.{p}" for a, p in surfaces],
            )

    patched_init._blackfrost_patched = True
    layer_cls.__init__ = patched_init

    _STATE["installed"] = True
    _STATE["backend"] = backend_name
    _STATE["surfaces"] = [f"{a}.{p}" for a, p in surfaces]
    _STATE["n_layers"] = max(dirs) + 1 if dirs else 0
    logger.warning(
        "[rt-intervention] INSTALLED on %d layers via %s (surfaces=%s, alpha=%.3f, toggle=%s)",
        len(dirs), backend_name,
        ",".join(f"{a}.{p}" for a, p in surfaces),
        _STATE["alpha"], _STATE["toggle_path"] or "none",
    )


try:
    _install()
except Exception as e:  # never break serving
    logger.warning("[rt-intervention] install failed: %s", e)


def status() -> dict:
    return {k: v for k, v in _STATE.items() if not k.startswith("_")}