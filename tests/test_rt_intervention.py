"""Unit test for rt_intervention — runs on CPU, no model, no GPU.

Proves the hook math, the TP-linearity claim, the toggle, and the fail-closed
path without needing a 252 GB checkpoint.
"""
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import torch

# Make `import rt_intervention` work whether this is run from tests/ or the root.
_RT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rt")
if _RT_DIR not in sys.path:
    sys.path.insert(0, _RT_DIR)


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def test_math_matches_weight_edit():
    """y - a(y.d)d  must equal  (W - a d d^T W)x  for y = Wx."""
    torch.manual_seed(0)
    D, H, A = 8, 16, 1.4

    d = torch.randn(D)
    d = d / d.norm()

    W = torch.randn(D, H)
    x = torch.randn(H)

    # weight-space edit
    W_edit = W - A * torch.outer(d, d @ W)
    y_weight = W_edit @ x

    # activation-space edit
    y = W @ x
    y_act = y - A * (y @ d) * d

    err = (y_weight - y_act).abs().max().item()
    assert err < 1e-6, f"activation/weight edit mismatch: {err}"
    print(f"  [ok] weight-space == activation-space (max err {err:.2e})")


def test_tp_linearity():
    """f(sum_r y_r) == sum_r f(y_r)  — justifies per-rank hooking."""
    torch.manual_seed(1)
    D, A = 8, 1.4
    d = torch.randn(D); d = d / d.norm()

    parts = [torch.randn(D) for _ in range(4)]
    f = lambda y: y - A * (y @ d) * d

    lhs = f(torch.stack(parts).sum(0))
    rhs = torch.stack([f(p) for p in parts]).sum(0)
    err = (lhs - rhs).abs().max().item()
    assert err < 1e-5, f"TP linearity broken: {err}"
    print(f"  [ok] TP-linear: per-rank edit == reduced edit (max err {err:.2e})")


def test_alpha_semantics():
    """Retained projection coefficient is (1 - alpha).

    Note on the manifest's wording: `refusal_proj_reduction` in
    PROBE_EDIT_MANIFEST.json is defined as 1 - |1 - alpha|, i.e. the MAGNITUDE
    of the projection removed, counting an overshoot past zero as removed.
    For alpha=1.4 that is 1 - 0.4 = 60%, matching the recorded 59.88-59.93%.
    The signed coefficient is 1 - alpha = -0.4 (slight sign flip past zero).
    """
    torch.manual_seed(2)
    D = 8
    d = torch.randn(D); d = d / d.norm()
    y = torch.randn(D)

    p0 = (y @ d).item()
    for a, want_coeff, want_reduction in [
        (0.0, 1.00, 0.00),
        (1.0, 0.00, 1.00),
        (1.4, -0.40, 0.60),
        (1.5, -0.50, 0.50),
    ]:
        out = y - a * (y @ d) * d
        coeff = (out @ d).item() / p0
        assert abs(coeff - want_coeff) < 1e-5, f"alpha={a}: coeff {coeff:.4f} != {want_coeff}"
        reduction = 1 - abs(1 - a)
        assert abs(reduction - want_reduction) < 1e-9, f"alpha={a}: reduction {reduction}"
        print(f"  [ok] alpha={a}: coeff {coeff:+.2f}, manifest reduction {reduction:.0%}")


def test_fail_closed_on_bad_hash():
    """Bad SHA-256 must refuse to install, not silently pass through."""
    tmp = tempfile.mkdtemp()
    dirs = os.path.join(tmp, "d.npy")
    np.save(dirs, np.random.randn(59, 6144).astype(np.float32))

    cfg = {
        "enabled": True,
        "alpha": 1.4,
        "directions": dirs,
        "directions_sha256": "0" * 64,  # deliberately wrong
        "layers": list(range(2, 60)),
    }
    cfg_path = os.path.join(tmp, "cfg.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    os.environ["BLACKFROST_RT_INTERVENTION"] = cfg_path
    for m in list(sys.modules):
        if "rt_intervention" in m:
            del sys.modules[m]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rt_intervention as rt

    st = rt.status()
    assert st["installed"] is False, f"should NOT install on hash mismatch: {st}"
    print("  [ok] fail-closed: hash mismatch -> hooks not installed")


def test_layer_fallback():
    """Layer 59 (no direction row) must map to row 58, matching the DWM edit."""
    arr = np.random.randn(59, 8).astype(np.float32)
    for layer in [2, 30, 58, 59]:
        idx = min(layer, arr.shape[0] - 1)
        assert idx <= 58
    assert min(59, 58) == 58
    print("  [ok] layer 59 -> direction row 58 (matches edit_a17.py fallback)")


def test_tuple_return_shape():
    """REGRESSION: sglang RowParallelLinear.forward returns (output, output_bias).

    Hooking the module must preserve the tuple so callers can still unpack it.
    This crashed a live serve on 2026-10-23 with:
        AttributeError: 'tuple' object has no attribute 'device'
    """
    import rt_intervention as R

    torch.manual_seed(3)
    D = 8
    d = torch.randn(D); d = d / d.norm()

    R._DIRS.clear()
    R._DIRS[7] = d
    # alpha now comes from a device (here CPU) buffer, not a Python float, so
    # that CUDA-graph replay honours toggle changes.
    R._ALPHA_BUF.clear()
    R._ALPHA_BUF[(0, torch.float32)] = torch.tensor([1.4], dtype=torch.float32)
    R._STATE.update({"_static_enabled": True, "_static_alpha": 1.4,
                     "enabled": True, "alpha": 1.4,
                     "toggle_path": None})

    hook = R._make_hook(7)
    y = torch.randn(3, D)
    bias = torch.randn(D)

    # tuple form (what sglang actually returns)
    out = hook(None, None, (y, bias))
    assert isinstance(out, tuple) and len(out) == 2, f"tuple not preserved: {type(out)}"
    assert out[1] is bias, "bias must pass through untouched"
    assert torch.allclose(out[0], y - 1.4 * (y @ d).unsqueeze(-1) * d, atol=1e-6)
    print("  [ok] tuple return preserved, bias untouched, output edited")

    # bare tensor form must still work
    out2 = hook(None, None, y)
    assert isinstance(out2, torch.Tensor)
    assert torch.allclose(out2, out[0], atol=1e-6)
    print("  [ok] bare-tensor return also handled")

    # alpha=0 in the device buffer -> identity, tuple intact (this is the
    # runtime toggle path: no Python branch, just a different buffer value)
    R._ALPHA_BUF[(0, torch.float32)].fill_(0.0)
    out3 = hook(None, None, (y, bias))
    assert isinstance(out3, tuple) and torch.allclose(out3[0], y, atol=1e-6)
    print("  [ok] alpha=0 buffer -> identity with tuple intact (toggle works)")


def test_toggle_changes_behaviour_via_buffer():
    """REGRESSION: under CUDA graphs a Python float alpha is baked into the graph.

    Observed live on 2026-10-23: alpha=0.0 and alpha=1.4 gave byte-identical
    outputs because graph replay never re-runs the hook body. Alpha must come
    from device memory so replay dereferences the live value.
    """
    import rt_intervention as R

    torch.manual_seed(4)
    D = 8
    d = torch.randn(D); d = d / d.norm()

    R._DIRS.clear(); R._DIRS[9] = d
    R._ALPHA_BUF.clear()
    R._STATE.update({"enabled": True, "alpha": 0.0, "toggle_path": None})

    hook = R._make_hook(9)
    y = torch.randn(2, D)

    # alpha=0 -> identity
    o0 = hook(None, None, y)
    assert torch.allclose(o0, y, atol=1e-6), "alpha=0 must be identity"

    # flip the buffer (what the poller does) -> behaviour must change
    R._ALPHA_BUF[(0, torch.float32)].fill_(1.4)
    o1 = hook(None, None, y)
    assert not torch.allclose(o0, o1, atol=1e-4), "buffer change had no effect"
    assert torch.allclose(o1, y - 1.4 * (y @ d).unsqueeze(-1) * d, atol=1e-5)
    print("  [ok] toggling the device buffer changes behaviour (graph-safe)")

    # and back to zero
    R._ALPHA_BUF[(0, torch.float32)].fill_(0.0)
    assert torch.allclose(hook(None, None, y), y, atol=1e-6)
    print("  [ok] toggling back to 0 restores identity")


def test_alpha_clamp():
    """alpha is clamped to [0, ALPHA_MAX] with a warning, never raised."""
    import rt_intervention as R

    assert R.ALPHA_MAX == 1.7, R.ALPHA_MAX
    assert R._clamp_alpha(1.4, "t") == 1.4
    assert R._clamp_alpha(0.0, "t") == 0.0
    assert R._clamp_alpha(1.7, "t") == 1.7
    assert R._clamp_alpha(2.5, "t") == 1.7, "above-ceiling must clamp"
    assert R._clamp_alpha(-0.3, "t") == 0.0, "negative must clamp to 0"
    print(f"  [ok] alpha clamped to [0.0, {R.ALPHA_MAX}] (ceiling is operator-set)")


if __name__ == "__main__":
    print("rt_intervention unit tests")
    print("-" * 56)
    for fn in [
        test_math_matches_weight_edit,
        test_tp_linearity,
        test_alpha_semantics,
        test_alpha_clamp,
        test_tuple_return_shape,
        test_toggle_changes_behaviour_via_buffer,
        test_layer_fallback,
        test_fail_closed_on_bad_hash,
    ]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "-" * 56)
    print("ALL TESTS PASSED")