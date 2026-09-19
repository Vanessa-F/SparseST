"""Checks on the sparse cell and the encoder-decoder that wraps it.

Runs standalone (``python tests/test_model.py``) and under pytest if it is installed.
Tests needing a GPU are skipped when none is available.
"""

import pickle
import sys

import torch

from sparsest.config import load_config
from sparsest.models import SparseST
from sparsest.models.cell import signed_shrink
from sparsest.utils import build_model

TOL = 1e-6


def test_config_survives_pickle():
    """Configs are handed to torch.multiprocessing.spawn, so they must pickle.

    Regression test: Config.__getattr__ used to resolve pickle's __setstate__ probe
    through the config dict, recursing until the stack blew and training died on launch.
    """
    cfg = load_config("configs/ipad.yaml")
    back = pickle.loads(pickle.dumps(cfg))
    assert back.window == cfg.window
    assert back.train.lr == cfg.train.lr
    assert back.data.root == cfg.data.root


def test_signed_shrink_zeroes_below_threshold():
    delta = torch.tensor([-0.15, -0.05, 0.0, 0.05, 0.15])
    out = signed_shrink(delta, torch.tensor(0.2))
    assert torch.all(out == 0), out


def test_signed_shrink_keeps_sign_and_shrinks():
    delta = torch.tensor([-0.50, 0.50, -0.80, 0.80])
    threshold = torch.tensor(0.2)
    out = signed_shrink(delta, threshold)
    expected = torch.tensor([-0.30, 0.30, -0.60, 0.60])
    assert torch.allclose(out, expected, atol=TOL), (out, expected)
    # Sign is preserved, which is what separates this from magnitude thresholding.
    assert torch.all(torch.sign(out) == torch.sign(delta))


def test_signed_shrink_is_differentiable_in_threshold():
    delta = torch.tensor([0.5, -0.5])
    threshold = torch.tensor(0.2, requires_grad=True)
    signed_shrink(delta, threshold).abs().sum().backward()
    # Raising the threshold shrinks both survivors, so the gradient is -2.
    assert torch.allclose(threshold.grad, torch.tensor(-2.0), atol=TOL), threshold.grad


def test_zero_threshold_is_identity():
    delta = torch.randn(64)
    out = signed_shrink(delta, torch.tensor(0.0))
    assert torch.allclose(out, delta, atol=TOL)


def test_clamp_thresholds_removes_negatives():
    cell = SparseST(
        window=4, shape=(8, 8), batch_size=2, input_channels=1, filter_size=3,
        num_features=4,
    )
    with torch.no_grad():
        cell.threshold_x.copy_(torch.tensor([-0.5, 0.1, -0.2, 0.3]))
    cell.clamp_thresholds()
    assert torch.all(cell.threshold_x >= 0), cell.threshold_x
    # Non-negative entries are left alone.
    assert torch.allclose(cell.threshold_x, torch.tensor([0.0, 0.1, 0.0, 0.3]), atol=TOL)


def test_divisor_and_threshold_shape_match_originals():
    # The per-dataset code hardcoded these divisors; they must fall out of the formula.
    for name, divisor, window in [("ipad", 168, 21), ("mnist", 80, 10)]:
        cfg = load_config(f"configs/{name}.yaml")
        model = build_model(cfg)
        assert model.normalizer == divisor, (name, model.normalizer, divisor)
        assert model.encoder.lstm1.threshold_x.shape == (window,)


def test_forward_shapes_on_gpu():
    if not torch.cuda.is_available():
        print("  (skipped: no CUDA device)")
        return

    cfg = load_config("configs/mnist.yaml")
    batch, window = 2, cfg.window
    model = build_model(cfg, device="cuda:0")
    model.batch_size = batch
    for module in model.modules():
        if isinstance(module, SparseST):
            module.batch_size = batch

    inputs = torch.rand(batch, window, cfg.input_channels, 64, 64, device="cuda:0")
    zero = torch.tensor(0.0, device="cuda:0")

    with torch.no_grad():
        l0, l1, pred = model(zero, zero, inputs)

    assert pred.shape == inputs.shape, (pred.shape, inputs.shape)
    # Both metrics are means of per-unit quantities, so they stay in [0, 1].
    assert 0.0 <= l0.item() <= 1.0, l0.item()
    assert 0.0 <= l1.item() <= 1.0, l1.item()
    print(f"  forward ok: pred={tuple(pred.shape)} l0={l0.item():.4f} l1={l1.item():.4f}")


def test_zero_threshold_gives_dense_deltas_on_gpu():
    """With thresholds at zero nothing is suppressed, so l0 sparsity stays low."""
    if not torch.cuda.is_available():
        print("  (skipped: no CUDA device)")
        return

    cfg = load_config("configs/mnist.yaml")
    batch = 2
    model = build_model(cfg, device="cuda:0")
    for module in model.modules():
        if isinstance(module, SparseST):
            module.batch_size = batch
            with torch.no_grad():
                module.threshold_x.zero_()
                module.threshold_h.zero_()

    inputs = torch.rand(batch, cfg.window, cfg.input_channels, 64, 64, device="cuda:0")
    zero = torch.tensor(0.0, device="cuda:0")
    with torch.no_grad():
        l0_low, _, _ = model(zero, zero, inputs)

    for module in model.modules():
        if isinstance(module, SparseST):
            with torch.no_grad():
                module.threshold_x.fill_(10.0)
                module.threshold_h.fill_(10.0)
    with torch.no_grad():
        l0_high, _, _ = model(zero, zero, inputs)

    # A huge threshold suppresses everything, so sparsity must rise.
    assert l0_high.item() > l0_low.item(), (l0_low.item(), l0_high.item())
    print(f"  sparsity responds to threshold: {l0_low.item():.4f} -> {l0_high.item():.4f}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - standalone runner reports and continues
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
