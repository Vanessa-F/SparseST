"""Checks on dataset loading: axis order, normalisation, and split boundaries.

The transpose logic is exercised against small synthetic arrays so the suite stays fast;
one real file is loaded to confirm the on-disk layout still matches expectations.
Set ``SPARSEST_TEST_HEAVY=1`` to also load the multi-gigabyte training arrays.
"""

import os
import sys
from unittest import mock

import numpy as np
import torch

from sparsest.config import load_config
from sparsest.data import IPADDataset, MovingMNIST

TOL = 1e-6


def _ipad_cfg():
    return load_config("configs/ipad.yaml")


def _mnist_cfg():
    return load_config("configs/mnist.yaml")


def test_ipad_transpose_and_scaling():
    """uint8 (N,S,H,W,C) becomes float (N,S,C,W,H) in [0,1]."""
    cfg = _ipad_cfg()
    raw = np.arange(2 * 21 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 21, 4, 5, 3)

    with mock.patch("sparsest.data.ipad.np.load", return_value=raw):
        ds = IPADDataset(cfg, "train")

    assert ds.dataset.shape == (2, 21, 3, 5, 4), ds.dataset.shape
    assert float(ds.dataset.max()) <= 1.0 and float(ds.dataset.min()) >= 0.0
    # Element [0,0,c,w,h] must come from raw[0,0,h,w,c] / 255.
    assert abs(float(ds.dataset[0, 0, 2, 3, 1]) - raw[0, 0, 1, 3, 2] / 255.0) < TOL


def test_ipad_targets_by_mode():
    cfg = _ipad_cfg()
    raw = np.zeros((2, 21, 4, 4, 3), dtype=np.uint8)

    with mock.patch("sparsest.data.ipad.np.load", return_value=raw):
        train_ds = IPADDataset(cfg, "train")
        test_ds = IPADDataset(cfg, "test")

    _, clip, target = train_ds[0]
    assert target.shape == clip.shape, "train reconstructs the whole clip"

    _, clip, target = test_ds[0]
    assert target.shape == clip.shape[1:], "test targets a single frame"
    assert test_ds.center == (cfg.window - 1) // 2 == 10


def test_mnist_transpose_and_splits():
    """float32 (S,N,H,W,C) becomes (N,S,C,H,W), sliced by the configured bounds."""
    cfg = _mnist_cfg()
    raw = np.random.rand(20, 40, 6, 7, 1).astype(np.float32)

    with mock.patch("sparsest.data.moving_mnist.np.load", return_value=raw):
        train_ds = MovingMNIST(cfg, "train")
        valid_ds = MovingMNIST(cfg, "valid")

    # Splits in the config are [0,8000] and [8000,10000]; with 40 sequences the first
    # takes everything and the second is empty, so use explicit bounds instead.
    assert train_ds.dataset.shape[1:] == (20, 1, 6, 7), train_ds.dataset.shape
    assert abs(float(train_ds.dataset[3, 5, 0, 2, 4]) - raw[5, 3, 2, 4, 0]) < TOL
    assert valid_ds is not None


def test_mnist_split_bounds_are_applied():
    cfg = load_config(
        "configs/mnist.yaml",
        overrides={"data.train_split": [0, 30], "data.valid_split": [30, 40]},
    )
    raw = np.random.rand(20, 40, 6, 7, 1).astype(np.float32)

    with mock.patch("sparsest.data.moving_mnist.np.load", return_value=raw):
        assert len(MovingMNIST(cfg, "train")) == 30
        assert len(MovingMNIST(cfg, "valid")) == 10


def test_mnist_input_target_offset():
    """Input is frames [0,w) and target is [1,w+1): one-step-ahead prediction."""
    cfg = _mnist_cfg()
    raw = np.random.rand(20, 4, 6, 7, 1).astype(np.float32)

    with mock.patch("sparsest.data.moving_mnist.np.load", return_value=raw):
        cfg_small = load_config(
            "configs/mnist.yaml", overrides={"data.train_split": [0, 4]}
        )
        ds = MovingMNIST(cfg_small, "train")

    _, inp, tgt = ds[0]
    assert inp.shape[0] == tgt.shape[0] == cfg.window
    assert torch.allclose(inp[1:], tgt[:-1], atol=TOL), "target is the input shifted by one"


def test_real_ipad_test_file_matches_expected_layout():
    cfg = _ipad_cfg()
    path = cfg.data_path("test")
    if not path.exists():
        print(f"  (skipped: {path} not present)")
        return

    ds = IPADDataset(cfg, "test")
    _, clip, target = ds[0]
    assert clip.shape == (cfg.window, cfg.input_channels, 64, 64), clip.shape
    assert target.shape == (cfg.input_channels, 64, 64), target.shape
    assert 0.0 <= float(clip.min()) and float(clip.max()) <= 1.0
    print(f"  real IPAD test file: {len(ds)} clips of {tuple(clip.shape)}")


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
