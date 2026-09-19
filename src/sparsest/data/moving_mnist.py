"""Moving-MNIST dataset.

Stored as ``(S, N, H, W, C)`` float32 already scaled to ``[0, 1]``, with 20 frames per
sequence. Train and validation are index slices of the same array; the model consumes
the first ``window`` frames and predicts the next ``window``, offset by one.
"""

import numpy as np
import torch
import torch.utils.data as data

# Modes that read the training array; the rest read the test array.
_TRAIN_MODES = ("train", "valid")


def load_fixed_set(cfg, mode):
    """Load one split as ``(N, S, C, H, W)`` floats in ``[0, 1]``."""
    if mode in _TRAIN_MODES:
        path = cfg.data_path("train")
    elif mode in ("test", "visualize", "unit"):
        path = cfg.data_path("test")
    else:
        raise ValueError(f"unknown mode {mode!r}")

    dataset = np.load(path)
    # (S, N, H, W, C) -> (N, S, C, H, W)
    dataset = np.transpose(dataset, (1, 0, 4, 2, 3)).astype(np.float32)

    if mode in _TRAIN_MODES:
        start, stop = cfg.data[f"{mode}_split"]
        dataset = dataset[start:stop]

    return dataset


class MovingMNIST(data.Dataset):
    """Sequences of ``2 * window`` frames, served as a one-step-ahead prediction pair."""

    def __init__(self, cfg, mode, transform=None):
        super().__init__()
        self.dataset = torch.from_numpy(load_fixed_set(cfg, mode))
        self.length = self.dataset.shape[0]
        self.mode = mode
        self.transform = transform
        self.window = cfg.window

    def __getitem__(self, idx):
        images = self.dataset[idx]  # (S, C, H, W)
        input_img = images[0 : self.window]
        output_img = images[1 : self.window + 1]
        return [idx, input_img, output_img]

    def __len__(self):
        return self.length
