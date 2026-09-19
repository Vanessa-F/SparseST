"""IPAD video dataset.

Stored as ``(N, window, H, W, C)`` uint8 clips. Train and validation live in separate
``.npy`` files; each test file is one recorded sequence, evaluated on its own so that
per-frame anomaly scores line up with the label files.
"""

import numpy as np
import torch
import torch.utils.data as data

# Modes that read the training array.
_TRAIN_MODES = ("train", "unit")


def load_fixed_set(cfg, mode):
    """Load one split as ``(N, S, C, W, H)`` floats in ``[0, 1]``."""
    if mode in _TRAIN_MODES:
        path = cfg.data_path("train")
    elif mode == "valid":
        path = cfg.data_path("valid")
    elif mode in ("test", "visualize"):
        path = cfg.data_path("test")
    else:
        raise ValueError(f"unknown mode {mode!r}")

    dataset = np.load(path)
    # (N, S, H, W, C) -> (N, S, C, W, H); the spatial axes are swapped to match the
    # orientation the model was trained on.
    dataset = dataset.transpose(0, 1, 4, 3, 2) / 255.0
    return dataset


class IPADDataset(data.Dataset):
    """Clips of ``window`` frames.

    For train and validation the target is the whole clip (reconstruction). For test the
    target is the centre frame, which is the one the anomaly score is reported against.
    """

    def __init__(self, cfg, mode, transform=None):
        super().__init__()
        self.dataset = torch.from_numpy(load_fixed_set(cfg, mode)).float()
        self.length = self.dataset.shape[0]
        self.mode = mode
        self.transform = transform
        self.center = (cfg.window - 1) // 2

    def __getitem__(self, idx):
        input_images = self.dataset[idx]

        if self.mode in ("train", "valid"):
            target = input_images
        else:
            target = input_images[self.center]

        return [idx, input_images, target]

    def __len__(self):
        return self.length
