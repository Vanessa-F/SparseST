"""Dataset lookup by config name."""

from .ipad import IPADDataset
from .moving_mnist import MovingMNIST

DATASETS = {
    "ipad": IPADDataset,
    "moving_mnist": MovingMNIST,
}


def build_dataset(cfg, mode):
    """Instantiate the dataset named by ``cfg.dataset`` for ``mode``."""
    try:
        dataset_cls = DATASETS[cfg.dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {cfg.dataset!r} (available: {sorted(DATASETS)})"
        ) from None
    return dataset_cls(cfg, mode)
