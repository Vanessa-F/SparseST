"""Dataset implementations and the config-driven registry."""

from .ipad import IPADDataset
from .moving_mnist import MovingMNIST
from .registry import DATASETS, build_dataset

__all__ = ["IPADDataset", "MovingMNIST", "DATASETS", "build_dataset"]
