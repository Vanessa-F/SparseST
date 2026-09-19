"""SparseST: multi-objective threshold search for sparse spatio-temporal models.

A delta-thresholded ConvLSTM learns, per timestep, how large a change must be before it
is worth computing on. Those thresholds trade reconstruction error against activation
occupancy, and a multi-task Gaussian process searches the preference weight between the
two objectives to trace out the Pareto front.
"""

__version__ = "0.1.0"

from .config import Config, load_config
from .losses import stch_loss
from .models import ED, Decoder, Encoder, SparseST

__all__ = [
    "Config",
    "load_config",
    "stch_loss",
    "SparseST",
    "Encoder",
    "Decoder",
    "ED",
    "__version__",
]
