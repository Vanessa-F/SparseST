"""Scalarisation of the two competing objectives: reconstruction error and occupancy."""

import torch


def stch_loss(w_mse, mse, occupancy, mu=1.0):
    """Smooth Tchebycheff scalarisation of ``(mse, occupancy)``.

    ``w_mse`` is the preference weight for reconstruction error; ``1 - w_mse`` weights
    occupancy. Sweeping it traces out the Pareto front that the GP search models.

    At the extremes (``w_mse`` of exactly 0 or 1) the smooth maximum degenerates -- it
    would still mix in the ignored objective through the log-sum-exp -- so those two
    points fall back to the plain weighted sum.

    Implemented via ``logsumexp`` rather than ``log(exp(..) + exp(..))``; the two are
    mathematically identical but the former does not overflow for large inputs.
    """
    if w_mse in (0.0, 1.0):
        return w_mse * mse + (1.0 - w_mse) * occupancy

    terms = torch.stack(
        [
            torch.as_tensor(w_mse * mse / mu),
            torch.as_tensor((1.0 - w_mse) * occupancy / mu),
        ]
    )
    return mu * torch.logsumexp(terms, dim=0)


# Backwards-compatible alias matching the original function name.
STCH_loss = stch_loss
