#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Bayesian-optimisation search over the preference weight.

A multi-task Gaussian process models both objectives -- reconstruction error and
occupancy -- as a function of the preference weight ``w``. Each iteration fits the GP to
the points measured so far, picks the untried weight the GP is least certain about,
trains a model at that weight, and folds the result back in. Sweeping ``w`` this way
traces out the Pareto front far more cheaply than a uniform grid.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import gpytorch
import torch

from .cli import parse_args
from .config import REPO_ROOT

# Two weights are treated as the same candidate within this tolerance. Comparing floats
# exactly would leave already-evaluated points in the pool and eventually exhaust it.
W_TOLERANCE = 1e-6


class MultitaskGPModel(gpytorch.models.ExactGP):
    """Exact GP over ``w`` with two correlated outputs.

    Matern-1.5 rather than RBF: the objectives are measured from finite training runs and
    are not smooth in the preference weight, and an RBF's infinitely differentiable
    sample paths over-smooth that. ``rank=1`` lets the two outputs share a single
    correlation direction, which is what makes measuring one informative about the other.

    The analysis notebook fits the same kernel, so the plotted posterior matches the
    surrogate the search actually used.
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(nu=1.5), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)


def load_seed_points(cfg) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the initial design as ``(w, objectives)`` tensors.

    Reads ``search.seed_points`` when set; otherwise bootstraps from the metrics jsonl,
    keeping the last record for each weight.
    """
    seed_path = cfg.search.get("seed_points")

    if seed_path:
        source = Path(seed_path)
        if not source.is_absolute():
            source = REPO_ROOT / source
        if not source.exists():
            raise FileNotFoundError(f"search.seed_points points at {source}, which does not exist")
        with open(source) as handle:
            records = json.load(handle)
    else:
        source = cfg.output_path("metrics", create=False)
        if not source.exists():
            raise FileNotFoundError(
                f"no metrics at {source}. The search seeds itself from completed "
                f"training runs, so train a spread of weights first (see README, "
                f"step 1 of Running), or set search.seed_points to a JSON file."
            )
        records = _records_from_metrics(source)

    if not records:
        raise ValueError(f"{source} has no records; run an initial weight sweep first")

    records = sorted(records, key=lambda r: r["w"])
    train_w = torch.tensor([[float(r["w"])] for r in records])
    train_objs = torch.tensor(
        [[float(r["mse"]), float(r["occupancy"])] for r in records]
    )
    return train_w, train_objs


def _records_from_metrics(metrics_path):
    """Collapse a metrics jsonl to one record per weight, keeping the last."""
    latest = {}
    with open(metrics_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                entry = json.loads(line)
                latest[float(entry["w"])] = entry
    return list(latest.values())


def read_latest_metrics(metrics_path) -> torch.Tensor:
    """Read the objectives from the most recent line of the metrics jsonl."""
    with open(metrics_path) as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        raise RuntimeError(f"no metrics were written to {metrics_path}")
    entry = json.loads(lines[-1])
    return torch.tensor([float(entry["mse"]), float(entry["occupancy"])])


def build_candidate_pool(cfg, evaluated_w) -> torch.Tensor:
    """Build the grid of candidate weights, minus those already evaluated."""
    grid = torch.arange(
        cfg.search.candidate_start,
        cfg.search.candidate_stop,
        cfg.search.candidate_step,
    ).reshape(-1, 1)
    return update_pool(grid, evaluated_w)


def update_pool(candidate_pool, evaluated_w) -> torch.Tensor:
    """Drop every candidate that matches an evaluated weight within tolerance."""
    if candidate_pool.numel() == 0:
        return candidate_pool.reshape(-1, 1)

    pool = candidate_pool.flatten()
    evaluated = evaluated_w.flatten()

    if evaluated.numel() == 0:
        return pool.reshape(-1, 1)

    distance = (pool.unsqueeze(1) - evaluated.unsqueeze(0)).abs()
    keep = distance.min(dim=1).values > W_TOLERANCE
    return pool[keep].reshape(-1, 1)


def select_query_point(model, likelihood, candidate_pool) -> torch.Tensor:
    """Return the candidate with the highest total predictive variance."""
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(candidate_pool))
        total_uncertainty = pred.variance.sum(dim=1)
        idx = torch.argmax(total_uncertainty)
    return candidate_pool[idx]


def run_trial(args, cfg, w_mse) -> torch.Tensor:
    """Train at ``w_mse`` in a subprocess and return the resulting objectives.

    Kept as a subprocess so each trial gets a clean process group; an in-process call
    would have to tear down and re-initialise NCCL between trials.
    """
    trial = cfg.search.trial
    metrics_path = cfg.output_path("metrics")

    cmd = [
        sys.executable,
        "-m",
        "sparsest.train",
        "--config", args.config,
        "-mode", "train",
        "-w_mse", str(float(w_mse)),
        "-window", str(cfg.window),
        "-batch_size", str(trial.batch_size),
        "-epochs", str(trial.epochs),
        "-begin_valid", str(trial.begin_valid),
        "--output_file", str(metrics_path),
    ]
    if trial.fp16:
        cmd.append("-fp16")

    subprocess.run(cmd, check=True)
    return read_latest_metrics(metrics_path)


def main(argv=None):
    args, cfg = parse_args(argv, description="GP search over the preference weight")

    train_w, train_objs = load_seed_points(cfg)
    gp_dir = cfg.output_path("gp_models")

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPModel(train_w, train_objs, likelihood)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.search.gp_lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    w_pool = build_candidate_pool(cfg, train_w)
    print(f"seeded with {train_w.numel()} points; {w_pool.numel()} candidates remain")

    for iteration in range(cfg.search.gp_iter):

        if w_pool.numel() == 0:
            print("candidate pool exhausted; stopping early")
            break

        model.train()
        likelihood.train()
        for _ in range(cfg.search.gp_steps):
            optimizer.zero_grad()
            loss = -mll(model(train_w), train_objs)
            loss.backward()
            optimizer.step()

        new_w = select_query_point(model, likelihood, w_pool)
        new_objs = run_trial(args, cfg, new_w.item())

        train_w = torch.cat([train_w, new_w.reshape(1, 1)], dim=0)
        train_objs = torch.cat([train_objs, new_objs.reshape(1, 2)], dim=0)
        w_pool = update_pool(w_pool, new_w)
        model.set_train_data(train_w, train_objs, strict=False)

        print(
            f"GP Iteration {iteration + 1}: Added x = {new_w.item():.3f}, "
            f"y = {new_objs.tolist()}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "likelihood_state_dict": likelihood.state_dict(),
                "X_train": train_w,
                "Y_train": train_objs,
            },
            gp_dir / f"gp_iter_{iteration}.pth",
        )


if __name__ == "__main__":
    main()
