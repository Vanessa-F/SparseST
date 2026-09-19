#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Evaluation entry point.

Four modes, selected with ``--eval-mode`` or the config's ``eval.mode``:

``anomaly_score``
    Score each frame of one test sequence by the reconstruction error of the clip
    centred on it. Used for the IPAD ROC curves.
``test_loss``
    Mean reconstruction error over the whole test split.
``unit``
    Run a single batch with per-timestep sparsity and threshold printing.
``visualize``
    Roll the model forward and write input/target/prediction image grids.
"""

import torch
from torch import nn
from tqdm import tqdm

from .cli import parse_args
from .data import build_dataset
from .utils import build_model, set_seed, strip_ddp_prefix


def load_model(cfg, device):
    """Build the model and load the configured checkpoint."""
    model = build_model(cfg, device=device)

    checkpoint_path = cfg.eval.checkpoint
    if not checkpoint_path:
        raise ValueError(
            "no checkpoint configured; pass --checkpoint or set eval.checkpoint"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_ddp_prefix(checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _loader(cfg, mode, batch_size):
    return torch.utils.data.DataLoader(
        build_dataset(cfg, mode),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def anomaly_score(cfg, model, device):
    """Print one anomaly score per frame of the configured test sequence.

    The model reconstructs a whole clip; the score for a frame is the error on the
    centre position, which is the frame the clip is aligned to.
    """
    loader = _loader(cfg, "test", batch_size=1)
    criterion = nn.MSELoss(reduction="mean").to(device)
    center = (cfg.window - 1) // 2

    with torch.no_grad():
        for i, (idx, inputVar, targetVar) in enumerate(loader):
            sum_l0 = torch.tensor(0.0).to(device)
            sum_l1 = torch.tensor(0.0).to(device)

            inputs = inputVar.to(device)
            label = targetVar.to(device)

            _, _, pred = model(sum_l0, sum_l1, inputs)
            loss = criterion(pred[:, center, ...], label)
            print(f"anomaly score for image {i + center} is {loss}")


def test_loss(cfg, model, device):
    """Report mean reconstruction error over the test split."""
    loader = _loader(cfg, "test", cfg.batch_size)
    criterion = nn.MSELoss(reduction="mean").to(device)
    losses = []

    with torch.no_grad():
        iterator = tqdm(loader, leave=False, total=len(loader))
        for idx, inputVar, targetVar in iterator:
            sum_l0 = torch.tensor(0.0).to(device)
            sum_l1 = torch.tensor(0.0).to(device)

            inputs = inputVar.to(device)
            label = targetVar.to(device)

            _, _, pred = model(sum_l0, sum_l1, inputs)
            loss = criterion(pred, label)
            losses.append(loss.item())
            iterator.set_postfix({"testloss/batch": "{:.6f}".format(loss)})

    total = sum(losses) / len(losses)
    print(f"total test loss: {total}")
    return total


def unit(cfg, model, device):
    """Run one batch, printing per-timestep sparsity and threshold values."""
    loader = _loader(cfg, "unit", batch_size=1)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Parameter memory = {param_bytes / 1024 ** 2:.5f} MB")

    with torch.no_grad():
        for idx, inputVar, targetVar in loader:
            sum_l0 = torch.tensor(0.0).to(device)
            sum_l1 = torch.tensor(0.0).to(device)
            l0, l1, _ = model(sum_l0, sum_l1, inputVar.to(device))
            print(f"l0_sparsity: {l0.item():.6f}  l1_occupancy: {l1.item():.6f}")
            break


def visualize(cfg, model, device):
    """Write input/target/prediction grids for the first few test batches."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = cfg.window
    out_dir = cfg.output_path("figures")
    loader = _loader(cfg, "visualize", cfg.batch_size)
    n_batches = cfg.eval.get("visualize_batches", 2)

    with torch.no_grad():
        for i, (idx, inputVar, targetVar) in enumerate(loader):
            if i >= n_batches:
                break

            inputs = inputVar.to(device)
            label = targetVar.to(device)
            output = torch.zeros(label.shape).to(device)

            # Slide a window across the concatenated sequence and keep the last
            # predicted frame at each position.
            sample = torch.cat((inputs, label), axis=1)
            for timestep in range(window):
                sum_l0 = torch.tensor(0.0).to(device)
                sum_l1 = torch.tensor(0.0).to(device)
                clip = sample[:, timestep : timestep + window].to(device)
                output[:, timestep] = model(sum_l0, sum_l1, clip)[2][:, -1, ...]

            for k in range(inputs.shape[0]):
                fig, ax = plt.subplots(3, window, figsize=(window, 3))
                for j in range(window):
                    ax[0, j].imshow(inputs[k][j][0].cpu().numpy() * 255, cmap="gray")
                    ax[1, j].imshow(label[k][j][0].cpu().numpy() * 255, cmap="gray")
                    ax[2, j].imshow(output[k][j][0].cpu().numpy() * 255, cmap="gray")
                    for row in range(3):
                        ax[row, j].axis("off")

                fig.savefig(out_dir / f"batch_{i}_sample_{k}.png", bbox_inches="tight")
                plt.close(fig)

    print(f"wrote figures to {out_dir}")


MODES = {
    "anomaly_score": anomaly_score,
    "test_loss": test_loss,
    "unit": unit,
    "visualize": visualize,
}


def main(argv=None):
    args, cfg = parse_args(argv, description="Evaluate a trained checkpoint")
    set_seed(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mode = cfg.eval.mode

    if mode not in MODES:
        raise ValueError(f"unknown eval mode {mode!r} (available: {sorted(MODES)})")

    model = load_model(cfg, device)
    MODES[mode](cfg, model, device)


if __name__ == "__main__":
    main()
