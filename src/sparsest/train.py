#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Distributed training of the sparse spatio-temporal model.

One run optimises a single preference weight ``w_mse``. The search driver in
:mod:`sparsest.search` launches many such runs to trace out the Pareto front.
"""

import json
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .cli import parse_args
from .data import build_dataset
from .losses import stch_loss
from .utils import build_model, set_seed


def setup(rank, world_size, cfg):
    os.environ.setdefault("MASTER_ADDR", cfg.distributed.master_addr)
    os.environ.setdefault("MASTER_PORT", str(cfg.distributed.master_port))
    torch.cuda.set_device(rank)
    dist.init_process_group(cfg.distributed.backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def ddp_dataloader(rank, mode, world_size, cfg):
    dataset = build_dataset(cfg, mode)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler,
    )
    return loader


def build_ddp_model(rank, cfg):
    model = build_model(cfg, device=rank)
    return DDP(model, device_ids=[rank], find_unused_parameters=True)


def _clamp_all_thresholds(model):
    """Keep every cell's thresholds non-negative, outside the autograd graph."""
    for module in model.modules():
        if hasattr(module, "clamp_thresholds"):
            module.clamp_thresholds()


def train(rank, world_size, cfg, w_mse):
    save_dir = cfg.output_path("checkpoints")
    metrics_path = cfg.output_path("metrics")

    setup(rank, world_size, cfg)
    ddp_model = build_ddp_model(rank, cfg)

    mse_loss = nn.MSELoss(reduction="mean").to(rank)
    optimizer = optim.Adam(ddp_model.parameters(), lr=cfg.train.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)

    fp16 = bool(cfg.train.fp16)
    scaler = GradScaler() if fp16 else None
    best_valid_loss = float("inf")

    # Reference values for the two objectives, captured on the very first step and held
    # fixed thereafter so the scalarised loss is comparable across epochs.
    max_mse = None
    max_occupancy = None

    if cfg.train.get("resume"):
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        checkpoint = torch.load(cfg.train.resume, map_location=map_location)
        ddp_model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        # Restore the objective references; without these the first step after a resume
        # would have nothing to normalise against.
        max_mse = checkpoint.get("max_mse")
        max_occupancy = checkpoint.get("max_occupancy")
        best_valid_loss = checkpoint.get("best_valid_loss", best_valid_loss)
        print(f"[Rank {rank}] Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Built once: rebuilding per epoch would re-read and re-transpose the whole array
    # every time. The sampler is left unseeded per epoch, preserving the original
    # fixed iteration order.
    train_loader = ddp_dataloader(rank, "train", world_size, cfg)
    valid_loader = None
    if rank == 0:
        valid_loader = torch.utils.data.DataLoader(
            build_dataset(cfg, "valid"),
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=True,
        )

    for epoch in range(start_epoch, cfg.train.epochs + 1):

        train_losses = []
        ddp_model.train()

        iterator = (
            tqdm(train_loader, leave=False, total=len(train_loader))
            if rank == 0
            else train_loader
        )

        for i, (idx, inputVar, targetVar) in enumerate(iterator):

            # Thresholds only receive gradients every `step` iterations, so they move on
            # a slower schedule than the convolution weights.
            for name, param in ddp_model.named_parameters():
                if "threshold" in name:
                    param.requires_grad_(i % cfg.train.step == 0)

            inputs = inputVar.to(rank)
            label = targetVar.to(rank)
            optimizer.zero_grad()

            with autocast() if fp16 else nullcontext():
                sum_l0 = torch.tensor(0.0).to(rank)
                sum_l1 = torch.tensor(0.0).to(rank)

                l0_sparsity, l1_occupancy, pred = ddp_model(sum_l0, sum_l1, inputs)
                mse = mse_loss(pred, label)

                if max_mse is None:
                    loss = stch_loss(w_mse, mse, l1_occupancy)

                    max_mse_tensor = mse.detach().clone()
                    max_occ_tensor = l1_occupancy.detach().clone()

                    if w_mse not in (0.0, 1.0):
                        dist.broadcast(max_mse_tensor, src=0)
                        dist.broadcast(max_occ_tensor, src=0)
                        max_mse = max_mse_tensor.item()
                        max_occupancy = max_occ_tensor.item()
                    else:
                        max_mse = 0.0
                        max_occupancy = 0.0
                else:
                    loss = stch_loss(
                        w_mse, mse - max_mse, l1_occupancy - max_occupancy
                    )

            if fp16:
                assert loss.dtype is torch.float32
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(ddp_model.parameters(), clip_value=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_value_(ddp_model.parameters(), clip_value=10.0)
                optimizer.step()

            _clamp_all_thresholds(ddp_model)
            train_losses.append(loss.item())

            if rank == 0:
                iterator.set_postfix(
                    {
                        "epoch": "{:02d}".format(epoch),
                        "trainloss": "{:.6f}".format(loss),
                        "w_mse": w_mse,
                        "mse": "{:.5f}".format(mse.item()),
                        "l0_sparsity": "{:.5f}".format(l0_sparsity.item()),
                        "l1_occupancy": "{:.5f}".format(l1_occupancy.item()),
                    }
                )

        torch.cuda.empty_cache()
        train_loss = np.mean(train_losses)

        if epoch < cfg.train.begin_valid:
            continue

        if rank == 0:
            ddp_model.eval()
            valid_losses, valid_mses = [], []
            l0_sparsities, l1_occupancies = [], []

            with torch.no_grad():
                for i, (idx, inputVar, targetVar) in enumerate(
                    tqdm(valid_loader, leave=False, total=len(valid_loader))
                ):
                    inputs = inputVar.to(rank)
                    label = targetVar.to(rank)

                    # Zeroed per batch: these accumulate across the whole encoder/decoder
                    # stack and must not carry over from the training loop.
                    sum_l0 = torch.tensor(0.0).to(rank)
                    sum_l1 = torch.tensor(0.0).to(rank)

                    l0_sparsity, l1_occupancy, pred = ddp_model(sum_l0, sum_l1, inputs)
                    mse = mse_loss(pred, label)
                    loss = stch_loss(w_mse, mse, l1_occupancy)

                    valid_losses.append(loss.item())
                    valid_mses.append(mse.item())
                    l0_sparsities.append(l0_sparsity.item())
                    l1_occupancies.append(l1_occupancy.item())

            torch.cuda.empty_cache()
            valid_loss = float(np.mean(valid_losses))
            valid_mse = float(np.mean(valid_mses))
            valid_sparsity = float(np.mean(l0_sparsities))
            valid_occupancy = float(np.mean(l1_occupancies))
            valid_loss_tensor = torch.tensor(valid_loss, device=rank)

            print(f"learning rate: {optimizer.param_groups[0]['lr']}")
            print(f"w_mse: {w_mse}")
            print(f"train STCH for epoch {epoch}: {train_loss}.")
            print(f"valid STCH for epoch {epoch}: {valid_loss}")
            print(f"valid mse for epoch {epoch}: {valid_mse}.")
            print(f"valid l0 sparsity of epoch {epoch}: {valid_sparsity}.")
            print(f"valid l1 occupancy of epoch {epoch}: {valid_occupancy}.")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": ddp_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "max_mse": max_mse,
                        "max_occupancy": max_occupancy,
                        "best_valid_loss": best_valid_loss,
                        "w_mse": w_mse,
                    },
                    os.path.join(
                        save_dir, f"checkpoint_{w_mse}_{epoch:03d}_{valid_loss:04f}.pt"
                    ),
                )

                result = {
                    "w": w_mse,
                    "mse": valid_mse,
                    "occupancy": 1.0 - valid_sparsity,
                }
                with open(metrics_path, "a") as handle:
                    handle.write(json.dumps(result) + "\n")
        else:
            valid_loss_tensor = torch.tensor(0.0, device=rank)

        dist.broadcast(valid_loss_tensor, src=0)
        pla_lr_scheduler.step(valid_loss_tensor.item())

    cleanup()


def main(argv=None):
    args, cfg = parse_args(argv, description="Train the sparse spatio-temporal model")
    set_seed(cfg.seed)

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("training requires at least one CUDA device")

    mp.spawn(train, args=(world_size, cfg, cfg.w_mse), nprocs=world_size)


if __name__ == "__main__":
    main()
