"""Seeding, distributed metric reduction, and network construction."""

import random

import numpy as np
import torch
import torch.distributed as dist

from .models import SparseST


def set_seed(seed):
    """Seed every RNG and force deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_metric(rank, world_size, metric):
    """Average a scalar metric across all ranks."""
    metrics_tensor = torch.tensor(metric).to(rank)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    return metrics_tensor.item() / world_size


def build_network(cfg):
    """Build the encoder and decoder cell stacks described by ``cfg``.

    The encoder widens ``input_channels`` through ``cfg.model.features``; the decoder
    mirrors it back down, ending at ``input_channels`` so the output matches the input.
    With the default ``features`` of ``[8, 16, 32, 64]`` this reproduces the original
    4+4 layer stack.
    """
    features = list(cfg.model.features)
    input_channels = cfg.input_channels
    shape = tuple(cfg.model.shape)
    filter_size = cfg.model.filter_size
    debug_unit = cfg.get("debug_unit", False)

    def make(in_ch, out_ch):
        return SparseST(
            window=cfg.window,
            shape=shape,
            batch_size=cfg.batch_size,
            input_channels=in_ch,
            filter_size=filter_size,
            num_features=out_ch,
            debug_unit=debug_unit,
        )

    encoder_params = []
    for i, out_ch in enumerate(features):
        in_ch = input_channels if i == 0 else features[i - 1]
        encoder_params.append(make(in_ch, out_ch))

    n = len(features)
    decoder_params = []
    in_ch = features[-1]
    for k in range(n):
        out_ch = features[-1] if k == 0 else (input_channels if k == n - 1 else features[n - 1 - k])
        decoder_params.append(make(in_ch, out_ch))
        in_ch = out_ch

    return encoder_params, decoder_params


def build_model(cfg, device=None):
    """Build the full encoder-decoder model on ``device``."""
    from .models import ED, Decoder, Encoder

    encoder_params, decoder_params = build_network(cfg)
    model = ED(Encoder(encoder_params), Decoder(decoder_params), window=cfg.window)
    return model if device is None else model.to(device)


def strip_ddp_prefix(state_dict):
    """Drop the ``module.`` prefix that ``DistributedDataParallel`` adds."""
    from collections import OrderedDict

    if any(key.startswith("module.") for key in state_dict):
        return OrderedDict((key.replace("module.", "", 1), value) for key, value in state_dict.items())
    return state_dict
