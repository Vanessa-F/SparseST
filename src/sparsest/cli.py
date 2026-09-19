"""Shared command-line interface.

Every entry point takes ``--config configs/<dataset>.yaml``. Individual flags override
the config, so a sweep can vary one value without editing a file. The original
single-dash flag spellings are kept so existing commands still work.
"""

import argparse
import json

from .config import load_config

# CLI destination -> dotted config key. Only values the user actually passed are
# applied, so unset flags leave the config alone.
ARG_TO_CONFIG = {
    "window": "window",
    "batch_size": "batch_size",
    "w_mse": "w_mse",
    "mode": "mode",
    "lr": "train.lr",
    "epochs": "train.epochs",
    "begin_valid": "train.begin_valid",
    "step": "train.step",
    "fp16": "train.fp16",
    "resume": "train.resume",
    "gp_iter": "search.gp_iter",
    "output_file": "output.metrics",
    "checkpoint": "eval.checkpoint",
    "eval_mode": "eval.mode",
    "debug_unit": "debug_unit",
}


def build_parser(description=None):
    """Build the argument parser shared by every entry point."""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--config", required=True, help="dataset config, e.g. configs/ipad.yaml"
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the fully resolved config and exit",
    )

    parser.add_argument("-batch_size", "--batch-size", type=int, help="mini-batch size")
    parser.add_argument("-window", "--window", type=int, help="timesteps unrolled per clip")
    parser.add_argument("-lr", "--lr", type=float, help="learning rate")
    # The original scripts pass -epoch while the parser defined -epochs; both are
    # accepted here rather than relying on argparse prefix matching.
    parser.add_argument("-epochs", "-epoch", "--epochs", type=int, help="number of epochs")
    parser.add_argument(
        "-fp16", "--fp16", action="store_true", default=None, help="mixed-precision training"
    )
    parser.add_argument("-mode", "--mode", help="train, test, unit or visualize")
    parser.add_argument("-resume", "--resume", help="checkpoint to resume training from")
    parser.add_argument(
        "-w_mse", "--w-mse", type=float, dest="w_mse", help="preference weight for MSE"
    )
    parser.add_argument(
        "-step", "--step", type=int, help="iterations between threshold updates"
    )
    parser.add_argument(
        "-begin_valid", "--begin-valid", type=int, help="first epoch that runs validation"
    )
    parser.add_argument(
        "-gp_iter", "--gp-iter", type=int, help="Bayesian-optimisation iterations"
    )
    parser.add_argument(
        "--output_file", "--output-file", dest="output_file", help="metrics jsonl path"
    )
    parser.add_argument("--checkpoint", help="checkpoint to evaluate")
    parser.add_argument(
        "--eval-mode",
        dest="eval_mode",
        choices=["anomaly_score", "test_loss", "visualize", "unit"],
        help="evaluation to run (defaults to the config value)",
    )
    parser.add_argument(
        "--debug-unit",
        dest="debug_unit",
        action="store_true",
        default=None,
        help="print per-timestep sparsity and thresholds",
    )

    return parser


def parse_args(argv=None, description=None):
    """Parse arguments and return ``(args, cfg)`` with CLI overrides applied."""
    parser = build_parser(description)
    args = parser.parse_args(argv)

    overrides = {
        dotted: getattr(args, dest, None) for dest, dotted in ARG_TO_CONFIG.items()
    }
    cfg = load_config(args.config, overrides)

    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        raise SystemExit(0)

    return args, cfg
