"""Checks on the Bayesian-optimisation driver: seeding, candidate pool, stopping."""

import json
import sys
import tempfile
from pathlib import Path

import torch

from sparsest.config import load_config
from sparsest.search import (
    MultitaskGPModel,
    build_candidate_pool,
    load_seed_points,
    read_latest_metrics,
    select_query_point,
    update_pool,
)

TOL = 1e-6

# A stand-in initial design. The shipped configs bootstrap from the metrics file rather
# than a checked-in seed file, so tests that need a design build their own.
SEED_W = torch.tensor([[0.0], [0.1], [0.25], [0.5], [0.75], [0.9], [0.99], [1.0]])
SEED_OBJS = torch.tensor(
    [[0.24, 0.01], [0.011, 0.14], [0.007, 0.29], [0.005, 0.57],
     [0.004, 0.71], [0.0035, 0.69], [0.003, 0.81], [0.0025, 0.92]]
)


def _config_with_metrics(tmp, records, name="ipad"):
    """Write a temp config whose metrics file holds ``records``, seeded by bootstrap.

    Nulling ``search.seed_points`` is done in YAML rather than via an override, because a
    None override means "this CLI flag was not passed" and is deliberately ignored.
    """
    import yaml

    tmp = Path(tmp)
    (tmp / "output_metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    config = yaml.safe_load(Path(f"configs/{name}.yaml").read_text())
    config["search"]["seed_points"] = None
    config["output"] = {
        "root": str(tmp),
        "metrics": "output_metrics.jsonl",
        "checkpoints": "checkpoints",
        "gp_models": "gp_model",
        "figures": "figures",
    }
    (tmp / "boot.yaml").write_text(yaml.safe_dump(config))
    return load_config(tmp / "boot.yaml")


def test_update_pool_removes_evaluated_within_tolerance():
    """Exact float equality would leave 0.15000000000000002 in the pool forever."""
    pool = torch.arange(0.05, 1.05, 0.05).reshape(-1, 1)
    evaluated = torch.tensor([[0.05], [0.15], [0.5]])
    out = update_pool(pool, evaluated)

    assert out.numel() == pool.numel() - 3, (out.numel(), pool.numel())
    for w in evaluated.flatten().tolist():
        assert (out.flatten() - w).abs().min() > TOL, f"{w} survived"

    # A drifted duplicate is still recognised as the same candidate.
    assert update_pool(torch.tensor([[0.15000000000000002]]),
                       torch.tensor([[0.15]])).numel() == 0


def test_exhausted_pool_is_empty_not_a_crash():
    """Exhausting the pool used to raise 'cannot reshape tensor of 0 elements'."""
    out = update_pool(torch.tensor([[0.1], [0.2]]), torch.tensor([[0.1], [0.2]]))
    assert out.shape == (0, 1), out.shape

    # Either side being empty is also fine.
    empty = torch.tensor([]).reshape(-1, 1)
    assert update_pool(empty, torch.tensor([[0.5]])).numel() == 0
    assert update_pool(torch.tensor([[0.1], [0.2]]), empty).numel() == 2


def test_build_candidate_pool_excludes_seed_points():
    cfg = load_config("configs/ipad.yaml")
    pool = build_candidate_pool(cfg, SEED_W)

    for w in SEED_W.flatten().tolist():
        assert (pool.flatten() - w).abs().min() > TOL, f"seed {w} still a candidate"
    assert pool.numel() > 0


def test_bootstrap_works_for_both_datasets():
    """Both shipped configs seed the search from their own metrics file."""
    records = [
        {"w": w, "mse": m, "occupancy": o}
        for w, (m, o) in zip(SEED_W.flatten().tolist(), SEED_OBJS.tolist())
    ]
    for name in ("ipad", "mnist"):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config_with_metrics(tmp, records, name=name)
            train_w, train_objs = load_seed_points(cfg)

        assert train_w.shape == (len(records), 1), (name, train_w.shape)
        assert train_objs.shape == (len(records), 2), (name, train_objs.shape)
        assert torch.all(train_w[1:] > train_w[:-1]), f"{name} seeds not sorted"
        assert torch.all(torch.isfinite(train_objs)) and torch.all(train_objs >= 0)


def test_shipped_configs_bootstrap_by_default():
    """seed_points is null in both configs, so no seed file needs to be distributed."""
    for name in ("ipad", "mnist"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg.search.get("seed_points") is None, name


def test_missing_metrics_file_reports_clearly():
    """Searching before any training run should say so, not raise something cryptic."""
    cfg = load_config("configs/ipad.yaml")
    try:
        load_seed_points(cfg)
    except (FileNotFoundError, ValueError):
        pass
    except Exception as exc:  # noqa: BLE001 - we want to know if it is something else
        raise AssertionError(f"unexpected error type: {type(exc).__name__}: {exc}")


def test_bootstrap_from_metrics_keeps_last_per_weight():
    """A weight retrained later supersedes its earlier record."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config_with_metrics(
            tmp,
            [
                {"w": 0.5, "mse": 0.9, "occupancy": 0.1},
                {"w": 0.1, "mse": 0.2, "occupancy": 0.3},
                {"w": 0.5, "mse": 0.4, "occupancy": 0.6},  # later wins
            ],
        )
        train_w, train_objs = load_seed_points(cfg)

    # Compared with tolerance: weights round-trip through float32, which is why the
    # original checkpoint names carry values like 0.44999998807907104.
    assert torch.allclose(
        train_w.flatten(), torch.tensor([0.1, 0.5]), atol=1e-6
    ), train_w.flatten().tolist()
    assert abs(train_objs[1, 0].item() - 0.4) < 1e-6, "kept the stale record"


def test_read_latest_metrics_skips_blank_lines():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.jsonl"
        path.write_text(
            json.dumps({"w": 0.1, "mse": 0.5, "occupancy": 0.6}) + "\n\n"
            + json.dumps({"w": 0.2, "mse": 0.7, "occupancy": 0.8}) + "\n\n"
        )
        objs = read_latest_metrics(path)
    assert torch.allclose(objs, torch.tensor([0.7, 0.8]), atol=TOL), objs


def test_select_query_point_returns_a_pool_member():
    cfg = load_config("configs/ipad.yaml")

    import gpytorch

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPModel(SEED_W, SEED_OBJS, likelihood)

    pool = build_candidate_pool(cfg, SEED_W)
    chosen = select_query_point(model, likelihood, pool)

    assert chosen.numel() == 1
    assert (pool.flatten() - chosen.item()).abs().min() < TOL, "chose outside the pool"
    print(f"  selected w={chosen.item():.3f} from {pool.numel()} candidates")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - standalone runner reports and continues
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
