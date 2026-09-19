# SparseST: Exploiting Data Sparsity in Spatiotemporal Modeling and Prediction

A ConvLSTM wastes work recomputing dense convolutions on inputs that barely changed
between frames. SparseST feeds each gate convolution the *change* since the last step and
zeroes changes below a threshold, so the convolution runs on a sparse tensor. The
thresholds are learned, one per timestep per layer, not hand-tuned. Raising them buys
sparsity at the cost of reconstruction error, so training minimises a smooth Tchebycheff
scalarisation controlled by a preference weight `w_mse`. Because each point on the
resulting Pareto front costs a full training run, a multi-task Gaussian process models
both objectives against `w_mse` and picks the next weight by maximum predictive variance.

One implementation covers both datasets; they differ only in configuration.

| | IPAD | Moving-MNIST |
|---|---|---|
| Task | video anomaly detection | next-frame prediction |
| Clip length (`window`) | 21 | 10 |
| Channels | 3 (RGB) | 1 (grayscale) |
| Evaluation | per-frame anomaly score → ROC | mean test MSE |
| Config | `configs/ipad.yaml` | `configs/mnist.yaml` |

## Layout

```
configs/
  ipad.yaml  mnist.yaml     dataset configs: model shape, data paths, training, search
  local.example.yaml        copy to local.yaml for machine-specific paths (gitignored)

src/sparsest/
  models/
    cell.py                 the delta-thresholded ConvLSTM cell; learns threshold_x/h
    encoder.py  decoder.py  stacks of cells, bottom-up and top-down
    ed.py                   joins them; normalises sparsity by n_cells x window
  data/
    ipad.py                 IPAD clips, (N,S,H,W,C) uint8 -> float in [0,1]
    moving_mnist.py         Moving-MNIST, (S,N,H,W,C) float32, train/valid index splits
    registry.py             dataset lookup by cfg.dataset
  train.py                  DDP training for one preference weight      [sparsest-train]
  evaluate.py               anomaly_score | test_loss | unit | visualize [sparsest-eval]
  search.py                 multi-task GP / Bayesian optimisation loop  [sparsest-search]
  config.py                 YAML + env-var resolution, path joining
  cli.py                    shared argument parser
  losses.py                 smooth Tchebycheff scalarisation
  utils.py                  seeding, network construction, DDP helpers

scripts/eval_sweep.sh       score every IPAD test sequence with one checkpoint
notebooks/                  01_pareto_front  02_training_curves  03_thresholds  04_roc_anomaly
tests/run_all.py            runs the three test modules; no pytest needed
```

Runs are written outside the source tree, under `output.root` (default `runs/<dataset>/`):

```
runs/ipad/
  checkpoints/              checkpoint_<w>_<epoch>_<loss>.pt
  gp_model/                 gp_iter_<i>.pth, with X_train / Y_train embedded
  figures/                  notebook and visualize output
  output_metrics.jsonl      {"w", "mse", "occupancy"} per improving epoch
```

## Install

PyTorch and spconv are CUDA-version specific, so they are installed explicitly.

```bash
conda env create -n SparseST -f environment.yml && conda activate SparseST
```

or with pip:

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install spconv-cu120==2.3.6     # match your CUDA: spconv-cu118, spconv-cu120, ...
pip install -r requirements.txt && pip install -e .
python tests/run_all.py             # verify
```

`environment.lock.yml` is a full export if you need an exact match. A CUDA device is
required; the cell allocates state at a fixed batch size, so loaders use `drop_last=True`
and the batch size must stay constant within a run.

## Data

Datasets are never copied into the repo. Point `data.root` at them, either by copying
`configs/local.example.yaml` to `configs/local.yaml`, or per run with `SPARSEST_DATA_ROOT`.
Expected layout under `data.root`:

```
# IPAD                                    # Moving-MNIST
mod_image/                                2_digit/
  train/total_train.npy                     train.npy    (S,N,H,W,C) float32 in [0,1]
  valid/total_valid.npy                     test.npy
  test/type_1/npy/006_mod.npy
       type_1/labels/006_label_mod.npy
```

Always confirm what resolved where before launching anything long:

```bash
sparsest-train --config configs/ipad.yaml --print-config
```

## Running

Three entry points, each taking `--config`. Any config key can be overridden on the
command line. Training uses every visible GPU via `DistributedDataParallel`; restrict with
`CUDA_VISIBLE_DEVICES`.

```bash
# 1. Initial design: train a spread of preference weights.
#    Each run appends to output_metrics.jsonl, which seeds the search.
for w in 0.0 0.1 0.25 0.5 0.75 0.9 0.99 1.0; do
  sparsest-train --config configs/ipad.yaml -w_mse $w
done

# 2. Search. Each round refits the GP, trains at the most uncertain untried weight,
#    and folds the result back in. Rerunning resumes from the metrics file.
sparsest-search --config configs/ipad.yaml -gp_iter 10

# 3. Evaluate.
sparsest-eval --config configs/ipad.yaml --checkpoint <ckpt>.pt --eval-mode anomaly_score
scripts/eval_sweep.sh configs/ipad.yaml <ckpt>.pt runs/ipad/scores   # all test sequences

sparsest-eval --config configs/mnist.yaml --checkpoint <ckpt>.pt --eval-mode test_loss
sparsest-eval --config configs/mnist.yaml --checkpoint <ckpt>.pt --eval-mode visualize

# 4. Figures. Set CONFIG_NAME in the first cell to switch datasets.
jupyter lab notebooks/
```

Useful overrides:

```bash
sparsest-train --config configs/mnist.yaml -w_mse 1.0 -batch_size 32 -epochs 100
sparsest-train --config configs/ipad.yaml  -w_mse 0.75 -resume runs/ipad/checkpoints/<ckpt>.pt
sparsest-eval  --config configs/ipad.yaml  --checkpoint <ckpt>.pt --eval-mode unit --debug-unit
SPARSEST_DATA_ROOT=/other/data SPARSEST_OUTPUT_ROOT=/scratch/runs sparsest-train --config configs/ipad.yaml
```

## Configuration

Resolved highest precedence first: command-line flags, then `SPARSEST_DATA_ROOT` /
`SPARSEST_OUTPUT_ROOT` / `SPARSEST_TEST_NPY` / `SPARSEST_CHECKPOINT`, then
`configs/local.yaml` and `configs/local.<dataset>.yaml`, then the `--config` file.

| Key | Meaning |
|---|---|
| `window` | timesteps per clip; also the number of learned thresholds per layer |
| `w_mse` | preference weight; `1 - w_mse` weights occupancy |
| `train.step` | iterations between threshold updates (weights update every iteration) |
| `train.begin_valid` | first epoch that runs validation and may checkpoint |
| `model.features` | encoder widths; the decoder mirrors them back to `input_channels` |
| `search.seed_points` | `null` bootstraps the design from `output_metrics.jsonl` |
| `search.trial` | shorter training settings used for each search round |

Worth knowing:

- Every improvement in validation loss writes its own checkpoint, so a long run produces
  many gigabytes. Keep the best per weight, or point `output.root` at scratch space.
- IPAD scores a frame by the error of the clip centred on it, so scores start at frame
  `(window-1)/2`; the label arrays are sliced to match.
- `-resume` restores weights, optimiser state, and the objective reference values captured
  on the first step — without those the scalarised loss changes scale after a resume.
- Results vary slightly across GPU counts: the batch is split across ranks and validation
  runs on rank 0 only.

## Citation

J. Wu, H. Benmeziane, K. El Maghraoui, L. Liu, and Y. Wang, "SparseST: Exploiting Data
Sparsity in Spatiotemporal Modeling and Prediction," *IEEE Transactions on Automation
Science and Engineering*, accepted for publication.

```bibtex
@article{wu2026sparsest,
  title   = {SparseST: Exploiting Data Sparsity in Spatiotemporal Modeling and Prediction},
  author  = {Wu, Junfeng and Benmeziane, Hadjer and El Maghraoui, Kaoutar and Liu, Liu and Wang, Yinan},
  journal = {IEEE Transactions on Automation Science and Engineering},
  year    = {2026},
  note    = {Accepted for publication}
}
```

## License

MIT — see [LICENSE](LICENSE).
