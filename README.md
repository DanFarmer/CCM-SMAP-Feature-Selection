# CCM/S-Map–Guided LSTM Feature Selection

Supplementary code for the paper. `ccm_lstm_optimizer3.py` selects predictive
features for a target time series using Convergent Cross Mapping (CCM) and
S-Map nonlinearity, sweeps the selection threshold against LSTM validation
loss, and compares the result against Pearson, Spearman, Mutual Information,
and All-Features baselines under an identical evaluation protocol.

## What the script does

For each target column in the input CSV:

1. Finds the optimal embedding dimension `E` via Simplex Projection.
2. Finds the optimal S-Map `theta` (nonlinearity) for the target and for each
   feature.
3. Computes CCM `rho` (causal coupling strength) for every candidate feature,
   validated with a Mann-Kendall convergence test on the rho-vs-library-size
   curve (genuine causal skill should increase with library size).
4. Forms a composite score per feature: `|CCM_rho| x (1 + S-Map nonlinearity
   bonus)`. Features that fail the convergence test fall back to a tiny
   Pearson-correlation tiebreaker instead of scoring zero outright.
5. Sweeps a 2-D grid of threshold `tau` x top-k fraction. At each grid point,
   features are selected by a hybrid rule: "fishing net" (composite score >=
   tau, after min-max normalization) union "fishing line" (top-k% by score),
   so at least one feature is always selected regardless of tau.
6. Trains an LSTM at every `(tau, top_k)` grid point and records validation
   loss.
7. Re-trains the best-validation `(tau, top_k)` configuration, evaluates it
   on the held-out test split, and saves weights, predictions, and metrics.

Baselines (All Features, Pearson, Spearman, Mutual Information) go through
the same top-k sweep and val-loss-based model selection, so every method is
compared on equal footing rather than against a single fixed CCM threshold.

## Requirements

- Python 3.9+
- `pyEDM` (required — CCM/Simplex/S-Map implementation; the script exits if
  it isn't installed)
- `torch` >= 2.0 (for optional `torch.compile`; earlier 1.x versions work
  with `--compile` simply unavailable)
- `numpy`, `pandas`, `scipy`, `scikit-learn`

```bash
pip install pyEDM torch numpy pandas scipy scikit-learn
```

GPU acceleration (CUDA) is optional but strongly recommended for the full
threshold sweep — see the DGX / multi-GPU section below.

## Basic usage

```bash
python ccm_lstm_optimizer3.py --csv Data/RothamstedData.csv --output results/
```

Runs every numeric column in the CSV as a target in turn, using default
sweep grids and LSTM hyperparameters (see reference below).

### Narrow targets, custom thresholds

```bash
python ccm_lstm_optimizer3.py --csv data.csv \
    --targets "Flow (l/s) [Catchment 1]" "Soil Moisture @ 10cm Depth (%) [Catchment 1]" \
    --thresholds 0.0 0.2 0.4 0.6 0.8 1.0 --output results/
```

### Large multi-GPU run (fine-grained sweep, AMP, compile)

```bash
python ccm_lstm_optimizer3.py --csv Data/RothamstedData.csv \
    --hidden 128 --layers 2 --epochs 150 --patience 20 --batch 256 \
    --thresholds 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 \
                 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 \
    --top-k-values 0.1 0.2 0.3 0.4 0.5 \
    --baseline-thresholds 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --smap-thetas 0 0.5 1 2 4 8 16 \
    --amp --num-gpus 8 --num-workers 4 --compile \
    --resume --output results/
```

With `--num-gpus > 1`, the script spawns one worker process per GPU and
distributes *targets* round-robin across them (CCM scoring on CPU and the
LSTM sweep on GPU both run in parallel across targets). It also detects
`torchrun`/`torch.distributed` launches (`WORLD_SIZE` in the environment)
and splits targets by rank instead.

`--resume` skips any target whose `summary.json` already exists, and reuses
a cached `ccm_scores.json` if present, so interrupted runs can continue
without re-running CCM from scratch.

## Command-line arguments

**Data**
| Flag | Default | Description |
|---|---|---|
| `--csv` | *required* | Input CSV path |
| `--output` | `results` | Root output directory |
| `--targets` | all numeric columns | Target column names to process |
| `--skip` | `[]` | Column names to exclude as targets |
| `--datetime-col` | `None` | Datetime column to drop before processing |

**CCM / S-Map**
| Flag | Default | Description |
|---|---|---|
| `--max-e` | `10` | Max embedding dimension searched by Simplex Projection |
| `--smap-thetas` | `0 0.5 1 2 4 8` | S-Map theta values (0 = linear) |
| `--ccm-lib-sizes` | 10 log-spaced sizes up to N | Fixed CCM library sizes |
| `--ccm-samples` | `100` | Surrogate resamples per CCM library size |
| `--ccm-subsample` | `0` | Subsample N timepoints for CCM scoring (0 = full) |
| `--mk-alpha` | `0.10` | Mann-Kendall significance level for convergence |
| `--detrend` / `--no-detrend` | on | Remove rolling-mean trend before CCM |
| `--detrend-window` | `672` | Rolling-mean window length (timesteps) |

**Data quality robustness**
| Flag | Default | Description |
|---|---|---|
| `--quality-filter` / `--no-quality-filter` | on | Null values whose companion `<col> Quality` flag isn't in `--good-quality-flags` |
| `--good-quality-flags` | `Acceptable` | Quality flag values treated as trustworthy |
| `--max-gap` | `8` | Max gap length (timesteps) eligible for interpolation; longer gaps are left as missing (0 = unlimited bridging) |
| `--min-ccm-points` | `500` | Minimum clean (target, feature) overlap required to attempt CCM |
| `--log1p-zero-inflated` | off | Apply log1p to non-negative, zero-heavy columns before detrending |
| `--zero-inflation-thresh` | `0.10` | Zero-fraction threshold that triggers log1p |

**CCM threshold sweep (tau x top-k grid)**
| Flag | Default | Description |
|---|---|---|
| `--thresholds` | `0.0, 0.1, ..., 1.0` | CCM score thresholds (tau) to sweep |
| `--top-k-values` | `0.30` | Fishing-line top-k fractions to co-sweep with tau |

**Baseline threshold sweep**
| Flag | Default | Description |
|---|---|---|
| `--baseline-thresholds` | `0.10, 0.15, ..., 1.0` | Top-k fractions swept for Pearson/Spearman/Mutual Info baselines |

**LSTM architecture and training**
| Flag | Default | Description |
|---|---|---|
| `--seq-len` | `48` | Look-back window (timesteps) |
| `--hidden` | `64` | LSTM hidden units |
| `--layers` | `2` | LSTM stacked layers |
| `--dropout` | `0.2` | Dropout rate |
| `--lr` | `1e-3` | Initial learning rate |
| `--batch` | `64` | Mini-batch size |
| `--epochs` | `100` | Max training epochs |
| `--patience` | `12` | Early-stopping patience |

**Data splitting**
| Flag | Default | Description |
|---|---|---|
| `--train-frac` | `0.70` | Fraction of rows (chronological) for training |
| `--val-frac` | `0.15` | Fraction of rows for validation (remainder is test) |

**System / DGX**
| Flag | Default | Description |
|---|---|---|
| `--seed` | `42` | Random seed (offset per target in multi-target runs) |
| `--device` | auto | PyTorch device string, e.g. `cuda:0` (overrides `--num-gpus`) |
| `--num-gpus` | all available | GPUs used for target-level parallelism |
| `--num-workers` | `4` | DataLoader worker threads per training process |
| `--prefetch-factor` | `2` | DataLoader prefetch factor per worker |
| `--amp` / `--no-amp` | off | Automatic mixed precision |
| `--compile` | off | `torch.compile` the model (PyTorch >= 2.0) |
| `--jobs-per-gpu` | `1` | Concurrent sweep jobs allowed per GPU |
| `--resume` | off | Skip targets with a completed results file |
| `--log-file` | `None` | Mirror log output to this file |
| `--verbose` | off | Show per-epoch training progress |

## Output structure

```
results/
├── summary.csv                     # one row per target, all methods compared
└── <target_name>/
    ├── ccm_scores.json             # E, S-Map thetas, per-feature CCM/composite scores
    ├── threshold_sweep.csv         # every (method, threshold, top_k) run: val/test metrics
    ├── best_model.pt               # state_dict of the best CCM-selected LSTM
    ├── best_predictions.npy        # test-set predictions of the best model
    ├── best_ground_truth.npy       # corresponding test-set ground truth
    └── summary.json                # best CCM config + all baseline results for this target
```

`summary.csv` reports, per target: best CCM `(tau, top_k)` and its test
MSE/RMSE/MAE/R², plus test MSE/R² for All Features, Pearson, Spearman, and
Mutual Information, alongside CCM diagnostics (`E`, target nonlinearity,
convergence count). The run log also prints a per-metric "wins" tally
(how many targets each method won on test R² / MSE).

## Reproducibility notes

- Seeds (`--seed`) are set for `torch` and `numpy`, offset per target in
  multi-GPU/target-parallel runs so targets don't share identical
  initializations.
- `--amp` and `--compile` trade determinism for speed; CuDNN determinism
  flags are not forced, so bit-exact reproduction across hardware/driver
  versions isn't guaranteed even with a fixed seed.
- Each target is evaluated on a single chronological train/val/test split
  (`--train-frac` / `--val-frac`), not cross-validated or repeated across
  seeds — per-target win margins between methods should be read with that
  in mind.
- `pyEDM` behavior (Simplex/S-Map/CCM output columns) can vary by version;
  pin the version used for the paper's results when publishing this code.
