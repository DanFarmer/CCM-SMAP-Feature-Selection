#!/usr/bin/env python3
"""
ccm_lstm_optimizer.py — CCM/S-Map feature selection with LSTM threshold sweep

For each target column in a CSV this script:
  1. Finds the optimal CCM embedding dimension E via Simplex Projection
  2. Finds the optimal S-Map theta (nonlinearity) for each feature->target pair
  3. Computes CCM rho (causal coupling) for every feature, validated with a
     Mann-Kendall convergence test
  4. Forms a composite score:  |CCM_rho| x (1 + S-Map nonlinearity bonus)
  5. Sweeps tau x top-k grid -- at each (tau, top_k), selects features via the
     hybrid fishing-net (score >= tau) union fishing-line (top-k% fallback)
  6. Trains an LSTM at every (tau, top_k) and records validation loss
  7. Re-trains the best-(tau, top_k) model, evaluates on test split, saves
     weights + full metrics + CCM scores to disk

Baselines (All Features, Pearson, Spearman, Mutual Info) now also sweep
through --baseline-thresholds top-k fractions, choosing the fraction that
minimises validation MSE — giving each method its own optimal threshold
rather than an arbitrary fixed value.

DGX optimisations:
  - Automatic mixed precision (AMP) via --amp
  - DataLoader pin_memory + persistent workers + prefetch
  - torch.compile (--compile, PyTorch >= 2.0)
  - Multi-target GPU parallelism (--num-gpus N): spawns N worker processes,
    each pinned to one GPU, processing different targets simultaneously.
    CCM scoring (CPU) and LSTM sweep (GPU) both run in parallel across targets.

Usage:
    python ccm_lstm_optimizer.py --csv Data/RothamstedData.csv --output results/

    # Narrow targets, custom thresholds
    python ccm_lstm_optimizer.py --csv data.csv \\
        --targets "Flow (l/s) [Catchment 1]" "Soil Moisture @ 10cm Depth (%) [Catchment 1]" \\
        --thresholds 0.0 0.2 0.4 0.6 0.8 1.0 --output results/

    # DGX weekend run — all 8 GPUs, AMP, fine-grained sweeps, compile
    python ccm_lstm_optimizer.py --csv Data/RothamstedData.csv \\
        --hidden 128 --layers 2 --epochs 150 --patience 20 --batch 256 \\
        --thresholds 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 \\
                     0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 \\
        --top-k-values 0.1 0.2 0.3 0.4 0.5 \\
        --baseline-thresholds 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \\
        --smap-thetas 0 0.5 1 2 4 8 16 \\
        --amp --num-gpus 8 --num-workers 4 --compile \\
        --resume --output results/
"""
import argparse
import gc
import json
import logging
import multiprocessing as _mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

try:
    import pyEDM
    HAS_PYEDM = True
except ImportError:
    HAS_PYEDM = False


# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CCM/S-Map feature selection + LSTM threshold optimisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Data")
    g.add_argument("--csv",          required=True, help="Input CSV path")
    g.add_argument("--output",       default="results", help="Root output directory")
    g.add_argument("--targets",      nargs="*", default=None,
                   help="Target column names (default: all numeric columns)")
    g.add_argument("--skip",         nargs="*", default=[],
                   help="Column names to exclude as targets")
    g.add_argument("--datetime-col", default=None,
                   help="Datetime column to drop before processing")

    g = p.add_argument_group("CCM / S-Map")
    g.add_argument("--max-e",          type=int,   default=10,
                   help="Max embedding dimension searched by Simplex Projection")
    g.add_argument("--smap-thetas",    nargs="+",  type=float,
                   default=[0, 0.5, 1, 2, 4, 8],
                   help="S-Map theta values; 0=linear, higher=more nonlinear")
    g.add_argument("--ccm-lib-sizes",  nargs="+",  type=int, default=None,
                   help="Fixed CCM library sizes (default: 10 log-spaced up to N)")
    g.add_argument("--ccm-samples",    type=int,   default=100,
                   help="Surrogate resamples per CCM library size")
    g.add_argument("--ccm-subsample",  type=int,   default=0,
                   help="Subsample N timepoints for CCM scoring (0=full data)")
    g.add_argument("--mk-alpha",       type=float, default=0.10,
                   help="Mann-Kendall significance level for convergence test")
    g.add_argument("--detrend",        action="store_true", default=True,
                   help="Remove rolling-mean trend before CCM")
    g.add_argument("--no-detrend",     dest="detrend", action="store_false")
    g.add_argument("--detrend-window", type=int,   default=672,
                   help="Rolling-mean window length for detrending (timesteps)")

    g = p.add_argument_group("Data quality robustness")
    g.add_argument("--quality-filter", action="store_true", default=True,
                   help="Null out values whose companion '<col> Quality' "
                        "column isn't in --good-quality-flags before any "
                        "processing (previously these flags were silently "
                        "dropped by select_dtypes and never consulted)")
    g.add_argument("--no-quality-filter", dest="quality_filter", action="store_false")
    g.add_argument("--good-quality-flags", nargs="+", default=["Acceptable"],
                   help="Quality flag values to treat as trustworthy")
    g.add_argument("--max-gap",        type=int,   default=8,
                   help="Max gap length (timesteps) eligible for interpolation. "
                        "Longer gaps are left as missing rather than bridged, "
                        "so CCM analyzes real observed dynamics instead of "
                        "fabricated straight-line fill across sensor outages. "
                        "0 = old unlimited-bridge behavior.")
    g.add_argument("--min-ccm-points", type=int,   default=500,
                   help="Minimum length of the longest clean (target,feature) "
                        "overlap required to attempt CCM for a feature; below "
                        "this it's skipped as insufficient signal rather than "
                        "run on a too-short, noise-dominated window")
    g.add_argument("--log1p-zero-inflated", action="store_true", default=False,
                   help="Apply log1p to non-negative columns whose zero "
                        "fraction exceeds --zero-inflation-thresh, before "
                        "detrending. Helps Simplex/S-Map on flashy, "
                        "zero-heavy series like intermittent stream flow.")
    g.add_argument("--zero-inflation-thresh", type=float, default=0.10,
                   help="Zero-fraction threshold that triggers log1p when "
                        "--log1p-zero-inflated is set")

    g = p.add_argument_group("CCM threshold sweep (2-D grid: tau x top-k)")
    g.add_argument("--thresholds", nargs="+", type=float,
                   default=[round(x * 0.1, 2) for x in range(11)],
                   help="CCM score thresholds tau to sweep")
    g.add_argument("--top-k-values", nargs="+", type=float, default=[0.30],
                   help="Fishing-line top-k fractions to co-sweep with tau. "
                        "Multiple values create a 2-D (tau x top_k) grid. "
                        "Default [0.30] reproduces the original single sweep.")

    g = p.add_argument_group("Baseline threshold sweep")
    g.add_argument("--baseline-thresholds", nargs="+", type=float,
                   default=[round(x * 0.05, 2) for x in range(2, 21)],
                   help="Top-k fractions to sweep for Pearson/Spearman/MI baselines. "
                        "The fraction yielding the lowest val MSE is chosen for each method.")

    g = p.add_argument_group("LSTM architecture and training")
    g.add_argument("--seq-len",  type=int,   default=48,   help="Look-back window (timesteps)")
    g.add_argument("--hidden",   type=int,   default=64,   help="LSTM hidden units")
    g.add_argument("--layers",   type=int,   default=2,    help="LSTM stacked layers")
    g.add_argument("--dropout",  type=float, default=0.2,  help="Dropout rate")
    g.add_argument("--lr",       type=float, default=1e-3, help="Initial learning rate")
    g.add_argument("--batch",    type=int,   default=64,   help="Mini-batch size")
    g.add_argument("--epochs",   type=int,   default=100,  help="Max training epochs")
    g.add_argument("--patience", type=int,   default=12,   help="Early-stopping patience")

    g = p.add_argument_group("Data splitting")
    g.add_argument("--train-frac", type=float, default=0.70)
    g.add_argument("--val-frac",   type=float, default=0.15)

    g = p.add_argument_group("System / DGX")
    g.add_argument("--seed",        type=int,  default=42)
    g.add_argument("--device",      default=None,
                   help="PyTorch device string, e.g. cuda:0 (overrides --num-gpus)")
    g.add_argument("--num-gpus",    type=int,  default=None,
                   help="GPUs to use for target-level parallelism (default: all available). "
                        "Each GPU runs a separate worker process handling different targets.")
    g.add_argument("--num-workers", type=int,  default=4,
                   help="DataLoader worker threads per training process")
    g.add_argument("--prefetch-factor", type=int, default=2,
                   help="DataLoader prefetch factor (per worker)")
    g.add_argument("--amp",         action="store_true", default=False,
                   help="Enable automatic mixed precision (AMP) — ~2x speedup on A100")
    g.add_argument("--no-amp",      dest="amp", action="store_false")
    g.add_argument("--compile",     action="store_true", default=False,
                   help="torch.compile the model (PyTorch >= 2.0, ~30%% faster after warmup)")
    g.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="Number of concurrent sweep jobs allowed per GPU. "
                        "Useful on DGX for keeping GPUs busy when each job is small.")
    g.add_argument("--resume",      action="store_true",
                   help="Skip targets that already have a completed results file")
    g.add_argument("--log-file",    default=None, help="Mirror log output to this file")
    g.add_argument("--verbose",     action="store_true",
                   help="Show per-epoch training progress")
    return p


def setup_logging(log_file: Optional[str], verbose: bool,
                  prefix: str = "") -> logging.Logger:
    lvl = logging.DEBUG if verbose else logging.INFO
    fmt = f"%(asctime)s{' ' + prefix if prefix else ''} %(levelname)-8s %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=lvl, format=fmt, datefmt="%H:%M:%S", handlers=handlers)
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STATISTICAL HELPERS
# ---------------------------------------------------------------------------

def mann_kendall(series: np.ndarray) -> Tuple[float, float]:
    n = len(series)
    if n < 4:
        return 0.0, 1.0
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = float(series[j]) - float(series[i])
            s += (1 if d > 0 else -1 if d < 0 else 0)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    z = ((s - 1) / var_s**0.5 if s > 0 else
         (s + 1) / var_s**0.5 if s < 0 else 0.0)
    p   = 2.0 * (1.0 - norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2.0)
    return float(tau), float(p)


def apply_quality_mask(
    df_raw: pd.DataFrame,
    good_flags: Tuple[str, ...] = ("Acceptable",),
) -> Tuple[pd.DataFrame, int]:
    """
    Every measurement column in this dataset has a companion '<col> Quality'
    column (Acceptable / Suspicious / Reject / Not set / Level Reset).
    select_dtypes(include=[np.number]) silently drops those Quality columns
    without ever consulting them, so flagged-bad sensor readings currently
    flow straight into CCM as if they were trustworthy. This nulls out any
    value whose companion flag isn't in good_flags, turning it into a normal
    missing value that the gap-aware cleaning below can then handle honestly
    instead of treating corrupted readings as real dynamics.
    """
    df = df_raw.copy()
    n_masked = 0
    for col in df.columns:
        qcol = f"{col} Quality"
        if qcol in df.columns:
            bad = ~df[qcol].isin(good_flags)
            bad &= df[col].notna()
            n = int(bad.sum())
            if n:
                df.loc[bad, col] = np.nan
                n_masked += n
    return df, n_masked


def safe_clean(v: np.ndarray, max_gap: int = 0) -> np.ndarray:
    """
    Interpolate NaNs. With max_gap=0 (legacy behavior) any gap length is
    bridged, including multi-day sensor outages -- CCM then sees fabricated
    "dynamics" across gaps that never happened. With max_gap>0, only gaps of
    that length (in timesteps) or shorter are interpolated; longer gaps are
    left as NaN so longest_clean_run() can route around them instead of
    quietly reconstructing them.
    """
    s = pd.Series(v.astype(float))
    if max_gap and max_gap > 0:
        return s.interpolate(limit=max_gap, limit_area="inside").values.astype(float)
    return (s.interpolate(limit_direction="both").ffill().bfill()
             .values.astype(float))


def longest_clean_run(*series_list: np.ndarray) -> Tuple[int, int]:
    """
    Given one or more equal-length, index-aligned series, return the
    [start, end) slice of the longest stretch where ALL of them are
    simultaneously non-NaN. This is how we "shorten the window" for CCM:
    rather than patch over long gaps with interpolation, we just analyze the
    longest run of genuinely-observed, mutually-aligned data.
    """
    n = len(series_list[0])
    valid = np.ones(n, dtype=bool)
    for s in series_list:
        valid &= ~pd.isna(s)
    if valid.all():
        return 0, n
    best_start = best_len = cur_start = cur_len = 0
    for i, v in enumerate(valid):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_start, best_start + best_len


def detrend_series(series: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(series.astype(float))
    trend = s.rolling(window=window, center=True, min_periods=window // 4).mean()
    # NOTE: does NOT ffill/bfill remaining NaNs -- gaps left by safe_clean's
    # max_gap cap should stay NaN here too, so longest_clean_run() sees them.
    return (s - trend).values


# ---------------------------------------------------------------------------
# CCM: EMBEDDING DIMENSION (SIMPLEX PROJECTION)
# ---------------------------------------------------------------------------

def find_optimal_E(series: np.ndarray, max_E: int) -> Tuple[int, float]:
    """
    Sweep E in [1, max_E] with Simplex Projection (train first half, predict
    second half).  The optimal E captures the attractor geometry — reused for
    all CCM / S-Map calls on this target.

    NOTE: expects `series` to already be a clean (no-NaN) slice -- the caller
    is responsible for windowing to a longest_clean_run() before calling this,
    so we don't undo gap-capping by blindly re-filling here.
    """
    n      = len(series)
    half   = n // 2
    df_s   = pd.DataFrame({"time": range(n), "x": series})
    best_rho, best_E = -1.0, 2

    for e in range(1, max_E + 1):
        try:
            res = pyEDM.Simplex(
                dataFrame=df_s, columns="x", target="x",
                E=e, Tp=1,
                lib=f"1 {half}", pred=f"{half + 1} {n}",
                showPlot=False,
            )
            obs, pred = res["Observations"], res["Predictions"]
            idx = obs.dropna().index.intersection(pred.dropna().index)
            rho = float(np.corrcoef(obs.loc[idx], pred.loc[idx])[0, 1]) if len(idx) > 5 else -1.0
            if not np.isnan(rho) and rho > best_rho:
                best_rho, best_E = rho, e
        except Exception:
            pass
    return best_E, float(best_rho)


# ---------------------------------------------------------------------------
# CCM: S-MAP THETA (NONLINEARITY)
# ---------------------------------------------------------------------------

def find_optimal_smap_theta(
    series: np.ndarray,
    E: int,
    thetas: List[float],
) -> Tuple[float, float, Dict[float, float]]:
    """
    theta=0 is global-linear; larger theta gives locally-weighted nonlinear fits.
    nonlinearity_index = rho(best_theta) - rho(theta=0) quantifies state-dependent
    coupling — targets with high index benefit most from CCM-based selection.

    NOTE: expects `series` to already be a clean (no-NaN) slice, same as
    find_optimal_E above.
    """
    n      = len(series)
    half   = n // 2
    df_s   = pd.DataFrame({"time": range(n), "x": series})
    rho_by_theta: Dict[float, float] = {}

    for theta in thetas:
        try:
            result = pyEDM.SMap(
                dataFrame=df_s, columns="x", target="x",
                E=E, theta=theta, Tp=1,
                lib=f"1 {half}", pred=f"{half + 1} {n}",
                showPlot=False,
            )
            preds = result["predictions"] if isinstance(result, dict) else result
            obs   = preds["Observations"].dropna()
            pred  = preds["Predictions"].dropna()
            idx   = obs.index.intersection(pred.index)
            rho   = (float(np.corrcoef(obs.loc[idx], pred.loc[idx])[0, 1])
                     if len(idx) > 5 else 0.0)
            rho_by_theta[float(theta)] = 0.0 if np.isnan(rho) else rho
        except Exception:
            rho_by_theta[float(theta)] = 0.0

    best_theta   = max(rho_by_theta, key=rho_by_theta.get) if rho_by_theta else 0.0
    nonlinearity = max(0.0, rho_by_theta.get(best_theta, 0.0) - rho_by_theta.get(0.0, 0.0))
    return float(best_theta), float(nonlinearity), rho_by_theta


# ---------------------------------------------------------------------------
# CCM: CAUSAL COUPLING STRENGTH WITH CONVERGENCE VALIDATION
# ---------------------------------------------------------------------------

def compute_ccm_rho(
    df_pair: pd.DataFrame,
    feature:   str,
    target:    str,
    E:         int,
    lib_sizes: List[int],
    n_samples: int,
    mk_alpha:  float,
) -> Tuple[float, bool, float, float]:
    """
    CCM cross-maps FROM the target's shadow manifold BACK to the feature.
    Genuine causation produces skill that INCREASES with library size (convergence),
    validated by Mann-Kendall across the rho-vs-library-size sequence.
    Shared forcing produces flat/declining skill.
    """
    lib_str = " ".join(str(s) for s in lib_sizes)
    ccm_key = f"{target}:{feature}"
    try:
        result = pyEDM.CCM(
            dataFrame=df_pair, E=E, Tp=1,
            columns=target, target=feature,
            libSizes=lib_str, sample=n_samples,
            showPlot=False,
        )
        if ccm_key not in result.columns:
            return 0.0, False, 0.0, 1.0
        rho_series = result[ccm_key].fillna(0).values.astype(float)
    except Exception:
        return 0.0, False, 0.0, 1.0

    if len(rho_series) < 3:
        val = float(max(0.0, rho_series[-1])) if len(rho_series) else 0.0
        return val, False, 0.0, 1.0

    tau, p     = mann_kendall(rho_series)
    converged  = (tau > 0) and (p < mk_alpha)
    rho_final  = float(max(0.0, rho_series[-1]))
    return (rho_final if converged else 0.0), bool(converged), float(tau), float(p)


# ---------------------------------------------------------------------------
# COMPOSITE FEATURE SCORING
# ---------------------------------------------------------------------------

def _maybe_log1p(v: np.ndarray, name: str, args, log: logging.Logger) -> np.ndarray:
    if not args.log1p_zero_inflated:
        return v
    finite = v[~np.isnan(v)]
    if len(finite) == 0 or np.any(finite < 0):
        return v  # log1p undefined for negative values -- leave as-is
    zero_frac = float(np.mean(finite == 0))
    if zero_frac >= args.zero_inflation_thresh:
        log.debug(f"  [log1p] {name[:40]:40s}  zero_frac={zero_frac:.2f} -> log1p applied")
        return np.log1p(v)
    return v


def compute_feature_scores(
    df_train:     pd.DataFrame,
    target_col:   str,
    feature_cols: List[str],
    args,
    log: logging.Logger,
) -> Dict:
    """
    Composite CCM+S-Map score for every feature vs target, using ONLY the
    training partition to prevent data leakage.

    Score = |CCM_rho| x (1 + S-Map nonlinearity bonus)
    Non-converged features score 0, with tiny Pearson x 1e-3 tiebreaker
    so the fishing-line still ranks them by correlation strength.

    Robustness changes vs. the original version:
      - Values are gap-capped (--max-gap) rather than blindly interpolated
        across arbitrary-length outages, so we don't feed CCM fabricated
        straight-line "dynamics" across multi-day sensor gaps.
      - The target's own E/theta search runs on its longest clean
        contiguous run (longest_clean_run), not the full noisy series.
      - Each feature's CCM computation runs on the longest run where BOTH
        the feature and target are simultaneously clean -- a per-pair
        "shortened window" rather than one global window forced on every
        feature regardless of its own gap pattern.
      - Features whose clean overlap with the target is too short
        (--min-ccm-points) skip CCM outright rather than running it on a
        noise-dominated window that can't possibly converge.
    """
    if not HAS_PYEDM:
        raise RuntimeError("pyEDM is required. pip install pyEDM")

    all_cols = [target_col] + list(feature_cols)
    raw      = {c: df_train[c].values.astype(float) for c in all_cols}
    raw      = {c: _maybe_log1p(v, c, args, log) for c, v in raw.items()}

    cleaned = {c: safe_clean(v, max_gap=args.max_gap) for c, v in raw.items()}
    if args.detrend:
        series = {c: detrend_series(v, args.detrend_window) for c, v in cleaned.items()}
    else:
        series = cleaned

    # Target's own window: longest run where the target alone is clean.
    # Used only for the target-level E/theta search below.
    t_start, t_end = longest_clean_run(series[target_col])
    t_len = t_end - t_start
    n_full = len(series[target_col])
    if t_len < n_full:
        log.info(f"  [Window] Target longest clean run: {t_len}/{n_full} "
                 f"points ({t_start}:{t_end})")
    if t_len < args.min_ccm_points:
        log.warning(f"  [Window] Target has only {t_len} clean points "
                    f"(< --min-ccm-points {args.min_ccm_points}) -- "
                    f"E/theta search will be unreliable")

    target_window = series[target_col][t_start:t_end]

    log.info("  [Simplex] Searching optimal E...")
    E, simplex_rho = find_optimal_E(target_window, args.max_e)
    log.info(f"  [Simplex] E = {E}  rho = {simplex_rho:.4f}")

    best_theta_tgt, target_nl, target_smap_rhos = find_optimal_smap_theta(
        target_window, E, args.smap_thetas,
    )
    log.info(f"  [S-Map]   target nonlinearity = {target_nl:.4f}  "
             f"best theta = {best_theta_tgt}")

    # Shared, single-window series used ONLY for the non-converged Pearson
    # tiebreak, so ranking is apples-to-apples across features -- matching
    # how compute_baseline_scores ranks Pearson (one consistent window for
    # everyone) rather than each feature being scored on its own different
    # clean-overlap slice with the target. CCM itself (above/below) still
    # uses the honest per-pair shortened window; this is purely for the
    # composite-score fallback when a feature doesn't converge.
    corr_series = {c: pd.Series(v).ffill().bfill().values for c, v in cleaned.items()}

    feature_scores: Dict[str, Dict] = {}
    nf = len(feature_cols)
    n_skipped_short = 0

    for fi, feat in enumerate(feature_cols):
        # Per-pair window: longest run where BOTH target and this feature
        # are clean, not the target's solo window and not the full series.
        f_start, f_end = longest_clean_run(series[target_col], series[feat])
        n_pair = f_end - f_start

        if n_pair < args.min_ccm_points:
            n_skipped_short += 1
            r = float(np.corrcoef(corr_series[feat], corr_series[target_col])[0, 1])
            composite = abs(r) * 1e-3 if not np.isnan(r) else 0.0
            feature_scores[feat] = {
                "ccm_rho": 0.0, "converged": False, "mk_tau": 0.0, "mk_p": 1.0,
                "smap_nl": 0.0, "smap_rhos": {}, "composite": float(composite),
                "skipped_short_window": True, "n_clean_overlap": int(n_pair),
            }
            log.debug(f"  [{fi+1:3d}/{nf}] {feat[:40]:40s}  "
                      f"SKIPPED (clean overlap {n_pair} < {args.min_ccm_points})")
            continue

        feat_win = series[feat][f_start:f_end]
        tgt_win  = series[target_col][f_start:f_end]

        if args.ccm_lib_sizes:
            lib_sizes = sorted(s for s in args.ccm_lib_sizes if s <= n_pair)
        else:
            lib_sizes = sorted(set(
                int(x) for x in np.geomspace(20, n_pair, 10).tolist() + [n_pair]
                if int(x) <= n_pair
            ))

        sub = pd.DataFrame({
            "time":     range(n_pair),
            feat:       feat_win,
            target_col: tgt_win,
        })
        ccm_rho, converged, mk_tau, mk_p = compute_ccm_rho(
            sub, feat, target_col, E, lib_sizes, args.ccm_samples, args.mk_alpha,
        )
        _, feat_nl, feat_smap = find_optimal_smap_theta(
            feat_win, E, args.smap_thetas,
        )

        if converged and ccm_rho > 0:
            composite = abs(ccm_rho) * (1.0 + feat_nl)
        else:
            r = float(np.corrcoef(corr_series[feat], corr_series[target_col])[0, 1])
            composite = abs(r) * 1e-3 if not np.isnan(r) else 0.0

        feature_scores[feat] = {
            "ccm_rho":   float(ccm_rho),
            "converged": bool(converged),
            "mk_tau":    float(mk_tau),
            "mk_p":      float(mk_p),
            "smap_nl":   float(feat_nl),
            "smap_rhos": {str(k): float(v) for k, v in feat_smap.items()},
            "composite": float(composite),
            "skipped_short_window": False,
            "n_clean_overlap": int(n_pair),
        }
        log.debug(
            f"  [{fi+1:3d}/{nf}] {feat[:40]:40s}  "
            f"rho={ccm_rho:.4f}  conv={'Y' if converged else 'N'}  "
            f"nl={feat_nl:.3f}  score={composite:.4f}  n={n_pair}"
        )

    if n_skipped_short:
        log.info(f"  [Window] {n_skipped_short}/{nf} features skipped CCM "
                 f"(clean overlap with target < {args.min_ccm_points} points)")

    return {
        "E":                 E,
        "simplex_rho":       float(simplex_rho),
        "target_nl":         float(target_nl),
        "best_theta_target": float(best_theta_tgt),
        "target_smap_rhos":  {str(k): float(v) for k, v in target_smap_rhos.items()},
        "feature_scores":    feature_scores,
        "target_clean_window": [int(t_start), int(t_end)],
        "n_skipped_short_window": int(n_skipped_short),
        "lib_sizes_used":    lib_sizes,
        "n_converged":       int(sum(1 for v in feature_scores.values() if v["converged"])),
        "n_features":        nf,
    }


# ---------------------------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------------------------

def select_by_threshold(
    feature_cols: List[str],
    ccm_scores:   Dict[str, Dict],
    threshold:    float,
    top_k:        float,
) -> Tuple[List[int], List[str]]:
    """
    Hybrid fishing-net (score >= tau) UNION fishing-line (top k%).
    The union guarantees at least one feature regardless of tau.

    To make the tau sweep meaningful, scores are normalized to the [0, 1]
    range before thresholding. This avoids a situation where raw CCM scores
    (which can be > 1) cause nearly every tau value to select the full set.
    """
    composite = {c: ccm_scores[c]["composite"] for c in feature_cols if c in ccm_scores}
    if composite:
        max_score = max(composite.values())
        if max_score > 0:
            score_scale = {c: s / max_score for c, s in composite.items()}
        else:
            score_scale = {c: 0.0 for c in composite}
    else:
        score_scale = {}

    net = {c for c, s in score_scale.items() if s >= threshold}
    nk = max(1, int(len(feature_cols) * top_k))
    line = set(sorted(score_scale, key=score_scale.get, reverse=True)[:nk])
    selected = sorted(net | line)
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    idx = sorted(col_to_idx[c] for c in selected if c in col_to_idx)
    return idx, selected


def select_top_k(
    scores:       np.ndarray,
    feature_cols: List[str],
    top_k:        float,
) -> Tuple[List[int], List[str]]:
    nk  = max(1, int(len(feature_cols) * top_k))
    idx = sorted(np.argsort(scores)[::-1][:nk].tolist())
    return idx, [feature_cols[i] for i in idx]


def compute_baseline_scores(
    X_tr_sc:      np.ndarray,
    y_tr_sc:      np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute raw ranking scores for each baseline method (train data only)."""
    pearson_sc = np.array([
        abs(float(np.corrcoef(X_tr_sc[:, i], y_tr_sc)[0, 1]))
        for i in range(X_tr_sc.shape[1])
    ])
    spearman_sc = np.array([
        abs(float(spearmanr(X_tr_sc[:, i], y_tr_sc)[0]))
        for i in range(X_tr_sc.shape[1])
    ])
    mi_sc = mutual_info_regression(X_tr_sc, y_tr_sc, random_state=42)
    return {
        "Pearson":     pearson_sc,
        "Spearman":    spearman_sc,
        "Mutual Info": mi_sc,
    }


# ---------------------------------------------------------------------------
# LSTM MODEL
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            batch_first=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        return self.fc(x).squeeze(-1)


def make_windows(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(y) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool,
            num_workers: int, pin_memory: bool, prefetch_factor: int) -> DataLoader:
    """Build a DataLoader keeping tensors on CPU; caller moves to GPU non-blocking."""
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    pf = prefetch_factor if num_workers > 0 else None
    return DataLoader(
        ds, batch_size=batch, shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=pf,
    )


def train_lstm(
    X_tr, y_tr, X_val, y_val,
    hidden:         int,
    layers:         int,
    dropout:        float,
    lr:             float,
    batch:          int,
    epochs:         int,
    patience:       int,
    device:         torch.device,
    verbose:        bool  = False,
    amp:            bool  = False,
    num_workers:    int   = 0,
    prefetch_factor:int   = 2,
    compile_model:  bool  = False,
    use_data_parallel: bool = False,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Train LSTMPredictor with Huber loss, AdamW + cosine-annealing LR,
    gradient clipping, and early stopping on validation loss.

    DGX additions vs original:
      - AMP via torch.cuda.amp (--amp flag, ~2x on A100)
      - Pinned DataLoaders with persistent workers for async CPU→GPU transfer
      - torch.compile support (--compile, requires PyTorch >= 2.0)
      - DataParallel only when use_data_parallel=True (disabled in multi-GPU
        target-parallel mode where each worker owns one GPU)
    """
    use_amp = amp and device.type == "cuda"
    pin_mem = device.type == "cuda"

    model = LSTMPredictor(X_tr.shape[2], hidden, layers, dropout)
    if use_data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit   = nn.HuberLoss()
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    tr_dl  = _loader(X_tr,  y_tr,  batch, True,  num_workers, pin_mem, prefetch_factor)
    val_dl = _loader(X_val, y_val, batch, False, num_workers, pin_mem, prefetch_factor)

    best_val, wait, best_state = float("inf"), 0, None
    tl: List[float] = []
    vl: List[float] = []

    for epoch in range(epochs):
        model.train()
        bl = []
        for xb, yb in tr_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            bl.append(loss.item())
        tl.append(float(np.mean(bl)))

        model.eval()
        vbl = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    vbl.append(crit(model(xb), yb).item())
        vl.append(float(np.mean(vbl)))
        sched.step()

        if vl[-1] < best_val:
            best_val  = vl[-1]
            core      = model.module if hasattr(model, "module") else model
            # unwrap compiled model to get plain state_dict
            raw_core  = getattr(core, "_orig_mod", core)
            best_state = {k: v.clone() for k, v in raw_core.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if verbose and epoch % 10 == 0:
            logging.getLogger(__name__).debug(
                f"    epoch {epoch+1:4d}  tr={tl[-1]:.5f}  val={vl[-1]:.5f}"
            )

    core     = model.module if hasattr(model, "module") else model
    raw_core = getattr(core, "_orig_mod", core)
    raw_core.load_state_dict(best_state)
    return raw_core, tl, vl


def evaluate_model(
    model:    nn.Module,
    X_te:     np.ndarray,
    y_te:     np.ndarray,
    scaler_y: RobustScaler,
    device:   torch.device,
    amp:      bool = False,
) -> Dict:
    use_amp = amp and device.type == "cuda"
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        preds = model(torch.tensor(X_te).to(device)).cpu().numpy()
    y_pred = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()
    return {
        "mse":     float(mean_squared_error(y_true, y_pred)),
        "rmse":    float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":     float(mean_absolute_error(y_true, y_pred)),
        "r2":      float(r2_score(y_true, y_pred)),
        "max_err": float(np.max(np.abs(y_true - y_pred))),
        "y_true":  y_true,
        "y_pred":  y_pred,
    }


# ---------------------------------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------------------------------

def prepare_data(
    df:           pd.DataFrame,
    target_col:   str,
    feature_cols: List[str],
    train_frac:   float,
    val_frac:     float,
    seq_len:      int,
    max_gap:      int = 0,
):
    """
    Chronological split with RobustScaler fit ONLY on training portion.
    RobustScaler (median/IQR) is resistant to outliers common in environmental data.

    Cleaning: short gaps (<= max_gap timesteps) get real interpolation via
    safe_clean; any longer gaps still fall back to ffill/bfill afterward,
    since LSTM sequences need to be fully populated -- but this is now the
    fallback of last resort rather than the only mechanism, so short/medium
    gaps aren't flattened into stale carried-forward values the way the old
    ffill()-only version did it.
    """
    d = df.copy()
    for c in d.columns:
        d[c] = safe_clean(d[c].values.astype(float), max_gap=max_gap)
    d = d.ffill().bfill()
    X_raw = d[feature_cols].values.astype(float)
    y_raw = d[[target_col]].values.astype(float)
    n     = len(y_raw)
    t1    = int(n * train_frac)
    t2    = int(n * (train_frac + val_frac))

    sc_X = RobustScaler()
    sc_y = RobustScaler()

    X_tr_sc  = sc_X.fit_transform(X_raw[:t1]).astype(np.float32)
    y_tr_sc  = sc_y.fit_transform(y_raw[:t1]).ravel().astype(np.float32)
    X_val_sc = sc_X.transform(X_raw[t1:t2]).astype(np.float32)
    y_val_sc = sc_y.transform(y_raw[t1:t2]).ravel().astype(np.float32)
    X_te_sc  = sc_X.transform(X_raw[t2:]).astype(np.float32)
    y_te_sc  = sc_y.transform(y_raw[t2:]).ravel().astype(np.float32)

    X_tr_w,  y_tr_w  = make_windows(X_tr_sc,  y_tr_sc,  seq_len)
    X_val_w, y_val_w = make_windows(X_val_sc, y_val_sc, seq_len)
    X_te_w,  y_te_w  = make_windows(X_te_sc,  y_te_sc,  seq_len)

    splits = {"train_end": t1, "val_end": t2, "n_total": n}
    return (X_tr_w, y_tr_w, X_val_w, y_val_w, X_te_w, y_te_w,
            sc_y, splits, X_tr_sc, y_tr_sc)


# ---------------------------------------------------------------------------
# THRESHOLD SWEEP PER TARGET
# ---------------------------------------------------------------------------

def run_sweep_for_target(
    df:           pd.DataFrame,
    target_col:   str,
    feature_cols: List[str],
    ccm_info:     Dict,
    args,
    device:       torch.device,
    out_dir:      Path,
    log:          logging.Logger,
) -> Dict:
    """
    Full 2-D CCM sweep (tau x top_k) plus per-method baseline threshold sweep.

    CCM: for every (tau, top_k) pair selects features via the hybrid
    fishing-net / fishing-line strategy, trains an LSTM, records val_mse.
    The pair (tau*, top_k*) minimising val_mse is the CCM optimum.

    Baselines: for each method (Pearson, Spearman, Mutual Info), sweeps
    through --baseline-thresholds top-k fractions and picks the fraction
    minimising val_mse.  All Features always uses all features.

    All results are saved to threshold_sweep.csv.  The best CCM model is
    re-trained from scratch, evaluated on the test split, and saved to disk.
    """
    (X_tr_w, y_tr_w, X_val_w, y_val_w, X_te_w, y_te_w,
     scaler_y, splits, X_tr_sc, y_tr_sc) = prepare_data(
        df, target_col, feature_cols,
        args.train_frac, args.val_frac, args.seq_len,
        max_gap=args.max_gap,
    )

    min_windows = args.seq_len + 5
    if any(a.shape[0] < min_windows for a in (X_tr_w, X_val_w, X_te_w)):
        log.warning(f"  Too few windows (need {min_windows}) -- skipping {target_col}")
        return {}

    # Shared kwargs for train_lstm
    lstm_kw = dict(
        hidden=args.hidden, layers=args.layers, dropout=args.dropout,
        lr=args.lr, batch=args.batch, epochs=args.epochs, patience=args.patience,
        device=device, verbose=args.verbose,
        amp=args.amp,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        compile_model=args.compile,
        use_data_parallel=False,  # target-parallel mode: one GPU per target
    )

    def _train_eval(feat_idx: List[int]) -> Optional[Dict]:
        if not feat_idx:
            feat_idx = list(range(len(feature_cols)))
        try:
            model, tl, vl = train_lstm(
                X_tr_w[:, :, feat_idx], y_tr_w,
                X_val_w[:, :, feat_idx], y_val_w,
                **lstm_kw,
            )
            m = evaluate_model(model, X_te_w[:, :, feat_idx], y_te_w,
                               scaler_y, device, amp=args.amp)
            result = {
                "val_mse":      float(min(vl)),
                "test_mse":     m["mse"],
                "test_rmse":    m["rmse"],
                "test_mae":     m["mae"],
                "test_r2":      m["r2"],
                "test_maxerr":  m["max_err"],
                "n_features":   len(feat_idx),
                "epochs_used":  len(vl),
                "y_true":       m["y_true"].tolist(),
                "y_pred":       m["y_pred"].tolist(),
                "train_losses": tl,
                "val_losses":   vl,
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            return result
        except Exception as e:
            log.warning(f"    ERROR: {e}")
            return None

    def _compact(r: Optional[Dict]) -> Dict:
        if r is None:
            return {}
        return {k: v for k, v in r.items()
                if k not in ("y_true", "y_pred", "train_losses", "val_losses")}

    # ---- Baseline scores (computed once from training data) ----------------
    bl_scores = compute_baseline_scores(X_tr_sc, y_tr_sc)
    baseline_thresholds = list(args.baseline_thresholds)

    sweep_rows: List[Dict] = []

    # ---- 2-D CCM sweep: tau x top_k ----------------------------------------
    top_k_values = list(args.top_k_values)
    n_ccm_runs   = len(args.thresholds) * len(top_k_values)
    log.info(f"  CCM sweep: {len(args.thresholds)} tau x {len(top_k_values)} top_k "
             f"= {n_ccm_runs} runs")

    best_val_mse  = float("inf")
    best_tau      = None
    best_top_k    = None
    best_compact  = None
    # In distributed torchrun mode, each rank is already pinned to one GPU.
    # Do not spawn sweep subprocesses across all GPUs from every rank.
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if "WORLD_SIZE" in os.environ else 1
    if device.type == "cuda" and world_size > 1:
        n_sweep_gpus = 1
    else:
        n_sweep_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_sweep_gpus > 1:
        jobs_per_gpu = max(1, args.jobs_per_gpu)
        log.info(
            f"  Parallel sweep: dispatching jobs across {n_sweep_gpus} GPUs "
            f"with up to {jobs_per_gpu} concurrent job(s) per GPU"
        )
        jobs = []
        for tau in args.thresholds:
            for top_k in top_k_values:
                feat_idx, feat_names = select_by_threshold(
                    feature_cols, ccm_info["feature_scores"], tau, top_k,
                )
                jobs.append(("CCM", tau, top_k, feat_idx, feat_names))

        # Also add baseline jobs and All Features later (below) via the same mechanism
        # Prepare executor
        ctx = _mp.get_context("spawn")
        max_workers = min(len(jobs), n_sweep_gpus * jobs_per_gpu)
        lstm_kw_no_device = {k: v for k, v in lstm_kw.items() if k != "device"}

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            futures = {}
            for i, (method, tau, top_k, feat_idx, feat_names) in enumerate(jobs):
                job = {
                    "gpu_id": i % n_sweep_gpus,
                    "X_tr_w": X_tr_w, "y_tr_w": y_tr_w,
                    "X_val_w": X_val_w, "y_val_w": y_val_w,
                    "X_te_w": X_te_w, "y_te_w": y_te_w,
                    "scaler_y": scaler_y,
                    "feat_idx": feat_idx,
                    "lstm_kw": lstm_kw_no_device,
                }
                fut = pool.submit(_sweep_job_worker, job)
                futures[fut] = (method, tau, top_k, feat_names)

            for fut in as_completed(futures):
                method, tau, top_k, feat_names = futures[fut]
                r = fut.result()
                compact = _compact(r)
                row = {"method": method, "threshold": tau, "top_k": top_k,
                       "feature_names": feat_names, **compact}
                sweep_rows.append(row)
                if r and r.get("val_mse") is not None:
                    log.info(
                        f"    {method} tau={tau:.2f} top_k={top_k:.2f}  "
                        f"n={r['n_features']:3d}  val={r['val_mse']:.5f}  "
                        f"test={r['test_mse']:.5f}  R2={r['test_r2']:.4f}"
                    )
                    if r["val_mse"] < best_val_mse:
                        best_val_mse = r["val_mse"]
                        best_tau     = tau
                        best_top_k   = top_k
                        best_compact = row

    else:
        for tau in args.thresholds:
            for top_k in top_k_values:
                feat_idx, feat_names = select_by_threshold(
                    feature_cols, ccm_info["feature_scores"], tau, top_k,
                )
                r       = _train_eval(feat_idx)
                compact = _compact(r)
                row     = {
                    "method": "CCM", "threshold": tau, "top_k": top_k,
                    "feature_names": feat_names, **compact,
                }
                sweep_rows.append(row)
                if r:
                    log.info(
                        f"    CCM tau={tau:.2f} top_k={top_k:.2f}  "
                        f"n={r['n_features']:3d}  val={r['val_mse']:.5f}  "
                        f"test={r['test_mse']:.5f}  R2={r['test_r2']:.4f}"
                    )
                    if r["val_mse"] < best_val_mse:
                        best_val_mse = r["val_mse"]
                        best_tau     = tau
                        best_top_k   = top_k
                        best_compact = row

    # ---- Baseline threshold sweeps -----------------------------------------
    baselines_raw: Dict[str, Optional[Dict]] = {}

    # If parallel sweep available, run baseline jobs in parallel as well
    if n_sweep_gpus > 1:
        log.info("  Parallel baseline sweeps across GPUs")
        baseline_jobs = []
        for method_name, scores_arr in bl_scores.items():
            for tk in baseline_thresholds:
                bidx, bnames = select_top_k(scores_arr, feature_cols, tk)
                baseline_jobs.append((method_name, tk, bidx, bnames))

        ctx = _mp.get_context("spawn")
        jobs_per_gpu = max(1, args.jobs_per_gpu)
        max_workers = min(len(baseline_jobs), n_sweep_gpus * jobs_per_gpu)
        lstm_kw_no_device = {k: v for k, v in lstm_kw.items() if k != "device"}
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            futures = {}
            for i, (method_name, tk, bidx, bnames) in enumerate(baseline_jobs):
                job = {
                    "gpu_id": i % n_sweep_gpus,
                    "X_tr_w": X_tr_w, "y_tr_w": y_tr_w,
                    "X_val_w": X_val_w, "y_val_w": y_val_w,
                    "X_te_w": X_te_w, "y_te_w": y_te_w,
                    "scaler_y": scaler_y,
                    "feat_idx": bidx,
                    "lstm_kw": lstm_kw_no_device,
                }
                fut = pool.submit(_sweep_job_worker, job)
                futures[fut] = (method_name, tk, bnames)

            # Track best per-method
            best_for_method = {}
            for fut in as_completed(futures):
                method_name, tk, bnames = futures[fut]
                r = fut.result()
                compact = _compact(r)
                sweep_rows.append({
                    "method": method_name, "threshold": None, "top_k": tk,
                    "feature_names": bnames, **compact,
                })
                if r and r.get("val_mse") is not None:
                    log.info(
                        f"    {method_name} top_k={tk:.2f}  "
                        f"n={r['n_features']:3d}  val={r['val_mse']:.5f}  "
                        f"R2={r['test_r2']:.4f}"
                    )
                    cur_best = best_for_method.get(method_name, (float('inf'), None, None))
                    if r["val_mse"] < cur_best[0]:
                        best_for_method[method_name] = (r["val_mse"], r, tk)

        # Populate baselines_raw from best_for_method
        for method_name, _scores in bl_scores.items():
            entry = best_for_method.get(method_name)
            if entry:
                _, r_obj, best_tk = entry
                r_obj["best_top_k"] = best_tk
                r_obj["best_n_features"] = r_obj.get("n_features", 0)
                baselines_raw[method_name] = r_obj
                log.info(f"  Best {method_name}: top_k={best_tk}  val={r_obj['val_mse']:.5f}  R2={r_obj['test_r2']:.4f}")
            else:
                baselines_raw[method_name] = None
                log.info(f"  Best {method_name}: (no valid run)")

    else:
        # Sweepable baselines (sequential)
        for method_name, scores_arr in bl_scores.items():
            log.info(f"  Baseline sweep: {method_name}  "
                     f"({len(baseline_thresholds)} top-k fractions)")
            bl_best_val  = float("inf")
            bl_best_r    = None
            bl_best_tk   = None
            bl_best_names = None

            for tk in baseline_thresholds:
                bidx, bnames = select_top_k(scores_arr, feature_cols, tk)
                r = _train_eval(bidx)
                compact = _compact(r)
                sweep_rows.append({
                    "method": method_name, "threshold": None, "top_k": tk,
                    "feature_names": bnames, **compact,
                })
                if r:
                    log.info(
                        f"    {method_name} top_k={tk:.2f}  "
                        f"n={r['n_features']:3d}  val={r['val_mse']:.5f}  "
                        f"R2={r['test_r2']:.4f}"
                    )
                    if r["val_mse"] < bl_best_val:
                        bl_best_val   = r["val_mse"]
                        bl_best_r     = r
                        bl_best_tk    = tk
                        bl_best_names = bnames

            if bl_best_r:
                bl_best_r["best_top_k"] = bl_best_tk
                bl_best_r["best_n_features"] = len(bl_best_names or [])
            baselines_raw[method_name] = bl_best_r
            log.info(f"  Best {method_name}: top_k={bl_best_tk}  "
                     f"val={bl_best_val:.5f}  "
                     f"R2={bl_best_r['test_r2']:.4f}" if bl_best_r else "  (no valid run)")

    # All Features — no threshold to sweep
    log.info("  Baseline: All Features")
    all_idx   = list(range(len(feature_cols)))
    r_all     = _train_eval(all_idx)
    baselines_raw["All Features"] = r_all
    sweep_rows.append({
        "method": "All Features", "threshold": None, "top_k": 1.0,
        "feature_names": feature_cols, **_compact(r_all),
    })
    if r_all:
        log.info(f"    All Features  n={r_all['n_features']:3d}  "
                 f"val={r_all['val_mse']:.5f}  R2={r_all['test_r2']:.4f}")

    # ---- Save sweep table --------------------------------------------------
    pd.DataFrame(sweep_rows).to_csv(out_dir / "threshold_sweep.csv", index=False)

    # ---- Re-train and save best CCM model ----------------------------------
    if best_tau is not None:
        log.info(f"  Best CCM: tau={best_tau:.2f}  top_k={best_top_k:.2f}  "
                 f"val_mse={best_val_mse:.5f}")
        best_feat_idx, best_feat_names = select_by_threshold(
            feature_cols, ccm_info["feature_scores"], best_tau, best_top_k,
        )
        final_model, _, _ = train_lstm(
            X_tr_w[:, :, best_feat_idx], y_tr_w,
            X_val_w[:, :, best_feat_idx], y_val_w,
            **lstm_kw,
        )
        torch.save(final_model.state_dict(), out_dir / "best_model.pt")

        final_model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', 
            enabled=args.amp and device.type == "cuda"
        ):
            best_preds = final_model(
                torch.tensor(X_te_w[:, :, best_feat_idx]).to(device)
            ).cpu().numpy()
        y_pred_inv = scaler_y.inverse_transform(best_preds.reshape(-1, 1)).ravel()
        y_true_inv = scaler_y.inverse_transform(y_te_w.reshape(-1, 1)).ravel()
        np.save(out_dir / "best_predictions.npy",  y_pred_inv)
        np.save(out_dir / "best_ground_truth.npy", y_true_inv)
        log.info(f"  Saved best model -> {out_dir / 'best_model.pt'}")
    else:
        best_feat_names = []
        log.warning("  No valid CCM threshold found")

    def _base(method):
        r = baselines_raw.get(method) or {}
        return {k: v for k, v in r.items()
                if k not in ("y_true", "y_pred", "train_losses", "val_losses")}

    return {
        "target":            target_col,
        "best_tau":          best_tau,
        "best_top_k":        best_top_k,
        "best_val_mse":      best_val_mse if best_val_mse < float("inf") else None,
        "best_n_features":   len(best_feat_names),
        "best_features":     best_feat_names,
        "best_test_mse":     best_compact.get("test_mse")    if best_compact else None,
        "best_test_rmse":    best_compact.get("test_rmse")   if best_compact else None,
        "best_test_mae":     best_compact.get("test_mae")    if best_compact else None,
        "best_test_r2":      best_compact.get("test_r2")     if best_compact else None,
        "ccm_E":             ccm_info["E"],
        "target_nl":         ccm_info["target_nl"],
        "n_converged":       ccm_info["n_converged"],
        "n_features_total":  len(feature_cols),
        "data_splits":       splits,
        "baselines": {
            "All Features": _base("All Features"),
            "Pearson":      _base("Pearson"),
            "Spearman":     _base("Spearman"),
            "Mutual Info":  _base("Mutual Info"),
        },
    }


# ---------------------------------------------------------------------------
# MULTI-GPU WORKER  (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _target_worker_fn(kwargs: Dict) -> Optional[Dict]:
    """
    Spawned worker process: processes one target end-to-end on the assigned GPU.
    Must be a module-level function (not nested) to be picklable by 'spawn'.
    """
    import logging as _logging
    import json as _json

    gpu_id     = kwargs["gpu_id"]
    target_col = kwargs["target_col"]
    col_idx    = kwargs["col_idx"]
    n_total    = kwargs["n_total"]
    df         = kwargs["df"]
    args       = argparse.Namespace(**kwargs["args_dict"])
    out_dir    = Path(kwargs["out_dir"])

    # Each spawned process needs its own logging setup
    _logging.basicConfig(
        level=_logging.INFO,
        format=f"%(asctime)s [GPU{gpu_id}] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[_logging.StreamHandler(sys.stdout)],
    )
    log = _logging.getLogger(f"worker_{gpu_id}")

    torch.manual_seed(args.seed + col_idx)  # different seed per target
    np.random.seed(args.seed + col_idx)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    log.info(f"[{col_idx+1}/{n_total}] {target_col}  device={device}")

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        return None

    # Resume check
    summary_file = out_dir / "summary.json"
    if args.resume and summary_file.exists():
        log.info("  Already complete -- skipping (--resume)")
        with open(summary_file) as f:
            return _json.load(f)

    # Phase 1: CCM scoring
    ccm_scores_file = out_dir / "ccm_scores.json"
    if args.resume and ccm_scores_file.exists():
        with open(ccm_scores_file) as f:
            ccm_info = _json.load(f)
    else:
        n_train  = int(len(df) * args.train_frac)
        df_train = df.iloc[:n_train].reset_index(drop=True)
        try:
            ccm_info = compute_feature_scores(df_train, target_col, feature_cols, args, log)
        except Exception as e:
            log.error(f"  CCM failed: {e}")
            return None
        with open(ccm_scores_file, "w") as f:
            _json.dump(ccm_info, f, indent=2)
        log.info(
            f"  CCM done: E={ccm_info['E']}  "
            f"converged={ccm_info['n_converged']}/{len(feature_cols)}  "
            f"NL={ccm_info['target_nl']:.4f}"
        )

    # Phase 2: Sweep
    summary = run_sweep_for_target(
        df, target_col, feature_cols, ccm_info, args, device, out_dir, log,
    )
    if summary:
        with open(summary_file, "w") as f:
            _json.dump(summary, f, indent=2, default=str)
    return summary


# ---------------------------------------------------------------------------
# SWEEP JOB WORKER
# ---------------------------------------------------------------------------

def _sweep_job_worker(kwargs: Dict) -> Optional[Dict]:
    """Train and evaluate one LSTM sweep job on the assigned GPU/CPU."""
    import torch as _torch
    import numpy as _np

    gpu_id = kwargs["gpu_id"]
    device = (_torch.device(f"cuda:{gpu_id}") if _torch.cuda.is_available()
              else _torch.device("cpu"))
    if device.type == "cuda":
        _torch.cuda.set_device(device)

    X_tr_w = kwargs["X_tr_w"]
    y_tr_w = kwargs["y_tr_w"]
    X_val_w = kwargs["X_val_w"]
    y_val_w = kwargs["y_val_w"]
    X_te_w = kwargs["X_te_w"]
    y_te_w = kwargs["y_te_w"]
    scaler_y = kwargs["scaler_y"]
    feat_idx = kwargs["feat_idx"]
    lstm_kw = kwargs["lstm_kw"]

    try:
        X_tr_w = _np.array(X_tr_w) if not isinstance(X_tr_w, _np.ndarray) else X_tr_w
        X_val_w = _np.array(X_val_w) if not isinstance(X_val_w, _np.ndarray) else X_val_w
        X_te_w = _np.array(X_te_w) if not isinstance(X_te_w, _np.ndarray) else X_te_w
        y_tr_w = _np.array(y_tr_w) if not isinstance(y_tr_w, _np.ndarray) else y_tr_w
        y_val_w = _np.array(y_val_w) if not isinstance(y_val_w, _np.ndarray) else y_val_w
        y_te_w = _np.array(y_te_w) if not isinstance(y_te_w, _np.ndarray) else y_te_w

        model, tl, vl = train_lstm(
            X_tr_w[:, :, feat_idx], y_tr_w,
            X_val_w[:, :, feat_idx], y_val_w,
            device=device,
            **lstm_kw,
        )
        m = evaluate_model(model, X_te_w[:, :, feat_idx], y_te_w, scaler_y, device, amp=lstm_kw.get("amp", False))
        if device.type == "cuda":
            _torch.cuda.empty_cache()

        return {
            "val_mse": float(min(vl)) if vl else None,
            "test_mse": m["mse"],
            "test_rmse": m["rmse"],
            "test_mae": m["mae"],
            "test_r2": m["r2"],
            "test_maxerr": m["max_err"],
            "n_features": len(feat_idx),
            "epochs_used": len(vl),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    log  = setup_logging(args.log_file, args.verbose)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not HAS_PYEDM:
        log.error("pyEDM is required but not installed. Run: pip install pyEDM")
        sys.exit(1)

    # Device / GPU count resolution
    # Support torchrun / torch.distributed multi-process multi-node runs
    # If launched with torchrun, WORLD_SIZE and RANK (and LOCAL_RANK) are set.
    if args.device:
        device = torch.device(args.device)
        n_gpus = 1
        dist_rank = 0
        dist_world = 1
    elif "WORLD_SIZE" in os.environ:
        # Running under torchrun / torch.distributed.launch
        dist_world = int(os.environ.get("WORLD_SIZE", "1"))
        dist_rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            n_gpus_avail = torch.cuda.device_count()
            n_gpus = 1
        else:
            device = torch.device("cpu")
            n_gpus = 1
        log.info(f"Distributed run: world_size={dist_world} rank={dist_rank} local_rank={local_rank} device={device}")
    elif torch.cuda.is_available():
        try:
            test = torch.zeros(1, device="cuda")
            del test
            n_gpus_avail = torch.cuda.device_count()
            n_gpus = min(args.num_gpus or n_gpus_avail, n_gpus_avail)
            device = torch.device("cuda:0")
            dist_rank = 0
            dist_world = 1
            log.info(f"CUDA available: {n_gpus_avail} GPU(s), using {n_gpus} for target parallelism")
        except (RuntimeError, AssertionError) as e:
            log.warning(f"CUDA not usable ({e}), falling back to CPU")
            device = torch.device("cpu")
            n_gpus = 1
            dist_rank = 0
            dist_world = 1
    else:
        device = torch.device("cpu")
        n_gpus = 1
        dist_rank = 0
        dist_world = 1

    log.info(f"Device: {device}  |  Workers: {n_gpus}  |  PyTorch {torch.__version__}")
    log.info(f"AMP: {args.amp}  |  compile: {args.compile}  |  workers/loader: {args.num_workers}")

    # Load data
    log.info(f"Loading: {args.csv}")
    df_raw = pd.read_csv(args.csv)
    if args.datetime_col and args.datetime_col in df_raw.columns:
        df_raw = df_raw.drop(columns=[args.datetime_col])

    if args.quality_filter:
        df_raw, n_masked = apply_quality_mask(df_raw, tuple(args.good_quality_flags))
        log.info(f"Quality filter: {n_masked} values nulled "
                 f"(flag not in {args.good_quality_flags})")

    df = df_raw.select_dtypes(include=[np.number]).copy()
    log.info(f"Numeric columns: {df.shape[1]}   Rows: {df.shape[0]}")

    skip_set = set(args.skip or [])
    if args.targets:
        target_cols = [t for t in args.targets if t in df.columns]
        missing = [t for t in args.targets if t not in df.columns]
        if missing:
            log.warning(f"Targets not found: {missing}")
    else:
        target_cols = [c for c in df.columns if c not in skip_set]

    log.info(f"Targets: {len(target_cols)}")
    log.info(f"CCM tau sweep: {args.thresholds}")
    log.info(f"CCM top-k sweep: {args.top_k_values}")
    log.info(f"Baseline top-k sweep: {args.baseline_thresholds}")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    def _safe_name(col):
        return (col[:50].replace(" ", "_").replace("/", "_")
                .replace("(", "").replace(")", "").replace("%", "pct"))

    # Build per-target output dirs up front
    target_dirs = {}
    for tc in target_cols:
        d = out_root / _safe_name(tc)
        d.mkdir(parents=True, exist_ok=True)
        target_dirs[tc] = d

    all_summaries: List[Dict] = []
    t_start = time.time()

    # ---- Multi-GPU: one worker process per GPU, targets distributed round-robin
    # If running in torch.distributed/torchrun mode, each process handles a subset
    # of targets (index % world_size == rank). Otherwise, use local ProcessPoolExecutor
    if "WORLD_SIZE" in os.environ:
        # Distributed launch: each process gets its subset of targets
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_targets = [t for i, t in enumerate(target_cols) if i % world_size == rank]
        log.info(f"Distributed assignment: rank {rank} handling {len(local_targets)} targets")
        for col_idx, target_col in enumerate(local_targets):
            sep = "=" * 70
            log.info(f"\n{sep}")
            log.info(f"[{col_idx+1}/{len(local_targets)}]  Target: {target_col}")
            log.info(sep)

            out_dir      = target_dirs[target_col]
            summary_file = out_dir / "summary.json"

            if args.resume and summary_file.exists():
                log.info("  Already complete -- skipping (--resume)")
                with open(summary_file) as f:
                    all_summaries.append(json.load(f))
                continue

            feature_cols = [c for c in df.columns if c != target_col]
            if not feature_cols:
                log.warning("  No feature columns -- skipping")
                continue

            # Phase 1: CCM scoring
            ccm_scores_file = out_dir / "ccm_scores.json"
            if args.resume and ccm_scores_file.exists():
                log.info("  Loading cached CCM scores...")
                with open(ccm_scores_file) as f:
                    ccm_info = json.load(f)
            else:
                log.info(f"  Computing CCM+S-Map scores for {len(feature_cols)} features...")
                n_train  = int(len(df) * args.train_frac)
                df_train = df.iloc[:n_train].reset_index(drop=True)
                t0 = time.time()
                try:
                    ccm_info = compute_feature_scores(
                        df_train, target_col, feature_cols, args, log,
                    )
                except Exception as e:
                    log.error(f"  CCM failed: {e}")
                    continue
                with open(ccm_scores_file, "w") as f:
                    json.dump(ccm_info, f, indent=2)
                log.info(
                    f"  CCM done in {time.time()-t0:.1f}s  "
                    f"E={ccm_info['E']}  "
                    f"converged={ccm_info['n_converged']}/{len(feature_cols)}  "
                    f"NL={ccm_info['target_nl']:.4f}"
                )

            # Phase 2: Sweep
            t1 = time.time()
            summary = run_sweep_for_target(
                df, target_col, feature_cols, ccm_info, args, device, out_dir, log,
            )
            log.info(f"  Sweep done in {time.time()-t1:.1f}s")

            if not summary:
                continue

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            all_summaries.append(summary)

        # After distributed processes finish, one can merge summaries externally.
    elif n_gpus > 1:
        log.info(f"\nLaunching {n_gpus} GPU worker processes...")

        # Convert args to a plain dict so it's picklable
        args_dict = vars(args)

        tasks = []
        for col_idx, tc in enumerate(target_cols):
            tasks.append({
                "gpu_id":    col_idx % n_gpus,
                "target_col": tc,
                "col_idx":   col_idx,
                "n_total":   len(target_cols),
                "df":        df,
                "args_dict": args_dict,
                "out_dir":   str(target_dirs[tc]),
            })

        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx) as pool:
            futures = {pool.submit(_target_worker_fn, task): task for task in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                tc   = task["target_col"]
                try:
                    summary = fut.result()
                    if summary:
                        all_summaries.append(summary)
                        elapsed = time.time() - t_start
                        log.info(
                            f"  Finished: {tc}  "
                            f"R2={summary.get('best_test_r2', 'n/a')}  "
                            f"elapsed={elapsed/3600:.2f}h"
                        )
                except Exception as e:
                    log.error(f"  Target FAILED: {tc}  {e}")

    # ---- Single GPU / CPU: sequential processing ---------------------------
    else:
        for col_idx, target_col in enumerate(target_cols):
            sep = "=" * 70
            log.info(f"\n{sep}")
            log.info(f"[{col_idx+1}/{len(target_cols)}]  Target: {target_col}")
            log.info(sep)

            out_dir      = target_dirs[target_col]
            summary_file = out_dir / "summary.json"

            if args.resume and summary_file.exists():
                log.info("  Already complete -- skipping (--resume)")
                with open(summary_file) as f:
                    all_summaries.append(json.load(f))
                continue

            feature_cols = [c for c in df.columns if c != target_col]
            if not feature_cols:
                log.warning("  No feature columns -- skipping")
                continue

            # Phase 1: CCM scoring
            ccm_scores_file = out_dir / "ccm_scores.json"
            if args.resume and ccm_scores_file.exists():
                log.info("  Loading cached CCM scores...")
                with open(ccm_scores_file) as f:
                    ccm_info = json.load(f)
            else:
                log.info(f"  Computing CCM+S-Map scores for {len(feature_cols)} features...")
                n_train  = int(len(df) * args.train_frac)
                df_train = df.iloc[:n_train].reset_index(drop=True)
                t0 = time.time()
                try:
                    ccm_info = compute_feature_scores(
                        df_train, target_col, feature_cols, args, log,
                    )
                except Exception as e:
                    log.error(f"  CCM failed: {e}")
                    continue
                with open(ccm_scores_file, "w") as f:
                    json.dump(ccm_info, f, indent=2)
                log.info(
                    f"  CCM done in {time.time()-t0:.1f}s  "
                    f"E={ccm_info['E']}  "
                    f"converged={ccm_info['n_converged']}/{len(feature_cols)}  "
                    f"NL={ccm_info['target_nl']:.4f}"
                )

            # Phase 2: Sweep
            t1 = time.time()
            summary = run_sweep_for_target(
                df, target_col, feature_cols, ccm_info, args, device, out_dir, log,
            )
            log.info(f"  Sweep done in {time.time()-t1:.1f}s")

            if not summary:
                continue

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            all_summaries.append(summary)

            elapsed   = time.time() - t_start
            done      = col_idx + 1
            remaining = len(target_cols) - done
            eta_h     = (elapsed / done * remaining) / 3600
            log.info(
                f"  Elapsed {elapsed/3600:.2f}h  ETA {eta_h:.2f}h  "
                f"best_tau={summary.get('best_tau')}  "
                f"best_top_k={summary.get('best_top_k')}  "
                f"R2={summary.get('best_test_r2', 'n/a')}"
            )

    # ---- Global summary CSV ------------------------------------------------
    if all_summaries:
        rows = []
        for s in all_summaries:
            if not s:
                continue
            b = s.get("baselines", {})

            def _bv(method, key):
                return (b.get(method) or {}).get(key)

            rows.append({
                "target":             s["target"],
                "best_tau":           s.get("best_tau"),
                "best_top_k":         s.get("best_top_k"),
                "best_n_features":    s.get("best_n_features"),
                "ccm_test_mse":       s.get("best_test_mse"),
                "ccm_test_rmse":      s.get("best_test_rmse"),
                "ccm_test_mae":       s.get("best_test_mae"),
                "ccm_test_r2":        s.get("best_test_r2"),
                "all_test_mse":       _bv("All Features", "test_mse"),
                "all_test_r2":        _bv("All Features", "test_r2"),
                "pearson_test_mse":   _bv("Pearson",      "test_mse"),
                "pearson_test_r2":    _bv("Pearson",      "test_r2"),
                "pearson_best_top_k": _bv("Pearson",      "best_top_k"),
                "spearman_test_mse":  _bv("Spearman",     "test_mse"),
                "spearman_test_r2":   _bv("Spearman",     "test_r2"),
                "spearman_best_top_k":_bv("Spearman",     "best_top_k"),
                "mi_test_mse":        _bv("Mutual Info",  "test_mse"),
                "mi_test_r2":         _bv("Mutual Info",  "test_r2"),
                "mi_best_top_k":      _bv("Mutual Info",  "best_top_k"),
                "ccm_E":              s.get("ccm_E"),
                "target_nl":          s.get("target_nl"),
                "n_converged":        s.get("n_converged"),
                "n_features_total":   s.get("n_features_total"),
            })

        summary_df = pd.DataFrame(rows)
        csv_path   = out_root / "summary.csv"
        summary_df.to_csv(csv_path, index=False)
        log.info(f"\nSummary -> {csv_path}")

        log.info("\n" + "=" * 80)
        log.info("RESULTS SUMMARY (test R2 -- higher is better)")
        log.info("=" * 80)
        show = [
            "target", "best_tau", "best_top_k",
            "ccm_test_r2", "all_test_r2",
            "pearson_test_r2", "spearman_test_r2", "mi_test_r2",
            "ccm_E", "target_nl",
        ]
        avail = [c for c in show if c in summary_df.columns]
        with pd.option_context("display.max_rows", None, "display.width", 140,
                               "display.float_format", "{:.4f}".format):
            log.info("\n" + summary_df[avail].to_string(index=False))

        # Win counts on R2 and MSE
        for metric, better in [("r2", "max"), ("mse", "min")]:
            mse_cols = [c for c in summary_df.columns if c.endswith(f"test_{metric}")]
            if len(mse_cols) > 1:
                fn   = summary_df[mse_cols].idxmax if better == "max" else summary_df[mse_cols].idxmin
                wins = fn(axis=1).value_counts()
                log.info(f"\nWins (best test {metric.upper()} per target):")
                for m, cnt in wins.items():
                    log.info(f"  {m:<35s}: {cnt}")

    log.info(f"\nTotal runtime: {(time.time()-t_start)/3600:.2f} hours")
    log.info(f"Results in:    {out_root}/")


if __name__ == "__main__":
    main()
