"""
Shared loaders and feature builders for the BGL and Thunderbird detector experiments.

Provides:
  - loaders for the clean and injected BGL / Thunderbird parquets (sorted by node, then time,
    with a stable `row_id`),
  - the chronological train/test split for Thunderbird (BGL's lives in run_baseline_detectors.py),
  - per-node timing baselines fit on clean, train-period, normal-to-normal gaps, and timing
    features for an injected frame scored against that clean baseline,
  - score_detector: thresholded detector output scored on the shared 60 s evaluation grid.

This module has no entry point; it is imported by the experiment scripts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import numpy as np
import pandas as pd

from src.detectors.timing_detector import add_log_ratio_feature, build_features
from src.metrics import evaluate_binary, rows_to_grid
from src.timing_baseline import add_sequence_context, compute_node_baselines

BGL_INJECTED_STALL_PATH = Path("data/processed/bgl_injected_stall.parquet")
BGL_INJECTED_BURST_PATH = Path("data/processed/bgl_injected_burst.parquet")
TB_CLEAN_PATH = Path("data/processed/thunderbird_parsed.parquet")
TB_TRAIN_FRAC = 0.7  # same chronological 70/30 split as BGL


def load_bgl_injected_stall():
    df = pd.read_parquet(BGL_INJECTED_STALL_PATH)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def load_bgl_injected_burst():
    df = pd.read_parquet(BGL_INJECTED_BURST_PATH)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def load_tb_clean():
    df = pd.read_parquet(TB_CLEAN_PATH)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def chronological_split_tb(df, train_frac=TB_TRAIN_FRAC):
    cutoff = df["timestamp"].quantile(train_frac)
    is_train = df["timestamp"] <= cutoff
    return is_train, cutoff


def load_tb_injected(inj_type):
    path = Path(f"data/processed/thunderbird_injected_{inj_type}.parquet")
    df = pd.read_parquet(path)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def build_timing_features(df_clean, is_train_clean, df_injected):
    """Timing features for `df_injected`, scored against per-node baselines fit on the clean
    frame's train-period normal-to-normal gaps. Returns (clean train-normal features, injected
    features)."""
    df_clean_ctx = add_sequence_context(df_clean.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))

    cutoff = df_clean.loc[is_train_clean, "timestamp"].max()
    is_train_ctx = df_clean_ctx["timestamp"] <= cutoff

    baseline_mask = is_train_ctx & (~df_clean_ctx["anomaly"]) & (~df_clean_ctx["prev_anomaly"].fillna(True)) & df_clean_ctx["gap_prev_s"].notna()
    features_clean_full, baselines = build_features(df_clean_ctx, baseline_mask=baseline_mask)
    features_clean_train_normal = features_clean_full.loc[is_train_ctx & (~df_clean_ctx["anomaly"])]

    df_injected_ctx = add_sequence_context(df_injected.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    features_injected, _ = build_features(df_injected_ctx, baselines=baselines)

    return features_clean_train_normal, features_injected


def build_timing_features_tb(df_clean, is_train_clean, df_injected):
    """Thunderbird variant of build_timing_features. The pooled fallback baseline is computed
    from non-zero gaps only (exclude_zero_from_pooled=True): Thunderbird timestamps have
    one-second resolution, so most gaps are exactly 0 and the all-gaps pooled median and MAD
    would both be 0. Also adds the log-ratio feature."""
    df_clean_ctx = add_sequence_context(df_clean.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    cutoff = df_clean.loc[is_train_clean, "timestamp"].max()
    is_train_ctx = df_clean_ctx["timestamp"] <= cutoff

    baseline_mask = (
        is_train_ctx & (~df_clean_ctx["anomaly"]) & (~df_clean_ctx["prev_anomaly"].fillna(True)) & df_clean_ctx["gap_prev_s"].notna()
    )
    baselines = compute_node_baselines(df_clean_ctx, baseline_mask, exclude_zero_from_pooled=True)

    features_clean_full, _ = build_features(df_clean_ctx, baselines=baselines)
    features_clean_full = add_log_ratio_feature(features_clean_full, baselines)
    features_clean_train_normal = features_clean_full.loc[is_train_ctx & (~df_clean_ctx["anomaly"])]

    df_injected_ctx = add_sequence_context(df_injected.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    features_injected, _ = build_features(df_injected_ctx, baselines=baselines)
    features_injected = add_log_ratio_feature(features_injected, baselines)

    return features_clean_train_normal, features_injected, baselines


def compute_clean_train_baselines(df_clean, is_train_clean):
    """Per-node (median, MAD) baselines from the clean frame's train-period normal-to-normal gaps."""
    df_clean_ctx = add_sequence_context(df_clean.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    cutoff = df_clean.loc[is_train_clean, "timestamp"].max()
    is_train_ctx = df_clean_ctx["timestamp"] <= cutoff
    baseline_mask = (
        is_train_ctx & (~df_clean_ctx["anomaly"]) & (~df_clean_ctx["prev_anomaly"].fillna(True)) & df_clean_ctx["gap_prev_s"].notna()
    )
    return compute_node_baselines(df_clean_ctx, baseline_mask)


def score_detector(df_eval, row_pred_by_id, row_true_aligned, grid_labels, injection_ids):
    """Score row-level boolean predictions on the eval grid. Returns (metrics dict, predicted grid).

    lift = recall / fraction of grid cells flagged, i.e. recall relative to a detector that flags
    the same fraction of cells at random. An injection counts as detected if any of its labeled
    grid cells is flagged.
    """
    row_predicted_aligned = df_eval["row_id"].map(row_pred_by_id).fillna(False)
    pred_grid = rows_to_grid(df_eval, row_predicted_aligned)
    true_grid = rows_to_grid(df_eval, row_true_aligned)
    aligned = pd.concat([true_grid.rename("y_true"), pred_grid.rename("y_pred")], axis=1).fillna(False)
    grid_result = evaluate_binary(aligned["y_true"], aligned["y_pred"])

    grid_flagged_frac = pred_grid.mean()
    lift = grid_result["recall"] / grid_flagged_frac if grid_flagged_frac > 0 else float("nan")

    n_detected = 0
    for inj_id in injection_ids:
        cells = set(
            zip(
                grid_labels.loc[grid_labels["injection_id"] == inj_id, "node"],
                grid_labels.loc[grid_labels["injection_id"] == inj_id, "window_idx"],
            )
        )
        if any(pred_grid.get(c, False) for c in cells):
            n_detected += 1

    return {
        "precision": grid_result["precision"],
        "recall": grid_result["recall"],
        "f1": grid_result["f1"],
        "grid_flagged_frac": grid_flagged_frac,
        "lift": lift,
        "detection_rate": n_detected / len(injection_ids),
        "n_injections_detected": n_detected,
        "n_injections_total": len(injection_ids),
    }, pred_grid
