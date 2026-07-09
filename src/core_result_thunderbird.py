"""
Port of src/core_result_v2_symmetric.py to Thunderbird -- reuses the exact same detector classes
and orchestration functions UNMODIFIED via direct import (fit_count_detectors, score_detector,
CountPCADetector, IsolationForestCountsDetector, ZScoreThresholdDetector,
LogRatioThresholdDetector, add_log_ratio_feature): none of these have any BGL-specific logic, they
already take dataframes/paths as parameters.

The ONE Thunderbird-specific piece is timing-baseline construction, which needs Part 1's fix
(exclude_zero_from_pooled=True) -- reimplemented here as build_timing_features_tb rather than
modifying src/core_result_stall.py's build_timing_features, so the BGL pipeline is never touched.

Ground truth is the injected-span grid-cell labels, same fixed-time 60s grid as BGL
(src.metrics.EVAL_WINDOW_SCHEME/EVAL_WINDOW_SIZE, unmodified).

Writes:
  results/thunderbird_core_result.csv

Usage:
    python src/core_result_thunderbird.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import numpy as np
import pandas as pd

from src.core_result_stall import fit_count_detectors, fit_zscore_threshold_detector
from src.core_result_stall_final import score_detector
from src.detectors.timing_detector import LogRatioThresholdDetector, add_log_ratio_feature, build_features
from src.metrics import assign_eval_grid, rows_to_grid
from src.run_baseline_detectors import THRESHOLD_PERCENTILE
from src.timing_baseline import add_sequence_context, compute_node_baselines

CLEAN_PATH = Path("data/processed/thunderbird_parsed.parquet")
TRAIN_FRAC = 0.7  # same convention as src.run_baseline_detectors
OUT_CSV = Path("results/thunderbird_core_result.csv")


def load_clean():
    df = pd.read_parquet(CLEAN_PATH)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def chronological_split(df, train_frac=TRAIN_FRAC):
    cutoff = df["timestamp"].quantile(train_frac)
    is_train = df["timestamp"] <= cutoff
    return is_train, cutoff


def load_injected(inj_type):
    path = Path(f"data/processed/thunderbird_injected_{inj_type}.parquet")
    df = pd.read_parquet(path)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def build_timing_features_tb(df_clean, is_train_clean, df_injected):
    """Same as src.core_result_stall.build_timing_features, except the baseline is computed with
    exclude_zero_from_pooled=True (Part 1's fix), since Thunderbird's global pooled fallback is
    otherwise degenerate (median=MAD=0, >50% of all gaps are exactly 0)."""
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


def fit_log_ratio_detector_tb(features_clean_train_normal, features_injected):
    detector = LogRatioThresholdDetector()
    detector.fit(features_clean_train_normal)
    scores = detector.score(features_injected)
    predicted = scores > detector.threshold
    return pd.Series(predicted.to_numpy(), index=features_injected["row_id"].to_numpy())


def run_fault_experiment(fault_type, df_clean, is_train):
    df_clean_train = df_clean.loc[is_train].reset_index(drop=True)
    df_injected = load_injected(fault_type)

    count_predictions, count_diag = fit_count_detectors(df_clean_train, df_injected)
    features_clean_train_normal, features_injected, baselines = build_timing_features_tb(df_clean, is_train, df_injected)
    zscore_predicted = fit_zscore_threshold_detector(features_injected)
    log_ratio_predicted = fit_log_ratio_detector_tb(features_clean_train_normal, features_injected)

    predictions = {
        "count_vector_pca": count_predictions["count_vector_pca"],
        "isolation_forest_counts": count_predictions["isolation_forest_counts"],
        "z_score_threshold": zscore_predicted,
        "log_ratio_threshold": log_ratio_predicted,
    }

    grid_labels = pd.read_csv(f"data/processed/thunderbird_injection_grid_labels_{fault_type}.csv")
    anomalous_cells = set(zip(grid_labels["node"], grid_labels["window_idx"]))
    injection_ids = sorted(grid_labels["injection_id"].unique())

    df_eval = assign_eval_grid(df_injected)
    row_true = pd.Series(
        [k in anomalous_cells for k in df_eval["eval_window_key"]], index=df_eval["row_id"].to_numpy()
    )
    row_true_aligned = df_eval["row_id"].map(row_true).fillna(False)

    rows = []
    for name, row_pred_by_id in predictions.items():
        metrics, _ = score_detector(df_eval, row_pred_by_id, row_true_aligned, grid_labels, injection_ids)
        metrics["detector"] = name
        metrics["fault_type"] = fault_type
        rows.append(metrics)

    return pd.DataFrame(rows), count_diag


def main():
    df_clean = load_clean()
    is_train, cutoff = chronological_split(df_clean)
    print(f"Thunderbird chronological split at {cutoff} -- {is_train.sum():,} train rows used for fitting only")

    all_results = []
    for fault_type in ["stall", "burst"]:
        print(f"\n=== Running Thunderbird {fault_type} experiment ===")
        results_df, count_diag = run_fault_experiment(fault_type, df_clean, is_train)
        all_results.append(results_df)
        print(results_df[["detector", "recall", "detection_rate", "grid_flagged_frac", "lift"]].to_string(index=False))

    combined = pd.concat(all_results, ignore_index=True)
    front_cols = ["fault_type", "detector", "recall", "detection_rate", "grid_flagged_frac", "lift", "n_injections_detected", "n_injections_total", "precision", "f1"]
    combined = combined[front_cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 220)
    print("\n=== Thunderbird core result (both fault types) ===")
    print(combined.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")
    return combined


if __name__ == "__main__":
    main()
