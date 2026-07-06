"""
Tests whether a compression-symmetric timing feature (log-ratio) closes the burst detection gap
left by z_score_threshold's additive bias (results/core_result_burst.csv found z_score_threshold
at lift 0.46 on bursts, vs 1.55 on stalls -- see src/detectors/timing_detector.py's
ZScoreThresholdDetector docstring for why: a stall ADDS intensity*node_scale directly, so its
z-score is unbounded by construction; a burst COMPRESSES the gap, and its z-score is bounded by
roughly -(node_median/node_scale) regardless of compression factor -- structurally below
Z_THRESH=3 for BGL's characteristic ~0.675 median/MAD ratio).

log_ratio = log(gap / node_baseline_median) is symmetric: a stall gives a large positive value, a
burst gives a large negative one of comparable magnitude, since "N times larger" and "N times
smaller" are equal-magnitude opposite-sign deviations in log space.

Does NOT touch the stall injector or the (locked) stall pipeline -- LogRatioThresholdDetector and
add_log_ratio_feature (src/detectors/timing_detector.py) are purely additive; this script reuses
src/core_result_stall.py's existing fitting utilities unchanged, and reads the existing injected
parquets without modifying them.

Writes:
  results/core_result_burst_v2.csv      -- burst detection: PCA, isolation_forest_counts,
                                            z_score_threshold, log_ratio_threshold
  results/timing_feature_comparison.csv -- z_score_threshold vs log_ratio_threshold on BOTH
                                            stall and burst -- the key comparison
  results/core_result_consolidated.csv  -- stall x burst x {PCA, z_score, log_ratio} summary

Usage:
    python src/core_result_v2_symmetric.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import numpy as np
import pandas as pd

from src.core_result_burst import load_injected_burst
from src.core_result_stall import build_timing_features, fit_count_detectors, fit_zscore_threshold_detector, load_injected as load_injected_stall
from src.detectors.timing_detector import LogRatioThresholdDetector, add_log_ratio_feature
from src.metrics import assign_eval_grid, evaluate_binary, rows_to_grid
from src.run_baseline_detectors import chronological_split, load
from src.timing_baseline import add_sequence_context, compute_node_baselines

OUT_BURST_V2_CSV = Path("results/core_result_burst_v2.csv")
OUT_COMPARISON_CSV = Path("results/timing_feature_comparison.csv")
OUT_CONSOLIDATED_CSV = Path("results/core_result_consolidated.csv")

FAULT_CONFIGS = {
    "stall": {
        "parquet": Path("data/processed/bgl_injected_stall.parquet"),
        "grid_labels": Path("data/processed/injection_grid_labels_stall.csv"),
        "loader": load_injected_stall,
    },
    "burst": {
        "parquet": Path("data/processed/bgl_injected_burst.parquet"),
        "grid_labels": Path("data/processed/injection_grid_labels_burst.csv"),
        "loader": load_injected_burst,
    },
}


def compute_clean_train_baselines(df_clean, is_train_clean):
    df_clean_ctx = add_sequence_context(df_clean.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    cutoff = df_clean.loc[is_train_clean, "timestamp"].max()
    is_train_ctx = df_clean_ctx["timestamp"] <= cutoff
    baseline_mask = (
        is_train_ctx & (~df_clean_ctx["anomaly"]) & (~df_clean_ctx["prev_anomaly"].fillna(True)) & df_clean_ctx["gap_prev_s"].notna()
    )
    return compute_node_baselines(df_clean_ctx, baseline_mask)


def fit_log_ratio_detector(df_clean, is_train, df_injected):
    features_clean_train_normal, features_injected = build_timing_features(df_clean, is_train, df_injected)
    baselines = compute_clean_train_baselines(df_clean, is_train)
    features_clean_train_normal = add_log_ratio_feature(features_clean_train_normal, baselines)
    features_injected = add_log_ratio_feature(features_injected, baselines)

    detector = LogRatioThresholdDetector()
    detector.fit(features_clean_train_normal)
    scores = detector.score(features_injected)
    predicted = scores > detector.threshold
    row_predicted = pd.Series(predicted.to_numpy(), index=features_injected["row_id"].to_numpy())
    return row_predicted, detector.threshold


def score_detector(df_eval, row_pred_by_id, row_true_aligned, grid_labels, injection_ids):
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
    }


def run_fault_experiment(fault_type, df_clean, is_train):
    cfg = FAULT_CONFIGS[fault_type]
    df_clean_train = df_clean.loc[is_train].reset_index(drop=True)
    df_injected = cfg["loader"]()

    count_predictions, _ = fit_count_detectors(df_clean_train, df_injected)
    _, features_injected = build_timing_features(df_clean, is_train, df_injected)
    zscore_predicted = fit_zscore_threshold_detector(features_injected)
    log_ratio_predicted, log_ratio_threshold = fit_log_ratio_detector(df_clean, is_train, df_injected)

    predictions = {
        "count_vector_pca": count_predictions["count_vector_pca"],
        "isolation_forest_counts": count_predictions["isolation_forest_counts"],
        "z_score_threshold": zscore_predicted,
        "log_ratio_threshold": log_ratio_predicted,
    }

    grid_labels = pd.read_csv(cfg["grid_labels"])
    anomalous_cells = set(zip(grid_labels["node"], grid_labels["window_idx"]))
    injection_ids = sorted(grid_labels["injection_id"].unique())

    df_eval = assign_eval_grid(df_injected)
    row_true = pd.Series(
        [k in anomalous_cells for k in df_eval["eval_window_key"]], index=df_eval["row_id"].to_numpy()
    )
    row_true_aligned = df_eval["row_id"].map(row_true).fillna(False)

    rows = []
    for name, row_pred_by_id in predictions.items():
        metrics = score_detector(df_eval, row_pred_by_id, row_true_aligned, grid_labels, injection_ids)
        metrics["detector"] = name
        metrics["fault_type"] = fault_type
        rows.append(metrics)

    return pd.DataFrame(rows), log_ratio_threshold


def main():
    df_clean = load(limit=None)
    is_train, cutoff = chronological_split(df_clean)

    all_results = {}
    log_ratio_thresholds = {}
    for fault_type in ["stall", "burst"]:
        print(f"\n=== Running {fault_type} experiment ===")
        results_df, lr_thresh = run_fault_experiment(fault_type, df_clean, is_train)
        all_results[fault_type] = results_df
        log_ratio_thresholds[fault_type] = lr_thresh
        print(results_df[["detector", "recall", "detection_rate", "grid_flagged_frac", "lift"]].to_string(index=False))

    # --- burst v2: full 4-detector table ---
    burst_df = all_results["burst"]
    front_cols = ["detector", "recall", "detection_rate", "grid_flagged_frac", "lift", "n_injections_detected", "n_injections_total", "precision", "f1"]
    OUT_BURST_V2_CSV.parent.mkdir(parents=True, exist_ok=True)
    burst_df[front_cols].to_csv(OUT_BURST_V2_CSV, index=False)

    # --- timing feature comparison: z_score vs log_ratio, both fault types ---
    comparison_rows = []
    for fault_type, results_df in all_results.items():
        for detector in ["z_score_threshold", "log_ratio_threshold"]:
            row = results_df[results_df["detector"] == detector].iloc[0]
            comparison_rows.append(
                {
                    "fault_type": fault_type,
                    "detector": detector,
                    "recall": row["recall"],
                    "detection_rate": row["detection_rate"],
                    "grid_flagged_frac": row["grid_flagged_frac"],
                    "lift": row["lift"],
                    "n_injections_detected": row["n_injections_detected"],
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_COMPARISON_CSV, index=False)

    # --- consolidated summary: stall x burst x {PCA, z_score, log_ratio} ---
    consolidated_rows = []
    for fault_type, results_df in all_results.items():
        row = {"fault_type": fault_type}
        for detector, label in [
            ("count_vector_pca", "pca"),
            ("z_score_threshold", "zscore"),
            ("log_ratio_threshold", "logratio"),
        ]:
            r = results_df[results_df["detector"] == detector].iloc[0]
            row[f"{label}_detection_rate"] = r["detection_rate"]
            row[f"{label}_lift"] = r["lift"]
        consolidated_rows.append(row)
    consolidated_df = pd.DataFrame(consolidated_rows)
    consolidated_df.to_csv(OUT_CONSOLIDATED_CSV, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== Timing feature comparison: z_score_threshold vs log_ratio_threshold ===")
    print(comparison_df.to_string(index=False))

    print("\n=== Consolidated: stall x burst x {PCA, z_score, log_ratio} ===")
    print(consolidated_df.to_string(index=False))

    # --- key question checks ---
    burst_log_ratio_lift = comparison_df.query("fault_type=='burst' and detector=='log_ratio_threshold'")["lift"].iloc[0]
    burst_zscore_lift = comparison_df.query("fault_type=='burst' and detector=='z_score_threshold'")["lift"].iloc[0]
    stall_log_ratio_lift = comparison_df.query("fault_type=='stall' and detector=='log_ratio_threshold'")["lift"].iloc[0]
    stall_zscore_lift = comparison_df.query("fault_type=='stall' and detector=='z_score_threshold'")["lift"].iloc[0]
    pca_stall_lift = consolidated_df.loc[consolidated_df["fault_type"] == "stall", "pca_lift"].iloc[0]
    pca_burst_lift = consolidated_df.loc[consolidated_df["fault_type"] == "burst", "pca_lift"].iloc[0]

    print(f"\nBurst: log_ratio lift={burst_log_ratio_lift:.3f} vs z_score lift={burst_zscore_lift:.3f} "
          f"({'CLOSES the gap' if burst_log_ratio_lift > 1.2 else 'does NOT close the gap'})")
    print(f"Stall: log_ratio lift={stall_log_ratio_lift:.3f} vs z_score lift={stall_zscore_lift:.3f} "
          f"({'log-ratio still catches stalls' if stall_log_ratio_lift > 1.2 else 'log-ratio REGRESSED on stalls'})")
    print(f"Content (PCA) lift: stall={pca_stall_lift:.3f}, burst={pca_burst_lift:.3f} "
          f"({'both at/near chance' if pca_stall_lift <= 1.2 and pca_burst_lift <= 1.2 else 'at least one above chance'})")

    print(f"\nWrote {OUT_BURST_V2_CSV}, {OUT_COMPARISON_CSV}, {OUT_CONSOLIDATED_CSV}")


if __name__ == "__main__":
    main()
