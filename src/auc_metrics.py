"""
Threshold-independent evaluation (AUC-PR and AUC-ROC) of the four fixed-time detectors on
injected BGL and Thunderbird data, stalls and bursts.

AUC scores each detector's raw continuous score, so it does not depend on each detector's
threshold choice (the z-score uses a fixed |z| > 3 rule, the others a train-percentile threshold).

Raw score per detector (higher = more anomalous):
  - count_vector_pca:        PCA reconstruction error (src.detectors.count_pca)
  - isolation_forest_counts: -IsolationForest.score_samples (src.detectors.isolation_forest_counts)
  - z_score_threshold:       |z_score| (src.detectors.timing_detector)
  - log_ratio_threshold:     |log_ratio| (src.detectors.timing_detector)

The count detectors score fixed-count windows of 20 events; the timing detectors score rows.
All scores are broadcast to rows and aggregated onto the shared 60 s per-node evaluation grid
(src.metrics.assign_eval_grid) by taking the max within each cell.

Ground truth is the injected-span grid-cell labels written by the injector. The AUC-PR no-skill
baseline is the positive-cell prevalence; it is reported alongside AUC-PR because positives are
very rare.

Writes:
  results/auc_metrics.csv -- dataset x fault_type x detector x
                              {auc_pr, no_skill_baseline, auc_pr_ratio, auc_roc,
                               n_eval_cells, n_positive_cells}

Usage:
    python src/auc_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.detectors.count_pca import CountPCADetector
from src.detectors.isolation_forest_counts import IsolationForestCountsDetector
from src.detectors.timing_detector import LogRatioThresholdDetector, ZScoreThresholdDetector, add_log_ratio_feature
from src.metrics import assign_eval_grid, rows_to_grid, rows_to_grid_max
from src.pipeline_common import (
    build_timing_features,
    build_timing_features_tb,
    chronological_split_tb,
    compute_clean_train_baselines,
    load_bgl_injected_burst as load_bgl_burst,
    load_bgl_injected_stall as load_bgl_stall,
    load_tb_clean,
    load_tb_injected,
)
from src.run_baseline_detectors import build_count_features, chronological_split as chronological_split_bgl, load as load_bgl_clean

OUT_CSV = Path("results/auc_metrics.csv")

DETECTOR_ORDER = ["count_vector_pca", "z_score_threshold", "log_ratio_threshold", "isolation_forest_counts"]


def fit_count_detector_scores(df_clean_train, df_injected):
    """Fit PCA and isolation forest on clean train-normal fixed-count (N=20) windows and return
    each detector's continuous score broadcast to the rows of `df_injected`."""
    X_train, X_all, windows_train, windows_all, df_all_w, _ = build_count_features(df_clean_train, df_injected, "fixed_count", 20)
    train_normal_mask = ~windows_train["label"].to_numpy()
    X_train_normal = X_train[train_normal_mask]

    scores = {}
    for name, detector in [
        ("count_vector_pca", CountPCADetector()),
        ("isolation_forest_counts", IsolationForestCountsDetector()),
    ]:
        detector.fit(X_train_normal)
        all_scores = detector.score(X_all)
        window_score = pd.Series(all_scores, index=windows_all["window_key"])
        row_score = pd.Series(df_all_w["window_key"].map(window_score).to_numpy(), index=df_all_w["row_id"].to_numpy())
        scores[name] = row_score
    return scores


def bgl_scores(fault_type, df_clean, is_train):
    df_clean_train = df_clean.loc[is_train].reset_index(drop=True)
    df_injected = load_bgl_stall() if fault_type == "stall" else load_bgl_burst()

    count_scores = fit_count_detector_scores(df_clean_train, df_injected)

    _, features_injected = build_timing_features(df_clean, is_train, df_injected)
    baselines = compute_clean_train_baselines(df_clean, is_train)
    features_injected = add_log_ratio_feature(features_injected, baselines)

    zscore_scores = pd.Series(ZScoreThresholdDetector().score(features_injected).to_numpy(), index=features_injected["row_id"].to_numpy())
    logratio_scores = pd.Series(LogRatioThresholdDetector().score(features_injected).to_numpy(), index=features_injected["row_id"].to_numpy())

    grid_labels_path = f"data/processed/injection_grid_labels_{fault_type}.csv"
    scores_by_detector = {**count_scores, "z_score_threshold": zscore_scores, "log_ratio_threshold": logratio_scores}
    return scores_by_detector, df_injected, grid_labels_path


def tb_scores(fault_type, df_clean, is_train):
    df_clean_train = df_clean.loc[is_train].reset_index(drop=True)
    df_injected = load_tb_injected(fault_type)

    count_scores = fit_count_detector_scores(df_clean_train, df_injected)

    _, features_injected, _ = build_timing_features_tb(df_clean, is_train, df_injected)  # already has z_score + log_ratio

    zscore_scores = pd.Series(ZScoreThresholdDetector().score(features_injected).to_numpy(), index=features_injected["row_id"].to_numpy())
    logratio_scores = pd.Series(LogRatioThresholdDetector().score(features_injected).to_numpy(), index=features_injected["row_id"].to_numpy())

    grid_labels_path = f"data/processed/thunderbird_injection_grid_labels_{fault_type}.csv"
    scores_by_detector = {**count_scores, "z_score_threshold": zscore_scores, "log_ratio_threshold": logratio_scores}
    return scores_by_detector, df_injected, grid_labels_path


def compute_grid_auc(df_eval, row_score_by_id, row_true_aligned):
    row_score_aligned = df_eval["row_id"].map(row_score_by_id)
    n_missing = int(row_score_aligned.isna().sum())
    if n_missing:
        # Every row is expected to have a score. All scores are >= 0, so filling with 0 can never
        # raise a cell's max above what its scored rows give it.
        print(f"    WARNING: {n_missing} of {len(row_score_aligned)} rows had no score -- filling with 0")
        row_score_aligned = row_score_aligned.fillna(0.0)

    score_grid = rows_to_grid_max(df_eval, row_score_aligned)
    true_grid = rows_to_grid(df_eval, row_true_aligned)

    aligned = pd.concat([true_grid.rename("y_true"), score_grid.rename("score")], axis=1)
    assert aligned["y_true"].notna().all() and aligned["score"].notna().all(), "score/truth grids must share the same cell universe"

    y_true = aligned["y_true"].astype(bool).to_numpy()
    y_score = aligned["score"].to_numpy()

    baseline = float(y_true.mean())
    auc_pr = float(average_precision_score(y_true, y_score))
    auc_roc = float(roc_auc_score(y_true, y_score)) if 0 < y_true.sum() < len(y_true) else float("nan")

    return {
        "auc_pr": auc_pr,
        "no_skill_baseline": baseline,
        "auc_pr_ratio": auc_pr / baseline if baseline > 0 else float("nan"),
        "auc_roc": auc_roc,
        "n_eval_cells": len(aligned),
        "n_positive_cells": int(y_true.sum()),
    }


def score_all_detectors(dataset, fault_type, scores_by_detector, df_injected, grid_labels_path):
    grid_labels = pd.read_csv(grid_labels_path)
    anomalous_cells = set(zip(grid_labels["node"], grid_labels["window_idx"]))

    df_eval = assign_eval_grid(df_injected)
    row_true = pd.Series([k in anomalous_cells for k in df_eval["eval_window_key"]], index=df_eval["row_id"].to_numpy())
    row_true_aligned = df_eval["row_id"].map(row_true).fillna(False)

    out = []
    for detector_name, row_score_by_id in scores_by_detector.items():
        metrics = compute_grid_auc(df_eval, row_score_by_id, row_true_aligned)
        metrics["dataset"] = dataset
        metrics["fault_type"] = fault_type
        metrics["detector"] = detector_name
        out.append(metrics)
        print(
            f"  {dataset:11s} {fault_type:5s} {detector_name:25s} "
            f"auc_pr={metrics['auc_pr']:.5f}  baseline={metrics['no_skill_baseline']:.6f}  "
            f"ratio={metrics['auc_pr_ratio']:7.2f}  auc_roc={metrics['auc_roc']:.4f}"
        )
    return out


def main():
    rows = []

    print("=== BGL ===")
    df_bgl = load_bgl_clean(limit=None)
    is_train_bgl, _ = chronological_split_bgl(df_bgl)
    for fault_type in ["stall", "burst"]:
        scores_by_detector, df_injected, grid_labels_path = bgl_scores(fault_type, df_bgl, is_train_bgl)
        rows.extend(score_all_detectors("BGL", fault_type, scores_by_detector, df_injected, grid_labels_path))

    print("\n=== Thunderbird ===")
    df_tb = load_tb_clean()
    is_train_tb, _ = chronological_split_tb(df_tb)
    for fault_type in ["stall", "burst"]:
        scores_by_detector, df_injected, grid_labels_path = tb_scores(fault_type, df_tb, is_train_tb)
        rows.extend(score_all_detectors("Thunderbird", fault_type, scores_by_detector, df_injected, grid_labels_path))

    out_df = pd.DataFrame(rows)
    out_df["detector_order"] = out_df["detector"].map({d: i for i, d in enumerate(DETECTOR_ORDER)})
    out_df["fault_order"] = out_df["fault_type"].map({"stall": 0, "burst": 1})
    out_df["dataset_order"] = out_df["dataset"].map({"BGL": 0, "Thunderbird": 1})
    out_df = out_df.sort_values(["dataset_order", "fault_order", "detector_order"]).drop(columns=["detector_order", "fault_order", "dataset_order"])
    out_df = out_df[["dataset", "fault_type", "detector", "auc_pr", "no_skill_baseline", "auc_pr_ratio", "auc_roc", "n_eval_cells", "n_positive_cells"]]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print("\n=== Full AUC-PR / AUC-ROC table ===")
    print(out_df.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
