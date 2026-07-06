"""
Core detection experiment: do the baseline detectors catch the INJECTED stall anomalies?

v2: re-run after fixing the timing detector's feature-scaling bug (see
src/detectors/timing_detector.py's module docstring and results/iforest_diagnosis.md for the same
failure mode in a different detector). v1 (results/core_result_stall.csv, frozen for comparison)
found the timing detector's isolation forest showed lift < 1 (no better than chance) despite the
injected rows' own z_score being enormous (0.84-87.6) -- the raw-seconds rolling features were
drowning the one properly-scaled feature. v2 adds a z_score-ALONE trivial threshold detector
(ZScoreThresholdDetector, |z|>3, no ML) as the ceiling the fixed ML detector should approach.

Detectors are fit on the CLEAN (pre-injection) train-period normal data -- the exact same fitting
methodology validated for results/baseline_final.csv -- then applied across the FULL injected
dataset (not restricted to a test split by timestamp, since most of the 15 synthetic stalls land
chronologically in what would have been the "train" period) to see whether each detector flags
them. Ground truth is the injected-span grid-cell labels (src/injector.py's
injection_grid_labels_stall.csv), NOT BGL's native content labels.

PCA is the content-detector representative (isolation_forest_counts is a confirmed dead baseline
per results/iforest_diagnosis.md -- kept in the table for completeness only, per instruction).

Writes:
  results/core_result_stall_v2.csv                  -- per-detector P/R/F1/lift/detection-rate + complementarity
  results/core_result_stall_v2_timing_feature_diag.csv -- the 15 injected rows' own features + scores

Usage:
    python src/core_result_stall.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import numpy as np
import pandas as pd

from src.detectors import windowing
from src.detectors.count_pca import CountPCADetector
from src.detectors.isolation_forest_counts import IsolationForestCountsDetector
from src.detectors.timing_detector import FEATURE_COLUMNS, TimingIsolationForestDetector, ZScoreThresholdDetector, Z_THRESH, build_features
from src.metrics import assign_eval_grid, evaluate_binary, rows_to_grid
from src.run_baseline_detectors import THRESHOLD_PERCENTILE, build_count_features, chronological_split, load
from src.timing_baseline import add_sequence_context

INJECTED_PATH = Path("data/processed/bgl_injected_stall.parquet")
GRID_LABELS_PATH = Path("data/processed/injection_grid_labels_stall.csv")
OLD_CSV = Path("results/core_result_stall.csv")  # v1, frozen for comparison -- never rewritten
OUT_CSV = Path("results/core_result_stall_v2.csv")
OUT_TIMING_DIAG_CSV = Path("results/core_result_stall_v2_timing_feature_diag.csv")


def load_injected():
    df = pd.read_parquet(INJECTED_PATH)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def fit_count_detectors(df_clean_train, df_injected):
    X_train, X_all, windows_train, windows_all, df_all_w, diag = build_count_features(
        df_clean_train, df_injected, "fixed_count", 20
    )
    train_normal_mask = ~windows_train["label"].to_numpy()
    X_train_normal = X_train[train_normal_mask]

    results = {}
    for name, detector in [
        ("count_vector_pca", CountPCADetector()),
        ("isolation_forest_counts", IsolationForestCountsDetector()),
    ]:
        detector.fit(X_train_normal)
        train_scores = detector.score(X_train_normal)
        threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)
        all_scores = detector.score(X_all)
        predicted_window = all_scores > threshold

        window_predicted = pd.Series(predicted_window, index=windows_all["window_key"])
        row_predicted = pd.Series(
            df_all_w["window_key"].map(window_predicted).to_numpy(), index=df_all_w["row_id"].to_numpy()
        )
        results[name] = row_predicted
    return results, diag


def build_timing_features(df_clean, is_train_clean, df_injected):
    """Shared feature-building step for both timing-family detectors (the ML one and the
    z_score-alone sanity check) -- avoids recomputing rolling features over 4.7M rows twice."""
    df_clean_ctx = add_sequence_context(df_clean.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))

    cutoff = df_clean.loc[is_train_clean, "timestamp"].max()
    is_train_ctx = df_clean_ctx["timestamp"] <= cutoff

    baseline_mask = is_train_ctx & (~df_clean_ctx["anomaly"]) & (~df_clean_ctx["prev_anomaly"].fillna(True)) & df_clean_ctx["gap_prev_s"].notna()
    features_clean_full, baselines = build_features(df_clean_ctx, baseline_mask=baseline_mask)
    features_clean_train_normal = features_clean_full.loc[is_train_ctx & (~df_clean_ctx["anomaly"])]

    df_injected_ctx = add_sequence_context(df_injected.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    features_injected, _ = build_features(df_injected_ctx, baselines=baselines)

    return features_clean_train_normal, features_injected


def fit_timing_ml_detector(features_clean_train_normal, features_injected):
    detector = TimingIsolationForestDetector()
    detector.fit(features_clean_train_normal)
    train_scores = detector.score(features_clean_train_normal)
    threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)

    all_scores = detector.score(features_injected)
    predicted = all_scores > threshold
    row_predicted = pd.Series(predicted, index=features_injected["row_id"].to_numpy())

    injected_diag = features_injected.loc[features_injected["injected_row"]].copy()
    injected_diag["score"] = detector.score(injected_diag)
    injected_diag["threshold"] = threshold
    injected_diag["flagged"] = injected_diag["score"] > threshold
    return row_predicted, injected_diag[["node"] + FEATURE_COLUMNS + ["score", "threshold", "flagged"]]


def fit_zscore_threshold_detector(features_injected):
    """The minimal-baseline sanity check: no fitting, no isolation forest, fixed |z|>Z_THRESH."""
    detector = ZScoreThresholdDetector()
    scores = detector.score(features_injected)
    predicted = scores > detector.z_thresh
    return pd.Series(predicted.to_numpy(), index=features_injected["row_id"].to_numpy())


def main():
    df_clean = load(limit=None)
    is_train, cutoff = chronological_split(df_clean)
    df_clean_train = df_clean.loc[is_train].reset_index(drop=True)
    print(f"Clean-data chronological split at {cutoff} -- {is_train.sum():,} train rows used for fitting only")

    df_injected = load_injected()
    print(f"Injected dataset: {len(df_injected):,} rows")

    count_predictions, count_diag = fit_count_detectors(df_clean_train, df_injected)

    features_clean_train_normal, features_injected = build_timing_features(df_clean, is_train, df_injected)
    timing_predicted, timing_row_diag = fit_timing_ml_detector(features_clean_train_normal, features_injected)
    zscore_predicted = fit_zscore_threshold_detector(features_injected)

    predictions = {
        **count_predictions,
        "timing_inter_arrival_isolation_forest": timing_predicted,
        "timing_zscore_threshold": zscore_predicted,
    }

    print("\n=== Diagnostic: the 15 injected rows' own (now node-normalized) timing features + ML score ===")
    pd.set_option("display.width", 200)
    print(timing_row_diag.to_string())
    print(
        f"\n{timing_row_diag['flagged'].sum()}/{len(timing_row_diag)} flagged by the ML detector "
        f"(z_score ranging {timing_row_diag['z_score'].min():.2f}-{timing_row_diag['z_score'].max():.2f}, "
        f"all >> Z_THRESH={Z_THRESH})."
    )

    # --- ground truth: injected-span grid-cell labels, mapped onto this same eval grid ---
    grid_labels = pd.read_csv(GRID_LABELS_PATH)
    anomalous_cells = set(zip(grid_labels["node"], grid_labels["window_idx"]))

    df_eval = assign_eval_grid(df_injected)
    row_true = pd.Series(
        [k in anomalous_cells for k in df_eval["eval_window_key"]], index=df_eval["row_id"].to_numpy()
    )
    row_true_aligned = df_eval["row_id"].map(row_true).fillna(False)

    injection_ids = sorted(grid_labels["injection_id"].unique())
    n_injections = len(injection_ids)

    results = []
    grid_predicted_by_detector = {}
    for name, row_pred_by_id in predictions.items():
        row_predicted_aligned = df_eval["row_id"].map(row_pred_by_id).fillna(False)

        pred_grid = rows_to_grid(df_eval, row_predicted_aligned)
        true_grid = rows_to_grid(df_eval, row_true_aligned)
        aligned = pd.concat([true_grid.rename("y_true"), pred_grid.rename("y_pred")], axis=1).fillna(False)
        grid_result = evaluate_binary(aligned["y_true"], aligned["y_pred"])
        grid_predicted_by_detector[name] = pred_grid

        grid_flagged_frac = pred_grid.mean()
        lift = grid_result["recall"] / grid_flagged_frac if grid_flagged_frac > 0 else float("nan")

        n_detected = 0
        chance_probs = []
        for inj_id in injection_ids:
            cells = set(
                zip(
                    grid_labels.loc[grid_labels["injection_id"] == inj_id, "node"],
                    grid_labels.loc[grid_labels["injection_id"] == inj_id, "window_idx"],
                )
            )
            discoverable = [c for c in cells if c in pred_grid.index]
            cell_flags = [pred_grid.get(c, False) for c in discoverable]
            if any(cell_flags):
                n_detected += 1
            chance_probs.append(1 - (1 - grid_flagged_frac) ** len(discoverable) if discoverable else 0.0)
        detection_rate = n_detected / n_injections
        expected_detection_rate_by_chance = float(np.mean(chance_probs))

        results.append(
            {
                "detector": name,
                "precision": grid_result["precision"],
                "recall": grid_result["recall"],
                "f1": grid_result["f1"],
                "grid_flagged_frac": grid_flagged_frac,
                "lift": lift,
                "detection_rate": detection_rate,
                "expected_detection_rate_by_chance": expected_detection_rate_by_chance,
                "n_injections_detected": n_detected,
                "n_injections_total": n_injections,
                "n_eval_cells": grid_result["tp"] + grid_result["fp"] + grid_result["fn"] + grid_result["tn"],
                "tp": grid_result["tp"],
                "fp": grid_result["fp"],
                "fn": grid_result["fn"],
                "tn": grid_result["tn"],
            }
        )

    results_df = pd.DataFrame(results)

    # --- complementarity: PCA vs the fixed ML timing detector ---
    pca_grid = grid_predicted_by_detector["count_vector_pca"]
    timing_grid = grid_predicted_by_detector["timing_inter_arrival_isolation_forest"]
    true_grid_ref = rows_to_grid(df_eval, row_true_aligned)

    all_cells = pca_grid.index.union(timing_grid.index)
    pca_aligned = pca_grid.reindex(all_cells, fill_value=False)
    timing_aligned = timing_grid.reindex(all_cells, fill_value=False)
    true_aligned_full = true_grid_ref.reindex(all_cells, fill_value=False)

    intersection = (pca_aligned & timing_aligned).sum()
    union = (pca_aligned | timing_aligned).sum()
    jaccard = intersection / union if union else float("nan")

    or_fusion = pca_aligned | timing_aligned
    fusion_result = evaluate_binary(true_aligned_full, or_fusion)
    pca_only_result = evaluate_binary(true_aligned_full, pca_aligned)
    timing_only_result = evaluate_binary(true_aligned_full, timing_aligned)

    pca_lift = results_df.loc[results_df["detector"] == "count_vector_pca", "lift"].iloc[0]
    timing_lift = results_df.loc[results_df["detector"] == "timing_inter_arrival_isolation_forest", "lift"].iloc[0]
    any_above_chance = (pca_lift > 1.2) or (timing_lift > 1.2)  # a modest margin above pure chance (lift=1)

    complementarity = {
        "jaccard_pca_timing": jaccard,
        "pca_recall": pca_only_result["recall"],
        "timing_recall": timing_only_result["recall"],
        "fusion_recall": fusion_result["recall"],
        "recall_gain_over_pca": fusion_result["recall"] - pca_only_result["recall"],
        "recall_gain_over_timing": fusion_result["recall"] - timing_only_result["recall"],
        "pca_precision": pca_only_result["precision"],
        "timing_precision": timing_only_result["precision"],
        "fusion_precision": fusion_result["precision"],
        "precision_cost_vs_pca": pca_only_result["precision"] - fusion_result["precision"],
        "precision_cost_vs_timing": timing_only_result["precision"] - fusion_result["precision"],
        "meaningful": any_above_chance,
    }

    for k, v in complementarity.items():
        results_df[k] = v

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_CSV, index=False)
    timing_row_diag.to_csv(OUT_TIMING_DIAG_CSV, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== Core result v2 (fixed timing detector + z_score-alone ceiling): detection of INJECTED stalls (60s grid) ===")
    print(results_df[["detector", "precision", "recall", "f1", "grid_flagged_frac", "lift", "detection_rate", "expected_detection_rate_by_chance", "n_injections_detected", "n_injections_total"]].to_string(index=False))

    print("\n=== Complementarity: PCA vs (fixed) timing ML detector ===")
    for k, v in complementarity.items():
        print(f"  {k}: {v}")
    print(
        "\nInterpreted as meaningful (at least one detector shows real above-chance lift)."
        if any_above_chance
        else "\nNOT interpreted as meaningful -- both detectors remain at/near chance-level lift; "
        "complementarity numbers above are mechanical, not evidence of real detection complementarity."
    )

    if OLD_CSV.exists():
        old = pd.read_csv(OLD_CSV)
        print(f"\n=== v1 vs v2: timing_inter_arrival_isolation_forest ===")
        cols = ["precision", "recall", "f1", "grid_flagged_frac", "lift", "detection_rate", "n_injections_detected"]
        old_row = old[old["detector"] == "timing_inter_arrival_isolation_forest"][cols].iloc[0]
        new_row = results_df[results_df["detector"] == "timing_inter_arrival_isolation_forest"][cols].iloc[0]
        comparison = pd.DataFrame({"v1 (broken scaling)": old_row, "v2 (fixed scaling)": new_row})
        print(comparison.to_string())

    print(f"\nWrote {OUT_CSV}, {OUT_TIMING_DIAG_CSV}")
    return results_df, complementarity


if __name__ == "__main__":
    main()
