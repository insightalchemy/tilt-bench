
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) 

import numpy as np
import pandas as pd

from src.core_result_stall import build_timing_features, fit_count_detectors, fit_zscore_threshold_detector
from src.metrics import assign_eval_grid, evaluate_binary, rows_to_grid
from src.run_baseline_detectors import chronological_split, load

INJECTED_PATH = Path("data/processed/bgl_injected_burst.parquet")
GRID_LABELS_PATH = Path("data/processed/injection_grid_labels_burst.csv")
OUT_CSV = Path("results/core_result_burst.csv")


def load_injected_burst():
    df = pd.read_parquet(INJECTED_PATH)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


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
    }, pred_grid


def main():
    df_clean = load(limit=None)
    is_train, cutoff = chronological_split(df_clean)
    df_clean_train = df_clean.loc[is_train].reset_index(drop=True)

    df_injected = load_injected_burst()
    print(f"Burst-injected dataset: {len(df_injected):,} rows")

    count_predictions, _ = fit_count_detectors(df_clean_train, df_injected)
    _, features_injected = build_timing_features(df_clean, is_train, df_injected)
    zscore_predicted = fit_zscore_threshold_detector(features_injected)

    predictions = {
        "count_vector_pca": count_predictions["count_vector_pca"],
        "z_score_threshold": zscore_predicted,
        "isolation_forest_counts": count_predictions["isolation_forest_counts"],
    }

    grid_labels = pd.read_csv(GRID_LABELS_PATH)
    anomalous_cells = set(zip(grid_labels["node"], grid_labels["window_idx"]))
    injection_ids = sorted(grid_labels["injection_id"].unique())

    df_eval = assign_eval_grid(df_injected)
    row_true = pd.Series(
        [k in anomalous_cells for k in df_eval["eval_window_key"]], index=df_eval["row_id"].to_numpy()
    )
    row_true_aligned = df_eval["row_id"].map(row_true).fillna(False)

    results = []
    grids = {}
    for name, row_pred_by_id in predictions.items():
        metrics, pred_grid = score_detector(df_eval, row_pred_by_id, row_true_aligned, grid_labels, injection_ids)
        metrics["detector"] = name
        results.append(metrics)
        grids[name] = pred_grid

    results_df = pd.DataFrame(results)[
        ["detector", "recall", "detection_rate", "grid_flagged_frac", "lift", "n_injections_detected", "n_injections_total", "precision", "f1"]
    ]

    pca_grid, z_grid = grids["count_vector_pca"], grids["z_score_threshold"]
    true_grid_ref = rows_to_grid(df_eval, row_true_aligned)
    all_cells = pca_grid.index.union(z_grid.index)
    pca_aligned = pca_grid.reindex(all_cells, fill_value=False)
    z_aligned = z_grid.reindex(all_cells, fill_value=False)
    true_aligned = true_grid_ref.reindex(all_cells, fill_value=False)

    jaccard = (pca_aligned & z_aligned).sum() / (pca_aligned | z_aligned).sum()
    fusion = pca_aligned | z_aligned
    fusion_result = evaluate_binary(true_aligned, fusion)
    pca_result = evaluate_binary(true_aligned, pca_aligned)
    z_result = evaluate_binary(true_aligned, z_aligned)

    complementarity = {
        "jaccard_pca_zscore": jaccard,
        "pca_recall": pca_result["recall"],
        "zscore_recall": z_result["recall"],
        "fusion_recall": fusion_result["recall"],
        "recall_gain_over_pca": fusion_result["recall"] - pca_result["recall"],
        "recall_gain_over_zscore": fusion_result["recall"] - z_result["recall"],
        "pca_precision": pca_result["precision"],
        "zscore_precision": z_result["precision"],
        "fusion_precision": fusion_result["precision"],
    }
    for k, v in complementarity.items():
        results_df[k] = v

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print("\n=== Burst result (results/core_result_burst.csv) ===")
    print(results_df[["detector", "recall", "detection_rate", "grid_flagged_frac", "lift"]].to_string(index=False))
    print("\n=== Complementarity: PCA vs z_score_threshold (burst) ===")
    for k, v in complementarity.items():
        print(f"  {k}: {v:.4f}")

    pca_lift = results_df.loc[results_df["detector"] == "count_vector_pca", "lift"].iloc[0]
    z_lift = results_df.loc[results_df["detector"] == "z_score_threshold", "lift"].iloc[0]
    pca_above = pca_lift > 1.2
    z_above = z_lift > 1.2
    print(f"\nPCA lift: {pca_lift:.3f} ({'above' if pca_above else 'at/near'} chance)  |  "
          f"z_score_threshold lift: {z_lift:.3f} ({'above' if z_above else 'at/near'} chance)")
    if z_above and not pca_above:
        print("Same blind-spot pattern as stall holds: timing above chance, content at/near chance.")
    elif z_above and pca_above:
        print("BOTH detectors show above-chance lift on bursts -- windowing-interaction evidence: "
              "fixed-time windows make bursts partially visible to the content detector too, unlike stalls.")
    elif pca_above and not z_above:
        print("REVERSED from stall: content shows above-chance lift, timing does not -- unexpected, report as-is.")
    else:
        print("Neither detector shows above-chance lift on bursts.")

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
