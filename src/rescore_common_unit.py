"""
Re-score all three baseline detectors under ONE common evaluation unit (see src/metrics.py's
"Common evaluation unit" docstring for the rationale) so their precision/recall/F1 numbers are
finally comparable, and Jaccard overlap / OR-fusion across detector families becomes meaningful.

This does NOT touch results/baseline_detection.csv (the original, native-granularity numbers,
frozen for comparison). It re-fits the exact same detectors, with the exact same chronological
split and thresholding, and evaluates on the shared grid instead of each detector's own native
unit. Writes results/baseline_detection_common_unit.csv and prints old vs new side by side.

Usage:
    python src/rescore_common_unit.py
    python src/rescore_common_unit.py --limit 200000   # fast smoke test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import pandas as pd

from src.detectors import windowing
from src.detectors.count_pca import CountPCADetector
from src.detectors.isolation_forest_counts import IsolationForestCountsDetector
from src.metrics import EVAL_WINDOW_SCHEME, EVAL_WINDOW_SIZE, assign_eval_grid, evaluate_common_unit
from src.run_baseline_detectors import (
    OUT_CSV as OLD_CSV,
    build_count_features,
    chronological_split,
    load,
    run_count_detector,
    run_timing_detector,
)
from src.timing_baseline import add_sequence_context

OUT_CSV = Path("results/baseline_detection_common_unit.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="Only use the first N rows (chronological), for a fast smoke test.")
    args = ap.parse_args()

    df = load(limit=args.limit)
    is_train, cutoff = chronological_split(df)
    df_train = df.loc[is_train].reset_index(drop=True)
    df_test = df.loc[~is_train].reset_index(drop=True)

    print(f"Chronological split at {cutoff} -- train: {is_train.sum():,} rows, test: {(~is_train).sum():,} rows")
    print(f"Common evaluation unit: {EVAL_WINDOW_SCHEME} grid, size={EVAL_WINDOW_SIZE} (per node)")

    # Fit + native-unit predictions, exactly as in run_baseline_detectors.py
    X_train, X_test, windows_train, windows_test, df_test_w, _ = build_count_features(df_train, df_test, "fixed_count", 20)
    _, row_pred_pca = run_count_detector("count_vector_pca", CountPCADetector(), X_train, X_test, windows_train, windows_test, df_test_w)
    _, row_pred_if = run_count_detector(
        "isolation_forest_counts", IsolationForestCountsDetector(), X_train, X_test, windows_train, windows_test, df_test_w
    )

    df_all = add_sequence_context(df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    is_train_all = df_all["timestamp"] <= cutoff
    _, row_pred_timing = run_timing_detector(df_all, is_train_all)

    # One shared grid over the test rows; every detector's row-level predictions get mapped onto it.
    df_test_eval = assign_eval_grid(df_test)
    row_true = df_test_eval["anomaly"]

    detectors = [
        ("count_vector_pca", row_pred_pca),
        ("isolation_forest_counts", row_pred_if),
        ("timing_inter_arrival_isolation_forest", row_pred_timing),
    ]

    results = []
    for name, row_pred_by_id in detectors:
        row_predicted = df_test_eval["row_id"].map(row_pred_by_id).fillna(False)
        result = evaluate_common_unit(df_test_eval, row_predicted, row_true)
        result["detector"] = name
        results.append(result)

    results_df = pd.DataFrame(results)
    front_cols = ["detector", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "n_eval_cells"]
    other_cols = [c for c in results_df.columns if c not in front_cols]
    results_df = results_df[front_cols + other_cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n=== NEW: common-unit results (results/baseline_detection_common_unit.csv) ===")
    print(results_df[front_cols].to_string(index=False))

    if OLD_CSV.exists():
        old = pd.read_csv(OLD_CSV)
        print(f"\n=== OLD: native-granularity results ({OLD_CSV}) ===")
        print(old[["detector", "granularity", "precision", "recall", "f1"]].to_string(index=False))

        old_map = {r["detector"]: r for r in old.to_dict("records")}
        new_map = {r["detector"]: r for r in results_df.to_dict("records")}
        comparison = pd.DataFrame(
            [
                {
                    "detector": name,
                    "old_granularity": old_map[name]["granularity"],
                    "old_precision": round(old_map[name]["precision"], 4),
                    "new_precision": round(new_map[name]["precision"], 4),
                    "old_recall": round(old_map[name]["recall"], 4),
                    "new_recall": round(new_map[name]["recall"], 4),
                    "old_f1": round(old_map[name]["f1"], 4),
                    "new_f1": round(new_map[name]["f1"], 4),
                }
                for name in new_map
            ]
        )
        print("\n=== side-by-side: old (mismatched units) vs new (common unit) ===")
        print(comparison.to_string(index=False))
    else:
        print(f"\n(no {OLD_CSV} found -- skipping old-vs-new comparison)")

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
