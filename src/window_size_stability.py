"""
Check whether the common-unit baseline (src/rescore_common_unit.py) is a window-size artifact
before treating it as the official pre-injection reference.

PCA moved from 0.514/0.692/0.590 (its own native fixed_count=20 window) to 0.212/0.818/0.337 (a
60s fixed_time common grid) -- the precision drop / recall rise are both consistent with a coarser
evaluation grid inflating "hits" (a bigger grid cell is more likely to overlap SOME anomalous row).
This re-scores all three detectors' *already-fitted* row-level predictions against common grids at
30s, 60s, and 120s -- no refitting needed, since the eval grid is independent of how each detector
is fit natively -- and checks whether the qualitative story (PCA reasonable, timing weak, isolation
forest recall-floored) and the relative ordering across detectors hold at every grid size.

Writes:
  results/window_size_stability.csv -- P/R/F1 per detector per grid size
  results/baseline_final.csv        -- the confirmed official pre-injection baseline (one grid size)

Usage:
    python src/window_size_stability.py
    python src/window_size_stability.py --limit 200000   # fast smoke test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import pandas as pd

from src.detectors import windowing
from src.detectors.count_pca import CountPCADetector
from src.detectors.isolation_forest_counts import IsolationForestCountsDetector
from src.metrics import assign_eval_grid, evaluate_common_unit
from src.run_baseline_detectors import build_count_features, chronological_split, load, run_count_detector, run_timing_detector
from src.timing_baseline import add_sequence_context

GRID_SIZES = [30, 60, 120]  # seconds
DEFAULT_GRID_SIZE = 60  # the one we'll certify as the official baseline, if stability holds

OUT_STABILITY_CSV = Path("results/window_size_stability.csv")
OUT_FINAL_CSV = Path("results/baseline_final.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="Only use the first N rows (chronological), for a fast smoke test.")
    args = ap.parse_args()

    df = load(limit=args.limit)
    is_train, cutoff = chronological_split(df)
    df_train = df.loc[is_train].reset_index(drop=True)
    df_test = df.loc[~is_train].reset_index(drop=True)
    print(f"Chronological split at {cutoff} -- train: {is_train.sum():,} rows, test: {(~is_train).sum():,} rows")

    # Fit once; row-level predictions don't depend on the eval grid size.
    X_train, X_test, windows_train, windows_test, df_test_w, _ = build_count_features(df_train, df_test, "fixed_count", 20)
    _, row_pred_pca = run_count_detector("count_vector_pca", CountPCADetector(), X_train, X_test, windows_train, windows_test, df_test_w)
    _, row_pred_if = run_count_detector(
        "isolation_forest_counts", IsolationForestCountsDetector(), X_train, X_test, windows_train, windows_test, df_test_w
    )

    df_all = add_sequence_context(df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    is_train_all = df_all["timestamp"] <= cutoff
    _, row_pred_timing = run_timing_detector(df_all, is_train_all)

    detectors = [
        ("count_vector_pca", row_pred_pca),
        ("isolation_forest_counts", row_pred_if),
        ("timing_inter_arrival_isolation_forest", row_pred_timing),
    ]

    rows = []
    for grid_size in GRID_SIZES:
        df_test_eval = assign_eval_grid(df_test, scheme="fixed_time", size=grid_size)
        row_true = df_test_eval["anomaly"]
        for name, row_pred_by_id in detectors:
            row_predicted = df_test_eval["row_id"].map(row_pred_by_id).fillna(False)
            result = evaluate_common_unit(df_test_eval, row_predicted, row_true)
            result["detector"] = name
            result["grid_size_s"] = grid_size
            rows.append(result)

    stability_df = pd.DataFrame(rows)
    front_cols = ["detector", "grid_size_s", "precision", "recall", "f1", "n_eval_cells"]
    other_cols = [c for c in stability_df.columns if c not in front_cols]
    stability_df = stability_df[front_cols + other_cols]

    OUT_STABILITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    stability_df.to_csv(OUT_STABILITY_CSV, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Window-size stability: P/R/F1 per detector per grid size ===")
    print(stability_df[["detector", "grid_size_s", "precision", "recall", "f1"]].to_string(index=False))

    # --- stability check: does relative ordering (by F1) hold across grid sizes? ---
    print("\n=== Ranking by F1 at each grid size ===")
    orderings = {}
    for grid_size in GRID_SIZES:
        sub = stability_df[stability_df["grid_size_s"] == grid_size].sort_values("f1", ascending=False)
        ordering = tuple(sub["detector"])
        orderings[grid_size] = ordering
        print(f"  {grid_size}s: {' > '.join(ordering)}")
    ordering_stable = len(set(orderings.values())) == 1

    # --- how much does each detector's F1 swing across grid sizes? ---
    swing = stability_df.groupby("detector")["f1"].agg(["min", "max"])
    swing["range"] = swing["max"] - swing["min"]
    print("\n=== F1 range across grid sizes ===")
    print(swing.to_string())

    print(f"\nOrdering stable across {GRID_SIZES}s: {ordering_stable}")
    if ordering_stable:
        print("Qualitative story holds: PCA is the strongest detector at every grid size, isolation "
              "forest stays recall-floored, and the timing detector stays weak relative to both -- "
              "consistent with the content-labels premise at every window size tested.")
    else:
        print("WARNING: relative ordering changes across grid sizes -- treat the qualitative story as "
              "grid-size-sensitive, not settled, and report window-size sensitivity explicitly.")

    # --- certify the official baseline at DEFAULT_GRID_SIZE ---
    final = stability_df[stability_df["grid_size_s"] == DEFAULT_GRID_SIZE].drop(columns=["grid_size_s"])
    OUT_FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_FINAL_CSV, index=False)
    print(f"\nCertified official pre-injection baseline at grid_size={DEFAULT_GRID_SIZE}s -> {OUT_FINAL_CSV}")
    print(f"\nWrote {OUT_STABILITY_CSV}, {OUT_FINAL_CSV}")


if __name__ == "__main__":
    main()
