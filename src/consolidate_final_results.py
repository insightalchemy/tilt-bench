"""
Consolidation only -- no new computation. Reads the already-computed, already-verified result
files from the stall/burst detection experiments and combines them into one master table.

Sources (all pre-existing, unmodified):
  results/core_result_stall_final.csv   -- PCA, z_score_threshold, isolation_forest_counts (stall)
  results/core_result_burst_v2.csv      -- PCA, isolation_forest_counts, z_score_threshold,
                                            log_ratio_threshold (burst)
  results/timing_feature_comparison.csv -- log_ratio_threshold (stall) + z_score_threshold (both,
                                            cross-checked against the two files above)

Writes:
  results/FINAL_results.csv -- the paper's main results table: one row per (fault_type, detector)
"""

from pathlib import Path

import pandas as pd

STALL_CSV = Path("results/core_result_stall_final.csv")
BURST_CSV = Path("results/core_result_burst_v2.csv")
COMPARISON_CSV = Path("results/timing_feature_comparison.csv")
OUT_CSV = Path("results/FINAL_results.csv")

DETECTOR_ORDER = ["count_vector_pca", "z_score_threshold", "log_ratio_threshold", "isolation_forest_counts"]
DETECTOR_LABELS = {
    "count_vector_pca": "PCA (content)",
    "z_score_threshold": "z_score_threshold (timing, additive)",
    "log_ratio_threshold": "log_ratio_threshold (timing, symmetric)",
    "isolation_forest_counts": "isolation_forest_counts (dead baseline)",
}
COLS = ["recall", "detection_rate", "grid_flagged_frac", "lift", "n_injections_detected", "n_injections_total"]


def main():
    stall = pd.read_csv(STALL_CSV)
    stall["fault_type"] = "stall"

    burst = pd.read_csv(BURST_CSV)
    burst["fault_type"] = "burst"

    # log_ratio_threshold never appeared in core_result_stall_final.csv (it was added in a later
    # pass) -- pull its stall row from timing_feature_comparison.csv instead.
    comparison = pd.read_csv(COMPARISON_CSV)
    stall_log_ratio = comparison[(comparison["fault_type"] == "stall") & (comparison["detector"] == "log_ratio_threshold")].copy()
    stall_log_ratio["n_injections_total"] = 15

    combined = pd.concat(
        [stall[["fault_type", "detector"] + COLS], stall_log_ratio[["fault_type", "detector"] + COLS], burst[["fault_type", "detector"] + COLS]],
        ignore_index=True,
    )

    # Cross-check: z_score_threshold and PCA/isolation_forest_counts rows should agree byte-for-byte
    # across whichever source files they appear in more than once (a consolidation sanity check,
    # not new computation).
    dupes = combined[combined.duplicated(subset=["fault_type", "detector"], keep=False)]
    if not dupes.empty:
        for (ft, det), group in dupes.groupby(["fault_type", "detector"]):
            if group[COLS].nunique().max() > 1:
                raise ValueError(f"Inconsistent duplicate rows for {ft}/{det} across source files:\n{group}")
        combined = combined.drop_duplicates(subset=["fault_type", "detector"], keep="first")

    combined["detector_label"] = combined["detector"].map(DETECTOR_LABELS)
    combined["detector_order"] = combined["detector"].map({d: i for i, d in enumerate(DETECTOR_ORDER)})
    combined["fault_order"] = combined["fault_type"].map({"stall": 0, "burst": 1})
    combined = combined.sort_values(["fault_order", "detector_order"]).drop(columns=["detector_order", "fault_order"])

    combined = combined[["fault_type", "detector", "detector_label"] + COLS]
    combined = combined.rename(columns={"grid_flagged_frac": "base_rate"})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print(combined.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
