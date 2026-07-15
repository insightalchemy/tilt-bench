from pathlib import Path

import pandas as pd

STALL_CSV = Path("results/core_result_stall_final.csv")
BURST_CSV = Path("results/core_result_burst_v2.csv")
COMPARISON_CSV = Path("results/timing_feature_comparison.csv")
AUC_CSV = Path("results/auc_metrics.csv")  # threshold-independent metrics, src/auc_metrics.py
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
  
    comparison = pd.read_csv(COMPARISON_CSV)
    stall_log_ratio = comparison[(comparison["fault_type"] == "stall") & (comparison["detector"] == "log_ratio_threshold")].copy()

    combined = pd.concat(
        [stall[["fault_type", "detector"] + COLS], stall_log_ratio[["fault_type", "detector"] + COLS], burst[["fault_type", "detector"] + COLS]],
        ignore_index=True,
    )

    dupes = combined[combined.duplicated(subset=["fault_type", "detector"], keep=False)]
    if not dupes.empty:
        for (ft, det), group in dupes.groupby(["fault_type", "detector"]):
            if group[COLS].nunique().max() > 1:
                raise ValueError(f"Inconsistent duplicate rows for {ft}/{det} across source files:\n{group}")
        combined = combined.drop_duplicates(subset=["fault_type", "detector"], keep="first")

    # Threshold-independent metrics (src/auc_metrics.py) -- BGL rows only, matched on
    # (fault_type, detector). Doesn't touch lift/detection_rate/etc above; purely additive columns.
    auc = pd.read_csv(AUC_CSV)
    auc_bgl = auc[auc["dataset"] == "BGL"][["fault_type", "detector", "auc_pr", "no_skill_baseline", "auc_pr_ratio", "auc_roc"]]
    combined = combined.merge(auc_bgl, on=["fault_type", "detector"], how="left", validate="one_to_one")
    if combined["auc_pr"].isna().any():
        missing = combined.loc[combined["auc_pr"].isna(), ["fault_type", "detector"]]
        raise ValueError(f"No AUC metrics found for:\n{missing}")

    combined["detector_label"] = combined["detector"].map(DETECTOR_LABELS)
    combined["detector_order"] = combined["detector"].map({d: i for i, d in enumerate(DETECTOR_ORDER)})
    combined["fault_order"] = combined["fault_type"].map({"stall": 0, "burst": 1})
    combined = combined.sort_values(["fault_order", "detector_order"]).drop(columns=["detector_order", "fault_order"])

    combined = combined[["fault_type", "detector", "detector_label"] + COLS + ["auc_pr", "no_skill_baseline", "auc_pr_ratio", "auc_roc"]]
    combined = combined.rename(columns={"grid_flagged_frac": "base_rate"})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print(combined.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
