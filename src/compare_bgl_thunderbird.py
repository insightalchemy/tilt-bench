"""
Pure consolidation, no new computation: combines the already-computed BGL results
(results/core_result_consolidated.csv) and Thunderbird results (results/thunderbird_core_result.csv)
into one side-by-side comparison table.

Writes:
  results/thunderbird_vs_bgl_comparison.csv
"""

from pathlib import Path

import pandas as pd

BGL_CSV = Path("results/core_result_consolidated.csv")
TB_CSV = Path("results/thunderbird_core_result.csv")
OUT_CSV = Path("results/thunderbird_vs_bgl_comparison.csv")

DETECTOR_LABEL = {"pca": "count_vector_pca", "zscore": "z_score_threshold", "logratio": "log_ratio_threshold"}


def main():
    bgl = pd.read_csv(BGL_CSV)
    tb = pd.read_csv(TB_CSV)

    rows = []
    for fault_type in ["stall", "burst"]:
        bgl_row = bgl[bgl["fault_type"] == fault_type].iloc[0]
        for short, detector in DETECTOR_LABEL.items():
            tb_row = tb[(tb["fault_type"] == fault_type) & (tb["detector"] == detector)].iloc[0]
            rows.append(
                {
                    "fault_type": fault_type,
                    "detector": detector,
                    "bgl_detection_rate": bgl_row[f"{short}_detection_rate"],
                    "bgl_lift": bgl_row[f"{short}_lift"],
                    "thunderbird_detection_rate": tb_row["detection_rate"],
                    "thunderbird_lift": tb_row["lift"],
                }
            )

    comparison = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print(comparison.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
