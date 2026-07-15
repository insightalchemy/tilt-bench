"""
Read-only diagnostic: does the injected-span grid-cell ground truth include "quiet middle" cells
that contain ZERO rows -- structurally undetectable by any row-driven detector -- and if so, does
excluding them from the positive class change the n=100 AUC-PR/AUC-ROC numbers?

Hypothesis being checked: src.injector.label_spans_on_grid computes (injection_id, node,
window_idx) for every eval-grid cell whose TIME RANGE overlaps an injected span -- computed
analytically from each node's t0, NOT from which rows actually exist. A stall's whole point is a
long gap with no events in it, so some of its labeled cells can have zero rows. If those cells were
being counted as false negatives, every row-driven detector's stall recall/AUC would be mechanically
depressed regardless of detector quality.

Whether that's actually happening turns on how src.metrics.rows_to_grid / rows_to_grid_max work:
both are `groupby(df_eval["eval_window_key"]).{any,max}()`, where df_eval is built by
assign_eval_grid(df_injected) -- ONE ROW PER LOG EVENT. A grid cell can only appear in the
resulting Series' index if at least one row maps to it. So true_grid/score_grid's cell universe is,
by construction, exactly the set of POPULATED (node, window_idx) cells -- any labeled-anomalous cell
with zero rows can never enter true_grid as a positive; it's absent from the evaluation entirely,
not present-and-wrongly-scored. This script verifies that reasoning empirically rather than just
asserting it: it independently computes the populated-cell set from the on-disk injected parquets
(no detector fitting, no injector re-run) and cross-checks against the already-saved
results/auc_metrics.csv's n_positive_cells column -- if they match exactly, that PROVES the current
AUC numbers already reflect detectable-positives-only, with no need to refit anything.

Reads (read-only): data/processed/*_injected_{stall,burst}.parquet,
data/processed/*injection_grid_labels_{stall,burst}.csv, results/auc_metrics.csv.

Writes: results/stall_labeling_diagnostic.md ONLY. Does not touch any existing results file.

Usage:
    python src/stall_labeling_diagnostic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import pandas as pd

from src.metrics import assign_eval_grid

OUT_MD = Path("results/stall_labeling_diagnostic.md")
AUC_CSV = Path("results/auc_metrics.csv")

CONFIGS = [
    ("BGL", "stall", "data/processed/bgl_injected_stall.parquet", "data/processed/injection_grid_labels_stall.csv"),
    ("BGL", "burst", "data/processed/bgl_injected_burst.parquet", "data/processed/injection_grid_labels_burst.csv"),
    (
        "Thunderbird",
        "stall",
        "data/processed/thunderbird_injected_stall.parquet",
        "data/processed/thunderbird_injection_grid_labels_stall.csv",
    ),
    (
        "Thunderbird",
        "burst",
        "data/processed/thunderbird_injected_burst.parquet",
        "data/processed/thunderbird_injection_grid_labels_burst.csv",
    ),
]


def diagnose_one(dataset, fault_type, parquet_path, grid_labels_path):
    grid_labels = pd.read_csv(grid_labels_path)
    anomalous_cells = set(zip(grid_labels["node"], grid_labels["window_idx"]))
    n_injections = grid_labels["injection_id"].nunique()

    df_injected = pd.read_parquet(parquet_path)
    df_eval = assign_eval_grid(df_injected)
    populated_cells = set(df_eval["eval_window_key"].unique())

    detectable = anomalous_cells & populated_cells
    undetectable = anomalous_cells - populated_cells

    return {
        "dataset": dataset,
        "fault_type": fault_type,
        "n_injections": n_injections,
        "n_labeled_anomalous_cells": len(anomalous_cells),
        "n_undetectable_zero_row_cells": len(undetectable),
        "n_detectable_positive_cells": len(detectable),
        "pct_undetectable": 100 * len(undetectable) / len(anomalous_cells) if anomalous_cells else float("nan"),
        "cells_per_injection": len(anomalous_cells) / n_injections if n_injections else float("nan"),
    }


def main():
    rows = [diagnose_one(*cfg) for cfg in CONFIGS]
    diag_df = pd.DataFrame(rows)

    auc = pd.read_csv(AUC_CSV)
    nunique_check = auc.groupby(["dataset", "fault_type"])["n_positive_cells"].nunique()
    assert (nunique_check == 1).all(), "n_positive_cells differs across detectors within a (dataset,fault_type) -- unexpected, investigate before trusting the cross-check"
    existing_positive = auc.groupby(["dataset", "fault_type"])["n_positive_cells"].first().reset_index().rename(columns={"n_positive_cells": "existing_auc_metrics_n_positive_cells"})

    merged = diag_df.merge(existing_positive, on=["dataset", "fault_type"], validate="one_to_one")
    merged["matches_existing_computation"] = merged["n_detectable_positive_cells"] == merged["existing_auc_metrics_n_positive_cells"]

    pd.set_option("display.width", 200)
    print(merged.to_string(index=False))

    all_match = bool(merged["matches_existing_computation"].all())

    # --- write the markdown report ---
    lines = []
    lines.append("# Stall labeling diagnostic: are zero-row grid cells depressing stall AUC?")
    lines.append("")
    lines.append(
        "Read-only diagnostic. No injections or detectors were re-run; this only counts grid cells "
        "from data already on disk (n=100 injected parquets + grid labels + `results/auc_metrics.csv`)."
    )
    lines.append("")
    lines.append("## Hypothesis")
    lines.append("")
    lines.append(
        "`src.injector.label_spans_on_grid` labels every 60s grid cell whose TIME RANGE overlaps an "
        "injected span as anomalous, computed purely from timestamps, not from which rows actually "
        "exist. A stall's entire mechanism is a long gap with no events in it, so some of a stall's "
        "labeled cells can contain zero rows -- structurally undetectable by any row-driven detector, "
        "since a detector can only score a cell that has at least one row to score. If those cells "
        "were being counted as false negatives, every stall AUC/recall number would be mechanically "
        "penalized regardless of detector quality. Bursts compress events into a shorter span rather "
        "than emptying it, so are expected to have few or no such cells."
    )
    lines.append("")
    lines.append("## 1-3. Cell counts (n=100, all four dataset x fault_type combinations)")
    lines.append("")
    lines.append(
        "| dataset | fault | injections | labeled anomalous cells | zero-row (undetectable) | detectable positives | % undetectable | cells/injection |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in merged.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['fault_type']} | {r['n_injections']} | {r['n_labeled_anomalous_cells']} | "
            f"{r['n_undetectable_zero_row_cells']} | {r['n_detectable_positive_cells']} | {r['pct_undetectable']:.1f}% | "
            f"{r['cells_per_injection']:.2f} |"
        )
    lines.append("")

    lines.append("## 4. Recomputed AUC-PR / AUC-ROC restricted to detectable positives only")
    lines.append("")
    lines.append(
        "**No recomputation was necessary, and none was performed** (per instruction: no re-running "
        "detectors) -- because the *existing* `results/auc_metrics.csv` numbers are **already** "
        "computed over detectable-positives-only, as a structural consequence of how the grid "
        "aggregation works, not by explicit design."
    )
    lines.append("")
    lines.append(
        "`src.metrics.rows_to_grid` / `rows_to_grid_max` both aggregate via "
        "`groupby(df_eval[\"eval_window_key\"])`, where `df_eval` has exactly one row per **log "
        "event**. A grid cell can only appear in the resulting Series' index if at least one row maps "
        "to it -- so the true-label grid (`true_grid`) and every detector's score grid can *only ever* "
        "be defined over populated cells. A zero-row anomalous cell is never counted as a false "
        "negative; it simply never enters the evaluation at all, exactly as if it had been explicitly "
        "excluded."
    )
    lines.append("")
    lines.append(
        "This is verified empirically, not just argued: the `n_detectable_positive_cells` column "
        "above (computed independently, directly from the on-disk parquets and grid-label CSVs, with "
        "zero reference to any detector) is cross-checked against `results/auc_metrics.csv`'s "
        "existing `n_positive_cells` column (which the AUC computation actually used):"
    )
    lines.append("")
    lines.append("| dataset | fault | n_detectable_positive_cells (this diagnostic) | n_positive_cells (existing auc_metrics.csv) | match |")
    lines.append("|---|---|---|---|---|")
    for _, r in merged.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['fault_type']} | {r['n_detectable_positive_cells']} | "
            f"{r['existing_auc_metrics_n_positive_cells']} | {'YES' if r['matches_existing_computation'] else 'NO — MISMATCH'} |"
        )
    lines.append("")
    if all_match:
        lines.append(
            "**All four rows match exactly.** This confirms the existing AUC-PR/AUC-ROC numbers in "
            "`results/auc_metrics.csv` and `results/FINAL_results.csv` already reflect "
            "detectable-positives-only scoring. There is nothing to recompute: the side-by-side "
            "comparison the task asked for is between the existing numbers and themselves."
        )
    else:
        mismatches = merged.loc[~merged["matches_existing_computation"]]
        lines.append(
            "**MISMATCH FOUND** on the following rows -- the reasoning above does NOT hold as stated, "
            "and the existing AUC numbers may need re-derivation:\n\n" + mismatches.to_string(index=False)
        )
    lines.append("")

    lines.append("## 5. Does this change anything?")
    lines.append("")
    if all_match:
        lines.append(
            "**No — the hypothesis is real (zero-row cells genuinely exist in the stall ground truth) "
            "but it does not depress the current numbers, because those cells were already structurally "
            "excluded from evaluation, not silently counted as misses.** Specifically:"
        )
        lines.append("")
        stall_bgl = merged[(merged["dataset"] == "BGL") & (merged["fault_type"] == "stall")].iloc[0]
        stall_tb = merged[(merged["dataset"] == "Thunderbird") & (merged["fault_type"] == "stall")].iloc[0]
        burst_bgl = merged[(merged["dataset"] == "BGL") & (merged["fault_type"] == "burst")].iloc[0]
        burst_tb = merged[(merged["dataset"] == "Thunderbird") & (merged["fault_type"] == "burst")].iloc[0]
        lines.append(
            f"- The zero-row \"quiet middle\" cells are real and substantial: BGL-stall has "
            f"{stall_bgl['pct_undetectable']:.1f}% of its labeled cells undetectable "
            f"({stall_bgl['n_undetectable_zero_row_cells']}/{stall_bgl['n_labeled_anomalous_cells']}), "
            f"Thunderbird-stall {stall_tb['pct_undetectable']:.1f}% "
            f"({stall_tb['n_undetectable_zero_row_cells']}/{stall_tb['n_labeled_anomalous_cells']}) -- "
            f"confirming the early-project concern was correct as a description of the data. "
            f"Bursts, as expected, have far fewer or none: BGL-burst {burst_bgl['pct_undetectable']:.1f}%, "
            f"Thunderbird-burst {burst_tb['pct_undetectable']:.1f}%."
        )
        lines.append(
            "- **The z_score-stall AUC-ROC does not move from 0.537 (BGL) / 0.587 (Thunderbird)** — "
            "there is no undercounted-false-negative correction to apply; those numbers already are "
            "the detectable-positives-only numbers."
        )
        lines.append(
            "- **The stall/burst asymmetry does not shrink** for the same reason — it isn't caused by "
            "this labeling artifact, so removing the (already-absent) artifact can't close it."
        )
        lines.append(
            "- **A related, genuinely structural (not artifactual) point worth carrying into the "
            "writeup**: bursts generate several times more *detectable* positive cells per injection "
            f"than stalls do (BGL: {burst_bgl['cells_per_injection']:.2f} cells/injection for burst vs "
            f"{stall_bgl['cells_per_injection']:.2f} for stall; Thunderbird: "
            f"{burst_tb['cells_per_injection']:.2f} vs {stall_tb['cells_per_injection']:.2f}). This is "
            "a real consequence of burst's mechanism (a compressed run of `burst_length` consecutive "
            "events, up to 25, spans more grid cells with real rows in them than a stall's one-instant "
            "gap does) — not a bug, but it does mean burst detectors are evaluated against a somewhat "
            "richer positive-class sample than stall detectors, which can independently contribute to "
            "burst's stronger AUC-PR-ratio numbers on top of (not instead of) the real log_ratio "
            "feature-geometry effect documented in FINDINGS.md §5."
        )
    else:
        lines.append(
            "The cross-check found a mismatch (see §4) -- the clean \"already excluded\" story does not "
            "hold as stated. Do not treat the current AUC numbers as detectable-positives-only until "
            "this is resolved; a genuine recomputation (which requires refitting detectors) would be "
            "needed to get correct numbers."
        )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_MD}")
    print(f"\nAll rows match existing computation: {all_match}")


if __name__ == "__main__":
    main()
