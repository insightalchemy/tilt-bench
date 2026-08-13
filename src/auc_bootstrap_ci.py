"""
Bootstrap 95% confidence intervals for the existing n=100 AUC-ROC results in results/auc_metrics.csv.

Does NOT re-inject and does NOT change n (100 injections per dataset x fault_type, unchanged) --
reuses the exact same scoring pipeline as src/auc_metrics.py (imported, not reimplemented), so the
point estimates recomputed here are guaranteed consistent with results/auc_metrics.csv. This script
only ADDS confidence intervals around those already-published numbers; it never modifies
results/auc_metrics.csv, results/FINAL_results.csv, or results/FINDINGS.md.

## Bootstrap unit: the injection, not the grid cell

The experimental unit that was actually randomized is the injection (100 per dataset x fault_type,
each an independent seeded (node, start, end, intensity) draw -- see data/processed/injection_labels_
*.csv). A single injection can span multiple 60s eval grid cells (src.metrics.assign_eval_grid), and
those sibling cells are NOT independent draws -- they're one event's downstream propagation. Bootstrap
resampling at the grid-cell level would treat correlated cells from the same injection as independent
evidence and understate the CI width. So this script resamples the 100 injection IDs with replacement
(cluster bootstrap), not individual cells.

The negative (non-anomalous) grid cells -- ~2.9M of them, several orders of magnitude more than the
100 injections -- are the fixed background population, not the thing under study, and are held fixed
across bootstrap replicates. Only the positive side (which injections "happened" in this resample) is
randomized. This mirrors "if we reran this same 100-injection experiment with different luck, which
subset/multiplicity of injections would we have drawn" -- the negatives are the dataset, not the
experiment.

## Efficient AUC computation via Mann-Whitney

AUC-ROC equals P(score of random positive > score of random negative) [+ 0.5 * P(tie)]. Because the
negative pool is fixed across all 1000 replicates, for each positive cell we precompute ONCE how many
negatives it outranks (count_less) and ties (count_eq) via searchsorted against the sorted negative
score array. Each bootstrap replicate then only needs to reweight the ~150-350 positive cells by their
injection's resample multiplicity (0, 1, 2, ...) and take a weighted sum -- no re-sorting of the ~2.9M
negative array per replicate. This is exactly equivalent to computing sklearn.roc_auc_score on the
literal resampled (with-replacement, duplicated) positive set each time, just without materializing
the duplicates.

Writes:
  results/auc_bootstrap_ci.md -- paper-ready table: dataset, fault_type, detector, auc_roc,
                                  ci_lower, ci_upper, crosses_chance (95% CI straddles 0.5)

Usage:
    python src/auc_bootstrap_ci.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.auc_metrics import (
    DETECTOR_ORDER,
    bgl_scores,
    chronological_split_bgl,
    chronological_split_tb,
    load_bgl_clean,
    load_tb_clean,
    tb_scores,
)
from src.metrics import assign_eval_grid, rows_to_grid, rows_to_grid_max

N_BOOT = 1000
SEED = 0
EXISTING_CSV = Path("results/auc_metrics.csv")
OUT_MD = Path("results/auc_bootstrap_ci.md")


def build_cell_arrays(scores_by_detector, df_injected, grid_labels_path):
    """For each detector, returns (y_true, y_score, injection_id) aligned arrays at the shared 60s
    eval-grid-cell granularity -- the same cells auc_metrics.py scores AUC-ROC over -- plus, for
    every positive cell, which injection_id produced it (needed for the cluster bootstrap)."""
    grid_labels = pd.read_csv(grid_labels_path)
    cell_to_injection = {(r.node, r.window_idx): int(r.injection_id) for r in grid_labels.itertuples(index=False)}
    anomalous_cells = set(cell_to_injection.keys())
    n_injections = int(grid_labels["injection_id"].nunique())

    df_eval = assign_eval_grid(df_injected)
    row_true = pd.Series([k in anomalous_cells for k in df_eval["eval_window_key"]], index=df_eval["row_id"].to_numpy())
    row_true_aligned = df_eval["row_id"].map(row_true).fillna(False)
    true_grid = rows_to_grid(df_eval, row_true_aligned)

    per_detector = {}
    for detector_name, row_score_by_id in scores_by_detector.items():
        row_score_aligned = df_eval["row_id"].map(row_score_by_id).fillna(0.0)
        score_grid = rows_to_grid_max(df_eval, row_score_aligned)

        aligned = pd.concat([true_grid.rename("y_true"), score_grid.rename("score")], axis=1)
        assert aligned["y_true"].notna().all() and aligned["score"].notna().all(), "score/truth grids must share the same cell universe"

        y_true = aligned["y_true"].astype(bool).to_numpy()
        y_score = aligned["score"].to_numpy()
        injection_ids = np.array([cell_to_injection.get(k, -1) for k in aligned.index], dtype=np.int64)
        per_detector[detector_name] = (y_true, y_score, injection_ids)

    return per_detector, n_injections


def bootstrap_ci(y_true, y_score, injection_ids, n_injections, n_boot, rng):
    """Cluster bootstrap over injection IDs, negatives held fixed. Returns
    (point_auc, ci_lower, ci_upper, n_degenerate_replicates)."""
    pos_mask = y_true
    neg_scores = np.sort(y_score[~pos_mask])
    n_neg = neg_scores.size

    pos_scores = y_score[pos_mask]
    pos_injection_ids = injection_ids[pos_mask]
    assert (pos_injection_ids >= 0).all(), "every positive cell must be traceable to an injection_id"

    # Per positive cell: how many negatives it outranks / ties, vs. the FIXED negative pool.
    count_less = np.searchsorted(neg_scores, pos_scores, side="left")
    count_leq = np.searchsorted(neg_scores, pos_scores, side="right")
    base_contrib = count_less + 0.5 * (count_leq - count_less)  # per unit weight

    point_auc = float(roc_auc_score(y_true, y_score))

    aucs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        draw = rng.integers(0, n_injections, size=n_injections)
        multiplicity = np.bincount(draw, minlength=n_injections)
        weights = multiplicity[pos_injection_ids]
        w_sum = weights.sum()
        if w_sum == 0:
            continue  # degenerate resample: none of the 100 injections were drawn (astronomically rare)
        aucs[b] = np.sum(weights * base_contrib) / (w_sum * n_neg)

    n_nan = int(np.isnan(aucs).sum())
    lo, hi = np.nanpercentile(aucs, [2.5, 97.5])
    return point_auc, float(lo), float(hi), n_nan


def run_dataset(rows, rng, dataset, fault_type, scores_by_detector, df_injected, grid_labels_path):
    per_detector, n_inj = build_cell_arrays(scores_by_detector, df_injected, grid_labels_path)
    for detector_name in DETECTOR_ORDER:
        y_true, y_score, injection_ids = per_detector[detector_name]
        point_auc, lo, hi, n_nan = bootstrap_ci(y_true, y_score, injection_ids, n_inj, N_BOOT, rng)
        crosses = lo <= 0.5 <= hi
        rows.append(
            {
                "dataset": dataset,
                "fault_type": fault_type,
                "detector": detector_name,
                "auc_roc": point_auc,
                "ci_lower": lo,
                "ci_upper": hi,
                "crosses_chance": crosses,
                "n_degenerate_boot": n_nan,
                "n_injections": n_inj,
            }
        )
        flag = "  <-- CROSSES 0.5" if crosses else ""
        print(
            f"  {dataset:11s} {fault_type:5s} {detector_name:25s} "
            f"auc_roc={point_auc:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]{flag}"
        )


def validate_against_existing(out_df):
    """Read-only sanity check against results/auc_metrics.csv -- never written to."""
    existing = pd.read_csv(EXISTING_CSV)
    merged = out_df.merge(existing[["dataset", "fault_type", "detector", "auc_roc"]], on=["dataset", "fault_type", "detector"], suffixes=("", "_existing"))
    diff = (merged["auc_roc"] - merged["auc_roc_existing"]).abs()
    max_diff = float(diff.max())
    print(f"\nValidation vs results/auc_metrics.csv: max |point AUC diff| = {max_diff:.2e}")
    if max_diff > 1e-6:
        print("  WARNING: point estimates recomputed here diverge from the existing CSV -- investigate before trusting the CIs.")
    return max_diff


def write_markdown(out_df, max_diff):
    lines = []
    lines.append("# Bootstrap 95% CIs for AUC-ROC (n=100 injections per dataset x fault type)\n")
    lines.append(
        "Cluster bootstrap over the 100 injection events (1000 resamples, seed=0), negatives held "
        "fixed as the background population. Point estimates recomputed here match "
        f"`results/auc_metrics.csv` to within {max_diff:.1e} (read-only cross-check; that file was not "
        "modified). See `src/auc_bootstrap_ci.py` module docstring for the full method.\n"
    )
    lines.append("| Dataset | Fault | Detector | AUC-ROC | 95% CI lower | 95% CI upper | Crosses chance (0.5)? |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in out_df.iterrows():
        crosses_str = "**yes**" if r["crosses_chance"] else "no"
        lines.append(
            f"| {r['dataset']} | {r['fault_type']} | {r['detector']} | {r['auc_roc']:.4f} | "
            f"{r['ci_lower']:.4f} | {r['ci_upper']:.4f} | {crosses_str} |"
        )

    n_crossing = int(out_df["crosses_chance"].sum())
    lines.append("")
    if n_crossing:
        lines.append(
            f"**{n_crossing} of {len(out_df)} cells have a 95% CI that crosses 0.5** -- for these, "
            "\"above chance\" or \"below chance\" is not statistically defensible at n=100 injections "
            "and the paper's wording should be softened accordingly (e.g. \"not distinguishable from "
            "chance at this sample size\" rather than asserting a direction)."
        )
        lines.append("")
        lines.append("Cells crossing chance:")
        for _, r in out_df[out_df["crosses_chance"]].iterrows():
            lines.append(f"- {r['dataset']} / {r['fault_type']} / {r['detector']}: AUC-ROC={r['auc_roc']:.4f}, CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    else:
        lines.append("No cells have a 95% CI crossing 0.5 -- every detector x fault x dataset cell is statistically distinguishable from chance at n=100 injections.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_MD}")


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    print("=== BGL ===")
    df_bgl = load_bgl_clean(limit=None)
    is_train_bgl, _ = chronological_split_bgl(df_bgl)
    for fault_type in ["stall", "burst"]:
        scores_by_detector, df_injected, grid_labels_path = bgl_scores(fault_type, df_bgl, is_train_bgl)
        run_dataset(rows, rng, "BGL", fault_type, scores_by_detector, df_injected, grid_labels_path)

    print("\n=== Thunderbird ===")
    df_tb = load_tb_clean()
    is_train_tb, _ = chronological_split_tb(df_tb)
    for fault_type in ["stall", "burst"]:
        scores_by_detector, df_injected, grid_labels_path = tb_scores(fault_type, df_tb, is_train_tb)
        run_dataset(rows, rng, "Thunderbird", fault_type, scores_by_detector, df_injected, grid_labels_path)

    out_df = pd.DataFrame(rows)
    out_df["detector_order"] = out_df["detector"].map({d: i for i, d in enumerate(DETECTOR_ORDER)})
    out_df["fault_order"] = out_df["fault_type"].map({"stall": 0, "burst": 1})
    out_df["dataset_order"] = out_df["dataset"].map({"BGL": 0, "Thunderbird": 1})
    out_df = out_df.sort_values(["dataset_order", "fault_order", "detector_order"]).drop(columns=["detector_order", "fault_order", "dataset_order"])
    out_df = out_df.reset_index(drop=True)

    max_diff = validate_against_existing(out_df)

    pd.set_option("display.width", 200)
    print("\n=== Full bootstrap CI table ===")
    print(out_df.to_string(index=False))

    write_markdown(out_df, max_diff)


if __name__ == "__main__":
    main()
