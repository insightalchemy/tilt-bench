"""
End-to-end fixed-time evaluation and injector checks for BGL and Thunderbird (stall and burst).

Runs, in one pass over the same data:
  1. AUC-PR / AUC-ROC of the four fixed-time detectors on the 60 s evaluation grid
     (src/auc_metrics.py),
  2. cluster-bootstrap 95% CIs on AUC-ROC over injection events (src/auc_bootstrap_ci.py),
  3. instrument validation: premise-audit signatures re-run on the injected rows
     (src/validate_injection.py),
  4. a whole-dataset ordering check on every injected parquet (zero violations expected).

Writes (under results/core_metrics/):
  auc_metrics.csv, auc_bootstrap_ci.csv, auc_bootstrap_ci.md,
  instrument_validation.csv, ordering_check.csv

Usage:
    python src/core_metrics.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.auc_bootstrap_ci import N_BOOT, SEED as BOOTSTRAP_SEED, build_cell_arrays, bootstrap_ci
from src.auc_metrics import (
    DETECTOR_ORDER,
    bgl_scores,
    chronological_split_bgl,
    chronological_split_tb,
    load_bgl_clean,
    load_tb_clean,
    score_all_detectors,
    tb_scores,
)
from src.pipeline_common import load_bgl_injected_burst, load_bgl_injected_stall, load_tb_injected
from src.validate_injection import check_ordering, instrument_validation

OUT_DIR = Path("results/core_metrics")

INJECTED_SOURCES = {
    "BGL": {"stall": load_bgl_injected_stall, "burst": load_bgl_injected_burst},
    "Thunderbird": {"stall": lambda: load_tb_injected("stall"), "burst": lambda: load_tb_injected("burst")},
}


def regenerate_auc_metrics():
    t0 = time.time()
    rows = []
    cached_scores = {}

    print("=== AUC metrics ===", flush=True)
    print("-- BGL --", flush=True)
    df_bgl = load_bgl_clean(limit=None)
    is_train_bgl, _ = chronological_split_bgl(df_bgl)
    for fault_type in ["stall", "burst"]:
        scores_by_detector, df_injected, grid_labels_path = bgl_scores(fault_type, df_bgl, is_train_bgl)
        cached_scores[("BGL", fault_type)] = (scores_by_detector, df_injected, grid_labels_path)
        rows.extend(score_all_detectors("BGL", fault_type, scores_by_detector, df_injected, grid_labels_path))
    print(f"  BGL done, elapsed {time.time() - t0:.0f}s", flush=True)

    print("-- Thunderbird --", flush=True)
    df_tb = load_tb_clean()
    is_train_tb, _ = chronological_split_tb(df_tb)
    for fault_type in ["stall", "burst"]:
        scores_by_detector, df_injected, grid_labels_path = tb_scores(fault_type, df_tb, is_train_tb)
        cached_scores[("Thunderbird", fault_type)] = (scores_by_detector, df_injected, grid_labels_path)
        rows.extend(score_all_detectors("Thunderbird", fault_type, scores_by_detector, df_injected, grid_labels_path))
    print(f"  Thunderbird done, elapsed {time.time() - t0:.0f}s", flush=True)

    out_df = pd.DataFrame(rows)
    out_df["detector_order"] = out_df["detector"].map({d: i for i, d in enumerate(DETECTOR_ORDER)})
    out_df["fault_order"] = out_df["fault_type"].map({"stall": 0, "burst": 1})
    out_df["dataset_order"] = out_df["dataset"].map({"BGL": 0, "Thunderbird": 1})
    out_df = out_df.sort_values(["dataset_order", "fault_order", "detector_order"]).drop(columns=["detector_order", "fault_order", "dataset_order"])
    out_df = out_df[["dataset", "fault_type", "detector", "auc_pr", "no_skill_baseline", "auc_pr_ratio", "auc_roc", "n_eval_cells", "n_positive_cells"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "auc_metrics.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}\n", flush=True)
    return out_df, cached_scores


def regenerate_bootstrap_ci(cached_scores):
    t0 = time.time()
    print("=== Cluster bootstrap CI ===", flush=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for dataset in ["BGL", "Thunderbird"]:
        for fault_type in ["stall", "burst"]:
            scores_by_detector, df_injected, grid_labels_path = cached_scores[(dataset, fault_type)]
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
                print(f"  {dataset:11s} {fault_type:5s} {detector_name:25s} auc_roc={point_auc:.4f} 95% CI=[{lo:.4f}, {hi:.4f}]", flush=True)

    out_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "auc_bootstrap_ci.csv"
    out_df.to_csv(out_csv, index=False)

    lines = ["# Bootstrap 95% CIs for AUC-ROC (n=100 injections per dataset x fault type)", ""]
    lines.append("Generated by src/core_metrics.py.")
    lines.append("")
    lines.append("| Dataset | Fault | Detector | AUC-ROC | 95% CI lower | 95% CI upper | Crosses chance (0.5)? |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in out_df.iterrows():
        crosses_str = "**yes**" if r["crosses_chance"] else "no"
        lines.append(f"| {r['dataset']} | {r['fault_type']} | {r['detector']} | {r['auc_roc']:.4f} | {r['ci_lower']:.4f} | {r['ci_upper']:.4f} | {crosses_str} |")
    out_md = OUT_DIR / "auc_bootstrap_ci.md"
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote {out_csv}, {out_md}, elapsed {time.time() - t0:.0f}s\n", flush=True)
    return out_df


def regenerate_instrument_and_ordering():
    t0 = time.time()
    print("=== Instrument validation + whole-dataset ordering check ===", flush=True)
    instrument_rows = []
    ordering_rows = []
    for dataset, faults in INJECTED_SOURCES.items():
        for fault_type, loader in faults.items():
            df_injected = loader()
            instrument = instrument_validation(df_injected)
            instrument["dataset"] = dataset
            instrument["fault_type"] = fault_type
            instrument_rows.append(instrument)

            ordering = check_ordering(df_injected)
            ordering["dataset"] = dataset
            ordering["fault_type"] = fault_type
            ordering_rows.append(ordering)
            print(f"  {dataset} {fault_type}: n_rows={ordering['n_rows']:,} n_violations={ordering['n_violations']}", flush=True)

    instrument_df = pd.concat(instrument_rows, ignore_index=True)
    instrument_df = instrument_df[["dataset", "fault_type", "signature", "count_flagged", "pct_of_injections", "count_insufficient_context"]]
    ordering_df = pd.DataFrame(ordering_rows)[["dataset", "fault_type", "n_rows", "n_violations", "violation_node_count"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    instrument_csv = OUT_DIR / "instrument_validation.csv"
    ordering_csv = OUT_DIR / "ordering_check.csv"
    instrument_df.to_csv(instrument_csv, index=False)
    ordering_df.to_csv(ordering_csv, index=False)

    any_violation = bool((ordering_df["n_violations"] > 0).any())
    if any_violation:
        print(f"CRITICAL: ordering violations found -- see {ordering_csv}", file=sys.stderr)

    print(f"Wrote {instrument_csv}, {ordering_csv}, elapsed {time.time() - t0:.0f}s\n", flush=True)
    return instrument_df, ordering_df


def main():
    t0 = time.time()
    _, cached_scores = regenerate_auc_metrics()
    regenerate_bootstrap_ci(cached_scores)
    regenerate_instrument_and_ordering()
    print(f"=== Done, total elapsed {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
