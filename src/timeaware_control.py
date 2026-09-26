"""
Positive control for fixed-count windowing (paper Section V-E): an isolation forest whose input
adds timestamp-derived features to the content features.

Feature vector per fixed-count window = the template count vector
(src.detectors.windowing.count_matrix) concatenated with four per-window inter-arrival statistics:
mean, max, min, and std of the log inter-arrival gap. Window membership stays position-based as
for every other fixed-count detector; only the feature values now depend on timestamps. Scored with
IsolationForestCountsDetector.

Proposition 1 does not cover this detector, so max_abs_score_diff between clean and injected data
is expected to be nonzero. The placebo-corrected delta = AUC_injected - AUC_placebo, with the
paired cluster bootstrap of src/placebo_sweep.py, measures fault-attributable detection.

Writes:
  results/placebo/timeaware_control_{dataset}.csv / .md
  results/checkpoints/timeaware_{dataset}_{fault}_checkpoint.csv

Usage:
    python src/timeaware_control.py --dataset bgl
    python src/timeaware_control.py --dataset thunderbird
    # quick end-to-end check on a subsample:
    python src/timeaware_control.py --dataset bgl --window-sizes 20 50 \
        --subsample 50000 --subsample-include-injected 5
"""

import argparse
import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.deeplog import load_by_node, load_clean_by_node, select_injected_nodes
from src.detectors.isolation_forest_counts import IsolationForestCountsDetector
from src.detectors.windowing import build_vocabulary, count_matrix
from src.placebo_sweep import build_auc_arrays, make_run_config, paired_delta_bootstrap
from src.run_baseline_detectors import THRESHOLD_PERCENTILE
from src.timing_baseline import MAD_FLOOR_SEC, add_sequence_context
from src.windowing_sweep import (
    DATASET_CONFIG,
    assign_fixed_count_window,
    build_fixed_count_windows_table,
    compare_window_scores,
    count_grid_reassignments,
    evaluate_cell,
    label_rows_in_injected_span,
    load_pos,
    scores_by_window_key,
    verify_std_pos_equivalence,
)

FAULTS = ["stall", "burst"]
WINDOW_SIZES_DEFAULT = [20, 50, 100]
N_BOOT = 1000
BOOT_SEED = 0

OUT_DIR = Path("results/placebo")
CHECKPOINT_DIR = Path("results/checkpoints")


def add_log_gap(df_pos):
    df = add_sequence_context(df_pos.copy())
    gap = df["gap_prev_s"].to_numpy(dtype=float)
    log_gap = np.log(np.clip(gap, MAD_FLOOR_SEC, None))
    log_gap = np.where(np.isnan(gap), 0.0, log_gap)
    df["log_gap"] = log_gap
    return df


def gap_stat_matrix(df_windowed, windows):
    grouped = df_windowed.groupby("window_key")["log_gap"]
    stats = grouped.agg(["mean", "max", "min", "std"]).fillna(0.0)
    stats = stats.reindex(windows["window_key"]).fillna(0.0)
    return stats.to_numpy()


def build_feature_matrix(df_windowed, windows, vocabulary):
    counts = count_matrix(df_windowed, windows, vocabulary)
    gaps = gap_stat_matrix(df_windowed, windows)
    return np.concatenate([counts, gaps], axis=1)


def run_fault(dataset, config, fault, window_sizes, subsample, subsample_include_injected, seed, tmp_dir):
    print(f"=== {dataset} / {fault} (time-aware) ===", flush=True)
    t0 = time.time()

    must_include_nodes = None
    if subsample_include_injected is not None:
        must_include_nodes = set()
        for f in FAULTS:
            must_include_nodes |= select_injected_nodes(config["injected_loaders"][f], subsample_include_injected, seed)

    run_config = make_run_config(config, subsample, must_include_nodes, tmp_dir)

    df_clean_std = load_clean_by_node(run_config["clean_path"])
    is_train_std, _ = config["split_fn"](df_clean_std)
    df_train_std = df_clean_std.loc[is_train_std.to_numpy()].reset_index(drop=True)
    vocabulary = build_vocabulary(df_train_std)

    df_clean_pos = add_log_gap(load_pos(run_config["clean_path"]))
    eq_clean = verify_std_pos_equivalence(df_clean_std, df_clean_pos, f"{dataset} clean")
    is_train_pos = df_clean_pos["timestamp"] <= df_clean_std.loc[is_train_std.to_numpy(), "timestamp"].max()
    df_train_pos = df_clean_pos.loc[is_train_pos.to_numpy()].reset_index(drop=True)

    df_injected_std = load_by_node(run_config["injected_loaders"][fault]())
    df_injected_pos = add_log_gap(load_pos(run_config["injected_paths"][fault]))
    eq_injected = verify_std_pos_equivalence(df_injected_std, df_injected_pos, f"{dataset} {fault} injected")
    if not (eq_clean and eq_injected):
        print("CRITICAL: std/pos row-order equivalence failed.", file=sys.stderr)
        sys.exit(1)

    labels_df = pd.read_csv(config["injection_labels_path"][fault], parse_dates=["start", "end"])
    node_to_injection = labels_df.set_index("node")["injection_id"]
    n_injections = len(node_to_injection)

    row_true_std = label_rows_in_injected_span(df_injected_std, labels_df)
    row_true_by_id = pd.Series(row_true_std.to_numpy(), index=df_injected_std["row_id"].to_numpy())

    rng = np.random.default_rng(seed)
    rows = []
    for size in window_sizes:
        print(f"  N={size}, elapsed: {time.time() - t0:.0f}s", flush=True)
        df_train_w = assign_fixed_count_window(df_train_pos, size)
        windows_train = build_fixed_count_windows_table(df_train_w)
        df_clean_w = assign_fixed_count_window(df_clean_pos, size)
        windows_clean = build_fixed_count_windows_table(df_clean_w)
        df_inj_w = assign_fixed_count_window(df_injected_pos, size)
        windows_inj = build_fixed_count_windows_table(df_inj_w)

        X_train = build_feature_matrix(df_train_w, windows_train, vocabulary)
        train_normal_mask = ~windows_train["label"].to_numpy()
        detector = IsolationForestCountsDetector(random_state=0)
        detector.fit(X_train[train_normal_mask])
        threshold = np.percentile(detector.score(X_train[train_normal_mask]), THRESHOLD_PERCENTILE)

        X_clean = build_feature_matrix(df_clean_w, windows_clean, vocabulary)
        scores_clean = detector.score(X_clean)
        window_score_clean = pd.Series(scores_clean, index=windows_clean["window_key"])
        window_flag_clean = window_score_clean > threshold
        row_score_clean = pd.Series(df_clean_w["window_key"].map(window_score_clean).to_numpy(), index=df_clean_w["row_id"].to_numpy())
        row_flag_clean = pd.Series(df_clean_w["window_key"].map(window_flag_clean).to_numpy(), index=df_clean_w["row_id"].to_numpy())

        X_inj = build_feature_matrix(df_inj_w, windows_inj, vocabulary)
        scores_inj = detector.score(X_inj)
        window_score_inj = pd.Series(scores_inj, index=windows_inj["window_key"])
        window_flag_inj = window_score_inj > threshold
        row_score_inj = pd.Series(df_inj_w["window_key"].map(window_score_inj).to_numpy(), index=df_inj_w["row_id"].to_numpy())
        row_flag_inj = pd.Series(df_inj_w["window_key"].map(window_flag_inj).to_numpy(), index=df_inj_w["row_id"].to_numpy())

        metrics_injected = evaluate_cell(df_inj_w, row_score_inj, row_flag_inj, row_true_by_id)
        metrics_placebo = evaluate_cell(df_clean_w, row_score_clean, row_flag_clean, row_true_by_id)

        y_true_inj, y_score_inj_arr, ids_inj = build_auc_arrays(df_inj_w, row_score_inj, row_true_by_id, node_to_injection)
        y_true_pla, y_score_pla_arr, ids_pla = build_auc_arrays(df_clean_w, row_score_clean, row_true_by_id, node_to_injection)
        point_delta, ci_lo, ci_hi, auc_inj_check, auc_pla_check, n_degenerate = paired_delta_bootstrap(
            y_true_inj, y_score_inj_arr, ids_inj, y_true_pla, y_score_pla_arr, ids_pla, n_injections, N_BOOT, rng
        )

        score_clean_by_key = scores_by_window_key(df_clean_w, row_score_clean)
        score_inj_by_key = scores_by_window_key(df_inj_w, row_score_inj)
        invariance = compare_window_scores(score_clean_by_key, score_inj_by_key)
        n_grid_reassign, n_common_rows = count_grid_reassignments(df_clean_w, df_inj_w)

        rows.append(
            {
                "dataset": dataset,
                "fault_type": fault,
                "scheme": f"fixed_count_{size}",
                "scheme_size": size,
                "detector": "timeaware_isolation_forest",
                "auc_injected": metrics_injected["auc_roc"],
                "auc_placebo": metrics_placebo["auc_roc"],
                "delta": metrics_injected["auc_roc"] - metrics_placebo["auc_roc"],
                "delta_ci_lower": ci_lo,
                "delta_ci_upper": ci_hi,
                "n_bootstrap_degenerate": n_degenerate,
                "recall_injected": metrics_injected["recall"],
                "precision_injected": metrics_injected["precision"],
                "lift_injected": metrics_injected["lift"],
                **invariance,
                "n_grid_cell_reassignments": n_grid_reassign,
                "n_common_rows_for_grid_check": n_common_rows,
            }
        )
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(CHECKPOINT_DIR / f"timeaware_{dataset}_{fault}_checkpoint.csv", index=False)
        del df_train_w, df_clean_w, df_inj_w, X_train, X_clean, X_inj
        gc.collect()

    print(f"  {dataset}/{fault} done, elapsed: {time.time() - t0:.0f}s", flush=True)
    return rows


def write_markdown(df, path, args):
    lines = [
        "# Time-aware isolation forest -- positive control for fixed-count windowing",
        "",
        f"window_sizes={args.window_sizes}, subsample={args.subsample}, "
        f"subsample_include_injected={args.subsample_include_injected}, n_boot={N_BOOT}",
        "",
        df.to_markdown(index=False),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["bgl", "thunderbird"], required=True)
    ap.add_argument("--fault", choices=["stall", "burst"], default=None)
    ap.add_argument("--window-sizes", type=int, nargs="+", default=WINDOW_SIZES_DEFAULT)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--subsample-include-injected", type=int, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    config = DATASET_CONFIG[args.dataset]
    faults = [args.fault] if args.fault else FAULTS

    tmp_dir = Path(tempfile.mkdtemp(prefix="timeaware_control_")) if args.subsample is not None else None
    all_rows = []
    try:
        for fault in faults:
            all_rows.extend(
                run_fault(args.dataset, config, fault, args.window_sizes, args.subsample, args.subsample_include_injected, args.seed, tmp_dir)
            )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    df = pd.DataFrame(all_rows)
    out_csv = args.out_csv or OUT_DIR / f"timeaware_control_{args.dataset}.csv"
    out_md = args.out_md or OUT_DIR / f"timeaware_control_{args.dataset}.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out_csv}")
    write_markdown(df, out_md, args)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
