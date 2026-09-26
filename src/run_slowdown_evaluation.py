"""
Slowdown-fault evaluation on BGL or Spirit (faults from src/injector_slowdown.py):

  1. Fixed-count invariance (N = 20/50/100) for the three content detectors (count_vector_pca,
     isolation_forest_counts, LogAnomaly-style): clean vs injected window scores. This is the
     result reported in the paper (Table III, "slowdown" rows).
  2. A fixed-time 60 s placebo sweep (src/placebo_sweep.run_fault), or with --light, point AUC
     estimates only without the bootstrap (lower memory).

The fixed-count sizes skip the AUC/bootstrap pipeline, which only the fixed-time cell needs, to
keep memory manageable.

Writes (under results/slowdown/):
  slowdown_invariance_{dataset}.csv
  slowdown_placebo_{dataset}.csv / .md

Usage:
    python src/run_slowdown_evaluation.py --dataset bgl
    python src/run_slowdown_evaluation.py --dataset spirit --light
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.deeplog import load_by_node, load_clean_by_node
from src.detectors.windowing import build_vocabulary, build_windows
from src.loganomaly_invariance import LogAnomalyDetector, build_feature_matrix, build_template2vec
from src.placebo_sweep import ALL_DATASET_CONFIG, CONTENT_DETECTORS, TIMING_DETECTORS, run_fault, write_markdown
from src.windowing_sweep import (
    assign_fixed_count_window,
    build_fixed_count_windows_table,
    build_timing_scores,
    compare_window_scores,
    count_grid_reassignments,
    evaluate_cell,
    fit_count_detectors_on_scheme,
    label_rows_in_injected_span,
    load_pos,
    score_count_detectors,
    scores_by_window_key,
    verify_std_pos_equivalence,
)

FIXED_COUNT_SIZES = [20, 50, 100]
FIXED_TIME_SIZES = [60]
LOGANOMALY_SEED = 42
OUT_DIR = Path("results/slowdown")


def load_slowdown(dataset):
    path = Path(f"data/processed/{dataset}_injected_slowdown.parquet")
    df = pd.read_parquet(path)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def build_config(dataset):
    base = ALL_DATASET_CONFIG[dataset]
    config = dict(base)
    config["injected_loaders"] = {"slowdown": lambda: load_slowdown(dataset)}
    config["injected_paths"] = {"slowdown": Path(f"data/processed/{dataset}_injected_slowdown.parquet")}
    config["injection_labels_path"] = {"slowdown": Path(f"data/processed/{dataset}_injection_labels_slowdown.csv")}
    return config


def fixed_count_invariance(dataset, config):
    df_clean_std = load_clean_by_node(config["clean_path"])
    is_train_std, _ = config["split_fn"](df_clean_std)
    df_train_std = df_clean_std.loc[is_train_std.to_numpy()].reset_index(drop=True)
    vocabulary = build_vocabulary(df_train_std)
    embeddings = build_template2vec(vocabulary, seed=LOGANOMALY_SEED)
    del df_train_std
    gc.collect()

    df_clean_pos = load_pos(config["clean_path"])
    eq_clean = verify_std_pos_equivalence(df_clean_std, df_clean_pos, f"{dataset} clean")
    cutoff_ts = df_clean_std.loc[is_train_std.to_numpy(), "timestamp"].max()
    del df_clean_std, is_train_std
    gc.collect()
    is_train_pos = df_clean_pos["timestamp"] <= cutoff_ts
    df_train_pos = df_clean_pos.loc[is_train_pos.to_numpy()].reset_index(drop=True)

    df_injected_std = load_by_node(config["injected_loaders"]["slowdown"]())
    df_injected_pos = load_pos(config["injected_paths"]["slowdown"])
    eq_injected = verify_std_pos_equivalence(df_injected_std, df_injected_pos, f"{dataset} slowdown injected")
    del df_injected_std
    gc.collect()

    rows = []
    for size in FIXED_COUNT_SIZES:
        print(f"  fixed_count_{size}", flush=True)
        df_train_w = assign_fixed_count_window(df_train_pos, size)
        windows_train = build_fixed_count_windows_table(df_train_w)
        df_clean_w = assign_fixed_count_window(df_clean_pos, size)
        windows_clean = build_fixed_count_windows_table(df_clean_w)
        df_inj_w = assign_fixed_count_window(df_injected_pos, size)
        windows_inj = build_fixed_count_windows_table(df_inj_w)

        fitted = fit_count_detectors_on_scheme(df_train_w, windows_train, vocabulary)
        count_scores_clean = score_count_detectors(fitted, df_clean_w, windows_clean, vocabulary)
        count_scores_inj = score_count_detectors(fitted, df_inj_w, windows_inj, vocabulary)

        X_train = build_feature_matrix(df_train_w, windows_train, vocabulary, embeddings)
        train_normal_mask = ~windows_train["label"].to_numpy()
        loganomaly = LogAnomalyDetector(random_state=LOGANOMALY_SEED).fit(X_train[train_normal_mask])
        X_clean = build_feature_matrix(df_clean_w, windows_clean, vocabulary, embeddings)
        X_inj = build_feature_matrix(df_inj_w, windows_inj, vocabulary, embeddings)
        loganomaly_clean = loganomaly.score(X_clean)
        loganomaly_inj = loganomaly.score(X_inj)

        detector_scores = {
            "count_vector_pca": (count_scores_clean["count_vector_pca"][0], count_scores_inj["count_vector_pca"][0]),
            "isolation_forest_counts": (count_scores_clean["isolation_forest_counts"][0], count_scores_inj["isolation_forest_counts"][0]),
        }
        loganomaly_clean_series = pd.Series(loganomaly_clean, index=windows_clean["window_key"])
        loganomaly_inj_series = pd.Series(loganomaly_inj, index=windows_inj["window_key"])
        row_score_clean_ln = pd.Series(df_clean_w["window_key"].map(loganomaly_clean_series).to_numpy(), index=df_clean_w["row_id"].to_numpy())
        row_score_inj_ln = pd.Series(df_inj_w["window_key"].map(loganomaly_inj_series).to_numpy(), index=df_inj_w["row_id"].to_numpy())
        detector_scores["loganomaly"] = (row_score_clean_ln, row_score_inj_ln)

        for name, (score_clean, score_inj) in detector_scores.items():
            score_clean_by_key = scores_by_window_key(df_clean_w, score_clean)
            score_inj_by_key = scores_by_window_key(df_inj_w, score_inj)
            comparison = compare_window_scores(score_clean_by_key, score_inj_by_key)
            n_grid_reassign, n_common_rows = count_grid_reassignments(df_clean_w, df_inj_w)
            rows.append(
                {
                    "dataset": dataset,
                    "fault_type": "slowdown",
                    "scheme": f"fixed_count_{size}",
                    "detector": name,
                    **comparison,
                    "n_grid_cell_reassignments": n_grid_reassign,
                    "n_common_rows_for_grid_check": n_common_rows,
                    "std_pos_equivalence_clean": eq_clean,
                    "std_pos_equivalence_injected": eq_injected,
                }
            )

        del df_train_w, windows_train, df_clean_w, windows_clean, df_inj_w, windows_inj, fitted
        del count_scores_clean, count_scores_inj, X_train, X_clean, X_inj, loganomaly
        gc.collect()

    del df_clean_pos, df_train_pos, df_injected_pos
    gc.collect()
    return rows


def fixed_time_60_point_estimate(dataset, config):
    df_clean_std = load_clean_by_node(config["clean_path"])
    is_train_std, _ = config["split_fn"](df_clean_std)
    df_train_std = df_clean_std.loc[is_train_std.to_numpy()].reset_index(drop=True)
    vocabulary = build_vocabulary(df_train_std)
    embeddings = build_template2vec(vocabulary, seed=LOGANOMALY_SEED)

    drop_cols = [c for c in ("raw_message", "label") if c in df_clean_std.columns]
    df_clean_std = df_clean_std.drop(columns=drop_cols)
    df_train_std = df_train_std.drop(columns=[c for c in drop_cols if c in df_train_std.columns])
    gc.collect()

    df_injected_std = load_by_node(config["injected_loaders"]["slowdown"]())
    df_injected_std = df_injected_std.drop(columns=[c for c in drop_cols if c in df_injected_std.columns])
    gc.collect()
    labels_df = pd.read_csv(config["injection_labels_path"]["slowdown"], parse_dates=["start", "end"])
    row_true_std = label_rows_in_injected_span(df_injected_std, labels_df)
    row_true_by_id = pd.Series(row_true_std.to_numpy(), index=df_injected_std["row_id"].to_numpy())

    timing_scores_injected = build_timing_scores(dataset, df_clean_std, is_train_std, df_injected_std)
    timing_scores_placebo = build_timing_scores(dataset, df_clean_std, is_train_std, df_clean_std)

    df_train_w, windows_train = build_windows(df_train_std, scheme="fixed_time", size=60)
    df_clean_w, windows_clean = build_windows(df_clean_std, scheme="fixed_time", size=60)
    df_clean_w["eval_window_key"] = df_clean_w["window_key"]
    df_inj_w, windows_inj = build_windows(df_injected_std, scheme="fixed_time", size=60)
    df_inj_w["eval_window_key"] = df_inj_w["window_key"]
    del df_clean_std, df_train_std, is_train_std
    gc.collect()

    fitted = fit_count_detectors_on_scheme(df_train_w, windows_train, vocabulary)
    count_scores_clean = score_count_detectors(fitted, df_clean_w, windows_clean, vocabulary)
    count_scores_inj = score_count_detectors(fitted, df_inj_w, windows_inj, vocabulary)
    del fitted
    gc.collect()

    X_train = build_feature_matrix(df_train_w, windows_train, vocabulary, embeddings)
    train_normal_mask = ~windows_train["label"].to_numpy()
    loganomaly = LogAnomalyDetector(random_state=LOGANOMALY_SEED).fit(X_train[train_normal_mask])
    threshold = np.percentile(loganomaly.score(X_train[train_normal_mask]), 95)
    del X_train
    gc.collect()

    def score_loganomaly(df_w, windows):
        X = build_feature_matrix(df_w, windows, vocabulary, embeddings)
        scores = loganomaly.score(X)
        window_score = pd.Series(scores, index=windows["window_key"])
        window_flag = window_score > threshold
        row_score = pd.Series(df_w["window_key"].map(window_score).to_numpy(), index=df_w["row_id"].to_numpy())
        row_flag = pd.Series(df_w["window_key"].map(window_flag).to_numpy(), index=df_w["row_id"].to_numpy())
        return row_score, row_flag

    detector_scores = {
        "count_vector_pca": {"clean": count_scores_clean["count_vector_pca"], "injected": count_scores_inj["count_vector_pca"]},
        "isolation_forest_counts": {"clean": count_scores_clean["isolation_forest_counts"], "injected": count_scores_inj["isolation_forest_counts"]},
        "loganomaly": {"clean": score_loganomaly(df_clean_w, windows_clean), "injected": score_loganomaly(df_inj_w, windows_inj)},
    }
    for name in TIMING_DETECTORS:
        detector_scores[name] = {"clean": timing_scores_placebo[name], "injected": timing_scores_injected[name]}

    rows = []
    for name in CONTENT_DETECTORS + TIMING_DETECTORS:
        score_clean, flag_clean = detector_scores[name]["clean"]
        score_inj, flag_inj = detector_scores[name]["injected"]
        metrics_injected = evaluate_cell(df_inj_w, score_inj, flag_inj, row_true_by_id)
        metrics_placebo = evaluate_cell(df_clean_w, score_clean, flag_clean, row_true_by_id)
        rows.append(
            {
                "dataset": dataset,
                "fault_type": "slowdown",
                "scheme": "fixed_time_60",
                "detector": name,
                "auc_injected": metrics_injected["auc_roc"],
                "auc_placebo": metrics_placebo["auc_roc"],
                "delta": metrics_injected["auc_roc"] - metrics_placebo["auc_roc"],
                "recall_injected": metrics_injected["recall"],
                "precision_injected": metrics_injected["precision"],
                "note": "point estimate only, no bootstrap CI (--light)",
            }
        )
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["bgl", "spirit"], required=True)
    ap.add_argument("--light", action="store_true", help="Skip the bootstrap CI for fixed-time-60 and compute point AUC estimates only (lower memory footprint).")
    return ap.parse_args()


def main():
    args = parse_args()
    config = build_config(args.dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.dataset} / slowdown: fixed-count invariance ===", flush=True)
    invariance_rows = fixed_count_invariance(args.dataset, config)
    invariance_df = pd.DataFrame(invariance_rows)
    invariance_csv = OUT_DIR / f"slowdown_invariance_{args.dataset}.csv"
    invariance_df.to_csv(invariance_csv, index=False)
    print(invariance_df.to_string(index=False))
    print(f"Wrote {invariance_csv}")

    print(f"=== {args.dataset} / slowdown: fixed-time-60 placebo sweep (light={args.light}) ===", flush=True)
    placebo_csv = OUT_DIR / f"slowdown_placebo_{args.dataset}.csv"
    placebo_md = OUT_DIR / f"slowdown_placebo_{args.dataset}.md"

    if args.light:
        placebo_rows = fixed_time_60_point_estimate(args.dataset, config)
        placebo_df = pd.DataFrame(placebo_rows)
        placebo_df.to_csv(placebo_csv, index=False)
        print(placebo_df.to_string(index=False))
        print(f"Wrote {placebo_csv}")
        lines = [f"# Slowdown fixed-time-60 placebo sweep (point estimates only, no bootstrap CI) -- {args.dataset}", "", placebo_df.to_markdown(index=False)]
        placebo_md.parent.mkdir(parents=True, exist_ok=True)
        placebo_md.write_text("\n".join(lines) + "\n")
        print(f"Wrote {placebo_md}")
        return

    placebo_rows = run_fault(args.dataset, config, "slowdown", FIXED_TIME_SIZES, [], None, None, 0, None)
    placebo_df = pd.DataFrame(placebo_rows)
    placebo_df.to_csv(placebo_csv, index=False)
    print(placebo_df.to_string(index=False))
    print(f"Wrote {placebo_csv}")

    class Args:
        dataset = args.dataset
        fixed_time_sizes = FIXED_TIME_SIZES
        fixed_count_sizes = []
        subsample = None
        subsample_include_injected = None
        seed = 0
        out_md = placebo_md

    write_markdown(placebo_df, placebo_md, Args())
    print(f"Wrote {placebo_md}")


if __name__ == "__main__":
    main()
