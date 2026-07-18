"""
Windowing-scheme sweep: is timing-fault observability determined by the windowing scheme rather
than the detector family? Compares the project's standard per-node fixed-TIME grid (30s/60s/120s,
via src.detectors.windowing.build_windows / src.metrics) against a new per-node fixed-COUNT grid
(N=20/50/100 consecutive events, grouped strictly by ORIGINAL EVENT POSITION -- rows are ordered by
a stable sort on "node" alone, never re-sorted by timestamp, so fixed-count window membership is
constructed to be timestamp-independent rather than merely assumed to be). Reuses the existing
detector-fitting and AUC/lift scoring harness throughout (src.detectors.count_pca,
src.detectors.isolation_forest_counts, src.detectors.timing_detector, src.metrics.rows_to_grid /
rows_to_grid_max / evaluate_common_unit, src.auc_metrics.compute_grid_auc) -- only window
assignment and fixed-count ground-truth labeling are new. No re-injection; reads the existing n=100
injected parquets read-only.

Usage:
    python src/windowing_sweep.py
"""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.auc_metrics import compute_grid_auc
from src.core_result_burst import load_injected_burst as load_bgl_burst
from src.core_result_stall import build_timing_features, load_injected as load_bgl_stall
from src.core_result_thunderbird import (
    build_timing_features_tb,
    chronological_split as chronological_split_tb,
    load_injected as load_tb_injected,
)
from src.core_result_v2_symmetric import compute_clean_train_baselines
from src.deeplog import load_by_node, load_clean_by_node
from src.detectors.count_pca import CountPCADetector
from src.detectors.isolation_forest_counts import IsolationForestCountsDetector
from src.detectors.timing_detector import LogRatioThresholdDetector, ZScoreThresholdDetector, add_log_ratio_feature
from src.detectors.windowing import build_vocabulary, build_windows, count_matrix
from src.metrics import evaluate_common_unit
from src.run_baseline_detectors import THRESHOLD_PERCENTILE, chronological_split as chronological_split_bgl

OUT_CSV = Path("results/windowing_sweep.csv")
OUT_NOTES = Path("results/windowing_sweep_notes.md")
CHECKPOINT_DIR = Path("/Users/ishannagpal/.claude/jobs/68e2b0e9/tmp")

FIXED_TIME_SIZES = [30, 60, 120]
FIXED_COUNT_SIZES = [20, 50, 100]

DATASET_CONFIG = {
    "bgl": {
        "clean_path": Path("data/processed/bgl_parsed.parquet"),
        "split_fn": chronological_split_bgl,
        "injected_loaders": {"stall": load_bgl_stall, "burst": load_bgl_burst},
        "injected_paths": {
            "stall": Path("data/processed/bgl_injected_stall.parquet"),
            "burst": Path("data/processed/bgl_injected_burst.parquet"),
        },
        "injection_labels_path": {
            "stall": Path("data/processed/injection_labels_stall.csv"),
            "burst": Path("data/processed/injection_labels_burst.csv"),
        },
    },
    "thunderbird": {
        "clean_path": Path("data/processed/thunderbird_parsed.parquet"),
        "split_fn": chronological_split_tb,
        "injected_loaders": {"stall": lambda: load_tb_injected("stall"), "burst": lambda: load_tb_injected("burst")},
        "injected_paths": {
            "stall": Path("data/processed/thunderbird_injected_stall.parquet"),
            "burst": Path("data/processed/thunderbird_injected_burst.parquet"),
        },
        "injection_labels_path": {
            "stall": Path("data/processed/thunderbird_injection_labels_stall.csv"),
            "burst": Path("data/processed/thunderbird_injection_labels_burst.csv"),
        },
    },
}


def load_pos(path):
    df = pd.read_parquet(path)
    df = df.sort_values("node", kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def verify_std_pos_equivalence(df_std, df_pos, label):
    same_node = bool((df_std["node"].to_numpy() == df_pos["node"].to_numpy()).all())
    same_timestamp = bool((df_std["timestamp"].to_numpy() == df_pos["timestamp"].to_numpy()).all())
    same_template = bool((df_std["event_template"].to_numpy() == df_pos["event_template"].to_numpy()).all())
    result = same_node and same_timestamp and same_template
    print(f"std/pos row-order equivalence [{label}]: {result} (node={same_node}, timestamp={same_timestamp}, template={same_template})")
    return result


def assign_fixed_count_window(df, n):
    df = df.copy()
    window_idx = df.groupby("node", sort=False).cumcount() // n
    key = list(zip(df["node"], window_idx))
    df["window_key"] = key
    df["eval_window_key"] = key
    return df


def build_fixed_count_windows_table(df_with_window_key):
    return (
        df_with_window_key.groupby("window_key", sort=False)
        .agg(node=("node", "first"), n_events=("node", "size"), label=("anomaly", "any"))
        .reset_index()
    )


def label_rows_in_injected_span(df, labels_df):
    starts = labels_df.set_index("node")["start"]
    ends = labels_df.set_index("node")["end"]
    row_start = df["node"].map(starts)
    row_end = df["node"].map(ends)
    in_span = (df["timestamp"] >= row_start) & (df["timestamp"] <= row_end)
    return in_span.fillna(False)


def build_scheme_windows(scheme_kind, scheme_size, df_std, df_pos):
    if scheme_kind == "fixed_time":
        df_windowed, windows = build_windows(df_std, scheme="fixed_time", size=scheme_size)
        df_windowed["eval_window_key"] = df_windowed["window_key"]
        return df_windowed, windows
    df_windowed = assign_fixed_count_window(df_pos, scheme_size)
    windows = build_fixed_count_windows_table(df_windowed)
    return df_windowed, windows


def fit_count_detectors_on_scheme(df_train_windowed, windows_train, vocabulary):
    X_train = count_matrix(df_train_windowed, windows_train, vocabulary)
    train_normal_mask = ~windows_train["label"].to_numpy()
    X_train_normal = X_train[train_normal_mask]
    fitted = {}
    for name, detector in [("count_vector_pca", CountPCADetector()), ("isolation_forest_counts", IsolationForestCountsDetector())]:
        detector.fit(X_train_normal)
        train_scores = detector.score(X_train_normal)
        threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)
        fitted[name] = (detector, threshold)
    return fitted


def score_count_detectors(fitted, df_all_windowed, windows_all, vocabulary):
    X_all = count_matrix(df_all_windowed, windows_all, vocabulary)
    results = {}
    for name, (detector, threshold) in fitted.items():
        all_scores = detector.score(X_all)
        window_score = pd.Series(all_scores, index=windows_all["window_key"])
        window_flag = pd.Series(all_scores > threshold, index=windows_all["window_key"])
        row_score = pd.Series(df_all_windowed["window_key"].map(window_score).to_numpy(), index=df_all_windowed["row_id"].to_numpy())
        row_flag = pd.Series(df_all_windowed["window_key"].map(window_flag).to_numpy(), index=df_all_windowed["row_id"].to_numpy())
        results[name] = (row_score, row_flag)
    return results


def scores_by_window_key(df_windowed, row_score_by_id):
    row_score_aligned = df_windowed["row_id"].map(row_score_by_id)
    return pd.Series(row_score_aligned.to_numpy(), index=df_windowed["window_key"].to_numpy()).groupby(level=0).first()


def compare_window_scores(scores_clean_by_key, scores_injected_by_key):
    common_keys = sorted(set(scores_clean_by_key.index) & set(scores_injected_by_key.index))
    clean_common = scores_clean_by_key.loc[common_keys].to_numpy()
    injected_common = scores_injected_by_key.loc[common_keys].to_numpy()
    diff = np.abs(clean_common - injected_common)
    return {
        "n_common_windows": len(common_keys),
        "max_abs_score_diff": float(diff.max()) if len(diff) else float("nan"),
        "n_differing_windows": int((diff > 1e-9).sum()),
    }


def count_grid_reassignments(df_clean_windowed, df_injected_windowed):
    clean_key = df_clean_windowed.set_index("row_id")["eval_window_key"]
    injected_key = df_injected_windowed.set_index("row_id")["eval_window_key"]
    common_row_ids = clean_key.index.intersection(injected_key.index)
    n_diff = int((clean_key.loc[common_row_ids] != injected_key.loc[common_row_ids]).sum())
    return n_diff, len(common_row_ids)


def evaluate_cell(df_eval, row_score_by_id, row_flag_by_id, row_true_by_id):
    row_predicted = df_eval["row_id"].map(row_flag_by_id).fillna(False)
    row_true_aligned = df_eval["row_id"].map(row_true_by_id).fillna(False)
    common = evaluate_common_unit(df_eval, row_predicted, row_true_aligned)
    grid_flagged_frac = common["n_flagged"] / common["n_eval_cells"] if common["n_eval_cells"] else float("nan")
    lift = common["recall"] / grid_flagged_frac if grid_flagged_frac else float("nan")

    row_score_by_id_full = pd.Series(row_score_by_id)
    auc = compute_grid_auc(df_eval, row_score_by_id_full, row_true_aligned)

    return {
        "recall": common["recall"],
        "precision": common["precision"],
        "grid_flagged_frac": grid_flagged_frac,
        "lift": lift,
        "n_positive_windows": auc["n_positive_cells"],
        "auc_pr": auc["auc_pr"],
        "no_skill_baseline": auc["no_skill_baseline"],
        "auc_pr_ratio": auc["auc_pr_ratio"],
        "auc_roc": auc["auc_roc"],
    }


def build_timing_scores(dataset, df_clean_std, is_train_std, df_injected_std):
    if dataset == "bgl":
        features_train_normal, features_injected = build_timing_features(df_clean_std, is_train_std, df_injected_std)
        baselines = compute_clean_train_baselines(df_clean_std, is_train_std)
        features_train_normal = add_log_ratio_feature(features_train_normal, baselines)
        features_injected = add_log_ratio_feature(features_injected, baselines)
    else:
        features_train_normal, features_injected, _ = build_timing_features_tb(df_clean_std, is_train_std, df_injected_std)

    zscore_detector = ZScoreThresholdDetector()
    zscore_scores = zscore_detector.score(features_injected)
    zscore_flags = zscore_scores > zscore_detector.z_thresh

    logratio_detector = LogRatioThresholdDetector()
    logratio_detector.fit(features_train_normal)
    logratio_scores = logratio_detector.score(features_injected)
    logratio_flags = logratio_scores > logratio_detector.threshold

    row_id = features_injected["row_id"].to_numpy()
    return {
        "z_score_threshold": (
            pd.Series(zscore_scores.to_numpy(), index=row_id),
            pd.Series(zscore_flags.to_numpy(), index=row_id),
        ),
        "log_ratio_threshold": (
            pd.Series(logratio_scores.to_numpy(), index=row_id),
            pd.Series(logratio_flags.to_numpy(), index=row_id),
        ),
    }


def save_checkpoint(all_sweep_rows, all_invariance_rows, equivalence_checks):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_sweep_rows).to_csv(CHECKPOINT_DIR / "windowing_sweep_checkpoint.csv", index=False)
    pd.DataFrame(all_invariance_rows).to_csv(CHECKPOINT_DIR / "windowing_sweep_invariance_checkpoint.csv", index=False)
    pd.DataFrame(equivalence_checks).to_csv(CHECKPOINT_DIR / "windowing_sweep_equivalence_checkpoint.csv", index=False)


def score_one_fault_one_scheme(fitted, df_clean_scheme, count_scores_clean, per_fault_timing, dataset, fault, scheme_name, scheme_kind, scheme_size, df_eval, windows_eval, vocabulary):
    count_scores_injected = score_count_detectors(fitted, df_eval, windows_eval, vocabulary)
    row_true_by_id = per_fault_timing[fault]["row_true_by_id"]
    timing_scores = per_fault_timing[fault]["timing_scores"]

    sweep_rows, invariance_rows = [], []
    for name in ["count_vector_pca", "isolation_forest_counts"]:
        score_clean_by_key = scores_by_window_key(df_clean_scheme, count_scores_clean[name][0])
        score_injected_by_key = scores_by_window_key(df_eval, count_scores_injected[name][0])
        comparison = compare_window_scores(score_clean_by_key, score_injected_by_key)
        n_grid_reassign, n_common_rows = count_grid_reassignments(df_clean_scheme, df_eval)
        invariance_rows.append(
            {
                "dataset": dataset,
                "fault_type": fault,
                "scheme": scheme_name,
                "scheme_kind": scheme_kind,
                "scheme_size": scheme_size,
                "detector": name,
                **comparison,
                "n_grid_cell_reassignments": n_grid_reassign,
                "n_common_rows_for_grid_check": n_common_rows,
            }
        )
        row_score, row_flag = count_scores_injected[name]
        metrics = evaluate_cell(df_eval, row_score, row_flag, row_true_by_id)
        sweep_rows.append(
            {"dataset": dataset, "fault_type": fault, "scheme": scheme_name, "scheme_kind": scheme_kind, "scheme_size": scheme_size, "detector": name, **metrics}
        )

    for name in ["z_score_threshold", "log_ratio_threshold"]:
        row_score, row_flag = timing_scores[name]
        metrics = evaluate_cell(df_eval, row_score, row_flag, row_true_by_id)
        sweep_rows.append(
            {"dataset": dataset, "fault_type": fault, "scheme": scheme_name, "scheme_kind": scheme_kind, "scheme_size": scheme_size, "detector": name, **metrics}
        )

    del count_scores_injected
    return sweep_rows, invariance_rows


def process_time_scheme(dataset, config, scheme_name, size, df_clean_std, df_train_std, vocabulary, per_fault_timing):
    df_train_scheme, windows_train_scheme = build_windows(df_train_std, scheme="fixed_time", size=size)
    df_train_scheme["eval_window_key"] = df_train_scheme["window_key"]
    fitted = fit_count_detectors_on_scheme(df_train_scheme, windows_train_scheme, vocabulary)
    del df_train_scheme, windows_train_scheme

    df_clean_scheme, windows_clean_scheme = build_windows(df_clean_std, scheme="fixed_time", size=size)
    df_clean_scheme["eval_window_key"] = df_clean_scheme["window_key"]
    count_scores_clean = score_count_detectors(fitted, df_clean_scheme, windows_clean_scheme, vocabulary)
    del windows_clean_scheme

    all_sweep_rows, all_invariance_rows = [], []
    for fault in ["stall", "burst"]:
        df_injected_std = load_by_node(config["injected_loaders"][fault]())
        df_eval, windows_eval = build_windows(df_injected_std, scheme="fixed_time", size=size)
        df_eval["eval_window_key"] = df_eval["window_key"]

        sweep_rows, invariance_rows = score_one_fault_one_scheme(
            fitted, df_clean_scheme, count_scores_clean, per_fault_timing, dataset, fault, scheme_name, "fixed_time", size, df_eval, windows_eval, vocabulary
        )
        all_sweep_rows.extend(sweep_rows)
        all_invariance_rows.extend(invariance_rows)

        del df_injected_std, df_eval, windows_eval
        gc.collect()

    del df_clean_scheme, count_scores_clean, fitted
    gc.collect()
    return all_sweep_rows, all_invariance_rows


def process_count_scheme(dataset, config, scheme_name, size, df_clean_pos, df_train_pos, vocabulary, per_fault_timing):
    df_train_scheme = assign_fixed_count_window(df_train_pos, size)
    windows_train_scheme = build_fixed_count_windows_table(df_train_scheme)
    fitted = fit_count_detectors_on_scheme(df_train_scheme, windows_train_scheme, vocabulary)
    del df_train_scheme, windows_train_scheme

    df_clean_scheme = assign_fixed_count_window(df_clean_pos, size)
    windows_clean_scheme = build_fixed_count_windows_table(df_clean_scheme)
    count_scores_clean = score_count_detectors(fitted, df_clean_scheme, windows_clean_scheme, vocabulary)
    del windows_clean_scheme

    all_sweep_rows, all_invariance_rows = [], []
    for fault in ["stall", "burst"]:
        df_injected_pos = load_pos(config["injected_paths"][fault])
        df_eval = assign_fixed_count_window(df_injected_pos, size)
        windows_eval = build_fixed_count_windows_table(df_eval)

        sweep_rows, invariance_rows = score_one_fault_one_scheme(
            fitted, df_clean_scheme, count_scores_clean, per_fault_timing, dataset, fault, scheme_name, "fixed_count", size, df_eval, windows_eval, vocabulary
        )
        all_sweep_rows.extend(sweep_rows)
        all_invariance_rows.extend(invariance_rows)

        del df_injected_pos, df_eval, windows_eval
        gc.collect()

    del df_clean_scheme, count_scores_clean, fitted
    gc.collect()
    return all_sweep_rows, all_invariance_rows


def main():
    t0 = time.time()
    all_sweep_rows = []
    all_invariance_rows = []
    equivalence_checks = []

    datasets_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(DATASET_CONFIG.keys())
    for dataset in datasets_to_run:
        config = DATASET_CONFIG[dataset]
        print(f"\n=== {dataset} ===", flush=True)

        df_clean_std = load_clean_by_node(config["clean_path"])
        is_train_std, cutoff = config["split_fn"](df_clean_std)
        df_train_std = df_clean_std.loc[is_train_std.to_numpy()].reset_index(drop=True)
        vocabulary = build_vocabulary(df_train_std)
        print(f"  vocab size: {len(vocabulary)}, elapsed: {time.time() - t0:.0f}s", flush=True)

        df_clean_pos = load_pos(config["clean_path"])
        eq_clean = verify_std_pos_equivalence(df_clean_std, df_clean_pos, f"{dataset} clean")
        equivalence_checks.append({"dataset": dataset, "fault_type": None, "check": "clean std==pos", "identical": eq_clean})
        is_train_pos = df_clean_pos["timestamp"] <= cutoff
        df_train_pos = df_clean_pos.loc[is_train_pos.to_numpy()].reset_index(drop=True)

        per_fault_timing = {}
        for fault in ["stall", "burst"]:
            df_injected_std = load_by_node(config["injected_loaders"][fault]())
            df_injected_pos = load_pos(config["injected_paths"][fault])
            eq_injected = verify_std_pos_equivalence(df_injected_std, df_injected_pos, f"{dataset} {fault} injected")
            equivalence_checks.append({"dataset": dataset, "fault_type": fault, "check": "injected std==pos", "identical": eq_injected})
            del df_injected_pos
            gc.collect()

            labels_df = pd.read_csv(config["injection_labels_path"][fault], parse_dates=["start", "end"])
            row_true_std = label_rows_in_injected_span(df_injected_std, labels_df)
            row_true_by_id = pd.Series(row_true_std.to_numpy(), index=df_injected_std["row_id"].to_numpy())
            timing_scores = build_timing_scores(dataset, df_clean_std, is_train_std, df_injected_std)
            per_fault_timing[fault] = {"row_true_by_id": row_true_by_id, "timing_scores": timing_scores}
            del df_injected_std
            gc.collect()

        print(f"  equivalence + timing features done, elapsed: {time.time() - t0:.0f}s", flush=True)
        save_checkpoint(all_sweep_rows, all_invariance_rows, equivalence_checks)

        for scheme_name, size in [("fixed_time_30", 30), ("fixed_time_60", 60), ("fixed_time_120", 120)]:
            print(f"  scheme {scheme_name}, elapsed: {time.time() - t0:.0f}s", flush=True)
            sweep_rows, invariance_rows = process_time_scheme(dataset, config, scheme_name, size, df_clean_std, df_train_std, vocabulary, per_fault_timing)
            all_sweep_rows.extend(sweep_rows)
            all_invariance_rows.extend(invariance_rows)
            save_checkpoint(all_sweep_rows, all_invariance_rows, equivalence_checks)

        del df_clean_std, df_train_std
        gc.collect()

        for scheme_name, size in [("fixed_count_20", 20), ("fixed_count_50", 50), ("fixed_count_100", 100)]:
            print(f"  scheme {scheme_name}, elapsed: {time.time() - t0:.0f}s", flush=True)
            sweep_rows, invariance_rows = process_count_scheme(dataset, config, scheme_name, size, df_clean_pos, df_train_pos, vocabulary, per_fault_timing)
            all_sweep_rows.extend(sweep_rows)
            all_invariance_rows.extend(invariance_rows)
            save_checkpoint(all_sweep_rows, all_invariance_rows, equivalence_checks)

        del df_clean_pos, df_train_pos, per_fault_timing
        gc.collect()

    sweep_df = pd.DataFrame(all_sweep_rows)
    invariance_df = pd.DataFrame(all_invariance_rows)
    equivalence_df = pd.DataFrame(equivalence_checks)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print("\n=== std/pos equivalence checks ===", flush=True)
    print(equivalence_df.to_string(index=False), flush=True)
    print("\n=== Invariance check: PCA / isolation_forest_counts, clean vs injected, all schemes ===", flush=True)
    print(invariance_df.to_string(index=False), flush=True)
    print(f"\n=== Full sweep ({len(sweep_df)} rows) ===", flush=True)
    print(sweep_df.to_string(index=False), flush=True)
    print(f"\nWrote {OUT_CSV}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)

    return sweep_df, invariance_df, equivalence_df


if __name__ == "__main__":
    main()
