"""
BGL data loading, chronological split, and count-vector features shared by the detector
experiments; run as a script, evaluates the three baseline detectors against BGL's native alert
labels on clean data.

Split: rows with timestamp <= the TRAIN_FRAC quantile are train, the rest test (never random).
Count-vector detectors are evaluated per window; the timing detector per row, since inter-arrival
time is a per-event quantity. Detectors are fit on train-normal data only, and flag test instances
whose score exceeds the THRESHOLD_PERCENTILE of the train-normal score distribution. Metrics are
point-wise precision/recall/F1 (not point-adjusted).

Usage:
    python src/run_baseline_detectors.py
    python src/run_baseline_detectors.py --limit 200000   # chronological prefix, for a quick check
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import numpy as np
import pandas as pd

from src.detectors import windowing
from src.detectors.count_pca import CountPCADetector
from src.detectors.isolation_forest_counts import IsolationForestCountsDetector
from src.detectors.timing_detector import TimingIsolationForestDetector, build_features
from src.metrics import evaluate_binary
from src.timing_baseline import add_sequence_context

IN_PATH = Path("data/processed/bgl_parsed.parquet")
OUT_CSV = Path("results/baseline_detection.csv")

TRAIN_FRAC = 0.7  # chronological: first 70% of rows (by timestamp) train, rest test
THRESHOLD_PERCENTILE = 95  # flag if score exceeds this percentile of the train-normal score distribution


def load(limit=None):
    df = pd.read_parquet(IN_PATH)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    # Stable row identifier assigned in chronological order and carried through every downstream
    # re-sort, so detections from detectors that sort rows differently can be realigned.
    df["row_id"] = df.index.to_numpy()
    return df


def chronological_split(df, train_frac=TRAIN_FRAC):
    cutoff = df["timestamp"].quantile(train_frac)
    is_train = df["timestamp"] <= cutoff
    return is_train, cutoff


def build_count_features(df_train, df_test, window_scheme, window_size):
    """Windowing, vocabulary, and count matrices shared by both count-vector detectors."""
    df_train_w, windows_train = windowing.build_windows(df_train, scheme=window_scheme, size=window_size)
    df_test_w, windows_test = windowing.build_windows(df_test, scheme=window_scheme, size=window_size)

    vocabulary = windowing.build_vocabulary(df_train_w)
    X_train = windowing.count_matrix(df_train_w, windows_train, vocabulary)
    X_test = windowing.count_matrix(df_test_w, windows_test, vocabulary)

    y_test = windows_test["label"].to_numpy()
    train_templates = set(df_train_w["event_template"].unique())
    test_templates = set(df_test_w["event_template"].unique())
    diag = {
        "n_vocab": len(vocabulary),
        "n_unique_templates_train": len(train_templates),
        "n_unique_templates_test": len(test_templates),
        "n_test_templates_unseen_in_train": len(test_templates - train_templates),
        "test_event_coverage_pct": round(100 * df_test_w["event_template"].isin(vocabulary).mean(), 2),
        "pct_test_windows_all_zero": round(100 * (X_test.sum(axis=1) == 0).mean(), 2),
        "pct_test_windows_all_zero_anomalous": round(100 * (X_test[y_test].sum(axis=1) == 0).mean(), 2),
        "pct_test_windows_all_zero_normal": round(100 * (X_test[~y_test].sum(axis=1) == 0).mean(), 2),
    }
    return X_train, X_test, windows_train, windows_test, df_test_w, diag


def run_count_detector(name, detector, X_train, X_test, windows_train, windows_test, df_test_w):
    train_normal_mask = ~windows_train["label"].to_numpy()
    X_train_normal = X_train[train_normal_mask]

    detector.fit(X_train_normal)
    train_scores = detector.score(X_train_normal)
    threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)

    test_scores = detector.score(X_test)
    predicted = test_scores > threshold
    y_true = windows_test["label"].to_numpy()

    result = evaluate_binary(y_true, predicted)
    result.update(
        {
            "detector": name,
            "granularity": "window",
            "n_train_windows": len(windows_train),
            "n_train_normal_windows": int(train_normal_mask.sum()),
            "n_test_windows": len(windows_test),
            "test_window_anomaly_rate": round(100 * y_true.mean(), 2),
            "threshold": round(float(threshold), 4),
        }
    )

    # Row-level flags (each row inherits its window's flag), keyed by row_id.
    window_predicted = pd.Series(predicted, index=windows_test["window_key"])
    row_predicted = pd.Series(
        df_test_w["window_key"].map(window_predicted).to_numpy(), index=df_test_w["row_id"].to_numpy()
    )
    return result, row_predicted


def run_timing_detector(df_all, is_train):
    normal_seq = (
        is_train & (~df_all["anomaly"]) & (~df_all["prev_anomaly"].fillna(True)) & df_all["gap_prev_s"].notna()
    )
    features, baselines = build_features(df_all, normal_seq)

    train_normal_mask = is_train & (~df_all["anomaly"])
    features_train_normal = features.loc[train_normal_mask]
    features_test = features.loc[~is_train]

    detector = TimingIsolationForestDetector()
    detector.fit(features_train_normal)

    train_scores = detector.score(features_train_normal)
    threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)

    test_scores = detector.score(features_test)
    predicted = test_scores > threshold
    y_true = features_test["anomaly"].to_numpy()

    result = evaluate_binary(y_true, predicted)
    result.update(
        {
            "detector": "timing_inter_arrival_isolation_forest",
            "granularity": "row",
            "n_train_normal_rows": int(train_normal_mask.sum()),
            "n_test_rows": len(features_test),
            "test_row_anomaly_rate": round(100 * y_true.mean(), 2),
            "threshold": round(float(threshold), 4),
            "n_nodes_fallback_low_count": len(baselines["fallback_low_count_nodes"]),
            "n_nodes_fallback_zero_mad": len(baselines["fallback_zero_mad_nodes"]),
        }
    )

    row_predicted = pd.Series(predicted, index=features_test["row_id"].to_numpy())
    return result, row_predicted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="Only use the first N rows (chronological).")
    ap.add_argument("--window-scheme", choices=["fixed_count", "fixed_time"], default=windowing.WINDOW_SCHEME)
    ap.add_argument("--window-size", type=float, default=windowing.WINDOW_SIZE)
    args = ap.parse_args()

    df = load(limit=args.limit)
    is_train, cutoff = chronological_split(df)
    print(f"Chronological split at {cutoff} -- train: {is_train.sum():,} rows, test: {(~is_train).sum():,} rows")
    print(f"Window scheme: {args.window_scheme}, size: {args.window_size}")

    df_train = df.loc[is_train].reset_index(drop=True)
    df_test = df.loc[~is_train].reset_index(drop=True)

    X_train, X_test, windows_train, windows_test, df_test_w, count_diag = build_count_features(
        df_train, df_test, args.window_scheme, args.window_size
    )
    print("\n=== count-vector diagnostics ===")
    print(f"Vocabulary size (top-k train templates):        {count_diag['n_vocab']:,}")
    print(f"Unique templates -- train / test:                {count_diag['n_unique_templates_train']:,} / {count_diag['n_unique_templates_test']:,}")
    print(f"Test templates never seen in train:              {count_diag['n_test_templates_unseen_in_train']:,}")
    print(f"Vocabulary covers this % of TEST events:          {count_diag['test_event_coverage_pct']:.2f}%")
    print(f"% of TEST windows that are all-zero (no vocab hits): {count_diag['pct_test_windows_all_zero']:.2f}% overall "
          f"({count_diag['pct_test_windows_all_zero_anomalous']:.2f}% of anomalous windows, "
          f"{count_diag['pct_test_windows_all_zero_normal']:.2f}% of normal windows)")

    results = []
    result, _ = run_count_detector("count_vector_pca", CountPCADetector(), X_train, X_test, windows_train, windows_test, df_test_w)
    results.append(result)
    result, _ = run_count_detector(
        "isolation_forest_counts", IsolationForestCountsDetector(), X_train, X_test, windows_train, windows_test, df_test_w
    )
    results.append(result)

    df_all = add_sequence_context(df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    is_train_all = df_all["timestamp"] <= cutoff
    result, _ = run_timing_detector(df_all, is_train_all)
    results.append(result)

    results_df = pd.DataFrame(results)
    front_cols = ["detector", "granularity", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    other_cols = [c for c in results_df.columns if c not in front_cols]
    results_df = results_df[front_cols + other_cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Baseline detector results (clean BGL, vs native alert labels) ===")
    print(results_df[front_cols].to_string(index=False))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
