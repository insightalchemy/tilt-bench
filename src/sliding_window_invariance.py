"""
Fixed-count invariance check for overlapping (sliding) windows: size N=50, step S in {10, 25}.

A sliding window starts every S events on a node, so windows overlap when S < N. Membership is
still defined by event position only, so Proposition 1 predicts max_abs_score_diff = 0 between
clean and injected data for the content detectors (count_vector_pca, isolation_forest_counts,
LogAnomaly-style).

Windows are built per node with an index view of shape (n_windows, N), stepped by S, and turned
into count vectors with a single bincount, so no dense per-event matrix is materialized. This keeps
memory bounded on nodes with millions of events. Each window size's tables are freed before the
next one is built.

Writes results/sliding_window/sliding_window_invariance_{dataset}_{fault}.csv (rewritten after each
step).

Usage:
    python src/sliding_window_invariance.py --dataset bgl --fault stall
    python src/sliding_window_invariance.py --dataset bgl --fault burst
    python src/sliding_window_invariance.py --dataset spirit --fault stall
    python src/sliding_window_invariance.py --dataset spirit --fault burst
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.deeplog import load_by_node, load_clean_by_node
from src.detectors.count_pca import CountPCADetector
from src.detectors.isolation_forest_counts import IsolationForestCountsDetector
from src.detectors.windowing import build_vocabulary
from src.loganomaly_invariance import LogAnomalyDetector, build_template2vec
from src.placebo_sweep import ALL_DATASET_CONFIG
from src.run_baseline_detectors import THRESHOLD_PERCENTILE
from src.windowing_sweep import load_pos, verify_std_pos_equivalence

N = 50
STEPS = [10, 25]
LOGANOMALY_SEED = 42
OUT_DIR = Path("results/sliding_window")


def sliding_windows_for_node(template_ids, anomaly, n_window, step, vocab_size, embeddings):
    n = len(template_ids)
    if n < n_window:
        return None
    starts = np.arange(0, n - n_window + 1, step)
    n_windows = len(starts)
    idx = starts[:, None] + np.arange(n_window)[None, :]
    ids_in_windows = template_ids[idx]
    row_idx = np.repeat(np.arange(n_windows), n_window)
    col_idx = ids_in_windows.ravel()
    valid = col_idx >= 0
    flat_idx = row_idx[valid] * vocab_size + col_idx[valid]
    counts_flat = np.bincount(flat_idx, minlength=n_windows * vocab_size).astype(np.float32)
    count_matrix = counts_flat.reshape(n_windows, vocab_size)
    semantic_matrix = (count_matrix @ embeddings) / n_window
    feature_matrix = np.concatenate([count_matrix, semantic_matrix], axis=1)
    labels = anomaly[idx].any(axis=1)
    del idx, ids_in_windows, row_idx, col_idx, flat_idx, counts_flat, semantic_matrix
    return starts, feature_matrix, labels


def build_sliding_table(df_pos, step, vocab_index, embeddings):
    vocab_size = len(vocab_index)
    all_keys, all_features, all_labels = [], [], []
    for node, group in df_pos.groupby("node", sort=False):
        template_ids = group["event_template"].map(vocab_index).fillna(-1).astype(int).to_numpy()
        anomaly = group["anomaly"].to_numpy()
        result = sliding_windows_for_node(template_ids, anomaly, N, step, vocab_size, embeddings)
        if result is None:
            continue
        starts, feature_matrix, labels = result
        all_keys.extend((node, int(s)) for s in starts)
        all_features.append(feature_matrix)
        all_labels.append(labels)
        del template_ids, anomaly, result, starts, feature_matrix, labels
    if not all_features:
        return [], np.empty((0, 2 * vocab_size), dtype=np.float32), np.empty((0,), dtype=bool)
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    del all_features, all_labels
    gc.collect()
    return all_keys, features, labels


def fit_and_score(dataset, fault, config, step):
    df_clean_std = load_clean_by_node(config["clean_path"])
    is_train_std, cutoff = config["split_fn"](df_clean_std)
    df_train_std = df_clean_std.loc[is_train_std.to_numpy()].reset_index(drop=True)
    vocabulary = build_vocabulary(df_train_std)
    vocab_index = {t: i for i, t in enumerate(vocabulary)}
    embeddings = build_template2vec(vocabulary, seed=LOGANOMALY_SEED)
    del df_train_std
    gc.collect()

    df_clean_pos = load_pos(config["clean_path"])
    eq_clean = verify_std_pos_equivalence(df_clean_std, df_clean_pos, f"{dataset} clean")
    cutoff_ts = df_clean_std.loc[is_train_std.to_numpy(), "timestamp"].max()
    is_train_pos = df_clean_pos["timestamp"] <= cutoff_ts
    df_train_pos = df_clean_pos.loc[is_train_pos.to_numpy()].reset_index(drop=True)
    del df_clean_std, is_train_std
    gc.collect()

    df_injected_std = load_by_node(config["injected_loaders"][fault]())
    df_injected_pos = load_pos(config["injected_paths"][fault])
    eq_injected = verify_std_pos_equivalence(df_injected_std, df_injected_pos, f"{dataset} {fault} injected")
    del df_injected_std
    gc.collect()

    print(f"  step={step}: building train windows", flush=True)
    _, X_train, labels_train = build_sliding_table(df_train_pos, step, vocab_index, embeddings)
    del df_train_pos
    gc.collect()
    train_normal_mask = ~labels_train
    X_train_normal = X_train[train_normal_mask]
    del X_train, labels_train, train_normal_mask
    gc.collect()

    detectors = {}
    for name, det in [("count_vector_pca", CountPCADetector()), ("isolation_forest_counts", IsolationForestCountsDetector())]:
        det.fit(X_train_normal)
        detectors[name] = det
    loganomaly = LogAnomalyDetector(random_state=LOGANOMALY_SEED).fit(X_train_normal)
    detectors["loganomaly"] = loganomaly
    del X_train_normal
    gc.collect()

    print(f"  step={step}: building clean windows", flush=True)
    keys_clean, X_clean, _ = build_sliding_table(df_clean_pos, step, vocab_index, embeddings)
    del df_clean_pos
    gc.collect()

    print(f"  step={step}: building injected windows", flush=True)
    keys_inj, X_inj, _ = build_sliding_table(df_injected_pos, step, vocab_index, embeddings)
    del df_injected_pos
    gc.collect()

    rows = []
    for name, det in detectors.items():
        scores_clean = pd.Series(det.score(X_clean), index=pd.MultiIndex.from_tuples(keys_clean))
        scores_inj = pd.Series(det.score(X_inj), index=pd.MultiIndex.from_tuples(keys_inj))
        common_keys = scores_clean.index.intersection(scores_inj.index)
        diff = (scores_clean.loc[common_keys].to_numpy() - scores_inj.loc[common_keys].to_numpy())
        diff = np.abs(diff)
        rows.append(
            {
                "dataset": dataset,
                "fault_type": fault,
                "window_size": N,
                "step": step,
                "detector": name,
                "n_common_windows": len(common_keys),
                "max_abs_score_diff": float(diff.max()) if len(diff) else float("nan"),
                "n_differing_windows": int((diff > 1e-9).sum()),
                "std_pos_equivalence_clean": eq_clean,
                "std_pos_equivalence_injected": eq_injected,
            }
        )
        del scores_clean, scores_inj, common_keys, diff
        gc.collect()

    del X_clean, X_inj, detectors
    gc.collect()
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["bgl", "spirit"], required=True)
    ap.add_argument("--fault", choices=["stall", "burst"], default="stall")
    return ap.parse_args()


def main():
    args = parse_args()
    config = ALL_DATASET_CONFIG[args.dataset]
    print(f"=== sliding window invariance: {args.dataset} / {args.fault} ===", flush=True)

    all_rows = []
    for step in STEPS:
        all_rows.extend(fit_and_score(args.dataset, args.fault, config, step))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(OUT_DIR / f"sliding_window_invariance_{args.dataset}_{args.fault}.csv", index=False)
        gc.collect()

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / f"sliding_window_invariance_{args.dataset}_{args.fault}.csv"
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
