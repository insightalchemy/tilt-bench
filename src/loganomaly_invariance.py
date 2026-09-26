"""
LogAnomaly-style detector and its fixed-count invariance check (paper Table III, "LA" rows).

Detector (after Meng et al., "LogAnomaly: Unsupervised Detection of Sequential and Quantitative
Anomalies in Unstructured Logs", IJCAI 2019): per-window template count vectors concatenated with
Template2Vec-style semantic vectors (per-template averages of word embeddings, count-weighted per
window), scored with an isolation forest over fixed-count windows of N events per node. Word
embeddings are seeded random vectors, so the detector is deterministic given --seed.

Invariance check: under fixed-count windows a timing-only injection leaves every window's content
unchanged, so a detector that never reads timestamps must score clean and injected data
identically. For each (dataset, fault, N) cell the script reports max_abs_score_diff and
n_differing_windows between clean and injected windows (expected: exactly 0).

Datasets: bgl, thunderbird, and the Loghub datasets of src/multi_dataset_registry.py. Pass
--dataset explicitly; without it every configured dataset is run.

Writes:
  results/loganomaly_invariance.csv / .md  (or --out-csv / --out-md)

Usage:
    python src/loganomaly_invariance.py --dataset bgl
    python src/loganomaly_invariance.py --dataset openstack --window-sizes 20 50 100
    # quick end-to-end check on a subsample:
    python src/loganomaly_invariance.py --dataset bgl --subsample 50000 --subsample-include-injected 5
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.deeplog import load_by_node, load_clean_by_node, select_injected_nodes
from src.detectors.windowing import TOP_K_TEMPLATES, build_vocabulary, build_windows, count_matrix
from src.pipeline_common import (
    chronological_split_tb,
    load_bgl_injected_burst as load_bgl_burst,
    load_bgl_injected_stall as load_bgl_stall,
    load_tb_injected,
)
from src.run_baseline_detectors import chronological_split as chronological_split_bgl

EMBED_DIM = 20
SEED = 42
WINDOW_SIZES = [20, 50, 100]
FAULTS = ["stall", "burst"]

DATASET_CONFIG = {
    "bgl": {
        "clean_path": Path("data/processed/bgl_parsed.parquet"),
        "split_fn": chronological_split_bgl,
        "injected_loaders": {"stall": load_bgl_stall, "burst": load_bgl_burst},
    },
    "thunderbird": {
        "clean_path": Path("data/processed/thunderbird_parsed.parquet"),
        "split_fn": chronological_split_tb,
        "injected_loaders": {"stall": lambda: load_tb_injected("stall"), "burst": lambda: load_tb_injected("burst")},
    },
}

from src.multi_dataset_registry import DATASETS as LOGHUB_DATASETS
from src.multi_dataset_registry import make_generic_dataset_config as _make_loghub_config

DATASET_CONFIG.update({name: _make_loghub_config(name) for name in LOGHUB_DATASETS})

OUT_CSV = Path("results/loganomaly_invariance.csv")
OUT_MD = Path("results/loganomaly_invariance.md")


def tokenize_template(template):
    return re.findall(r"[a-z]+", str(template).lower())


def build_template2vec(vocabulary, dim=EMBED_DIM, seed=SEED):
    words = sorted({w for t in vocabulary for w in tokenize_template(t)})
    rng = np.random.default_rng(seed)
    word_vectors = {w: rng.normal(size=dim) for w in words}
    embeddings = np.zeros((len(vocabulary), dim), dtype=np.float64)
    for i, template in enumerate(vocabulary):
        toks = tokenize_template(template)
        if toks:
            embeddings[i] = np.mean([word_vectors[w] for w in toks], axis=0)
    return embeddings


def semantic_matrix(counts, embeddings):
    totals = counts.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    return (counts @ embeddings) / totals


def build_feature_matrix(df_windowed, windows, vocabulary, embeddings):
    counts = count_matrix(df_windowed, windows, vocabulary)
    semantic = semantic_matrix(counts, embeddings)
    return np.concatenate([counts, semantic], axis=1)


class LogAnomalyDetector:
    def __init__(self, n_estimators=100, random_state=SEED):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    def fit(self, X_train_normal):
        self.model.fit(X_train_normal)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


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


def run_dataset(dataset, config, window_sizes, subsample, subsample_include_injected, seed):
    must_include_nodes = None
    if subsample_include_injected is not None:
        must_include_nodes = select_injected_nodes(config["injected_loaders"]["stall"], subsample_include_injected, seed)

    df_clean = load_clean_by_node(config["clean_path"], subsample=subsample, must_include_nodes=must_include_nodes)
    is_train, _ = config["split_fn"](df_clean)
    df_train = df_clean.loc[is_train.to_numpy()].reset_index(drop=True)
    vocabulary = build_vocabulary(df_train, top_k=TOP_K_TEMPLATES)
    embeddings = build_template2vec(vocabulary, seed=seed)

    df_injected_by_fault = {
        fault: load_by_node(config["injected_loaders"][fault](), subsample=subsample, must_include_nodes=must_include_nodes)
        for fault in FAULTS
    }

    rows = []
    for n in window_sizes:
        df_train_win, windows_train = build_windows(df_train, scheme="fixed_count", size=n)
        X_train = build_feature_matrix(df_train_win, windows_train, vocabulary, embeddings)
        train_normal_mask = ~windows_train["label"].to_numpy()
        detector = LogAnomalyDetector(random_state=seed).fit(X_train[train_normal_mask])

        df_clean_win, windows_clean = build_windows(df_clean, scheme="fixed_count", size=n)
        X_clean = build_feature_matrix(df_clean_win, windows_clean, vocabulary, embeddings)
        scores_clean_by_key = pd.Series(detector.score(X_clean), index=windows_clean["window_key"])

        for fault in FAULTS:
            df_injected_win, windows_injected = build_windows(df_injected_by_fault[fault], scheme="fixed_count", size=n)
            X_injected = build_feature_matrix(df_injected_win, windows_injected, vocabulary, embeddings)
            scores_injected_by_key = pd.Series(detector.score(X_injected), index=windows_injected["window_key"])

            comparison = compare_window_scores(scores_clean_by_key, scores_injected_by_key)
            rows.append({"dataset": dataset, "fault_type": fault, "window_size": n, "detector": "loganomaly", **comparison})
            print(f"  {dataset} {fault} N={n}: {comparison}")

    return rows


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_CONFIG), default=None)
    ap.add_argument("--window-sizes", type=int, nargs="+", default=WINDOW_SIZES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--subsample-include-injected", type=int, default=None)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    return ap.parse_args()


def write_markdown(df, path, args):
    all_zero = bool((df["max_abs_score_diff"].fillna(0.0) == 0.0).all())
    lines = [
        "# LogAnomaly-style fixed-count windowing invariance check",
        "",
        f"seed={args.seed}, window_sizes={args.window_sizes}, subsample={args.subsample}, "
        f"subsample_include_injected={args.subsample_include_injected}",
        "",
        df.to_markdown(index=False),
        "",
        f"All-zero max_abs_score_diff: {all_zero}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIG.keys())
    all_rows = []
    for dataset in datasets:
        print(f"=== {dataset} ===")
        rows = run_dataset(dataset, DATASET_CONFIG[dataset], args.window_sizes, args.subsample, args.subsample_include_injected, args.seed)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {args.out_csv}")

    write_markdown(df, args.out_md, args)
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
