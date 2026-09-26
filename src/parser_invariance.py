"""
Parser-robustness variant of the fixed-count invariance check (paper Table III, "masked" and
"Drain 0.5/0.7" rows): N = 20/50/100, three content detectors (count_vector_pca,
isolation_forest_counts, LogAnomaly-style), stall faults.

Instead of re-injecting, the templates of an alternative parse of the same raw log (Spirit with
variable-field masking from src/parser_spirit_masked.py, or BGL at a different Drain similarity
threshold from `src/parser.py --sim-th`) are substituted into the existing injected parquet by row
identity. Injection never changes event_template, so the injected rows' templates at a given
original file position equal whatever the alternative parse produces there.

The substitution uses the injected parquet's `row_id` (assigned in original file order before
any sort; see src.injector.load_clean) and assumes the alternative parse is in the same file
order. Before substituting, it checks that the row counts are equal and that raw_message matches
at every position. A different similarity threshold changes only template mining, not line
splitting. If either check fails, the script stops.

Writes:
  --dataset spirit_masked: results/parser_robustness/masked_invariance_spirit.csv
  --dataset bgl:           results/parser_robustness/parser_robustness_bgl_simth{X}.csv

Usage:
    python src/parser_invariance.py --dataset spirit_masked
    python src/parser_invariance.py --dataset bgl --clean-parsed data/processed/bgl_parsed_simth0.5.parquet
    python src/parser_invariance.py --dataset bgl --clean-parsed data/processed/bgl_parsed_simth0.7.parquet
"""

import argparse
import gc
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.deeplog import load_clean_by_node
from src.detectors.windowing import build_vocabulary
from src.loganomaly_invariance import LogAnomalyDetector, build_feature_matrix, build_template2vec
from src.run_baseline_detectors import chronological_split
from src.windowing_sweep import (
    assign_fixed_count_window,
    build_fixed_count_windows_table,
    compare_window_scores,
    count_grid_reassignments,
    fit_count_detectors_on_scheme,
    load_pos,
    score_count_detectors,
    scores_by_window_key,
)

FIXED_COUNT_SIZES = [20, 50, 100]
LOGANOMALY_SEED = 42

DATASET_DEFAULTS = {
    "spirit_masked": {
        "clean_parsed": Path("data/processed/spirit_parsed_masked.parquet"),
        "injected_path": Path("data/processed/spirit_injected_stall.parquet"),
        "out_csv": Path("results/parser_robustness/masked_invariance_spirit.csv"),
    },
    "bgl": {
        "clean_parsed": None,
        "injected_path": Path("data/processed/bgl_injected_stall.parquet"),
        "out_csv": None,
    },
}


def sim_th_suffix_from_path(path: Path) -> str | None:
    m = re.search(r"simth([\d.]+)", path.stem)
    return m.group(1) if m else None


def load_reparsed_injected(injected_path: Path, clean_parsed_path: Path) -> pd.DataFrame:
    df_injected = pd.read_parquet(injected_path)
    df_reparsed = pd.read_parquet(clean_parsed_path, columns=["event_template", "raw_message"])

    if len(df_injected) != len(df_reparsed):
        raise ValueError(
            f"row count mismatch: {injected_path} has {len(df_injected):,} rows but "
            f"{clean_parsed_path} has {len(df_reparsed):,} -- these do not correspond to the same "
            "underlying parse and cannot be substituted by row identity."
        )

    injected_raw_by_pos = df_injected.sort_values("row_id")["raw_message"].to_numpy()
    reparsed_raw = df_reparsed["raw_message"].to_numpy()
    mismatched = injected_raw_by_pos != reparsed_raw
    if mismatched.any():
        n_mismatch = int(mismatched.sum())
        first_bad = int(np.flatnonzero(mismatched)[0])
        raise ValueError(
            f"raw_message mismatch at {n_mismatch:,} of {len(reparsed_raw):,} positions between "
            f"{injected_path} (by row_id order) and {clean_parsed_path} (file order) -- first "
            f"mismatch at position {first_bad}: {injected_raw_by_pos[first_bad]!r} vs "
            f"{reparsed_raw[first_bad]!r}. These are not the same underlying parse in the same "
            "order; refusing to substitute templates."
        )

    reparsed_templates = df_reparsed["event_template"].to_numpy()
    df_injected["event_template"] = reparsed_templates[df_injected["row_id"].to_numpy()]
    return df_injected


def to_pos(df):
    return df.sort_values("node", kind="mergesort").reset_index(drop=True)


def run(dataset: str, clean_parsed: Path, injected_path: Path, out_csv: Path):
    df_clean_std = load_clean_by_node(clean_parsed)
    is_train_std, cutoff = chronological_split(df_clean_std)
    df_train_std = df_clean_std.loc[is_train_std.to_numpy()].reset_index(drop=True)
    vocabulary = build_vocabulary(df_train_std)
    embeddings = build_template2vec(vocabulary, seed=LOGANOMALY_SEED)
    del df_train_std
    gc.collect()

    df_clean_pos = load_pos(clean_parsed)
    is_train_pos = df_clean_pos["timestamp"] <= cutoff
    df_train_pos = df_clean_pos.loc[is_train_pos.to_numpy()].reset_index(drop=True)
    del df_clean_std, is_train_std, is_train_pos
    gc.collect()

    df_injected_reparsed = load_reparsed_injected(injected_path, clean_parsed)
    df_injected_pos = to_pos(df_injected_reparsed)
    del df_injected_reparsed
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
        loganomaly_clean_by_key = pd.Series(loganomaly.score(X_clean), index=windows_clean["window_key"])
        loganomaly_inj_by_key = pd.Series(loganomaly.score(X_inj), index=windows_inj["window_key"])

        detector_score_by_key = {
            "count_vector_pca": (
                scores_by_window_key(df_clean_w, count_scores_clean["count_vector_pca"][0]),
                scores_by_window_key(df_inj_w, count_scores_inj["count_vector_pca"][0]),
            ),
            "isolation_forest_counts": (
                scores_by_window_key(df_clean_w, count_scores_clean["isolation_forest_counts"][0]),
                scores_by_window_key(df_inj_w, count_scores_inj["isolation_forest_counts"][0]),
            ),
            "loganomaly": (loganomaly_clean_by_key, loganomaly_inj_by_key),
        }

        for name, (score_clean_by_key, score_inj_by_key) in detector_score_by_key.items():
            comparison = compare_window_scores(score_clean_by_key, score_inj_by_key)
            n_grid_reassign, n_common_rows = count_grid_reassignments(df_clean_w, df_inj_w)
            rows.append(
                {
                    "dataset": dataset,
                    "fault_type": "stall",
                    "scheme": f"fixed_count_{size}",
                    "detector": name,
                    **comparison,
                    "n_grid_cell_reassignments": n_grid_reassign,
                    "n_common_rows_for_grid_check": n_common_rows,
                }
            )

        del df_train_w, windows_train, df_clean_w, windows_clean, df_inj_w, windows_inj
        del fitted, count_scores_clean, count_scores_inj, X_train, X_clean, X_inj, loganomaly
        gc.collect()

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out_csv}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="spirit_masked")
    ap.add_argument("--clean-parsed", type=Path, default=None)
    ap.add_argument("--injected-path", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    clean_parsed = args.clean_parsed or defaults["clean_parsed"]
    if clean_parsed is None:
        print(f"ERROR: --clean-parsed is required for --dataset {args.dataset}.", file=sys.stderr)
        sys.exit(1)
    injected_path = args.injected_path or defaults["injected_path"]

    out_csv = args.out_csv or defaults["out_csv"]
    if out_csv is None:
        sim_th = sim_th_suffix_from_path(clean_parsed)
        out_csv = Path(f"results/parser_robustness/parser_robustness_{args.dataset}_simth{sim_th}.csv") if sim_th else Path(f"results/parser_robustness/parser_robustness_{args.dataset}.csv")

    if out_csv.exists() and args.out_csv is None and args.dataset != "spirit_masked":
        print(f"ERROR: {out_csv} already exists -- refusing to overwrite. Pass --out-csv to write elsewhere.", file=sys.stderr)
        sys.exit(1)

    run(args.dataset, clean_parsed, injected_path, out_csv)


if __name__ == "__main__":
    main()
