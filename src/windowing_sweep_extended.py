"""
Windowing sweep over a wider grid of window sizes, using the per-scheme functions of
src/windowing_sweep.py (process_time_scheme / process_count_scheme). Default grid: fixed-count
N in {20, 50, 100, 200, 500}; fixed-time in {15, 30, 60, 120, 300, 600} seconds.

Reports, per (dataset, fault_type, scheme, size):
  1. Invariance (count_vector_pca, isolation_forest_counts; clean vs injected): n_common_windows,
     max_abs_score_diff, n_differing_windows, n_grid_cell_reassignments. Proposition 1 predicts
     exactly 0 for every fixed-count cell. A nonzero fixed-count cell indicates a
     timestamp-dependence bug in window construction; the script reports it and exits with
     status 1.
  2. Detection (all four detectors): precision / recall / lift / AUC-PR / AUC-ROC.

Writes:
  results/windowing_sweep_extended.csv
  results/windowing_sweep_extended.md

Usage:
    python src/windowing_sweep_extended.py --dataset bgl
    python src/windowing_sweep_extended.py --dataset thunderbird
    # quick end-to-end check on a subsample:
    python src/windowing_sweep_extended.py --dataset bgl --fixed-time-sizes 15 30 --fixed-count-sizes 20 50 \
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
from src.detectors.windowing import build_vocabulary
from src.windowing_sweep import (
    DATASET_CONFIG,
    build_timing_scores,
    label_rows_in_injected_span,
    load_pos,
    process_count_scheme,
    process_time_scheme,
    verify_std_pos_equivalence,
)

FAULTS = ["stall", "burst"]
FIXED_TIME_SIZES_DEFAULT = [15, 30, 60, 120, 300, 600]
FIXED_COUNT_SIZES_DEFAULT = [20, 50, 100, 200, 500]

OUT_CSV = Path("results/windowing_sweep_extended.csv")
OUT_MD = Path("results/windowing_sweep_extended.md")


def subsample_df(df, subsample, must_include_nodes):
    if subsample is None:
        return df
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    if must_include_nodes:
        included = df[df["node"].isin(must_include_nodes)]
        remainder_budget = max(subsample - len(included), 0)
        remainder = df[~df["node"].isin(must_include_nodes)].head(remainder_budget)
        df = pd.concat([included, remainder]).sort_values(["node", "timestamp"], kind="mergesort")
    else:
        df = df.head(subsample)
    return df.reset_index(drop=True)


def make_run_config(config, subsample, must_include_nodes, tmp_dir):
    if subsample is None:
        return config

    run_config = dict(config)

    clean_df = subsample_df(pd.read_parquet(config["clean_path"]), subsample, must_include_nodes)
    clean_tmp_path = tmp_dir / f"{config['clean_path'].stem}_subsample.parquet"
    clean_df.to_parquet(clean_tmp_path)
    run_config["clean_path"] = clean_tmp_path

    run_config["injected_loaders"] = {
        fault: (lambda loader_fn=loader_fn: subsample_df(loader_fn(), subsample, must_include_nodes))
        for fault, loader_fn in config["injected_loaders"].items()
    }

    injected_paths = {}
    for fault, path in config["injected_paths"].items():
        injected_df = subsample_df(pd.read_parquet(path), subsample, must_include_nodes)
        injected_tmp_path = tmp_dir / f"{path.stem}_subsample.parquet"
        injected_df.to_parquet(injected_tmp_path)
        injected_paths[fault] = injected_tmp_path
    run_config["injected_paths"] = injected_paths

    return run_config


def run_dataset(dataset, config, args, tmp_dir):
    print(f"\n=== {dataset} ===", flush=True)
    t0 = time.time()

    must_include_nodes = None
    if args.subsample_include_injected is not None:
        must_include_nodes = set()
        for fault in FAULTS:
            must_include_nodes |= select_injected_nodes(config["injected_loaders"][fault], args.subsample_include_injected, args.seed)

    run_config = make_run_config(config, args.subsample, must_include_nodes, tmp_dir)

    df_clean_std = load_clean_by_node(run_config["clean_path"])
    is_train_std, cutoff = config["split_fn"](df_clean_std)
    df_train_std = df_clean_std.loc[is_train_std.to_numpy()].reset_index(drop=True)
    vocabulary = build_vocabulary(df_train_std)
    print(f"  vocab size: {len(vocabulary)}, elapsed: {time.time() - t0:.0f}s", flush=True)

    df_clean_pos = load_pos(run_config["clean_path"])
    eq_clean = verify_std_pos_equivalence(df_clean_std, df_clean_pos, f"{dataset} clean")
    is_train_pos = df_clean_pos["timestamp"] <= cutoff
    df_train_pos = df_clean_pos.loc[is_train_pos.to_numpy()].reset_index(drop=True)

    equivalence_checks = [{"dataset": dataset, "fault_type": None, "check": "clean std==pos", "identical": eq_clean}]

    per_fault_timing = {}
    for fault in FAULTS:
        df_injected_std = load_by_node(run_config["injected_loaders"][fault]())
        df_injected_pos = load_pos(run_config["injected_paths"][fault])
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

    all_sweep_rows, all_invariance_rows = [], []
    for size in args.fixed_time_sizes:
        scheme_name = f"fixed_time_{size}"
        print(f"  scheme {scheme_name}, elapsed: {time.time() - t0:.0f}s", flush=True)
        sweep_rows, invariance_rows = process_time_scheme(dataset, run_config, scheme_name, size, df_clean_std, df_train_std, vocabulary, per_fault_timing)
        all_sweep_rows.extend(sweep_rows)
        all_invariance_rows.extend(invariance_rows)
        gc.collect()

    del df_clean_std, df_train_std
    gc.collect()

    for size in args.fixed_count_sizes:
        scheme_name = f"fixed_count_{size}"
        print(f"  scheme {scheme_name}, elapsed: {time.time() - t0:.0f}s", flush=True)
        sweep_rows, invariance_rows = process_count_scheme(dataset, run_config, scheme_name, size, df_clean_pos, df_train_pos, vocabulary, per_fault_timing)
        all_sweep_rows.extend(sweep_rows)
        all_invariance_rows.extend(invariance_rows)
        gc.collect()

    del df_clean_pos, df_train_pos, per_fault_timing
    gc.collect()

    print(f"  {dataset} done, elapsed: {time.time() - t0:.0f}s", flush=True)
    return all_sweep_rows, all_invariance_rows, equivalence_checks


def check_fixed_count_invariance(invariance_df):
    fixed_count = invariance_df[invariance_df["scheme_kind"] == "fixed_count"]
    return fixed_count[(fixed_count["max_abs_score_diff"] > 0) | (fixed_count["n_differing_windows"] > 0)]


def write_markdown(sweep_df, invariance_df, equivalence_df, bad_cells, args):
    lines = [
        "# Windowing sweep -- extended window-size grid",
        "",
        f"dataset={args.dataset}, fixed_time_sizes={args.fixed_time_sizes}, "
        f"fixed_count_sizes={args.fixed_count_sizes}, subsample={args.subsample}, "
        f"subsample_include_injected={args.subsample_include_injected}, seed={args.seed}",
        "",
        "## std/pos row-order equivalence checks",
        "",
        equivalence_df.to_markdown(index=False),
        "",
        "## 1. Invariance: count_vector_pca / isolation_forest_counts, clean vs injected, every scheme",
        "",
        invariance_df.to_markdown(index=False),
        "",
        f"All fixed-count cells exactly zero (max_abs_score_diff, n_differing_windows): {bad_cells.empty}",
    ]
    if not bad_cells.empty:
        lines += [
            "",
            "**CRITICAL: nonzero invariance score on a fixed-count cell -- this is a likely "
            "timestamp-dependence bug in windowing/count-matrix construction, not a finding.**",
            "",
            bad_cells.to_markdown(index=False),
        ]
    lines += [
        "",
        "## 2. Detection: AUC-ROC (and full precision/recall/lift) per detector, every scheme",
        "",
        sweep_df.to_markdown(index=False),
    ]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["bgl", "thunderbird", "both"], default="both")
    ap.add_argument("--fixed-time-sizes", type=int, nargs="+", default=FIXED_TIME_SIZES_DEFAULT)
    ap.add_argument("--fixed-count-sizes", type=int, nargs="+", default=FIXED_COUNT_SIZES_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--subsample-include-injected", type=int, default=None)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    return ap.parse_args()


def main():
    args = parse_args()
    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "both" else [args.dataset]

    tmp_dir = Path(tempfile.mkdtemp(prefix="windowing_sweep_extended_")) if args.subsample is not None else None

    all_sweep_rows, all_invariance_rows, all_equivalence = [], [], []
    try:
        for dataset in datasets:
            sweep_rows, invariance_rows, equivalence = run_dataset(dataset, DATASET_CONFIG[dataset], args, tmp_dir)
            all_sweep_rows.extend(sweep_rows)
            all_invariance_rows.extend(invariance_rows)
            all_equivalence.extend(equivalence)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    sweep_df = pd.DataFrame(all_sweep_rows)
    invariance_df = pd.DataFrame(all_invariance_rows)
    equivalence_df = pd.DataFrame(all_equivalence)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    bad_cells = check_fixed_count_invariance(invariance_df)
    write_markdown(sweep_df, invariance_df, equivalence_df, bad_cells, args)
    print(f"Wrote {args.out_md}")

    if not bad_cells.empty:
        print("\nCRITICAL: nonzero invariance score on a fixed-count cell -- treat as a bug, not a finding:", flush=True)
        print(bad_cells.to_string(index=False), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
