"""
Multi-seed stability harness for the threshold-independent AUC-PR / AUC-ROC results in
results/auc_metrics.csv. Re-plans and re-applies the n=100 stall/burst injections under N
different seeds -- same node-eligibility rule, same rate/density gate (MAX_MAD_EFF_S), same
intensity/burst-length choices, same fixed-time 60s eval grid, same four detectors -- to check
whether a single-seed AUC-ROC number is stable or an artifact of which 100 nodes/positions that
one seed happened to draw.

Reuses the existing injection and scoring code paths unmodified and in-memory (no new parquet/CSV
files are written for the injected data itself): src.injector.{plan_injections, apply_injections,
plan_burst_injections, apply_burst_injections, label_spans_on_grid} for injection, and
src.auc_metrics.fit_count_detector_scores / src.core_result_stall.build_timing_features /
src.core_result_thunderbird.build_timing_features_tb / src.core_result_v2_symmetric.
compute_clean_train_baselines / src.detectors.timing_detector.{ZScoreThresholdDetector,
LogRatioThresholdDetector, add_log_ratio_feature} / src.auc_metrics.compute_grid_auc for scoring --
identical to what produced results/auc_metrics.csv, just called once per seed instead of once.
Does not modify results/auc_metrics.csv or any other existing result file.

Clean data and its chronological train/test split and per-node baselines are loaded/computed once
per dataset (they do not depend on seed); only injection planning and application are repeated per
seed. Seeds run sequentially, with `del` + `gc.collect()` between seeds, and both output files are
rewritten after every seed so a crash mid-run keeps every seed already completed.

Server run command (full 5-seed run, both datasets, both fault types):
    python src/multiseed_auc.py --n-seeds 5 --dataset both --fault both

Smoke test (small subsample, 2 seeds, verifies the harness end-to-end without real numbers):
    python src/multiseed_auc.py --n-seeds 2 --subsample 300000 --n-injections 10
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.auc_metrics import compute_grid_auc, fit_count_detector_scores
from src.core_result_stall import build_timing_features as build_timing_features_bgl
from src.core_result_thunderbird import build_timing_features_tb, chronological_split as chronological_split_tb
from src.core_result_v2_symmetric import compute_clean_train_baselines
from src.detectors.timing_detector import LogRatioThresholdDetector, ZScoreThresholdDetector, add_log_ratio_feature
from src.injector import (
    BURST_MIN_NODE_EVENTS,
    MAX_MAD_EFF_S,
    MIN_NODE_EVENTS,
    N_INJECTIONS,
    apply_burst_injections,
    apply_injections,
    eligible_nodes,
    label_spans_on_grid,
    plan_burst_injections,
    plan_injections,
)
from src.injector_thunderbird import build_baselines as build_baselines_tb
from src.metrics import assign_eval_grid
from src.run_baseline_detectors import chronological_split as chronological_split_bgl, load as load_bgl_clean
from src.timing_baseline import add_sequence_context, compute_node_baselines

OUT_CSV = Path("results/multiseed_auc.csv")
OUT_MD = Path("results/multiseed_auc.md")

DETECTORS = ["count_vector_pca", "isolation_forest_counts", "z_score_threshold", "log_ratio_threshold"]
AUDIT_COLS = ["prev_template", "next_template", "prev_anomaly", "gap_prev_s", "gap_next_s"]
CHANCE = 0.5


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--dataset", choices=["bgl", "thunderbird", "both"], default="both")
    ap.add_argument("--fault", choices=["stall", "burst", "both"], default="both")
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--n-injections", type=int, default=None)
    return ap.parse_args()


def resolve_seeds(args):
    if args.seeds is not None:
        return args.seeds
    return list(range(42, 42 + args.n_seeds))


def resolve_datasets(args):
    return ["bgl", "thunderbird"] if args.dataset == "both" else [args.dataset]


def resolve_faults(args):
    return ["stall", "burst"] if args.fault == "both" else [args.fault]


def build_bgl_context(subsample):
    df_clean_raw = load_bgl_clean(limit=subsample)
    is_train, _ = chronological_split_bgl(df_clean_raw)

    df_for_planning = add_sequence_context(
        df_clean_raw.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True).copy()
    )
    normal_seq = (~df_for_planning["anomaly"]) & (~df_for_planning["prev_anomaly"].fillna(True)) & df_for_planning["gap_prev_s"].notna()
    node_baselines = compute_node_baselines(df_for_planning, normal_seq)

    return {
        "name": "BGL",
        "df_clean_raw": df_clean_raw,
        "is_train": is_train,
        "df_for_planning": df_for_planning,
        "node_baselines": node_baselines,
        "require_valid_baseline": False,
        "build_timing_features": build_timing_features_bgl,
    }


def build_thunderbird_context(subsample):
    df_clean_raw = pd.read_parquet("data/processed/thunderbird_parsed.parquet")
    df_clean_raw = df_clean_raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if subsample is not None:
        df_clean_raw = df_clean_raw.iloc[:subsample].reset_index(drop=True)
    df_clean_raw["row_id"] = np.arange(len(df_clean_raw))
    is_train, _ = chronological_split_tb(df_clean_raw)

    df_for_planning = add_sequence_context(
        df_clean_raw.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True).copy()
    )
    node_baselines = build_baselines_tb(df_for_planning)

    return {
        "name": "Thunderbird",
        "df_clean_raw": df_clean_raw,
        "is_train": is_train,
        "df_for_planning": df_for_planning,
        "node_baselines": node_baselines,
        "require_valid_baseline": True,
        "build_timing_features": build_timing_features_tb,
    }


def clamp_n_injections(requested, eligible_count, label):
    if requested is None:
        return None
    n = min(requested, eligible_count)
    if n < requested:
        print(f"    {label}: requested n_injections={requested} exceeds {eligible_count} eligible nodes, clamped to {n}")
    return n


def plan_and_apply(ctx, fault, seed, n_injections_override):
    min_node_events = MIN_NODE_EVENTS if fault == "stall" else BURST_MIN_NODE_EVENTS
    eligible_count = len(
        eligible_nodes(
            ctx["df_for_planning"],
            min_events=min_node_events,
            node_baselines=ctx["node_baselines"],
            require_valid_baseline=ctx["require_valid_baseline"],
            max_mad_eff=MAX_MAD_EFF_S,
        )
    )
    n_injections = clamp_n_injections(n_injections_override or N_INJECTIONS, eligible_count, f"{ctx['name']}/{fault}")
    n_injections = n_injections or N_INJECTIONS

    if fault == "stall":
        plans = plan_injections(
            ctx["df_for_planning"],
            ctx["node_baselines"],
            seed=seed,
            n_injections=n_injections,
            min_node_events=min_node_events,
            require_valid_baseline=ctx["require_valid_baseline"],
            max_mad_eff=MAX_MAD_EFF_S,
        )
        df_injected, labels_df = apply_injections(ctx["df_for_planning"], plans, ctx["node_baselines"])
    else:
        plans = plan_burst_injections(
            ctx["df_for_planning"],
            ctx["node_baselines"],
            seed=seed,
            n_injections=n_injections,
            min_node_events=min_node_events,
            require_valid_baseline=ctx["require_valid_baseline"],
            max_mad_eff=MAX_MAD_EFF_S,
        )
        df_injected, labels_df = apply_burst_injections(ctx["df_for_planning"], plans)

    grid_labels_df = label_spans_on_grid(labels_df, df_injected)
    df_injected_scored = df_injected.drop(columns=AUDIT_COLS)
    return df_injected_scored, grid_labels_df


def row_true_from_grid_labels(df_eval, grid_labels_df):
    anomalous_cells = set(zip(grid_labels_df["node"], grid_labels_df["window_idx"]))
    row_true = pd.Series([k in anomalous_cells for k in df_eval["eval_window_key"]], index=df_eval["row_id"].to_numpy())
    return df_eval["row_id"].map(row_true).fillna(False)


def score_one_seed(ctx, fault, seed, n_injections_override):
    df_injected, grid_labels_df = plan_and_apply(ctx, fault, seed, n_injections_override)

    df_clean_train = ctx["df_clean_raw"].loc[ctx["is_train"]].reset_index(drop=True)
    count_scores = fit_count_detector_scores(df_clean_train, df_injected)

    if ctx["name"] == "BGL":
        _, features_injected = ctx["build_timing_features"](ctx["df_clean_raw"], ctx["is_train"], df_injected)
        baselines = compute_clean_train_baselines(ctx["df_clean_raw"], ctx["is_train"])
        features_injected = add_log_ratio_feature(features_injected, baselines)
    else:
        _, features_injected, _ = ctx["build_timing_features"](ctx["df_clean_raw"], ctx["is_train"], df_injected)

    zscore_scores = pd.Series(ZScoreThresholdDetector().score(features_injected).to_numpy(), index=features_injected["row_id"].to_numpy())
    logratio_scores = pd.Series(LogRatioThresholdDetector().score(features_injected).to_numpy(), index=features_injected["row_id"].to_numpy())
    scores_by_detector = {**count_scores, "z_score_threshold": zscore_scores, "log_ratio_threshold": logratio_scores}

    df_eval = assign_eval_grid(df_injected)
    row_true_aligned = row_true_from_grid_labels(df_eval, grid_labels_df)

    rows = []
    for detector_name in DETECTORS:
        metrics = compute_grid_auc(df_eval, scores_by_detector[detector_name], row_true_aligned)
        rows.append(
            {
                "dataset": ctx["name"],
                "fault_type": fault,
                "seed": seed,
                "detector": detector_name,
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "no_skill_baseline": metrics["no_skill_baseline"],
                "auc_pr_ratio": metrics["auc_pr_ratio"],
                "n_eval_cells": metrics["n_eval_cells"],
                "n_positive_cells": metrics["n_positive_cells"],
            }
        )

    del df_injected, grid_labels_df, count_scores, features_injected, zscore_scores, logratio_scores, scores_by_detector, df_eval, row_true_aligned
    gc.collect()
    return rows


def aggregate(raw_df):
    grouped = raw_df.groupby(["dataset", "fault_type", "detector"])["auc_roc"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary["auc_roc_min"] = grouped.min().to_numpy()
    summary["auc_roc_max"] = grouped.max().to_numpy()
    summary["crosses_chance"] = (summary["mean"] - summary["std"] < CHANCE) & (summary["mean"] + summary["std"] > CHANCE)
    return summary


def write_outputs(raw_df, summary_df, seeds, args, elapsed_s, subsampled):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# Multi-seed AUC-ROC stability")
    lines.append("")
    if subsampled:
        lines.append(
            f"**SMOKE-TEST OUTPUT -- subsample={args.subsample}, n_injections_override={args.n_injections}. "
            "Numbers below are NOT representative of real detector performance; this run only verifies "
            "the harness executes end-to-end and the schema/aggregation are correct. Re-run without "
            "`--subsample`/`--n-injections` on the server for real numbers.**"
        )
        lines.append("")
    lines.append(f"Seeds used: {seeds}")
    lines.append("")
    lines.append("Server run command (full 5-seed run, both datasets, both fault types):")
    lines.append("")
    lines.append("```")
    lines.append("python src/multiseed_auc.py --n-seeds 5 --dataset both --fault both")
    lines.append("```")
    lines.append("")
    lines.append(f"Seeds completed so far: {sorted(raw_df['seed'].unique().tolist())}, elapsed {elapsed_s:.1f}s")
    lines.append("")
    lines.append("| dataset | fault | detector | mean AUC-ROC | std | min | max | n seeds | crosses chance (0.5)? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summary_df.sort_values(["dataset", "fault_type", "detector"]).iterrows():
        flag = "**YES**" if r["crosses_chance"] else "no"
        lines.append(
            f"| {r['dataset']} | {r['fault_type']} | {r['detector']} | {r['mean']:.4f} | {r['std']:.4f} | "
            f"{r['auc_roc_min']:.4f} | {r['auc_roc_max']:.4f} | {int(r['count'])} | {flag} |"
        )
    lines.append("")
    lines.append(f"Does NOT overwrite results/auc_metrics.csv. Raw per-seed values: {OUT_CSV}.")

    OUT_MD.write_text("\n".join(lines))


def main():
    args = parse_args()
    seeds = resolve_seeds(args)
    datasets = resolve_datasets(args)
    faults = resolve_faults(args)
    subsampled = args.subsample is not None or args.n_injections is not None

    t0 = time.time()
    contexts = {}
    if "bgl" in datasets:
        print("Loading BGL context...", flush=True)
        contexts["bgl"] = build_bgl_context(args.subsample)
    if "thunderbird" in datasets:
        print("Loading Thunderbird context...", flush=True)
        contexts["thunderbird"] = build_thunderbird_context(args.subsample)
    print(f"Contexts ready, elapsed {time.time() - t0:.1f}s", flush=True)

    all_rows = []
    for seed in seeds:
        for dataset_key in datasets:
            ctx = contexts[dataset_key]
            for fault in faults:
                print(f"seed={seed} dataset={ctx['name']} fault={fault} elapsed={time.time() - t0:.1f}s", flush=True)
                rows = score_one_seed(ctx, fault, seed, args.n_injections)
                all_rows.extend(rows)

        raw_df = pd.DataFrame(all_rows)
        summary_df = aggregate(raw_df)
        write_outputs(raw_df, summary_df, seeds, args, time.time() - t0, subsampled)
        print(f"Wrote partial results after seed={seed}, {len(all_rows)} rows so far", flush=True)
        gc.collect()

    print(f"\nDone. Total elapsed {time.time() - t0:.1f}s", flush=True)
    print(summary_df.sort_values(["dataset", "fault_type", "detector"]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
