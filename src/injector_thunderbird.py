"""
Port of src/injector.py to Thunderbird -- same design, same functions, reused unmodified except
for two additive, backward-compatible parameters added to the shared library
(src.timing_baseline.compute_node_baselines's exclude_zero_from_pooled, and
src.injector.{eligible_nodes,plan_injections,plan_burst_injections}'s require_valid_baseline),
both off by default so BGL's own invocation (`python src/injector.py`) is unaffected -- verified
by regression diff after the change.

PART 1 FIX: Thunderbird's pooled fallback baseline is itself degenerate (median=MAD=0) because
>50% of ALL gaps are exactly 0 (see results/thunderbird_setup_notes_v2.md). Two changes applied
here, both opt-in via the new parameters:
  1. compute_node_baselines(..., exclude_zero_from_pooled=True) -- the pooled fallback is computed
     from non-zero gaps only, so it's no longer degenerate.
  2. plan_injections/plan_burst_injections(..., require_valid_baseline=True) -- injection
     eligibility is restricted to nodes with a valid (non-degenerate) PER-NODE baseline, so no
     injection ever gets scaled against even the fixed pooled fallback; every injected fault's
     intensity is relative to that specific node's own measured variability, exactly as on BGL.

Everything else -- the stall/burst mechanics, seeds, intensities, eval grid -- is identical to
src/injector.py's BGL configuration; only the input/output paths and the two new flags differ.

Reads data/processed/thunderbird_parsed.parquet. Writes:
  data/processed/thunderbird_injected_stall.parquet / _burst.parquet
  data/processed/thunderbird_injection_labels_stall.csv / _burst.csv
  data/processed/thunderbird_injection_config_stall.json / _burst.json
  data/processed/thunderbird_injection_grid_labels_stall.csv / _burst.csv

Usage:
    python src/injector_thunderbird.py --type stall   # default
    python src/injector_thunderbird.py --type burst
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import numpy as np
import pandas as pd

from src.injector import apply_burst_injections, apply_injections, label_spans_on_grid, plan_burst_injections, plan_injections
from src.injector import (
    BURST_INTENSITY_CHOICES,
    BURST_LENGTH_CHOICES,
    BURST_MIN_NODE_EVENTS,
    BURST_SEED,
    INTENSITY_CHOICES,
    MIN_NODE_EVENTS,
    N_INJECTIONS,
    SEED,
)
from src.metrics import EVAL_WINDOW_SCHEME, EVAL_WINDOW_SIZE
from src.timing_baseline import add_sequence_context, compute_node_baselines

IN_PATH = Path("data/processed/thunderbird_parsed.parquet")

OUT_PARQUET = Path("data/processed/thunderbird_injected_stall.parquet")
OUT_LABELS_CSV = Path("data/processed/thunderbird_injection_labels_stall.csv")
OUT_CONFIG_JSON = Path("data/processed/thunderbird_injection_config_stall.json")
OUT_GRID_LABELS_CSV = Path("data/processed/thunderbird_injection_grid_labels_stall.csv")

OUT_BURST_PARQUET = Path("data/processed/thunderbird_injected_burst.parquet")
OUT_BURST_LABELS_CSV = Path("data/processed/thunderbird_injection_labels_burst.csv")
OUT_BURST_CONFIG_JSON = Path("data/processed/thunderbird_injection_config_burst.json")
OUT_BURST_GRID_LABELS_CSV = Path("data/processed/thunderbird_injection_grid_labels_burst.csv")


def load_clean():
    df = pd.read_parquet(IN_PATH)
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def build_baselines(df_clean):
    normal_seq = (~df_clean["anomaly"]) & (~df_clean["prev_anomaly"].fillna(True)) & df_clean["gap_prev_s"].notna()
    return compute_node_baselines(df_clean, normal_seq, exclude_zero_from_pooled=True)


def main_stall():
    df_clean = load_clean()
    df_clean = add_sequence_context(df_clean)
    node_baselines = build_baselines(df_clean)

    n_eligible = len(
        [n for n in df_clean.groupby("node").size()[lambda s: s >= MIN_NODE_EVENTS].index if n in node_baselines["valid_nodes"]]
    )
    print(f"Eligible nodes after fix (size>={MIN_NODE_EVENTS} AND valid per-node baseline): {n_eligible:,}")

    plans = plan_injections(df_clean, node_baselines, require_valid_baseline=True)
    df_injected, labels_df = apply_injections(df_clean, plans, node_baselines)

    grid_labels_df = label_spans_on_grid(labels_df, df_injected)

    audit_cols = ["prev_template", "next_template", "prev_anomaly", "gap_prev_s", "gap_next_s"]
    df_injected_out = df_injected.drop(columns=audit_cols)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_injected_out.to_parquet(OUT_PARQUET, index=False)
    labels_df.to_csv(OUT_LABELS_CSV, index=False)
    grid_labels_df.to_csv(OUT_GRID_LABELS_CSV, index=False)

    config = {
        "seed": SEED,
        "n_injections": N_INJECTIONS,
        "intensity_choices": INTENSITY_CHOICES,
        "min_node_events": MIN_NODE_EVENTS,
        "require_valid_baseline": True,
        "exclude_zero_from_pooled": True,
        "n_eligible_nodes": n_eligible,
        "eval_grid_scheme": EVAL_WINDOW_SCHEME,
        "eval_grid_size_s": EVAL_WINDOW_SIZE,
        "input": str(IN_PATH),
    }
    OUT_CONFIG_JSON.write_text(json.dumps(config, indent=2))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Injected {len(plans)} stalls across {len(plans)} distinct nodes.")
    print("\n=== injection config ===")
    print(json.dumps(config, indent=2))
    print("\n=== injection labels (ground truth) ===")
    print(labels_df.to_string(index=False))
    print("\nGrid cells labeled per injection (should be >=1 each):")
    print(grid_labels_df.groupby("injection_id").size().to_string())
    print(f"\nWrote {OUT_PARQUET}, {OUT_LABELS_CSV}, {OUT_CONFIG_JSON}, {OUT_GRID_LABELS_CSV}")


def main_burst():
    df_clean = load_clean()
    df_clean = add_sequence_context(df_clean)
    node_baselines = build_baselines(df_clean)

    n_eligible = len(
        [n for n in df_clean.groupby("node").size()[lambda s: s >= BURST_MIN_NODE_EVENTS].index if n in node_baselines["valid_nodes"]]
    )
    print(f"Eligible nodes after fix (size>={BURST_MIN_NODE_EVENTS} AND valid per-node baseline): {n_eligible:,}")

    plans = plan_burst_injections(df_clean, node_baselines, require_valid_baseline=True)
    df_injected, labels_df = apply_burst_injections(df_clean, plans)

    grid_labels_df = label_spans_on_grid(labels_df, df_injected)

    audit_cols = ["prev_template", "next_template", "prev_anomaly", "gap_prev_s", "gap_next_s"]
    df_injected_out = df_injected.drop(columns=audit_cols)

    OUT_BURST_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_injected_out.to_parquet(OUT_BURST_PARQUET, index=False)
    labels_df.to_csv(OUT_BURST_LABELS_CSV, index=False)
    grid_labels_df.to_csv(OUT_BURST_GRID_LABELS_CSV, index=False)

    config = {
        "seed": BURST_SEED,
        "n_injections": len(plans),
        "intensity_choices": BURST_INTENSITY_CHOICES,
        "burst_length_choices": BURST_LENGTH_CHOICES,
        "min_node_events": BURST_MIN_NODE_EVENTS,
        "require_valid_baseline": True,
        "exclude_zero_from_pooled": True,
        "n_eligible_nodes": n_eligible,
        "eval_grid_scheme": EVAL_WINDOW_SCHEME,
        "eval_grid_size_s": EVAL_WINDOW_SIZE,
        "input": str(IN_PATH),
    }
    OUT_BURST_CONFIG_JSON.write_text(json.dumps(config, indent=2))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Injected {len(plans)} bursts across {len(plans)} distinct nodes.")
    print("\n=== injection config ===")
    print(json.dumps(config, indent=2))
    print("\n=== injection labels (ground truth) ===")
    print(labels_df.to_string(index=False))
    print("\nGrid cells labeled per injection (should be >=1 each):")
    print(grid_labels_df.groupby("injection_id").size().to_string())
    print(f"\nWrote {OUT_BURST_PARQUET}, {OUT_BURST_LABELS_CSV}, {OUT_BURST_CONFIG_JSON}, {OUT_BURST_GRID_LABELS_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=["stall", "burst"], default="stall")
    args = ap.parse_args()
    main_stall() if args.type == "stall" else main_burst()
