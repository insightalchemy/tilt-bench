"""
Stall and burst injection for Spirit (see src/injector.py for the fault model).

Only the input/output paths and two options differ from the BGL configuration of src/injector.py
(same stall/burst mechanics, seeds, intensities, rate gate, and evaluation grid):
  1. compute_node_baselines(..., exclude_zero_from_pooled=True): with whole-second timestamps most
     gaps are exactly 0, so the pooled fallback is computed from non-zero gaps only.
  2. require_valid_baseline=True: only nodes with a valid (non-degenerate) per-node baseline are
     eligible, so every fault's intensity is relative to that node's own measured variability.

Reads data/processed/spirit_parsed.parquet. Writes:
  data/processed/spirit_injected_stall.parquet / _burst.parquet
  data/processed/spirit_injection_labels_stall.csv / _burst.csv
  data/processed/spirit_injection_config_stall.json / _burst.json
  data/processed/spirit_injection_grid_labels_stall.csv / _burst.csv

Usage:
    python src/injector_spirit.py --type stall   # default
    python src/injector_spirit.py --type burst
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.injector import apply_burst_injections, apply_injections, eligible_nodes, label_spans_on_grid, plan_burst_injections, plan_injections
from src.injector import (
    BURST_INTENSITY_CHOICES,
    BURST_LENGTH_CHOICES,
    BURST_MIN_NODE_EVENTS,
    BURST_SEED,
    INTENSITY_CHOICES,
    MAX_MAD_EFF_S,
    MAX_PLAUSIBLE_FAULT_DURATION_S,
    MIN_NODE_EVENTS,
    N_INJECTIONS,
    SEED,
)
from src.metrics import EVAL_WINDOW_SCHEME, EVAL_WINDOW_SIZE
from src.timing_baseline import add_sequence_context, compute_node_baselines

IN_PATH = Path("data/processed/spirit_parsed.parquet")

OUT_PARQUET = Path("data/processed/spirit_injected_stall.parquet")
OUT_LABELS_CSV = Path("data/processed/spirit_injection_labels_stall.csv")
OUT_CONFIG_JSON = Path("data/processed/spirit_injection_config_stall.json")
OUT_GRID_LABELS_CSV = Path("data/processed/spirit_injection_grid_labels_stall.csv")

OUT_BURST_PARQUET = Path("data/processed/spirit_injected_burst.parquet")
OUT_BURST_LABELS_CSV = Path("data/processed/spirit_injection_labels_burst.csv")
OUT_BURST_CONFIG_JSON = Path("data/processed/spirit_injection_config_burst.json")
OUT_BURST_GRID_LABELS_CSV = Path("data/processed/spirit_injection_grid_labels_burst.csv")


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
        eligible_nodes(df_clean, min_events=MIN_NODE_EVENTS, node_baselines=node_baselines, require_valid_baseline=True, max_mad_eff=MAX_MAD_EFF_S)
    )
    print(f"Eligible nodes after rate/density gate (size>={MIN_NODE_EVENTS} AND valid per-node baseline AND mad_eff<={MAX_MAD_EFF_S}s): {n_eligible:,}")

    plans = plan_injections(df_clean, node_baselines, require_valid_baseline=True, max_mad_eff=MAX_MAD_EFF_S)
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
        "max_plausible_fault_duration_s": MAX_PLAUSIBLE_FAULT_DURATION_S,
        "max_mad_eff_s": MAX_MAD_EFF_S,
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
    print("\nGrid cells labeled per injection (should be >=1 each):")
    print(grid_labels_df.groupby("injection_id").size().to_string())
    print(f"\nWrote {OUT_PARQUET}, {OUT_LABELS_CSV}, {OUT_CONFIG_JSON}, {OUT_GRID_LABELS_CSV}")


def main_burst():
    df_clean = load_clean()
    df_clean = add_sequence_context(df_clean)
    node_baselines = build_baselines(df_clean)

    n_eligible = len(
        eligible_nodes(
            df_clean, min_events=BURST_MIN_NODE_EVENTS, node_baselines=node_baselines, require_valid_baseline=True, max_mad_eff=MAX_MAD_EFF_S
        )
    )
    print(
        f"Eligible nodes after rate/density gate (size>={BURST_MIN_NODE_EVENTS} AND valid per-node baseline AND mad_eff<={MAX_MAD_EFF_S}s): {n_eligible:,}"
    )

    plans = plan_burst_injections(df_clean, node_baselines, require_valid_baseline=True, max_mad_eff=MAX_MAD_EFF_S)
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
        "max_plausible_fault_duration_s": MAX_PLAUSIBLE_FAULT_DURATION_S,
        "max_mad_eff_s": MAX_MAD_EFF_S,
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
    print("\nGrid cells labeled per injection (should be >=1 each):")
    print(grid_labels_df.groupby("injection_id").size().to_string())
    print(f"\nWrote {OUT_BURST_PARQUET}, {OUT_BURST_LABELS_CSV}, {OUT_BURST_CONFIG_JSON}, {OUT_BURST_GRID_LABELS_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=["stall", "burst"], default="stall")
    args = ap.parse_args()
    main_stall() if args.type == "stall" else main_burst()
