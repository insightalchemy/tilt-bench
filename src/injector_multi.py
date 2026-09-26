"""
Stall and burst injection for the additional Loghub datasets (src/multi_dataset_registry.py),
using the functions and constants of src/injector.py.

These datasets have no anomaly labels, and several have whole-second timestamps, so eligibility
always uses require_valid_baseline=True and exclude_zero_from_pooled=True (as for Thunderbird and
Spirit). n_injections = min(100, number of eligible streams). The whole-dataset ordering check
(src/validate_injection.py) runs before anything is written, and the script refuses to write output
if any violation is found. Exits with code 3 if no stream is eligible.

Writes (see src/multi_dataset_registry.py for exact paths):
  data/processed/<name>_injected_<type>.parquet, labels CSV, grid-labels CSV, config JSON

Usage:
    python src/injector_multi.py --dataset openstack --type stall
    python src/injector_multi.py --dataset openstack --type burst
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.injector import (
    BURST_INTENSITY_CHOICES,
    BURST_LENGTH_CHOICES,
    BURST_MIN_NODE_EVENTS,
    BURST_SEED,
    INTENSITY_CHOICES,
    MAX_MAD_EFF_S,
    MIN_NODE_EVENTS,
    SEED,
    apply_burst_injections,
    apply_injections,
    eligible_nodes,
    label_spans_on_grid,
    plan_burst_injections,
    plan_injections,
)
from src.metrics import EVAL_WINDOW_SCHEME, EVAL_WINDOW_SIZE
from src.multi_dataset_registry import (
    DATASETS,
    injected_path,
    injection_config_path,
    injection_grid_labels_path,
    injection_labels_path,
    parsed_path,
    require_known,
    results_dir,
)
from src.timing_baseline import add_sequence_context, compute_node_baselines
from src.validate_injection import check_ordering

MAX_INJECTIONS = 100


def load_clean(name: str):
    df = pd.read_parquet(parsed_path(name))
    df["anomaly"] = df["anomaly"].fillna(False).astype(bool)
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def inject(name: str, inj_type: str):
    df_clean = load_clean(name)
    df_clean = add_sequence_context(df_clean)

    normal_seq = (~df_clean["prev_anomaly"].fillna(True).astype(bool)) & df_clean["gap_prev_s"].notna()
    node_baselines = compute_node_baselines(df_clean, normal_seq, exclude_zero_from_pooled=True)

    min_events = MIN_NODE_EVENTS if inj_type == "stall" else BURST_MIN_NODE_EVENTS
    n_eligible = len(
        eligible_nodes(
            df_clean, min_events=min_events, node_baselines=node_baselines,
            require_valid_baseline=True, max_mad_eff=MAX_MAD_EFF_S,
        )
    )
    n_injections = min(MAX_INJECTIONS, n_eligible)
    print(f"[{name}/{inj_type}] eligible streams: {n_eligible:,} -- injecting n={n_injections}")
    if n_injections == 0:
        note = f"# {name} {inj_type} injection\n\nZero eligible streams after the rate/density + valid-baseline gate -- no injection possible.\n"
        out_dir = results_dir(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"injection_{inj_type}.md").write_text(note)
        print(note)
        return None

    if inj_type == "stall":
        plans = plan_injections(
            df_clean, node_baselines, seed=SEED, n_injections=n_injections, min_node_events=min_events,
            require_valid_baseline=True, max_mad_eff=MAX_MAD_EFF_S,
        )
        df_injected, labels_df = apply_injections(df_clean, plans, node_baselines)
        config = {
            "seed": SEED, "n_injections": n_injections, "intensity_choices": INTENSITY_CHOICES,
            "min_node_events": min_events, "max_mad_eff_s": MAX_MAD_EFF_S, "n_eligible_streams": n_eligible,
        }
    else:
        plans = plan_burst_injections(
            df_clean, node_baselines, seed=BURST_SEED, n_injections=n_injections, min_node_events=min_events,
            require_valid_baseline=True, max_mad_eff=MAX_MAD_EFF_S,
        )
        df_injected, labels_df = apply_burst_injections(df_clean, plans)
        config = {
            "seed": BURST_SEED, "n_injections": n_injections, "intensity_choices": BURST_INTENSITY_CHOICES,
            "burst_length_choices": BURST_LENGTH_CHOICES, "min_node_events": min_events,
            "max_mad_eff_s": MAX_MAD_EFF_S, "n_eligible_streams": n_eligible,
        }
    config.update({"eval_grid_scheme": EVAL_WINDOW_SCHEME, "eval_grid_size_s": EVAL_WINDOW_SIZE, "dataset": name, "type": inj_type})

    ordering = check_ordering(df_injected)
    print(f"[{name}/{inj_type}] ordering check: {ordering}")
    if ordering["n_violations"] != 0:
        raise RuntimeError(
            f"{name}/{inj_type}: {ordering['n_violations']} ordering violations across "
            f"{ordering['violation_node_count']} streams -- injector propagation bug, refusing to write output."
        )

    grid_labels_df = label_spans_on_grid(labels_df, df_injected)

    audit_cols = ["prev_template", "next_template", "prev_anomaly", "gap_prev_s", "gap_next_s"]
    df_injected_out = df_injected.drop(columns=audit_cols)

    out_parquet = injected_path(name, inj_type)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_injected_out.to_parquet(out_parquet, index=False)
    labels_df.to_csv(injection_labels_path(name, inj_type), index=False)
    grid_labels_df.to_csv(injection_grid_labels_path(name, inj_type), index=False)
    injection_config_path(name, inj_type).write_text(json.dumps(config, indent=2))

    print(f"Injected {len(plans)} {inj_type}s across {len(plans)} distinct streams.")
    print(f"Wrote {out_parquet}, {injection_labels_path(name, inj_type)}, {injection_config_path(name, inj_type)}, {injection_grid_labels_path(name, inj_type)}")
    return config


NO_ELIGIBLE_STREAMS_EXIT_CODE = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--type", choices=["stall", "burst"], default="stall")
    args = ap.parse_args()
    require_known(args.dataset)
    config = inject(args.dataset, args.type)
    if config is None:
        sys.exit(NO_ELIGIBLE_STREAMS_EXIT_CODE)


if __name__ == "__main__":
    main()
