"""
Slowdown fault (gradual stretch), used for the invariance check on BGL and Spirit.

Planning follows src.injector.plan_burst_injections (node, start position, run of L consecutive
gaps, intensity), with the same eligibility gate, n=100, run-length and intensity choices. For the
L gaps in the run (i = 1..L), each gap is multiplied by a factor rising linearly from 1 to the
intensity:
    factor(i) = 1 + (intensity - 1) * (i - 1) / (L - 1)      [L > 1]
    factor(i) = intensity                                     [L == 1]
new_gap(i) = orig_gap(i) * factor(i) >= orig_gap(i), so timestamps inside the run stay strictly
increasing, and every later event on the node is shifted forward by the total time added (a
uniform shift, so order in the trailing region is preserved).

Because a slowdown multiplies gaps by up to the maximum intensity, a run on a sparse node can
stretch far beyond the one-hour transient-fault bound (src.injector.MAX_PLAUSIBLE_FAULT_DURATION_S).
Candidates whose projected stretched span exceeds the bound are rejected, and run length,
intensity, and position are redrawn jointly. If no draw within MAX_START_POS_TRIES clears the
bound, the smallest-span candidate is used and the injection is labeled
exceeds_plausible_duration=True in the output labels.

SLOWDOWN_SEED (44) differs from the stall (42) and burst (43) seeds so the three fault types draw
independent nodes and positions.

--dataset bgl uses the BGL settings of src/injector.py; --dataset spirit uses
require_valid_baseline=True and exclude_zero_from_pooled=True, as in src/injector_spirit.py.

Writes data/processed/{dataset}_injected_slowdown.parquet,
{dataset}_injection_labels_slowdown.csv, {dataset}_injection_config_slowdown.json,
{dataset}_injection_grid_labels_slowdown.csv.

Usage:
    python src/injector_slowdown.py --dataset bgl
    python src/injector_slowdown.py --dataset spirit
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.injector import (
    BURST_LENGTH_CHOICES,
    BURST_MIN_NODE_EVENTS,
    BURST_START_FRACTION_RANGE,
    INTENSITY_CHOICES,
    MAD_FLOOR_SEC,
    MAX_MAD_EFF_S,
    MAX_ORIGINAL_GAP_Z,
    MAX_PLAUSIBLE_FAULT_DURATION_S,
    MAX_START_POS_TRIES,
    N_INJECTIONS,
    eligible_nodes,
    label_spans_on_grid,
)
from src.metrics import EVAL_WINDOW_SCHEME, EVAL_WINDOW_SIZE
from src.timing_baseline import add_sequence_context, compute_node_baselines

SLOWDOWN_SEED = 44

DATASET_PATHS = {
    "bgl": {
        "in_path": Path("data/processed/bgl_parsed.parquet"),
        "require_valid_baseline": False,
        "exclude_zero_from_pooled": False,
    },
    "spirit": {
        "in_path": Path("data/processed/spirit_parsed.parquet"),
        "require_valid_baseline": True,
        "exclude_zero_from_pooled": True,
    },
}


def load_clean(in_path):
    df = pd.read_parquet(in_path)
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def build_baselines(df_clean, exclude_zero_from_pooled):
    normal_seq = (~df_clean["anomaly"]) & (~df_clean["prev_anomaly"].fillna(True)) & df_clean["gap_prev_s"].notna()
    return compute_node_baselines(df_clean, normal_seq, exclude_zero_from_pooled=exclude_zero_from_pooled)


def projected_stretched_span(gaps, intensity):
    L = len(gaps)
    total = 0.0
    for j, g in enumerate(gaps):
        i = j + 1
        factor = intensity if L == 1 else 1.0 + (intensity - 1.0) * (i - 1) / (L - 1)
        total += g * factor
    return total


def plan_slowdown_injections(df, node_baselines, seed, n_injections, min_node_events, require_valid_baseline, max_mad_eff):
    master_rng = np.random.default_rng(seed)

    nodes = sorted(
        eligible_nodes(df, min_events=min_node_events, node_baselines=node_baselines, require_valid_baseline=require_valid_baseline, max_mad_eff=max_mad_eff)
    )
    chosen_nodes = master_rng.choice(nodes, size=n_injections, replace=False)
    injection_seeds = master_rng.integers(0, 2**31 - 1, size=n_injections)

    node_sizes = df.groupby("node").size()

    plans = []
    for i in range(n_injections):
        node = chosen_nodes[i]
        inj_seed = int(injection_seeds[i])
        inj_rng = np.random.default_rng(inj_seed)

        use_node_baseline = node in node_baselines["valid_nodes"]
        median = node_baselines["node_median"][node] if use_node_baseline else node_baselines["global_median"]
        mad = node_baselines["node_mad"][node] if use_node_baseline else node_baselines["global_mad"]
        mad_eff = max(mad * 1.4826, MAD_FLOOR_SEC)
        n = int(node_sizes[node])
        node_positions = df.index[df["node"] == node].to_numpy()

        start_pos, run_length, intensity = None, None, None
        best_candidate, best_run_length, best_intensity, best_span = None, None, None, float("inf")
        for _ in range(MAX_START_POS_TRIES):
            trial_run_length = int(inj_rng.choice(BURST_LENGTH_CHOICES))
            trial_intensity = float(inj_rng.choice(INTENSITY_CHOICES))
            lo = int(n * BURST_START_FRACTION_RANGE[0])
            hi = int(n * BURST_START_FRACTION_RANGE[1])
            hi = max(hi, lo + 1)
            hi = min(hi, n - trial_run_length - 1)
            candidate = int(inj_rng.integers(lo, hi))
            gaps = [df.at[node_positions[candidate + 1 + j], "gap_prev_s"] for j in range(trial_run_length)]
            zs = [abs(g - median) / mad_eff for g in gaps]
            projected_span = projected_stretched_span(gaps, trial_intensity)
            if projected_span < best_span:
                best_candidate, best_run_length, best_intensity, best_span = candidate, trial_run_length, trial_intensity, projected_span
            if max(zs) <= MAX_ORIGINAL_GAP_Z and projected_span <= MAX_PLAUSIBLE_FAULT_DURATION_S:
                start_pos, run_length, intensity = candidate, trial_run_length, trial_intensity
                break
        exceeds_plausible_duration = start_pos is None
        if start_pos is None:
            start_pos, run_length, intensity = best_candidate, best_run_length, best_intensity
        plans.append(
            {
                "injection_id": i,
                "node": node,
                "start_pos": start_pos,
                "burst_length": run_length,
                "intensity": intensity,
                "seed": inj_seed,
                "exceeds_plausible_duration": exceeds_plausible_duration,
            }
        )
    return plans


def apply_slowdown_injections(df, plans):
    df = df.copy()
    df["injected_row"] = False
    labels = []

    for plan in plans:
        node, start_pos, L, intensity = plan["node"], plan["start_pos"], plan["burst_length"], plan["intensity"]
        node_positions = df.index[df["node"] == node].to_numpy()
        i_global = node_positions[start_pos]
        run_positions = node_positions[start_pos + 1 : start_pos + 1 + L]

        t_prev_orig = df.at[i_global, "timestamp"]
        t_start = t_prev_orig
        new_ts = []
        t_prev_new = t_prev_orig
        total_added_us = 0
        for j, pos in enumerate(run_positions):
            i = j + 1
            factor = intensity if L == 1 else 1.0 + (intensity - 1.0) * (i - 1) / (L - 1)
            t_orig = df.at[pos, "timestamp"]
            orig_gap_us = round((t_orig - t_prev_orig).total_seconds() * 1_000_000)
            new_gap_us = round(orig_gap_us * factor)
            total_added_us += new_gap_us - orig_gap_us
            t_new = t_prev_new + pd.Timedelta(microseconds=new_gap_us)
            new_ts.append(t_new)
            t_prev_orig, t_prev_new = t_orig, t_new

        t_end = new_ts[-1]
        df.loc[run_positions, "timestamp"] = pd.Series(new_ts, index=run_positions).astype(df["timestamp"].dtype)
        df.loc[run_positions, "injected_row"] = True

        trailing_positions = node_positions[start_pos + 1 + L :]
        if len(trailing_positions) > 0:
            shift = pd.Timedelta(microseconds=total_added_us)
            df.loc[trailing_positions, "timestamp"] = (df.loc[trailing_positions, "timestamp"] + shift).astype(df["timestamp"].dtype)

        labels.append(
            {
                "injection_id": plan["injection_id"],
                "node": node,
                "type": "slowdown",
                "start": t_start,
                "end": t_end,
                "intensity": intensity,
                "seed": plan["seed"],
                "run_length": L,
                "total_time_added_s": total_added_us / 1_000_000,
                "stretched_span_s": (t_end - t_start).total_seconds(),
                "start_pos": start_pos,
                "node_n_events": len(node_positions),
                "exceeds_plausible_duration": plan.get("exceeds_plausible_duration", False),
            }
        )
    return df, pd.DataFrame(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["bgl", "spirit"], required=True)
    args = ap.parse_args()

    spec = DATASET_PATHS[args.dataset]
    df_clean = load_clean(spec["in_path"])
    df_clean = add_sequence_context(df_clean)
    node_baselines = build_baselines(df_clean, spec["exclude_zero_from_pooled"])

    n_eligible = len(
        eligible_nodes(
            df_clean,
            min_events=BURST_MIN_NODE_EVENTS,
            node_baselines=node_baselines,
            require_valid_baseline=spec["require_valid_baseline"],
            max_mad_eff=MAX_MAD_EFF_S,
        )
    )
    print(f"Eligible nodes (size>={BURST_MIN_NODE_EVENTS} AND mad_eff<={MAX_MAD_EFF_S}s AND require_valid_baseline={spec['require_valid_baseline']}): {n_eligible:,}")

    plans = plan_slowdown_injections(
        df_clean,
        node_baselines,
        seed=SLOWDOWN_SEED,
        n_injections=N_INJECTIONS,
        min_node_events=BURST_MIN_NODE_EVENTS,
        require_valid_baseline=spec["require_valid_baseline"],
        max_mad_eff=MAX_MAD_EFF_S,
    )
    df_injected, labels_df = apply_slowdown_injections(df_clean, plans)
    grid_labels_df = label_spans_on_grid(labels_df, df_injected)

    audit_cols = ["prev_template", "next_template", "prev_anomaly", "gap_prev_s", "gap_next_s"]
    df_injected_out = df_injected.drop(columns=audit_cols)

    out_parquet = Path(f"data/processed/{args.dataset}_injected_slowdown.parquet")
    out_labels_csv = Path(f"data/processed/{args.dataset}_injection_labels_slowdown.csv")
    out_config_json = Path(f"data/processed/{args.dataset}_injection_config_slowdown.json")
    out_grid_labels_csv = Path(f"data/processed/{args.dataset}_injection_grid_labels_slowdown.csv")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_injected_out.to_parquet(out_parquet, index=False)
    labels_df.to_csv(out_labels_csv, index=False)
    grid_labels_df.to_csv(out_grid_labels_csv, index=False)

    config = {
        "seed": SLOWDOWN_SEED,
        "n_injections": len(plans),
        "intensity_choices": INTENSITY_CHOICES,
        "run_length_choices": BURST_LENGTH_CHOICES,
        "min_node_events": BURST_MIN_NODE_EVENTS,
        "require_valid_baseline": spec["require_valid_baseline"],
        "exclude_zero_from_pooled": spec["exclude_zero_from_pooled"],
        "max_plausible_fault_duration_s": MAX_PLAUSIBLE_FAULT_DURATION_S,
        "max_mad_eff_s": MAX_MAD_EFF_S,
        "n_eligible_nodes": n_eligible,
        "eval_grid_scheme": EVAL_WINDOW_SCHEME,
        "eval_grid_size_s": EVAL_WINDOW_SIZE,
        "input": str(spec["in_path"]),
    }
    out_config_json.write_text(json.dumps(config, indent=2))

    print(f"Injected {len(plans)} slowdowns across {len(plans)} distinct nodes.")
    print(json.dumps(config, indent=2))
    print(grid_labels_df.groupby("injection_id").size().to_string())
    print(f"Wrote {out_parquet}, {out_labels_csv}, {out_config_json}, {out_grid_labels_csv}")


if __name__ == "__main__":
    main()
