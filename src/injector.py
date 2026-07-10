"""
Timing-anomaly injector, two types, on BGL: stall (widen a gap) and burst (compress a run of gaps).

Stall model: pick a node and a point in its chronological event sequence; insert a large gap at
that point by shifting EVERY subsequent timestamp for that node forward by a constant delta
(seconds). This is a uniform offset applied to all downstream events of that node, which provably
preserves per-node ordering -- pairwise gaps among the shifted events are exactly unchanged (adding
the same constant to both sides of any inequality preserves it); only the single gap at the
injection point widens. This is the opposite of "widen one gap in isolation," which would only
shift ONE timestamp and leave everything after it unmoved, potentially inverting order.

Burst model (the inverse): pick a node, a point, and a run of L consecutive gaps; compress each of
those L gaps by a factor (new_gap = orig_gap / intensity -- always positive, no floor needed, unlike
an additive reduction which could drive a gap negative), then shift EVERY event after the burst
window backward by the total time saved. Order is preserved by the same constant-shift argument as
the stall (within the burst, each new_gap > 0 so timestamps strictly increase; the shift after the
window is a single negative constant applied uniformly, so relative order/gaps in the trailing
region are exactly unchanged, and the boundary is continuous since the shift equals exactly the
compressed window's own total duration change).

Intensity is relative, not absolute. For stalls: delta = intensity * node_scale, where node_scale is
the exact per-node MAD-derived scale (with the SAME pooled fallback for low-data nodes, ~33% of
them) already validated in src/timing_baseline.py and used throughout the audit and detectors -- an
intensity of 20 means "20x this node's own normal inter-arrival variability." For bursts: intensity
is a compression factor (new_gap = orig_gap / intensity) -- relative to each gap's own size, which
itself reflects that node's pacing, so it's relative in the same spirit without risking non-positive
gaps the way an additive MAD-based reduction would.

Reads data/processed/bgl_parsed.parquet (untouched -- never mutate raw/clean data). Writes, per type
(stall shown; burst is the same filenames with `_burst` instead of `_stall`):
  data/processed/bgl_injected_stall.parquet      -- injected data (adds an `injected_row` bool col)
  data/processed/injection_labels_stall.csv      -- ground truth: start, end, node, type, intensity, seed
  data/processed/injection_config_stall.json     -- resolved generation config, for reproducibility
  data/processed/injection_grid_labels_stall.csv -- (injection_id, node, window_idx) cells overlapped
                                                     by each span on the SAME eval grid as baseline_final.csv

Fully seeded: one master SEED drives node selection and a per-injection seed array; each
injection's own seed alone reproduces that injection's start position and intensity. Per
CLAUDE.md: "Ground-truth labels recorded as (start, end, node, type, intensity, seed)."

Usage:
    python src/injector.py --type stall   # default
    python src/injector.py --type burst
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import numpy as np
import pandas as pd

from src.metrics import EVAL_WINDOW_SCHEME, EVAL_WINDOW_SIZE
from src.timing_baseline import MAD_FLOOR_SEC, add_sequence_context, compute_node_baselines

IN_PATH = Path("data/processed/bgl_parsed.parquet")
OUT_PARQUET = Path("data/processed/bgl_injected_stall.parquet")
OUT_LABELS_CSV = Path("data/processed/injection_labels_stall.csv")
OUT_CONFIG_JSON = Path("data/processed/injection_config_stall.json")
OUT_GRID_LABELS_CSV = Path("data/processed/injection_grid_labels_stall.csv")

SEED = 42
N_INJECTIONS = 40
INTENSITY_CHOICES = [10, 15, 20, 25, 30]  # multiples of the node's MAD-derived scale -- easily changeable
MIN_NODE_EVENTS = 30  # node eligibility: enough events for real before/after context
START_FRACTION_RANGE = (0.2, 0.8)  # place the stall well inside the node's own sequence
MAX_ORIGINAL_GAP_Z = 3.0  # reject injection points whose PRE-EXISTING gap already looks unusual
MAX_START_POS_TRIES = 200  # bounded retries before falling back to any position in range

BURST_SEED = 43  # distinct from stall's SEED, for an independently reproducible burst experiment
BURST_N_INJECTIONS = 40
BURST_INTENSITY_CHOICES = [10, 15, 20, 25, 30]  # compression factor: new_gap = orig_gap / intensity
BURST_LENGTH_CHOICES = [10, 15, 20, 25]  # number of consecutive gaps compressed per injection
BURST_MIN_NODE_EVENTS = MIN_NODE_EVENTS + max(BURST_LENGTH_CHOICES)  # room for the whole burst + margin
BURST_START_FRACTION_RANGE = (0.2, 0.7)  # leave room for burst_length + trailing events

OUT_BURST_PARQUET = Path("data/processed/bgl_injected_burst.parquet")
OUT_BURST_LABELS_CSV = Path("data/processed/injection_labels_burst.csv")
OUT_BURST_CONFIG_JSON = Path("data/processed/injection_config_burst.json")
OUT_BURST_GRID_LABELS_CSV = Path("data/processed/injection_grid_labels_burst.csv")


def load_clean():
    df = pd.read_parquet(IN_PATH)
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def eligible_nodes(df, min_events=MIN_NODE_EVENTS, node_baselines=None, require_valid_baseline=False):
    """require_valid_baseline=True additionally restricts to nodes with a non-degenerate per-node
    timing baseline (node_baselines["valid_nodes"]) -- needed on datasets where the pooled fallback
    itself is degenerate (see src.timing_baseline.compute_node_baselines's exclude_zero_from_pooled),
    so a low-data node never gets an injection intensity scaled against a broken (zero) baseline.
    Off by default -- BGL's pooled fallback is healthy, so this was never needed there."""
    counts = df.groupby("node").size()
    nodes = counts[counts >= min_events].index.tolist()
    if require_valid_baseline:
        if node_baselines is None:
            raise ValueError("require_valid_baseline=True requires node_baselines")
        nodes = [n for n in nodes if n in node_baselines["valid_nodes"]]
    return nodes


def plan_injections(
    df,
    node_baselines,
    seed=SEED,
    n_injections=N_INJECTIONS,
    min_node_events=MIN_NODE_EVENTS,
    require_valid_baseline=False,
):
    """Deterministic from `seed` alone. Node selection uses one draw from the master RNG (sampling
    without replacement needs shared state); each injection's start position and intensity are then
    drawn from that injection's OWN seeded RNG, so a single injection is independently regenerable
    given just (node, seed) without needing the rest of the batch.

    df must already have gap_prev_s (see src.timing_baseline.add_sequence_context), from the CLEAN
    (pre-injection) data -- start-position candidates are rejected if the node's PRE-EXISTING gap
    there is already an outlier relative to that node's own baseline (|z| > MAX_ORIGINAL_GAP_Z), so
    an injected stall lands on an otherwise-ordinary transition rather than stacking on top of some
    unrelated, already-anomalous natural gap.
    """
    master_rng = np.random.default_rng(seed)

    # sorted for determinism (set/groupby order isn't guaranteed)
    nodes = sorted(eligible_nodes(df, min_events=min_node_events, node_baselines=node_baselines, require_valid_baseline=require_valid_baseline))
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
        lo = int(n * START_FRACTION_RANGE[0])
        hi = int(n * START_FRACTION_RANGE[1])
        hi = max(hi, lo + 1)
        hi = min(hi, n - 1)  # must leave at least one event after the stall

        node_positions = df.index[df["node"] == node].to_numpy()
        start_pos = None
        for _ in range(MAX_START_POS_TRIES):
            candidate = int(inj_rng.integers(lo, hi))
            gap = df.at[node_positions[candidate + 1], "gap_prev_s"]
            if abs(gap - median) / mad_eff <= MAX_ORIGINAL_GAP_Z:
                start_pos = candidate
                break
        if start_pos is None:  # extremely unlikely given node sizes -- fall back rather than crash
            start_pos = int(inj_rng.integers(lo, hi))

        intensity = float(inj_rng.choice(INTENSITY_CHOICES))
        plans.append({"injection_id": i, "node": node, "start_pos": start_pos, "intensity": intensity, "seed": inj_seed})
    return plans


def apply_injections(df, plans, node_baselines):
    """Applies each planned stall by shifting all downstream same-node timestamps forward by a
    constant delta. Injections target distinct nodes (by construction of plan_injections), so
    application order doesn't matter -- each only touches its own node's rows."""
    df = df.copy()
    df["injected_row"] = False
    labels = []

    for plan in plans:
        node, start_pos, intensity = plan["node"], plan["start_pos"], plan["intensity"]
        node_positions = df.index[df["node"] == node].to_numpy()  # chronological, since df is (node,timestamp)-sorted
        i_global = node_positions[start_pos]
        next_global = node_positions[start_pos + 1]

        use_node_baseline = node in node_baselines["valid_nodes"]
        mad = node_baselines["node_mad"][node] if use_node_baseline else node_baselines["global_mad"]
        mad_eff = max(mad * 1.4826, MAD_FLOOR_SEC)
        delta_seconds = intensity * mad_eff

        t_start = df.at[i_global, "timestamp"]
        t_next_orig = df.at[next_global, "timestamp"]
        original_gap_s = (t_next_orig - t_start).total_seconds()

        # Build the delta directly in microseconds -- a Timedelta built from fractional seconds
        # defaults to nanosecond precision, which pandas 3.x refuses to downcast losslessly into
        # a coarser column dtype. Casting explicitly to the column's OWN dtype (datetime64[us] for
        # BGL, datetime64[ms] for Thunderbird -- whatever the parquet round-trip produced) makes
        # this portable across datasets instead of assuming BGL's specific resolution.
        delta = pd.Timedelta(microseconds=round(delta_seconds * 1_000_000))
        delta_seconds = delta.total_seconds()  # the exact applied value, for the ground-truth record
        shift_positions = node_positions[start_pos + 1 :]
        df.loc[shift_positions, "timestamp"] = (df.loc[shift_positions, "timestamp"] + delta).astype(df["timestamp"].dtype)

        t_next_new = df.at[next_global, "timestamp"]
        df.at[next_global, "injected_row"] = True

        labels.append(
            {
                "injection_id": plan["injection_id"],
                "node": node,
                "type": "stall",
                "start": t_start,
                "end": t_next_new,
                "intensity": intensity,
                "seed": plan["seed"],
                "node_mad_eff_s": mad_eff,
                "used_pooled_fallback": not use_node_baseline,
                "delta_seconds": delta_seconds,
                "original_gap_s": original_gap_s,
                "n_events_shifted": len(shift_positions),
                "start_pos": start_pos,
                "node_n_events": len(node_positions),
            }
        )
    return df, pd.DataFrame(labels)


def label_spans_on_grid(labels_df, df_injected, scheme=EVAL_WINDOW_SCHEME, size=EVAL_WINDOW_SIZE):
    """(injection_id, node, window_idx) for every eval-grid cell whose TIME RANGE overlaps an
    injected span -- computed analytically from each node's t0, not from existing rows. A stall's
    whole point is a quiet period with no rows in it, so a purely row-driven grid (groupby on rows
    that exist) can't represent the empty cells in the middle of a long stall; this can, and is
    the ground truth those future detector-vs-truth comparisons should join against."""
    if scheme != "fixed_time":
        raise NotImplementedError("span-based grid labeling is only implemented for the fixed_time scheme")

    node_t0 = df_injected.groupby("node")["timestamp"].min()
    rows = []
    for rec in labels_df.to_dict("records"):
        t0 = node_t0[rec["node"]]
        idx_start = int((rec["start"] - t0).total_seconds() // size)
        idx_end = int((rec["end"] - t0).total_seconds() // size)
        for idx in range(idx_start, idx_end + 1):
            rows.append({"injection_id": rec["injection_id"], "node": rec["node"], "window_idx": idx})
    return pd.DataFrame(rows)


def plan_burst_injections(
    df,
    node_baselines,
    seed=BURST_SEED,
    n_injections=BURST_N_INJECTIONS,
    min_node_events=BURST_MIN_NODE_EVENTS,
    require_valid_baseline=False,
):
    """Mirrors plan_injections, plus a burst_length draw. A candidate start position is rejected
    if ANY of its burst_length candidate gaps is already an outlier relative to the node's
    baseline (too large OR too small), so a burst compresses an otherwise-ordinary run of gaps
    rather than one that already contained something unusual."""
    master_rng = np.random.default_rng(seed)

    nodes = sorted(eligible_nodes(df, min_events=min_node_events, node_baselines=node_baselines, require_valid_baseline=require_valid_baseline))
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

        burst_length = int(inj_rng.choice(BURST_LENGTH_CHOICES))
        n = int(node_sizes[node])
        lo = int(n * BURST_START_FRACTION_RANGE[0])
        hi = int(n * BURST_START_FRACTION_RANGE[1])
        hi = max(hi, lo + 1)
        hi = min(hi, n - burst_length - 1)  # room for burst_length gaps + >=1 trailing event

        node_positions = df.index[df["node"] == node].to_numpy()
        start_pos = None
        for _ in range(MAX_START_POS_TRIES):
            candidate = int(inj_rng.integers(lo, hi))
            gaps = [df.at[node_positions[candidate + 1 + j], "gap_prev_s"] for j in range(burst_length)]
            zs = [abs(g - median) / mad_eff for g in gaps]
            if max(zs) <= MAX_ORIGINAL_GAP_Z:
                start_pos = candidate
                break
        if start_pos is None:  # extremely unlikely -- fall back rather than crash
            start_pos = int(inj_rng.integers(lo, hi))

        intensity = float(inj_rng.choice(BURST_INTENSITY_CHOICES))
        plans.append(
            {
                "injection_id": i,
                "node": node,
                "start_pos": start_pos,
                "burst_length": burst_length,
                "intensity": intensity,
                "seed": inj_seed,
            }
        )
    return plans


def apply_burst_injections(df, plans):
    """Compresses each planned burst window (new_gap = orig_gap / intensity, always positive) and
    shifts every downstream same-node event backward by the total time saved. Marks EVERY row
    inside the burst window as injected_row=True (unlike stall's single boundary row), since a
    burst affects a whole run of events, not one transition."""
    df = df.copy()
    df["injected_row"] = False
    labels = []

    for plan in plans:
        node, start_pos, L, intensity = plan["node"], plan["start_pos"], plan["burst_length"], plan["intensity"]
        node_positions = df.index[df["node"] == node].to_numpy()
        i_global = node_positions[start_pos]
        burst_positions = node_positions[start_pos + 1 : start_pos + 1 + L]

        t_prev_orig = df.at[i_global, "timestamp"]
        t_start = t_prev_orig
        new_ts = []
        t_prev_new = t_prev_orig
        total_saved_us = 0
        for pos in burst_positions:
            t_orig = df.at[pos, "timestamp"]
            orig_gap_us = round((t_orig - t_prev_orig).total_seconds() * 1_000_000)
            new_gap_us = round(orig_gap_us / intensity)
            total_saved_us += orig_gap_us - new_gap_us
            t_new = t_prev_new + pd.Timedelta(microseconds=new_gap_us)
            new_ts.append(t_new)
            t_prev_orig, t_prev_new = t_orig, t_new

        t_end = new_ts[-1]
        # Cast to the column's OWN dtype (datetime64[us] for BGL, datetime64[ms] for Thunderbird)
        # rather than a hardcoded unit, so this is portable across datasets.
        df.loc[burst_positions, "timestamp"] = pd.Series(new_ts, index=burst_positions).astype(df["timestamp"].dtype)
        df.loc[burst_positions, "injected_row"] = True

        trailing_positions = node_positions[start_pos + 1 + L :]
        if len(trailing_positions) > 0:
            shift = pd.Timedelta(microseconds=total_saved_us)
            df.loc[trailing_positions, "timestamp"] = (df.loc[trailing_positions, "timestamp"] - shift).astype(df["timestamp"].dtype)

        labels.append(
            {
                "injection_id": plan["injection_id"],
                "node": node,
                "type": "burst",
                "start": t_start,
                "end": t_end,
                "intensity": intensity,
                "seed": plan["seed"],
                "burst_length": L,
                "total_time_saved_s": total_saved_us / 1_000_000,
                "compressed_span_s": (t_end - t_start).total_seconds(),
                "start_pos": start_pos,
                "node_n_events": len(node_positions),
            }
        )
    return df, pd.DataFrame(labels)


def main_stall():
    df_clean = load_clean()
    df_clean = add_sequence_context(df_clean)

    normal_seq = (~df_clean["anomaly"]) & (~df_clean["prev_anomaly"].fillna(True)) & df_clean["gap_prev_s"].notna()
    node_baselines = compute_node_baselines(df_clean, normal_seq)

    plans = plan_injections(df_clean, node_baselines)
    df_injected, labels_df = apply_injections(df_clean, plans, node_baselines)

    grid_labels_df = label_spans_on_grid(labels_df, df_injected)

    # Drop the audit bookkeeping columns added by add_sequence_context -- injected_row is the only
    # new column the injected parquet should carry (gap_prev_s etc. get recomputed post-injection
    # by any consumer, since injection changes them).
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
        "start_fraction_range": list(START_FRACTION_RANGE),
        "eval_grid_scheme": EVAL_WINDOW_SCHEME,
        "eval_grid_size_s": EVAL_WINDOW_SIZE,
        "input": str(IN_PATH),
    }
    OUT_CONFIG_JSON.write_text(json.dumps(config, indent=2))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Injected {len(plans)} stalls across {len(plans)} distinct nodes.")
    print(f"\n=== injection config ===")
    print(json.dumps(config, indent=2))
    print(f"\n=== injection labels (ground truth) ===")
    print(labels_df.to_string(index=False))
    print(f"\nGrid cells labeled per injection (should be >=1 each):")
    print(grid_labels_df.groupby("injection_id").size().to_string())

    print(f"\nWrote {OUT_PARQUET}, {OUT_LABELS_CSV}, {OUT_CONFIG_JSON}, {OUT_GRID_LABELS_CSV}")


def main_burst():
    df_clean = load_clean()
    df_clean = add_sequence_context(df_clean)

    normal_seq = (~df_clean["anomaly"]) & (~df_clean["prev_anomaly"].fillna(True)) & df_clean["gap_prev_s"].notna()
    node_baselines = compute_node_baselines(df_clean, normal_seq)

    plans = plan_burst_injections(df_clean, node_baselines)
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
        "n_injections": BURST_N_INJECTIONS,
        "intensity_choices": BURST_INTENSITY_CHOICES,
        "burst_length_choices": BURST_LENGTH_CHOICES,
        "min_node_events": BURST_MIN_NODE_EVENTS,
        "start_fraction_range": list(BURST_START_FRACTION_RANGE),
        "eval_grid_scheme": EVAL_WINDOW_SCHEME,
        "eval_grid_size_s": EVAL_WINDOW_SIZE,
        "input": str(IN_PATH),
    }
    OUT_BURST_CONFIG_JSON.write_text(json.dumps(config, indent=2))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Injected {len(plans)} bursts across {len(plans)} distinct nodes.")
    print(f"\n=== injection config ===")
    print(json.dumps(config, indent=2))
    print(f"\n=== injection labels (ground truth) ===")
    print(labels_df.to_string(index=False))
    print(f"\nGrid cells labeled per injection (should be >=1 each):")
    print(grid_labels_df.groupby("injection_id").size().to_string())

    print(f"\nWrote {OUT_BURST_PARQUET}, {OUT_BURST_LABELS_CSV}, {OUT_BURST_CONFIG_JSON}, {OUT_BURST_GRID_LABELS_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=["stall", "burst"], default="stall")
    args = ap.parse_args()
    main_stall() if args.type == "stall" else main_burst()
