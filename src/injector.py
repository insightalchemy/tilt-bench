"""
TILT-Bench timing-fault injector for BGL: stall (widen a gap) and burst (compress a run of gaps).
Only timestamps change; event content and per-node order are preserved.

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

Intensity is relative, not absolute. For stalls: delta = intensity * node_scale, where node_scale
= max(1.4826 * MAD, floor) is the node's MAD-derived scale (Eq. 1 in the paper), with the pooled
fallback of src/timing_baseline.py for low-data nodes; an intensity of 20 means "20x this node's
normal inter-arrival variability." For bursts, intensity is a compression factor
(new_gap = orig_gap / intensity), relative to each gap's own size, which cannot produce a
non-positive gap the way an additive reduction could.

Reads data/processed/bgl_parsed.parquet (never modified). Writes, per type (stall shown; burst uses
the same filenames with `_burst` instead of `_stall`):
  data/processed/bgl_injected_stall.parquet      -- injected data (adds an `injected_row` bool col)
  data/processed/injection_labels_stall.csv      -- ground truth: start, end, node, type, intensity, seed
  data/processed/injection_config_stall.json     -- resolved generation config, for reproducibility
  data/processed/injection_grid_labels_stall.csv -- (injection_id, node, window_idx) cells overlapped
                                                     by each span on the shared 60 s evaluation grid

Fully seeded: one master SEED drives node selection and a per-injection seed array; each
injection's own seed alone reproduces that injection's start position and intensity. Ground-truth
labels are recorded as (start, end, node, type, intensity, seed).

Usage:
    python src/injector.py --type stall   # default
    python src/injector.py --type burst
    python src/injector.py --type stall --n-injections 300   # writes separate _n300-suffixed outputs
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
N_INJECTIONS = 100
INTENSITY_CHOICES = [10, 15, 20, 25, 30]  # multiples of the node's MAD-derived scale
MIN_NODE_EVENTS = 30  # node eligibility: enough events for real before/after context
START_FRACTION_RANGE = (0.2, 0.8)  # place the stall well inside the node's own sequence
MAX_ORIGINAL_GAP_Z = 3.0  # reject injection points whose PRE-EXISTING gap already looks unusual
MAX_START_POS_TRIES = 200  # bounded retries before falling back to any position in range

# --- rate/density eligibility gate (paper Section IV-B) ---
# MIN_NODE_EVENTS alone says nothing about how densely packed in time a node's events are: a node
# can clear it with a handful of events spread over weeks. Since a stall's
# delta_seconds = intensity * mad_eff, a sparse node with a large mad_eff would receive a fault
# spanning days, which is not a plausible transient timing anomaly.
#
# MAX_PLAUSIBLE_FAULT_DURATION_S is a domain assumption: a transient timing fault on an actively
# monitored node resolves on the order of an hour (60x the 60 s evaluation grid). Gating
# eligibility on mad_eff <= MAX_MAD_EFF_S guarantees that even the maximum intensity cannot
# produce a longer stall:
#   delta_seconds = intensity * mad_eff <= max(INTENSITY_CHOICES) * MAX_MAD_EFF_S = MAX_PLAUSIBLE_FAULT_DURATION_S.
# The same node-level gate is applied to bursts. Bursts compress observed local gaps rather than
# mad_eff, so the bound is only approximate there.
MAX_PLAUSIBLE_FAULT_DURATION_S = 3600.0  # 1 hour
MAX_MAD_EFF_S = MAX_PLAUSIBLE_FAULT_DURATION_S / max(INTENSITY_CHOICES)  # 120s

BURST_SEED = 43  # distinct from stall's SEED, for an independently reproducible burst experiment
BURST_N_INJECTIONS = 100
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


def eligible_nodes(df, min_events=MIN_NODE_EVENTS, node_baselines=None, require_valid_baseline=False, max_mad_eff=None):
    """Nodes with at least `min_events` events, optionally further restricted.

    require_valid_baseline=True keeps only nodes with a non-degenerate per-node baseline
    (node_baselines["valid_nodes"]). This is needed on datasets whose pooled fallback is itself
    degenerate (whole-second timestamps), so no injection is scaled against a zero baseline.

    max_mad_eff keeps only nodes whose scale mad_eff = max(node_mad * 1.4826, MAD_FLOOR_SEC)
    -- the quantity that sets a stall's delta_seconds -- is at most this cap (the rate gate; see
    MAX_MAD_EFF_S above). Nodes with no MAD on record are treated as ineligible.
    """
    counts = df.groupby("node").size()
    nodes = counts[counts >= min_events].index.tolist()
    if require_valid_baseline:
        if node_baselines is None:
            raise ValueError("require_valid_baseline=True requires node_baselines")
        nodes = [n for n in nodes if n in node_baselines["valid_nodes"]]
    if max_mad_eff is not None:
        if node_baselines is None:
            raise ValueError("max_mad_eff requires node_baselines")
        node_mad = node_baselines["node_mad"]

        def mad_eff_of(n):
            m = node_mad.get(n)
            if m is None or pd.isna(m):
                return float("inf")
            return max(m * 1.4826, MAD_FLOOR_SEC)

        nodes = [n for n in nodes if mad_eff_of(n) <= max_mad_eff]
    return nodes


def plan_injections(
    df,
    node_baselines,
    seed=SEED,
    n_injections=N_INJECTIONS,
    min_node_events=MIN_NODE_EVENTS,
    require_valid_baseline=False,
    max_mad_eff=None,
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
    nodes = sorted(
        eligible_nodes(
            df, min_events=min_node_events, node_baselines=node_baselines, require_valid_baseline=require_valid_baseline, max_mad_eff=max_mad_eff
        )
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

        # Build the delta in whole microseconds: a Timedelta from fractional seconds has nanosecond
        # precision, which pandas will not downcast losslessly into a coarser column. The result is
        # cast back to the column's own dtype (e.g. datetime64[us] or [ms]) so this works on any
        # dataset's timestamp resolution.
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
    """(injection_id, node, window_idx) for every eval-grid cell whose time range overlaps an
    injected span, computed analytically from each node's t0 rather than from existing rows. A
    stall creates a quiet period with no rows in it, so a row-driven grid could not represent the
    empty cells inside a long stall."""
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
    max_mad_eff=None,
):
    """Mirrors plan_injections, plus a burst_length draw. A candidate start position is rejected
    if ANY of its burst_length candidate gaps is already an outlier relative to the node's
    baseline (too large OR too small), so a burst compresses an otherwise-ordinary run of gaps
    rather than one that already contained something unusual."""
    master_rng = np.random.default_rng(seed)

    nodes = sorted(
        eligible_nodes(
            df, min_events=min_node_events, node_baselines=node_baselines, require_valid_baseline=require_valid_baseline, max_mad_eff=max_mad_eff
        )
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
        # Cast to the column's own dtype so this works on any timestamp resolution.
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


def _suffixed(path: Path, n_injections: int, default_n: int) -> Path:
    """Appends `_n{N}` to path's stem when n_injections differs from the default, so a
    non-default run writes separate files instead of overwriting the n=100 outputs that the
    downstream scripts read."""
    if n_injections == default_n:
        return path
    return path.with_name(f"{path.stem}_n{n_injections}{path.suffix}")


def main_stall(n_injections=N_INJECTIONS):
    df_clean = load_clean()
    df_clean = add_sequence_context(df_clean)

    normal_seq = (~df_clean["anomaly"]) & (~df_clean["prev_anomaly"].fillna(True)) & df_clean["gap_prev_s"].notna()
    node_baselines = compute_node_baselines(df_clean, normal_seq)

    n_eligible = len(eligible_nodes(df_clean, min_events=MIN_NODE_EVENTS, node_baselines=node_baselines, max_mad_eff=MAX_MAD_EFF_S))
    print(f"Eligible nodes after rate/density gate (size>={MIN_NODE_EVENTS} AND mad_eff<={MAX_MAD_EFF_S}s): {n_eligible:,}")

    plans = plan_injections(df_clean, node_baselines, n_injections=n_injections, max_mad_eff=MAX_MAD_EFF_S)
    df_injected, labels_df = apply_injections(df_clean, plans, node_baselines)

    grid_labels_df = label_spans_on_grid(labels_df, df_injected)

    # Drop the neighbor/gap columns added by add_sequence_context: injection changes them, so
    # consumers recompute them. injected_row is the only new column in the output.
    audit_cols = ["prev_template", "next_template", "prev_anomaly", "gap_prev_s", "gap_next_s"]
    df_injected_out = df_injected.drop(columns=audit_cols)

    out_parquet = _suffixed(OUT_PARQUET, n_injections, N_INJECTIONS)
    out_labels_csv = _suffixed(OUT_LABELS_CSV, n_injections, N_INJECTIONS)
    out_grid_labels_csv = _suffixed(OUT_GRID_LABELS_CSV, n_injections, N_INJECTIONS)
    out_config_json = _suffixed(OUT_CONFIG_JSON, n_injections, N_INJECTIONS)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_injected_out.to_parquet(out_parquet, index=False)
    labels_df.to_csv(out_labels_csv, index=False)
    grid_labels_df.to_csv(out_grid_labels_csv, index=False)

    config = {
        "seed": SEED,
        "n_injections": n_injections,
        "intensity_choices": INTENSITY_CHOICES,
        "min_node_events": MIN_NODE_EVENTS,
        "max_plausible_fault_duration_s": MAX_PLAUSIBLE_FAULT_DURATION_S,
        "max_mad_eff_s": MAX_MAD_EFF_S,
        "n_eligible_nodes": n_eligible,
        "start_fraction_range": list(START_FRACTION_RANGE),
        "eval_grid_scheme": EVAL_WINDOW_SCHEME,
        "eval_grid_size_s": EVAL_WINDOW_SIZE,
        "input": str(IN_PATH),
    }
    out_config_json.write_text(json.dumps(config, indent=2))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Injected {len(plans)} stalls across {len(plans)} distinct nodes.")
    print(f"\n=== injection config ===")
    print(json.dumps(config, indent=2))
    print(f"\n=== injection labels (ground truth) ===")
    print(labels_df.to_string(index=False))
    print(f"\nGrid cells labeled per injection (should be >=1 each):")
    print(grid_labels_df.groupby("injection_id").size().to_string())

    print(f"\nWrote {out_parquet}, {out_labels_csv}, {out_config_json}, {out_grid_labels_csv}")


def main_burst(n_injections=BURST_N_INJECTIONS):
    df_clean = load_clean()
    df_clean = add_sequence_context(df_clean)

    normal_seq = (~df_clean["anomaly"]) & (~df_clean["prev_anomaly"].fillna(True)) & df_clean["gap_prev_s"].notna()
    node_baselines = compute_node_baselines(df_clean, normal_seq)

    n_eligible = len(eligible_nodes(df_clean, min_events=BURST_MIN_NODE_EVENTS, node_baselines=node_baselines, max_mad_eff=MAX_MAD_EFF_S))
    print(f"Eligible nodes after rate/density gate (size>={BURST_MIN_NODE_EVENTS} AND mad_eff<={MAX_MAD_EFF_S}s): {n_eligible:,}")

    plans = plan_burst_injections(df_clean, node_baselines, n_injections=n_injections, max_mad_eff=MAX_MAD_EFF_S)
    df_injected, labels_df = apply_burst_injections(df_clean, plans)

    grid_labels_df = label_spans_on_grid(labels_df, df_injected)

    audit_cols = ["prev_template", "next_template", "prev_anomaly", "gap_prev_s", "gap_next_s"]
    df_injected_out = df_injected.drop(columns=audit_cols)

    out_parquet = _suffixed(OUT_BURST_PARQUET, n_injections, BURST_N_INJECTIONS)
    out_labels_csv = _suffixed(OUT_BURST_LABELS_CSV, n_injections, BURST_N_INJECTIONS)
    out_grid_labels_csv = _suffixed(OUT_BURST_GRID_LABELS_CSV, n_injections, BURST_N_INJECTIONS)
    out_config_json = _suffixed(OUT_BURST_CONFIG_JSON, n_injections, BURST_N_INJECTIONS)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_injected_out.to_parquet(out_parquet, index=False)
    labels_df.to_csv(out_labels_csv, index=False)
    grid_labels_df.to_csv(out_grid_labels_csv, index=False)

    config = {
        "seed": BURST_SEED,
        "n_injections": n_injections,
        "intensity_choices": BURST_INTENSITY_CHOICES,
        "burst_length_choices": BURST_LENGTH_CHOICES,
        "min_node_events": BURST_MIN_NODE_EVENTS,
        "max_plausible_fault_duration_s": MAX_PLAUSIBLE_FAULT_DURATION_S,
        "max_mad_eff_s": MAX_MAD_EFF_S,
        "n_eligible_nodes": n_eligible,
        "start_fraction_range": list(BURST_START_FRACTION_RANGE),
        "eval_grid_scheme": EVAL_WINDOW_SCHEME,
        "eval_grid_size_s": EVAL_WINDOW_SIZE,
        "input": str(IN_PATH),
    }
    out_config_json.write_text(json.dumps(config, indent=2))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Injected {len(plans)} bursts across {len(plans)} distinct nodes.")
    print(f"\n=== injection config ===")
    print(json.dumps(config, indent=2))
    print(f"\n=== injection labels (ground truth) ===")
    print(labels_df.to_string(index=False))
    print(f"\nGrid cells labeled per injection (should be >=1 each):")
    print(grid_labels_df.groupby("injection_id").size().to_string())

    print(f"\nWrote {out_parquet}, {out_labels_csv}, {out_config_json}, {out_grid_labels_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=["stall", "burst"], default="stall")
    ap.add_argument(
        "--n-injections", type=int, default=None,
        help=f"Override the number of injections (default {N_INJECTIONS} for stall, {BURST_N_INJECTIONS} for burst). "
        "A non-default value writes to separate _n{N}-suffixed outputs.",
    )
    args = ap.parse_args()
    if args.type == "stall":
        main_stall(n_injections=args.n_injections if args.n_injections is not None else N_INJECTIONS)
    else:
        main_burst(n_injections=args.n_injections if args.n_injections is not None else BURST_N_INJECTIONS)
