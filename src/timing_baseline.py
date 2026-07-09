"""
Shared per-node timing-baseline utilities.

This is the validated methodology from src/premise_audit.py, factored out so
src/detectors/timing_detector.py can reuse the exact same per-node baseline + pooled-fallback
logic rather than re-deriving it. premise_audit.py imports from here too, so there is exactly one
implementation of this logic, not two that could drift apart.
"""

import numpy as np
import pandas as pd

MIN_NODE_BASELINE = 10  # min normal-to-normal gaps needed to trust a per-node baseline
MAD_FLOOR_SEC = 1e-3  # last-resort floor if even the pooled fallback MAD is 0


def add_sequence_context(df):
    """Per-node neighbor templates/labels and inter-arrival gaps (seconds), via groupby+shift."""
    g = df.groupby("node", sort=False)
    df["prev_template"] = g["event_template"].shift(1)
    df["next_template"] = g["event_template"].shift(-1)
    df["prev_anomaly"] = g["anomaly"].shift(1)
    prev_ts = g["timestamp"].shift(1)
    next_ts = g["timestamp"].shift(-1)
    df["gap_prev_s"] = (df["timestamp"] - prev_ts).dt.total_seconds()
    df["gap_next_s"] = (next_ts - df["timestamp"]).dt.total_seconds()
    return df


def compute_node_baselines(df, normal_seq_mask, min_samples=MIN_NODE_BASELINE, exclude_zero_from_pooled=False):
    """Per-node (median, MAD) of the gaps selected by normal_seq_mask, with pooled-global
    fallback for nodes with too few samples or a degenerate (zero) per-node MAD.

    normal_seq_mask should select rows whose gap_prev_s reflects a normal-to-normal transition
    (both the row and its predecessor unlabeled) -- callers restrict this to a training period
    to avoid leaking test-period timing into the baseline.

    exclude_zero_from_pooled=True computes the pooled (global) fallback from NON-ZERO gaps only.
    Off by default -- BGL's zero-gap rate is ~0%, so this never mattered there. On datasets with a
    high zero-gap rate (Thunderbird: >50% of gaps are exactly 0, since its timestamps are
    second-resolution only), the zero value can dominate the pooled median/MAD too, making even
    the fallback degenerate (median=MAD=0) exactly when it's needed most -- for the low-data nodes
    that couldn't get a per-node baseline in the first place.
    """
    normal_gaps = df.loc[normal_seq_mask, ["node", "gap_prev_s"]]

    node_counts = normal_gaps.groupby("node")["gap_prev_s"].size()
    node_median = normal_gaps.groupby("node")["gap_prev_s"].median()
    node_mad = normal_gaps.groupby("node")["gap_prev_s"].apply(lambda s: (s - s.median()).abs().median())

    enough_samples = node_counts >= min_samples
    nonzero_mad = node_mad > 0
    valid_nodes = set(node_counts[enough_samples & nonzero_mad].index)
    fallback_low_count_nodes = set(node_counts[~enough_samples].index)
    fallback_zero_mad_nodes = set(node_counts[enough_samples & ~nonzero_mad].index)

    pooled_gaps = normal_gaps.loc[normal_gaps["gap_prev_s"] > 0, "gap_prev_s"] if exclude_zero_from_pooled else normal_gaps["gap_prev_s"]
    global_median = pooled_gaps.median()
    global_mad = (pooled_gaps - global_median).abs().median()

    return {
        "node_median": node_median,
        "node_mad": node_mad,
        "valid_nodes": valid_nodes,
        "global_median": global_median,
        "global_mad": global_mad,
        "fallback_low_count_nodes": fallback_low_count_nodes,
        "fallback_zero_mad_nodes": fallback_zero_mad_nodes,
    }


def score_gap_zscore(gap: pd.Series, node: pd.Series, baselines: dict, mad_floor: float = MAD_FLOOR_SEC) -> pd.Series:
    """Robust z-score of `gap` against each row's node baseline, or the pooled fallback."""
    use_node = node.isin(baselines["valid_nodes"])
    med = np.where(use_node, node.map(baselines["node_median"]), baselines["global_median"])
    mad = np.where(use_node, node.map(baselines["node_mad"]), baselines["global_mad"])
    mad_eff = np.maximum(mad * 1.4826, mad_floor)
    return (gap - med) / mad_eff
