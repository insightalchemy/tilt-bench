"""
Detector 3 (timing family): lightweight inter-arrival features -> isolation forest.

Per event: a trailing rolling mean/std of inter-arrival gaps, an EWMA-predicted gap and its
residual, and a robust z-score against the per-node baseline (with pooled fallback for low-data
nodes) validated in src/premise_audit.py -- reused via src/timing_baseline.py so this detector is
consistent with what the audit already checked out.

All rolling/EWMA statistics are computed causally (shifted by one event first, per node) over the
FULL continuous per-node stream (train and test together, in chronological order) so a test-period
row's features reflect its true preceding history, same as a real streaming deployment would see --
they just never look at future events. Only the z-score BASELINE (median/MAD per node) is fit on
train-normal data alone, to avoid leaking test-period timing into that baseline.

FEATURE SCALING (fixed after results/core_result_stall.csv v1): the raw rolling_mean/rolling_std/
ewma_residual are in seconds and span ~5 orders of magnitude across nodes (results/
core_result_stall_timing_feature_diag.csv showed rolling_mean from 1.18s to 351,226s across the 15
injected-stall nodes) -- feeding those raw values into IsolationForest alongside the properly
node-normalized z_score let the raw-seconds features dominate the ensemble's splits and drown the
one feature that actually carries signal (the same failure mode diagnosed for
isolation_forest_counts in results/iforest_diagnosis.md). All four features are now expressed on
the SAME node-relative, dimensionless scale, using the identical per-node (median, MAD-derived
scale) baseline -- with the same pooled fallback for low-data nodes -- that z_score already used:
  - rolling_mean_z     = (rolling_mean - node_median) / node_scale
  - rolling_std_ratio  = rolling_std / node_scale
  - ewma_residual_z    = ewma_residual / node_scale
  - z_score            = (gap - node_median) / node_scale   (unchanged)
The raw columns are still computed and kept in the dataframe for diagnostics/interpretability;
only the FEATURE_COLUMNS below (normalized) are fed to the isolation forest.

This operates at row/event granularity, not window granularity -- inter-arrival time is naturally a
per-event quantity (see CLAUDE.md's "BGL is a continuous per-node stream").
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.timing_baseline import MAD_FLOOR_SEC, compute_node_baselines, score_gap_zscore

ROLL_WINDOW = 20  # events, trailing -- explicit, easily-changeable
EWMA_SPAN = 20  # events
MIN_ROLL_PERIODS = 3
N_ESTIMATORS = 100
Z_THRESH = 3.0  # for the z_score-alone sanity-check detector; matches premise_audit.py's convention

FEATURE_COLUMNS = ["rolling_mean_z", "rolling_std_ratio", "ewma_residual_z", "z_score"]


def add_rolling_features(df: pd.DataFrame, roll_window: int = ROLL_WINDOW, ewma_span: int = EWMA_SPAN) -> pd.DataFrame:
    """df must already have `gap_prev_s` (see src.timing_baseline.add_sequence_context) and be
    sorted by (node, timestamp). Adds rolling_mean, rolling_std, ewma_residual -- all causal."""
    df = df.copy()
    prior_gap = df.groupby("node", sort=False)["gap_prev_s"].shift(1)  # excludes the row's own gap
    df["_prior_gap"] = prior_gap

    roll = df.groupby("node", sort=False)["_prior_gap"]
    df["rolling_mean"] = roll.transform(lambda s: s.rolling(roll_window, min_periods=MIN_ROLL_PERIODS).mean())
    df["rolling_std"] = roll.transform(lambda s: s.rolling(roll_window, min_periods=MIN_ROLL_PERIODS).std())
    ewma_pred = roll.transform(lambda s: s.ewm(span=ewma_span, min_periods=MIN_ROLL_PERIODS).mean())
    df["ewma_residual"] = df["gap_prev_s"] - ewma_pred

    df.drop(columns=["_prior_gap"], inplace=True)
    return df


def build_features(df: pd.DataFrame, baseline_mask: pd.Series = None, baselines: dict = None) -> tuple[pd.DataFrame, dict]:
    """Adds rolling features + a z-score against a baseline fit ONLY on rows selected by
    baseline_mask (callers pass a train-period, normal-to-normal mask to avoid leakage). Returns
    the augmented dataframe and the baseline dict (for diagnostics/reuse).

    Pass a precomputed `baselines` dict (e.g. fit on a different, clean dataset) to score THIS df
    against that baseline instead of recomputing one from baseline_mask -- e.g. scoring an injected
    dataset against a baseline fit purely on the pre-injection clean data's train period.
    """
    df = add_rolling_features(df)
    if baselines is None:
        baselines = compute_node_baselines(df, baseline_mask)

    use_node = df["node"].isin(baselines["valid_nodes"])
    node_median = np.where(use_node, df["node"].map(baselines["node_median"]), baselines["global_median"])
    node_mad = np.where(use_node, df["node"].map(baselines["node_mad"]), baselines["global_mad"])
    node_scale = np.maximum(node_mad * 1.4826, MAD_FLOOR_SEC)  # same MAD-derived scale as z_score

    # Fill rows with insufficient rolling history (start of a node's sequence) using that node's
    # (or the pooled-fallback) baseline, in RAW units, before normalizing -- so "no deviation info
    # yet" fills to a neutral 0 post-normalization rather than an arbitrary raw value.
    df["rolling_mean"] = df["rolling_mean"].fillna(pd.Series(node_median, index=df.index))
    df["rolling_std"] = df["rolling_std"].fillna(pd.Series(node_scale, index=df.index))
    df["ewma_residual"] = df["ewma_residual"].fillna(0.0)

    # Node-relative, dimensionless versions -- the actual isolation-forest inputs (FEATURE_COLUMNS).
    df["rolling_mean_z"] = (df["rolling_mean"] - node_median) / node_scale
    df["rolling_std_ratio"] = df["rolling_std"] / node_scale
    df["ewma_residual_z"] = df["ewma_residual"] / node_scale
    df["z_score"] = score_gap_zscore(df["gap_prev_s"], df["node"], baselines).fillna(0.0)

    return df, baselines


class TimingIsolationForestDetector:
    def __init__(self, n_estimators: int = N_ESTIMATORS, random_state: int = 0):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    def fit(self, features_train_normal: pd.DataFrame):
        self.model.fit(features_train_normal[FEATURE_COLUMNS])
        return self

    def score(self, features: pd.DataFrame):
        return -self.model.score_samples(features[FEATURE_COLUMNS])


class ZScoreThresholdDetector:
    """Minimal-baseline sanity check: |z_score| > Z_THRESH, no isolation forest, no fitting at
    all. This is the ceiling TimingIsolationForestDetector should be matching (or approaching)
    once its features are properly node-normalized -- if even this trivial rule doesn't catch the
    injected stalls, the raw timing signal itself is the problem, not the ML wrapper around it.
    fit() is a no-op (kept only so this plugs into the same fit/score pipeline as every other
    detector); the threshold is the fixed Z_THRESH, not a train-score percentile.

    NOTE (found via the burst experiment): z_score = (gap - node_median) / node_scale is
    ADDITIVE-biased. A stall adds intensity*node_scale directly, so its z-score is unbounded and
    scales with intensity by construction. A burst instead COMPRESSES the gap
    (new_gap = orig_gap / intensity); its z-score is bounded by roughly -(node_median / node_scale)
    as intensity -> infinity (found empirically to be ~-0.675 for BGL nodes), structurally far
    below Z_THRESH regardless of how aggressive the compression is. This detector is therefore
    stall-only by construction, not because compression is a weaker fault -- see
    LogRatioThresholdDetector below for a symmetric alternative.
    """

    def __init__(self, z_thresh: float = Z_THRESH):
        self.z_thresh = z_thresh

    def fit(self, features_train_normal: pd.DataFrame):
        return self

    def score(self, features: pd.DataFrame):
        return features["z_score"].abs()


def add_log_ratio_feature(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """log(gap / node_baseline_median) -- symmetric to multiplicative change, unlike z_score's
    additive form. A stall (gap >> baseline) gives a large POSITIVE log-ratio; a burst/compression
    (gap << baseline) gives a large NEGATIVE one of comparable magnitude, since a log-ratio treats
    "N times larger" and "N times smaller" as equal-magnitude, opposite-sign deviations. Purely
    additive to the module -- does not touch build_features, FEATURE_COLUMNS, or anything the
    (locked) stall pipeline depends on. Requires df to already have `gap_prev_s`
    (src.timing_baseline.add_sequence_context) and uses the SAME per-node baseline (with pooled
    fallback) as z_score, via the same `baselines` dict from build_features.
    """
    df = df.copy()
    use_node = df["node"].isin(baselines["valid_nodes"])
    node_median = np.where(use_node, df["node"].map(baselines["node_median"]), baselines["global_median"])
    node_median_safe = np.maximum(node_median, MAD_FLOOR_SEC)
    gap_safe = df["gap_prev_s"].clip(lower=MAD_FLOOR_SEC)
    log_ratio = np.log(gap_safe / node_median_safe)
    df["log_ratio"] = pd.Series(log_ratio, index=df.index).fillna(0.0)  # no gap yet -> neutral
    return df


class LogRatioThresholdDetector:
    """Symmetric timing detector: |log(gap / node_baseline_median)| > threshold, no isolation
    forest. Threshold is calibrated from the train-normal |log_ratio| distribution (percentile),
    mirroring how every ML detector's threshold is set elsewhere in this project -- log-ratio has
    no natural "3-sigma"-like convention the way z_score does, so a data-driven percentile is the
    fairer choice here rather than picking an arbitrary fixed constant.
    """

    def __init__(self, threshold_percentile: float = 95):
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    def fit(self, features_train_normal: pd.DataFrame):
        self.threshold = np.percentile(features_train_normal["log_ratio"].abs(), self.threshold_percentile)
        return self

    def score(self, features: pd.DataFrame):
        return features["log_ratio"].abs()
