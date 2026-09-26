"""
Timing detectors on per-node inter-arrival gaps.

Features per event: a trailing rolling mean/std of inter-arrival gaps, an EWMA-predicted gap and
its residual, and a robust z-score against the node's (median, MAD) baseline with pooled fallback
(src/timing_baseline.py). Rolling and EWMA statistics are causal (shifted by one event per node)
and computed over the full per-node stream, so every row sees only its true preceding history.
Only the baseline itself is fit on train-period normal data, to avoid leakage.

All isolation-forest inputs are node-relative and dimensionless, using the same MAD-derived scale:
  - rolling_mean_z     = (rolling_mean - node_median) / node_scale
  - rolling_std_ratio  = rolling_std / node_scale
  - ewma_residual_z    = ewma_residual / node_scale
  - z_score            = (gap - node_median) / node_scale
Raw inter-arrival gaps span several orders of magnitude across nodes, so unnormalized features
would dominate the forest's splits.

Detectors:
  - ZScoreThresholdDetector:   |z_score| > Z_THRESH (Eq. 2 in the paper).
  - LogRatioThresholdDetector: |log(gap / node_median)| above a train-percentile threshold (Eq. 3).
  - TimingIsolationForestDetector: isolation forest over FEATURE_COLUMNS.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.timing_baseline import MAD_FLOOR_SEC, compute_node_baselines, score_gap_zscore

ROLL_WINDOW = 20  # events, trailing
EWMA_SPAN = 20  # events
MIN_ROLL_PERIODS = 3
N_ESTIMATORS = 100
Z_THRESH = 3.0  # same outlier threshold as the premise audit's timing-gap signature

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

    # Rows without enough rolling history (start of a node's sequence) are filled with the node's
    # baseline in raw units, so they normalize to a neutral 0.
    df["rolling_mean"] = df["rolling_mean"].fillna(pd.Series(node_median, index=df.index))
    df["rolling_std"] = df["rolling_std"].fillna(pd.Series(node_scale, index=df.index))
    df["ewma_residual"] = df["ewma_residual"].fillna(0.0)

    # Node-relative, dimensionless isolation-forest inputs (FEATURE_COLUMNS).
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
    """|z_score| > Z_THRESH, with no fitting (fit() is a no-op kept for interface parity).

    The z-score is additive: a stall adds intensity * node_scale to a gap, so its z-score grows
    without bound. A burst divides the gap by the intensity, so its z-score is bounded below by
    about -(node_median / node_scale) no matter how severe the compression. The statistic is
    therefore structurally weak against bursts; see LogRatioThresholdDetector.
    """

    def __init__(self, z_thresh: float = Z_THRESH):
        self.z_thresh = z_thresh

    def fit(self, features_train_normal: pd.DataFrame):
        return self

    def score(self, features: pd.DataFrame):
        return features["z_score"].abs()


def add_log_ratio_feature(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """log(gap / node_baseline_median), symmetric under multiplicative change: a stall gives a
    large positive value and a burst a large negative value of comparable magnitude. Requires
    `gap_prev_s` (src.timing_baseline.add_sequence_context) and uses the same per-node baseline,
    with pooled fallback, as z_score.
    """
    df = df.copy()
    use_node = df["node"].isin(baselines["valid_nodes"])
    node_median = np.where(use_node, df["node"].map(baselines["node_median"]), baselines["global_median"])
    node_median_safe = np.maximum(node_median, MAD_FLOOR_SEC)
    gap_safe = df["gap_prev_s"].clip(lower=MAD_FLOOR_SEC)
    log_ratio = np.log(gap_safe / node_median_safe)
    df["log_ratio"] = pd.Series(log_ratio, index=df.index).fillna(0.0)  # first event of a node: neutral
    return df


class LogRatioThresholdDetector:
    """|log_ratio| > threshold, where the threshold is a percentile of the train-normal
    |log_ratio| distribution (the log-ratio has no natural 3-sigma convention)."""

    def __init__(self, threshold_percentile: float = 95):
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    def fit(self, features_train_normal: pd.DataFrame):
        self.threshold = np.percentile(features_train_normal["log_ratio"].abs(), self.threshold_percentile)
        return self

    def score(self, features: pd.DataFrame):
        return features["log_ratio"].abs()
