"""
Evaluation helpers shared by all detectors.

Scoring is plain point-wise precision/recall/F1 (not point-adjusted, which inflates time-series
AD scores): each unit is scored on whether it was flagged versus whether it was truly anomalous.

Detectors operate at different native granularities (count-vector detectors score windows, timing
detectors score individual events), so their outputs are compared on one common unit: a fixed
per-node time grid (60 s cells by default). Each detector's flag or score is broadcast down to
rows and re-aggregated onto the grid; a cell is positive if any of its rows is. Injected fault
spans are labeled with the same rule, so a grid cell overlapping an injected (start, end, node)
span is anomalous.
"""

import pandas as pd

from src.detectors.windowing import assign_window_id

EVAL_WINDOW_SCHEME = "fixed_time"
EVAL_WINDOW_SIZE = 60  # seconds


def assign_eval_grid(df: pd.DataFrame, scheme: str = EVAL_WINDOW_SCHEME, size: float = EVAL_WINDOW_SIZE) -> pd.DataFrame:
    """Adds `eval_window_key` (node, window_idx), the shared evaluation grid every detector's
    output is mapped onto."""
    df = assign_window_id(df, scheme=scheme, size=size)
    return df.rename(columns={"window_key": "eval_window_key"})


def rows_to_grid(df_with_eval_grid: pd.DataFrame, row_flags: pd.Series) -> pd.Series:
    """Aggregate a row-level boolean flag up to the eval grid: a cell is flagged if ANY of its
    rows are flagged. row_flags must be aligned (same index) with df_with_eval_grid."""
    flags = pd.Series(row_flags).to_numpy()
    return pd.Series(flags, index=df_with_eval_grid["eval_window_key"].to_numpy()).groupby(level=0).any()


def rows_to_grid_max(df_with_eval_grid: pd.DataFrame, row_scores: pd.Series) -> pd.Series:
    """Aggregate a row-level continuous score up to the eval grid by max -- the threshold-free
    analog of rows_to_grid's "any row flagged" rule, used for AUC scoring. row_scores must be
    aligned (same index) with df_with_eval_grid."""
    scores = pd.Series(row_scores).to_numpy()
    return pd.Series(scores, index=df_with_eval_grid["eval_window_key"].to_numpy()).groupby(level=0).max()


def evaluate_common_unit(df_with_eval_grid: pd.DataFrame, row_predicted: pd.Series, row_true: pd.Series) -> dict:
    """Map row-level predicted/true flags onto the shared eval grid and score there."""
    pred_grid = rows_to_grid(df_with_eval_grid, row_predicted).rename("y_pred")
    true_grid = rows_to_grid(df_with_eval_grid, row_true).rename("y_true")
    aligned = pd.concat([true_grid, pred_grid], axis=1).fillna(False)
    result = evaluate_binary(aligned["y_true"], aligned["y_pred"])
    result["n_eval_cells"] = len(aligned)
    return result


def evaluate_binary(y_true, y_pred) -> dict:
    y_true = pd.Series(y_true).astype(bool)
    y_pred = pd.Series(y_pred).astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_true_anomalous": tp + fn,
        "n_flagged": tp + fp,
        "n_total": tp + fp + fn + tn,
    }
