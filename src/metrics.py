"""
Shared evaluation helper for detector outputs.

Deliberately plain point-wise precision/recall/F1 -- NOT point-adjusted (CLAUDE.md flags
point-adjusted F1 as inflating time-series AD scores). Each row or window is scored on whether IT
was flagged vs IT was truly anomalous; a detector doesn't get credit for a whole contiguous
anomalous segment just because it caught one point inside it.

## Common evaluation unit

The three baseline detectors natively operate at different granularities (count-vector detectors:
their own fixed-count windows; the timing detector: individual rows/events), so their raw
precision/recall/F1 numbers are not comparable, and a Jaccard overlap or OR-fusion across them
would be meaningless -- a "detection" from a 20-event window and a "detection" from one row are not
the same unit of thing.

Chosen unit: **a single fixed per-node window grid** (fixed_time, 60s by default), used to score
every detector identically. Rationale for the grid over a region/event scheme built from BGL's
point labels: point labels have no natural span (each row is independently labeled), so turning
them into "events" would require inventing a grouping rule (a max-gap-to-merge parameter) with no
principled default -- one more free parameter with no ground truth to calibrate it against. A fixed
grid needs no such invention, reuses the exact windowing machinery already validated for the
count-vector detectors (src.detectors.windowing), and is trivial to apply uniformly: broadcast each
detector's native-granularity flag down to the row level, then re-aggregate rows up to the shared
grid (a grid cell is flagged if ANY row inside it is flagged -- same rule already used for window
labels, applied once instead of per-detector). This also directly determines how injected fault
spans will be labeled later: a grid cell overlapping an injected (start, end, node) span is
"anomalous," exactly the same rule.
"""

import pandas as pd

from src.detectors.windowing import assign_window_id

EVAL_WINDOW_SCHEME = "fixed_time"
EVAL_WINDOW_SIZE = 60  # seconds


def assign_eval_grid(df: pd.DataFrame, scheme: str = EVAL_WINDOW_SCHEME, size: float = EVAL_WINDOW_SIZE) -> pd.DataFrame:
    """Adds `eval_window_key` (node, window_idx) -- the shared evaluation grid every detector's
    output gets mapped onto. Reuses windowing.assign_window_id so this is the identical per-node
    grid logic used elsewhere, just decoupled from any one detector's own window size."""
    df = assign_window_id(df, scheme=scheme, size=size)
    return df.rename(columns={"window_key": "eval_window_key"})


def rows_to_grid(df_with_eval_grid: pd.DataFrame, row_flags: pd.Series) -> pd.Series:
    """Aggregate a row-level boolean flag up to the eval grid: a cell is flagged if ANY of its
    rows are flagged. row_flags must be aligned (same index) with df_with_eval_grid."""
    flags = pd.Series(row_flags).to_numpy()
    return pd.Series(flags, index=df_with_eval_grid["eval_window_key"].to_numpy()).groupby(level=0).any()


def rows_to_grid_max(df_with_eval_grid: pd.DataFrame, row_scores: pd.Series) -> pd.Series:
    """Aggregate a row-level CONTINUOUS score up to the eval grid by MAX -- the threshold-free
    analog of rows_to_grid's "any row flagged" rule, used for AUC-PR/AUC-ROC scoring where there is
    no boolean flag yet. row_scores must be aligned (same index) with df_with_eval_grid. Applied
    identically to every detector (src/auc_metrics.py), so cross-detector AUC comparisons use the
    same aggregation rule throughout."""
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
