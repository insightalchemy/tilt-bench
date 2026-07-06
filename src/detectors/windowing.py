"""
Shared per-node windowing for the two count-vector detectors (count_pca, isolation_forest_counts).

BGL is a continuous per-node stream (CLAUDE.md), so windows are built independently within each
node's own chronological sequence. Windowing scheme is a first-class, explicit parameter -- CLAUDE.md
flags windowing choice as something we'll want to sweep later, so it must not be hardcoded:

  - "fixed_count": every WINDOW_SIZE consecutive events on a node form one window.
  - "fixed_time":  every WINDOW_SIZE-second wall-clock bucket on a node forms one window.

Call build_windows() separately on your train and test frames (after you've already done the
chronological split) -- windows are numbered fresh within whatever frame you pass in, so a window
never straddles the train/test boundary.

A window's label is anomalous if ANY event inside it is anomalous (standard for window-level
detection evaluation; this is a window-level label, not a point-adjusted trick).
"""

import numpy as np
import pandas as pd

WINDOW_SCHEME = "fixed_count"  # "fixed_count" | "fixed_time" -- change here (or pass explicitly)
WINDOW_SIZE = 20  # events, if fixed_count; seconds, if fixed_time
TOP_K_TEMPLATES = 300  # count-vector vocabulary: this many most frequent templates in the training data


def assign_window_id(df: pd.DataFrame, scheme: str = WINDOW_SCHEME, size: float = WINDOW_SIZE) -> pd.DataFrame:
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    if scheme == "fixed_count":
        window_idx = df.groupby("node", sort=False).cumcount() // size
    elif scheme == "fixed_time":
        t0 = df.groupby("node", sort=False)["timestamp"].transform("min")
        window_idx = ((df["timestamp"] - t0).dt.total_seconds() // size).astype(int)
    else:
        raise ValueError(f"unknown window scheme: {scheme!r} (expected 'fixed_count' or 'fixed_time')")
    df = df.copy()
    df["window_key"] = list(zip(df["node"], window_idx))
    return df


def build_windows(df: pd.DataFrame, scheme: str = WINDOW_SCHEME, size: float = WINDOW_SIZE):
    """Returns (df_with_window_key, windows) where `windows` has one row per (node, window)
    with its time bounds, event count, and window-level anomaly label."""
    df = assign_window_id(df, scheme, size)
    windows = (
        df.groupby("window_key", sort=False)
        .agg(
            node=("node", "first"),
            window_start=("timestamp", "min"),
            window_end=("timestamp", "max"),
            n_events=("timestamp", "size"),
            label=("anomaly", "any"),
        )
        .reset_index()
    )
    return df, windows


def build_vocabulary(df_train: pd.DataFrame, top_k: int = TOP_K_TEMPLATES) -> list:
    """Top-k most frequent templates in the training data. Capped so the dense windows x vocab
    matrix stays small (BGL has ~2-2.4k unique templates total, heavy-tailed, so a few hundred
    of the most common ones already cover the large majority of event volume). Templates outside
    this vocabulary simply aren't counted -- a window whose events are dominated by rare/unseen
    templates will show up as having unusually LOW counts across the known vocabulary, which is
    itself a legitimate signal, not an artifact of the cap."""
    counts = df_train["event_template"].value_counts()
    return counts.head(top_k).index.tolist()


def count_matrix(df_with_window_key: pd.DataFrame, windows: pd.DataFrame, vocabulary: list) -> np.ndarray:
    """Dense (n_windows, len(vocabulary)) template-count matrix, row-aligned to `windows`."""
    vocab_index = {t: i for i, t in enumerate(vocabulary)}
    window_index = {k: i for i, k in enumerate(windows["window_key"])}

    mat = np.zeros((len(windows), len(vocabulary)), dtype=np.float32)
    in_vocab = df_with_window_key["event_template"].isin(vocab_index)
    rows = df_with_window_key.loc[in_vocab, "window_key"].map(window_index).to_numpy()
    cols = df_with_window_key.loc[in_vocab, "event_template"].map(vocab_index).to_numpy()
    np.add.at(mat, (rows, cols), 1.0)
    return mat
