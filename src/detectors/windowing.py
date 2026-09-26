"""
Per-node windowing for the count-vector detectors.

Logs are treated as continuous per-node streams, so windows are built independently within each
node's chronological sequence. Two schemes are supported:

  - "fixed_count": every `size` consecutive events on a node form one window.
  - "fixed_time":  every `size`-second wall-clock bucket on a node forms one window.

Call build_windows() separately on train and test frames so no window straddles the split.
A window is labeled anomalous if any event inside it is anomalous.
"""

import numpy as np
import pandas as pd

WINDOW_SCHEME = "fixed_count"  # "fixed_count" | "fixed_time"
WINDOW_SIZE = 20  # events, if fixed_count; seconds, if fixed_time
TOP_K_TEMPLATES = 300  # count-vector vocabulary size (most frequent training templates)


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
    """Top-k most frequent templates in the training data. The template distribution is heavy
    tailed, so a few hundred templates cover most event volume while keeping the dense count
    matrix small. Out-of-vocabulary templates are not counted; a window dominated by rare or
    unseen templates therefore shows unusually low counts, which is itself an anomaly signal."""
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
