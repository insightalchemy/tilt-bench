"""
Small LogBERT-style masked-template transformer (architecture after Guo et al., "LogBERT: Log
Anomaly Detection via BERT", IJCNN 2021; not the released toolkit): 2 encoder layers, 4 attention
heads, d_model=64, over per-node sliding windows of template IDs. One position per window is
masked and predicted from bidirectional context; a row is scored by how poorly its masked
template is predicted. Paper Section V-C and Table III ("TF" row).

--eval-target native:   AUC against native anomaly labels on the chronological test period
                        (BGL / Thunderbird).
--eval-target injected: (a) max_abs_score_diff between clean and injected data, row by row
                        (expected 0: the model never reads timestamps and injection does not
                        change template content or order), and (b) AUC against the injected-span
                        ground truth plus the placebo AUC (clean scores vs the same labels), whose
                        difference is the fault-attributable detection.

Metrics are printed to stdout. --self-test runs one train/score step on a small synthetic frame.

Usage:
    python src/logbert_small.py --dataset bgl --epochs 20 --eval-target native
    python src/logbert_small.py --dataset bgl --fault stall --epochs 20 --eval-target injected
    python src/logbert_small.py --self-test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view

from src.deeplog import (
    DATASET_CONFIG,
    OOV_TOKEN,
    build_vocabulary,
    encode_templates,
    evaluate_against_injected,
    evaluate_against_native_labels,
    full_row_arrays,
    load_by_node,
    load_clean_by_node,
    resolve_device,
    select_injected_nodes,
    set_seed,
)
from src.detectors.windowing import build_windows
from src.metrics import assign_eval_grid
from src.auc_metrics import compute_grid_auc

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 128
WINDOW_SIZE = 20  # h, events per side of context (per-node sliding window length)
SEED = 42
FAULTS = ["stall", "burst"]


def build_masked_windows(df, encoded_col, h):
    """Per-node sliding windows of length h (no separate next-token target, unlike
    src.deeplog.build_windows) -- masking is applied per-batch at train/score time so the SAME
    window array is reused for every random masking draw rather than baked in once."""
    X_chunks, row_id_chunks = [], []
    for _, group in df.groupby("node", sort=False):
        ids = group[encoded_col].to_numpy()
        row_ids = group["row_id"].to_numpy()
        if len(ids) < h:
            continue
        windows = sliding_window_view(ids, h)
        X_chunks.append(windows)
        row_id_chunks.append(row_ids[h - 1 :])  # window's own eval-row is its LAST position
    if not X_chunks:
        return np.empty((0, h), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_chunks), np.concatenate(row_id_chunks)


def mask_batch(X: np.ndarray, mask_id: int, rng: np.random.Generator):
    """Masks exactly one random position per window (a single-mask simplification of BERT's
    15%-of-tokens scheme, kept deliberately simple since windows here are short, h~20). Returns
    (X_masked, mask_positions, targets)."""
    n, h = X.shape
    positions = rng.integers(0, h, size=n)
    targets = X[np.arange(n), positions].copy()
    X_masked = X.copy()
    X_masked[np.arange(n), positions] = mask_id
    return X_masked, positions, targets


class LogBertSmall(nn.Module):
    def __init__(self, vocab_size, window_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dim_feedforward=DIM_FEEDFORWARD):
        super().__init__()
        self.mask_id = vocab_size  # one id past the real vocabulary (which already includes OOV)
        self.token_embed = nn.Embedding(vocab_size + 1, d_model)
        self.pos_embed = nn.Embedding(window_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.window_size = window_size

    def forward(self, x_masked, mask_positions):
        n, h = x_masked.shape
        pos_ids = torch.arange(h, device=x_masked.device).unsqueeze(0).expand(n, h)
        hidden = self.token_embed(x_masked) + self.pos_embed(pos_ids)
        hidden = self.encoder(hidden)
        masked_hidden = hidden[torch.arange(n, device=x_masked.device), mask_positions]
        return self.head(masked_hidden)


def train_one_epoch(model, X, mask_id, device, batch_size, lr, seed, optimizer=None):
    rng = np.random.default_rng(seed)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    n = len(X)
    order = rng.permutation(n)
    total_loss = 0.0
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        X_masked, positions, targets = mask_batch(X[idx], mask_id, rng)
        xb = torch.from_numpy(X_masked).to(device)
        pb = torch.from_numpy(positions).to(device)
        yb = torch.from_numpy(targets).to(device)
        logits = model(xb, pb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    return total_loss / n, optimizer


def predict_masked_scores(model, X, device, top_k, seed, batch_size=4096):
    """Scores each window once, masking the SAME position for every window in a given call
    (position 0) rather than a random one, so scores are reproducible run-to-run for a fixed
    model -- unlike training's per-epoch random masking, evaluation needs a fixed protocol."""
    model.eval()
    model.to(device)
    n = len(X)
    scores = np.empty(n, dtype=np.float64)
    flags = np.empty(n, dtype=bool)
    mask_id = model.mask_id
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = start + batch_size
            chunk = X[start:end]
            targets = chunk[:, 0].copy()
            X_masked = chunk.copy()
            X_masked[:, 0] = mask_id
            xb = torch.from_numpy(X_masked).to(device)
            pb = torch.zeros(len(chunk), dtype=torch.long, device=device)
            yb = torch.from_numpy(targets).to(device)
            logits = model(xb, pb)
            log_probs = F.log_softmax(logits, dim=1)
            true_log_prob = log_probs.gather(1, yb.unsqueeze(1)).squeeze(1)
            scores[start:end] = (-true_log_prob).cpu().numpy()
            k = min(top_k, logits.shape[1])
            topk = torch.topk(logits, k=k, dim=1).indices
            hit = (topk == yb.unsqueeze(1)).any(dim=1)
            flags[start:end] = (~hit).cpu().numpy()
    return scores, flags


def build_synthetic_df(n_nodes=4, n_per_node=60, n_templates=6, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2020-01-01")
    for node_i in range(n_nodes):
        node = f"node-{node_i}"
        t = t0
        for i in range(n_per_node):
            t = t + pd.Timedelta(seconds=rng.uniform(0.5, 1.5))
            rows.append(
                {
                    "timestamp": t, "node": node, "anomaly": False,
                    "event_template": f"tmpl{rng.integers(0, n_templates)}", "raw_message": "msg",
                }
            )
    df = pd.DataFrame(rows)
    df["row_id"] = np.arange(len(df))
    return df


def self_test():
    df = build_synthetic_df()
    vocab = build_vocabulary(df["event_template"])
    df["template_id"] = encode_templates(df["event_template"], vocab)

    h = 10
    X, row_ids = build_masked_windows(df, "template_id", h)
    assert X.shape[1] == h
    assert len(X) > 0, "synthetic df produced zero windows -- check n_per_node > window size"

    set_seed(SEED)
    device = resolve_device("cpu")
    model = LogBertSmall(vocab_size=len(vocab), window_size=h)
    loss, _ = train_one_epoch(model, X, model.mask_id, device, batch_size=16, lr=1e-3, seed=SEED)
    assert np.isfinite(loss), f"non-finite loss: {loss}"

    scores, flags = predict_masked_scores(model, X, device, top_k=3, seed=SEED)
    assert len(scores) == len(X)
    assert np.isfinite(scores).all()

    print(f"windows: {len(X)}, vocab_size: {len(vocab)}, one-epoch loss: {loss:.4f}")
    print(f"score range: [{scores.min():.4f}, {scores.max():.4f}], flagged fraction: {flags.mean():.2%}")
    print("\nSELF-TEST PASSED: one training step + one scoring pass ran without error on synthetic data.")


def run_native(dataset, config, epochs, window_size, subsample, seed, device_name):
    device = resolve_device(device_name)
    set_seed(seed)
    df_clean = load_clean_by_node(config["clean_path"], subsample=subsample)
    is_train, _ = config["split_fn"](df_clean)
    df_train = df_clean.loc[is_train.to_numpy()].reset_index(drop=True)

    vocab = build_vocabulary(df_train["event_template"])
    df_clean["template_id"] = encode_templates(df_clean["event_template"], vocab)

    train_mask = is_train.to_numpy() & (~df_clean["anomaly"].to_numpy())
    df_train_normal = df_clean.loc[train_mask].reset_index(drop=True)

    X_train, _ = build_masked_windows(df_train_normal, "template_id", window_size)
    model = LogBertSmall(vocab_size=len(vocab), window_size=window_size)
    for epoch in range(epochs):
        loss, _ = train_one_epoch(model, X_train, model.mask_id, device, batch_size=256, lr=1e-3, seed=seed + epoch)
        print(f"  epoch {epoch + 1}/{epochs}: loss={loss:.4f}")

    X_full, row_ids_full = build_masked_windows(df_clean, "template_id", window_size)
    scores, flags = predict_masked_scores(model, X_full, device, top_k=5, seed=seed)
    row_score, row_flag = full_row_arrays(len(df_clean), row_ids_full, scores, flags)

    metrics = evaluate_against_native_labels(df_clean, is_train, row_score, row_flag)
    print("native-anomaly metrics:", metrics)
    return metrics


def run_injected(dataset, config, fault, epochs, window_size, subsample, seed, device_name):
    if "grid_labels_path" not in config:
        raise ValueError(
            f"{dataset} has no grid_labels_path in src.deeplog.DATASET_CONFIG -- injected-target "
            "eval needs native alert labels' train/test split AND injection grid labels together "
            "(bgl/thunderbird only; the additional Loghub datasets carry no native labels, see "
            "src/multi_dataset_registry.py)."
        )
    device = resolve_device(device_name)
    set_seed(seed)

    must_include_nodes = select_injected_nodes(config["injected_loaders"][fault], n=20, seed=seed)
    df_clean = load_clean_by_node(config["clean_path"], subsample=subsample, must_include_nodes=must_include_nodes)
    is_train, _ = config["split_fn"](df_clean)
    train_mask = is_train.to_numpy() & (~df_clean["anomaly"].to_numpy())
    df_train = df_clean.loc[train_mask].reset_index(drop=True)

    vocab = build_vocabulary(df_train["event_template"])
    df_clean["template_id"] = encode_templates(df_clean["event_template"], vocab)
    df_train["template_id"] = encode_templates(df_train["event_template"], vocab)

    X_train, _ = build_masked_windows(df_train, "template_id", window_size)
    model = LogBertSmall(vocab_size=len(vocab), window_size=window_size)
    for epoch in range(epochs):
        loss, _ = train_one_epoch(model, X_train, model.mask_id, device, batch_size=256, lr=1e-3, seed=seed + epoch)
        print(f"  epoch {epoch + 1}/{epochs}: loss={loss:.4f}")

    df_injected = load_by_node(config["injected_loaders"][fault](), subsample=subsample, must_include_nodes=must_include_nodes)
    df_injected["template_id"] = encode_templates(df_injected["event_template"], vocab)

    X_clean, rows_clean = build_masked_windows(df_clean, "template_id", window_size)
    X_injected, rows_injected = build_masked_windows(df_injected, "template_id", window_size)

    scores_clean, flags_clean = predict_masked_scores(model, X_clean, device, top_k=5, seed=seed)
    scores_injected, flags_injected = predict_masked_scores(model, X_injected, device, top_k=5, seed=seed)

    row_score_clean, row_flag_clean = full_row_arrays(len(df_clean), rows_clean, scores_clean, flags_clean)
    row_score_injected, row_flag_injected = full_row_arrays(len(df_injected), rows_injected, scores_injected, flags_injected)

    # Invariance check: the model never reads timestamps and injection never changes template
    # content or order, so per-row scores keyed by row_id (a position-based join) must match.
    common_row_ids = np.intersect1d(df_clean["row_id"].to_numpy(), df_injected["row_id"].to_numpy())
    diff = np.abs(
        pd.Series(row_score_clean, index=df_clean["row_id"].to_numpy()).loc[common_row_ids].to_numpy()
        - pd.Series(row_score_injected, index=df_injected["row_id"].to_numpy()).loc[common_row_ids].to_numpy()
    )
    max_abs_score_diff = float(diff.max()) if len(diff) else float("nan")

    injected_metrics = evaluate_against_injected(df_injected, row_score_injected, row_flag_injected, config["grid_labels_path"][fault])

    # AUC_placebo: the same injected-span row-level truth, scored against clean scores on the
    # clean frame's own eval grid (row-identity mapping, as in src/placebo_sweep.py).
    grid_labels = pd.read_csv(config["grid_labels_path"][fault])
    anomalous_cells = set(zip(grid_labels["node"], grid_labels["window_idx"]))
    df_eval_injected = assign_eval_grid(df_injected)
    row_true_by_id = pd.Series(
        [k in anomalous_cells for k in df_eval_injected["eval_window_key"]], index=df_eval_injected["row_id"].to_numpy()
    )
    df_eval_clean = assign_eval_grid(df_clean)
    row_true_aligned_clean = df_eval_clean["row_id"].map(row_true_by_id).fillna(False)
    row_score_clean_by_id = pd.Series(row_score_clean, index=df_clean["row_id"].to_numpy())
    placebo_auc_metrics = compute_grid_auc(df_eval_clean, row_score_clean_by_id, row_true_aligned_clean)

    metrics = {
        "dataset": dataset, "fault": fault, "detector": "logbert_small",
        "max_abs_score_diff_fixed_count_predicted_zero": max_abs_score_diff,
        **{f"injected_{k}": v for k, v in injected_metrics.items()},
        **{f"placebo_{k}": v for k, v in placebo_auc_metrics.items()},
    }
    print("logbert_small injected-eval metrics:", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Run one train + score step on an in-memory synthetic DataFrame; touches no real data.")
    ap.add_argument("--dataset", choices=sorted(DATASET_CONFIG), default=None)
    ap.add_argument("--fault", choices=FAULTS, default="stall")
    ap.add_argument("--eval-target", choices=["native", "injected"], default="native")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.dataset is None:
        print("ERROR: --dataset is required for a real run.", file=sys.stderr)
        sys.exit(1)

    config = DATASET_CONFIG[args.dataset]
    if args.eval_target == "native":
        run_native(args.dataset, config, args.epochs, args.window_size, args.subsample, args.seed, args.device)
    else:
        run_injected(args.dataset, config, args.fault, args.epochs, args.window_size, args.subsample, args.seed, args.device)


if __name__ == "__main__":
    main()
