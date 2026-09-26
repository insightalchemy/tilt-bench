"""
Time-aware DeepLog variant on BGL (paper Section V-E). With --time-aware, each LSTM input step is
(template id, binned log inter-arrival gap) instead of the template id alone. Gaps are binned into
N_GAP_BINS quantile bins fit on the train period's log(gap) distribution, plus one reserved bin for
a node's first event (no preceding gap). Both are one-hot encoded and concatenated before the
LSTM. The model and training loop follow src/deeplog.py and are kept self-contained here.

Reports:
  - clean vs injected score differences (expected nonzero: gap bins depend on timestamps),
  - AUC on BGL's native anomaly labels (test period, as in deeplog.py --eval-target native),
  - AUC_injected and AUC_placebo on injected stalls, using the same trained model and the
    row-identity labels of src/placebo_sweep.py.

Writes results/deeplog_timeaware_bgl.csv and the model checkpoint results/deeplog_timeaware_bgl.pt.

Usage:
    python src/deeplog_timeaware.py --time-aware --epochs 20 --device cuda
    # quick end-to-end check on a subsample:
    python src/deeplog_timeaware.py --time-aware --subsample 50000 --subsample-include-injected 5 \
        --epochs 1 --device cpu --check-invariance
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view

from src.deeplog import full_row_arrays, load_by_node, load_clean_by_node, select_injected_nodes
from src.metrics import assign_eval_grid
from src.pipeline_common import load_bgl_injected_stall as load_bgl_stall
from src.run_baseline_detectors import chronological_split as chronological_split_bgl
from src.timing_baseline import MAD_FLOOR_SEC, add_sequence_context
from src.windowing_sweep import label_rows_in_injected_span

OOV_TOKEN = "<OOV>"
N_GAP_BINS = 16
MISSING_GAP_BIN = N_GAP_BINS

DATASET_CONFIG = {
    "bgl": {
        "clean_path": Path("data/processed/bgl_parsed.parquet"),
        "split_fn": chronological_split_bgl,
        "injected_loaders": {"stall": load_bgl_stall},
        "injection_labels_path": {"stall": Path("data/processed/injection_labels_stall.csv")},
    }
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_vocabulary(templates):
    uniq = sorted(templates.unique().tolist())
    vocab = {t: i + 1 for i, t in enumerate(uniq)}
    vocab[OOV_TOKEN] = 0
    return vocab


def encode_templates(templates, vocab):
    oov_id = vocab[OOV_TOKEN]
    return templates.map(lambda t: vocab.get(t, oov_id)).to_numpy(dtype=np.int64)


def fit_gap_bin_edges(df_train):
    df = add_sequence_context(df_train.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    gap = df["gap_prev_s"].dropna().to_numpy()
    gap = gap[gap >= 0]
    log_gap = np.log(np.clip(gap, MAD_FLOOR_SEC, None))
    quantiles = np.linspace(0, 1, N_GAP_BINS + 1)[1:-1]
    edges = np.quantile(log_gap, quantiles) if len(log_gap) else np.array([])
    return edges


def encode_gap_bins(df, bin_edges):
    df = add_sequence_context(df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True))
    gap = df["gap_prev_s"].to_numpy(dtype=float)
    log_gap = np.log(np.clip(gap, MAD_FLOOR_SEC, None))
    bins = np.digitize(log_gap, bin_edges) if len(bin_edges) else np.zeros(len(df), dtype=np.int64)
    bins = np.where(np.isnan(gap), MISSING_GAP_BIN, bins).astype(np.int64)
    df = df.copy()
    df["gap_bin"] = bins
    return df


def build_windows(df, template_col, gap_col, h):
    X_template_chunks, X_gap_chunks, y_chunks, row_id_chunks = [], [], [], []
    for _, group in df.groupby("node", sort=False):
        templates = group[template_col].to_numpy()
        gaps = group[gap_col].to_numpy()
        row_ids = group["row_id"].to_numpy()
        if len(templates) <= h:
            continue
        X_template_chunks.append(sliding_window_view(templates, h)[:-1])
        X_gap_chunks.append(sliding_window_view(gaps, h)[:-1])
        y_chunks.append(templates[h:])
        row_id_chunks.append(row_ids[h:])
    if not X_template_chunks:
        empty_i = np.empty((0, h), dtype=np.int64)
        return empty_i, empty_i, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return (
        np.concatenate(X_template_chunks),
        np.concatenate(X_gap_chunks),
        np.concatenate(y_chunks),
        np.concatenate(row_id_chunks),
    )


class TimeAwareDeepLog(nn.Module):
    def __init__(self, vocab_size, n_gap_bins, hidden_size, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_gap_bins = n_gap_bins
        input_size = vocab_size + n_gap_bins
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_template, x_gap):
        template_onehot = F.one_hot(x_template, num_classes=self.vocab_size).float()
        gap_onehot = F.one_hot(x_gap, num_classes=self.n_gap_bins).float()
        x = torch.cat([template_onehot, gap_onehot], dim=-1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def train_model(model, X_template, X_gap, y, epochs, batch_size, lr, device, seed):
    model.to(device)
    model.train()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_template), torch.from_numpy(X_gap), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    for _ in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for xb_t, xb_g, yb in loader:
            xb_t, xb_g, yb = xb_t.to(device), xb_g.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb_t, xb_g), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        history.append(total_loss / max(n_batches, 1))
    return history


@torch.no_grad()
def predict_scores(model, X_template, X_gap, y, device, top_k, batch_size=4096):
    model.eval()
    model.to(device)
    n = len(X_template)
    scores = np.empty(n, dtype=np.float64)
    flags = np.empty(n, dtype=bool)
    for start in range(0, n, batch_size):
        end = start + batch_size
        xb_t = torch.from_numpy(X_template[start:end]).to(device)
        xb_g = torch.from_numpy(X_gap[start:end]).to(device)
        yb = torch.from_numpy(y[start:end]).to(device)
        logits = model(xb_t, xb_g)
        log_probs = F.log_softmax(logits, dim=1)
        true_log_prob = log_probs.gather(1, yb.unsqueeze(1)).squeeze(1)
        scores[start:end] = (-true_log_prob).cpu().numpy()
        k = min(top_k, logits.shape[1])
        topk = torch.topk(logits, k=k, dim=1).indices
        hit = (topk == yb.unsqueeze(1)).any(dim=1)
        flags[start:end] = (~hit).cpu().numpy()
    return scores, flags


def run_invariance_check(model, df_clean, df_injected, h, top_k, device):
    X_t_clean, X_g_clean, y_clean, row_ids_clean = build_windows(df_clean, "template_id", "gap_bin", h)
    X_t_inj, X_g_inj, y_inj, row_ids_inj = build_windows(df_injected, "template_id", "gap_bin", h)

    n_common = min(len(X_t_clean), len(X_t_inj))
    row_ids_aligned = bool(np.array_equal(row_ids_clean[:n_common], row_ids_inj[:n_common]))

    scores_clean, flags_clean = predict_scores(model, X_t_clean[:n_common], X_g_clean[:n_common], y_clean[:n_common], device, top_k)
    scores_inj, flags_inj = predict_scores(model, X_t_inj[:n_common], X_g_inj[:n_common], y_inj[:n_common], device, top_k)
    score_diff = np.abs(scores_clean - scores_inj)

    result = {
        "n_common_windows": n_common,
        "row_id_alignment_identical": row_ids_aligned,
        "n_score_differences": int((score_diff > 1e-9).sum()),
        "n_topk_flag_differences": int((flags_clean != flags_inj).sum()),
        "max_abs_score_diff": float(score_diff.max()) if n_common else float("nan"),
    }
    print("=== Invariance check (time-aware) ===")
    for k_, v_ in result.items():
        print(f"  {k_}: {v_}")
    return result


def evaluate_against_native_labels(df_full, is_train, row_score, row_flag):
    from src.metrics import evaluate_common_unit
    from src.auc_metrics import compute_grid_auc

    df_test = df_full.loc[~is_train.to_numpy()].reset_index(drop=True)
    df_eval = assign_eval_grid(df_test)

    row_flag_by_id = pd.Series(row_flag, index=df_full["row_id"].to_numpy())
    row_score_by_id = pd.Series(row_score, index=df_full["row_id"].to_numpy())
    row_predicted = df_eval["row_id"].map(row_flag_by_id).fillna(False)
    row_true = df_eval["anomaly"]

    common_unit = evaluate_common_unit(df_eval, row_predicted, row_true)
    auc_metrics = compute_grid_auc(df_eval, row_score_by_id, row_true)
    return {"precision": common_unit["precision"], "recall": common_unit["recall"], "f1": common_unit["f1"], **auc_metrics}


def evaluate_placebo(df_full, row_score, row_true_by_id):
    from src.auc_metrics import compute_grid_auc

    df_eval = assign_eval_grid(df_full)
    row_score_by_id = pd.Series(row_score, index=df_full["row_id"].to_numpy())
    row_true_aligned = df_eval["row_id"].map(row_true_by_id).fillna(False)
    return compute_grid_auc(df_eval, row_score_by_id, row_true_aligned)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-aware", action="store_true", required=True)
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=9)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--subsample-include-injected", type=int, default=None)
    ap.add_argument("--check-invariance", action="store_true")
    ap.add_argument("--out-csv", type=Path, default=Path("results/deeplog_timeaware_bgl.csv"))
    ap.add_argument("--save-model", type=Path, default=Path("results/deeplog_timeaware_bgl.pt"))
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    config = DATASET_CONFIG["bgl"]
    t0 = time.time()

    must_include_nodes = None
    if args.subsample_include_injected is not None:
        must_include_nodes = select_injected_nodes(config["injected_loaders"]["stall"], args.subsample_include_injected, args.seed)

    df_clean = load_clean_by_node(config["clean_path"], subsample=args.subsample, must_include_nodes=must_include_nodes)
    is_train, cutoff = config["split_fn"](df_clean)
    train_mask = is_train.to_numpy() & (~df_clean["anomaly"].to_numpy())
    df_train = df_clean.loc[train_mask].copy()

    vocab = build_vocabulary(df_train["event_template"])
    df_clean["template_id"] = encode_templates(df_clean["event_template"], vocab)
    df_train["template_id"] = encode_templates(df_train["event_template"], vocab)

    bin_edges = fit_gap_bin_edges(df_train)
    df_train = encode_gap_bins(df_train, bin_edges)
    df_clean = encode_gap_bins(df_clean, bin_edges)

    X_t_train, X_g_train, y_train, _ = build_windows(df_train, "template_id", "gap_bin", args.window_size)
    print(f"train windows: {len(X_t_train)}, vocab size (incl. OOV): {len(vocab)}, gap bins: {N_GAP_BINS + 1}, cutoff: {cutoff}", flush=True)

    model = TimeAwareDeepLog(len(vocab), N_GAP_BINS + 1, args.hidden_size, args.num_layers)
    t_train0 = time.time()
    train_history = train_model(model, X_t_train, X_g_train, y_train, args.epochs, args.batch_size, args.lr, device, args.seed)
    train_time_s = time.time() - t_train0
    print(f"train done in {train_time_s:.0f}s, final_train_loss={train_history[-1]:.4f}", flush=True)

    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved model checkpoint to {args.save_model}", flush=True)

    df_injected = load_by_node(config["injected_loaders"]["stall"](), subsample=args.subsample, must_include_nodes=must_include_nodes)
    df_injected["template_id"] = encode_templates(df_injected["event_template"], vocab)
    df_injected = encode_gap_bins(df_injected, bin_edges)

    X_t_infer, X_g_infer, y_infer, target_row_ids = build_windows(df_injected, "template_id", "gap_bin", args.window_size)
    scores_inj, flags_inj = predict_scores(model, X_t_infer, X_g_infer, y_infer, device, args.top_k)
    row_score_inj, row_flag_inj = full_row_arrays(len(df_injected), target_row_ids, scores_inj, flags_inj)

    X_t_clean_infer, X_g_clean_infer, y_clean_infer, target_row_ids_clean = build_windows(df_clean, "template_id", "gap_bin", args.window_size)
    scores_clean, flags_clean = predict_scores(model, X_t_clean_infer, X_g_clean_infer, y_clean_infer, device, args.top_k)
    row_score_clean, row_flag_clean = full_row_arrays(len(df_clean), target_row_ids_clean, scores_clean, flags_clean)

    labels_df = pd.read_csv(config["injection_labels_path"]["stall"], parse_dates=["start", "end"])
    row_true_std = label_rows_in_injected_span(df_injected, labels_df)
    row_true_by_id = pd.Series(row_true_std.to_numpy(), index=df_injected["row_id"].to_numpy())

    native_metrics = evaluate_against_native_labels(df_clean, is_train, row_score_clean, row_flag_clean)
    injected_metrics = evaluate_placebo(df_injected, row_score_inj, row_true_by_id)
    placebo_metrics = evaluate_placebo(df_clean, row_score_clean, row_true_by_id)

    metrics = {
        "dataset": "bgl",
        "detector": "deeplog_timeaware",
        "window_size": args.window_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "subsample": args.subsample,
        "vocab_size": len(vocab),
        "n_train_windows": len(X_t_train),
        "train_time_s": train_time_s,
        "final_train_loss": train_history[-1],
        "native_auc_roc": native_metrics["auc_roc"],
        "native_precision": native_metrics["precision"],
        "native_recall": native_metrics["recall"],
        "auc_injected": injected_metrics["auc_roc"],
        "auc_placebo": placebo_metrics["auc_roc"],
        "delta": injected_metrics["auc_roc"] - placebo_metrics["auc_roc"],
        "total_time_s": time.time() - t0,
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.out_csv, index=False)
    print(pd.Series(metrics).to_string())
    print(f"Wrote {args.out_csv}")

    if args.check_invariance:
        run_invariance_check(model, df_clean, df_injected, args.window_size, args.top_k, device)


if __name__ == "__main__":
    main()
