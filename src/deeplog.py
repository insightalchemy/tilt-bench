"""
DeepLog baseline (Du et al., "DeepLog: Anomaly Detection and Diagnosis from System Logs through
Deep Learning", CCS 2017): stacked LSTM over one-hot per-node template-ID sequences, trained on
clean train-period native-normal rows only. Two evaluation targets, selected by --eval-target:
"injected" scores the trained model against the project's n=100 injected-span ground truth;
"native" scores it against BGL/Thunderbird's own native alert labels on the chronological
test-period rows only. Both reuse the existing eval-grid harness (src.metrics, src.auc_metrics,
src.core_result_stall_final) rather than reimplementing scoring. Never trains on injected data or
on native-anomalous rows.

Run (GPU server):
    python src/deeplog.py --dataset bgl --fault stall --eval-target injected --device cuda
    python src/deeplog.py --dataset bgl --fault burst --eval-target injected --device cuda
    python src/deeplog.py --dataset thunderbird --fault stall --eval-target injected --device cuda
    python src/deeplog.py --dataset thunderbird --fault burst --eval-target injected --device cuda
    python src/deeplog.py --dataset bgl --eval-target native --device cuda
    python src/deeplog.py --dataset thunderbird --eval-target native --device cuda

Local smoke test (never full training locally):
    python src/deeplog.py --dataset bgl --fault stall --eval-target injected --subsample 50000 --subsample-include-injected 5 --epochs 1 --device cpu --check-invariance
    python src/deeplog.py --dataset bgl --eval-target native --subsample 200000 --epochs 1 --device cpu
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

from src.auc_metrics import compute_grid_auc
from src.core_result_burst import load_injected_burst as load_bgl_burst
from src.core_result_stall import load_injected as load_bgl_stall
from src.core_result_stall_final import score_detector
from src.core_result_thunderbird import chronological_split as chronological_split_tb, load_clean as load_tb_clean, load_injected as load_tb_injected
from src.metrics import assign_eval_grid, evaluate_common_unit
from src.run_baseline_detectors import chronological_split as chronological_split_bgl

OOV_TOKEN = "<OOV>"

DATASET_CONFIG = {
    "bgl": {
        "clean_path": Path("data/processed/bgl_parsed.parquet"),
        "split_fn": chronological_split_bgl,
        "injected_loaders": {"stall": load_bgl_stall, "burst": load_bgl_burst},
        "grid_labels_path": {
            "stall": Path("data/processed/injection_grid_labels_stall.csv"),
            "burst": Path("data/processed/injection_grid_labels_burst.csv"),
        },
    },
    "thunderbird": {
        "clean_path": Path("data/processed/thunderbird_parsed.parquet"),
        "split_fn": chronological_split_tb,
        "injected_loaders": {"stall": lambda: load_tb_injected("stall"), "burst": lambda: load_tb_injected("burst")},
        "grid_labels_path": {
            "stall": Path("data/processed/thunderbird_injection_grid_labels_stall.csv"),
            "burst": Path("data/processed/thunderbird_injection_grid_labels_burst.csv"),
        },
    },
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


def load_by_node(df, subsample=None, must_include_nodes=None):
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    if subsample is not None:
        if must_include_nodes:
            included = df[df["node"].isin(must_include_nodes)]
            remainder_budget = max(subsample - len(included), 0)
            remainder = df[~df["node"].isin(must_include_nodes)].head(remainder_budget)
            df = pd.concat([included, remainder]).sort_values(["node", "timestamp"], kind="mergesort")
        else:
            df = df.head(subsample)
        df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def load_clean_by_node(path, subsample=None, must_include_nodes=None):
    return load_by_node(pd.read_parquet(path), subsample=subsample, must_include_nodes=must_include_nodes)


def select_injected_nodes(loader_fn, n, seed):
    df = loader_fn()
    injected_nodes = df.loc[df["injected_row"], "node"].unique().tolist()
    rng = np.random.default_rng(seed)
    size = min(n, len(injected_nodes))
    return set(rng.choice(injected_nodes, size=size, replace=False).tolist())


def build_vocabulary(templates):
    uniq = sorted(templates.unique().tolist())
    vocab = {t: i + 1 for i, t in enumerate(uniq)}
    vocab[OOV_TOKEN] = 0
    return vocab


def encode_templates(templates, vocab):
    oov_id = vocab[OOV_TOKEN]
    return templates.map(lambda t: vocab.get(t, oov_id)).to_numpy(dtype=np.int64)


def build_windows(df, encoded_col, h):
    X_chunks, y_chunks, row_id_chunks = [], [], []
    for _, group in df.groupby("node", sort=False):
        ids = group[encoded_col].to_numpy()
        row_ids = group["row_id"].to_numpy()
        if len(ids) <= h:
            continue
        windows = sliding_window_view(ids, h)[:-1]
        X_chunks.append(windows)
        y_chunks.append(ids[h:])
        row_id_chunks.append(row_ids[h:])
    if not X_chunks:
        return np.empty((0, h), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_chunks), np.concatenate(y_chunks), np.concatenate(row_id_chunks)


class DeepLog(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x_onehot = F.one_hot(x, num_classes=self.vocab_size).float()
        _, (h_n, _) = self.lstm(x_onehot)
        return self.fc(h_n[-1])


def train_model(model, X, y, epochs, batch_size, lr, device, seed):
    model.to(device)
    model.train()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    for _ in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        history.append(total_loss / max(n_batches, 1))
    return history


@torch.no_grad()
def predict_scores(model, X, y, device, top_k, batch_size=4096):
    model.eval()
    model.to(device)
    n = len(X)
    scores = np.empty(n, dtype=np.float64)
    flags = np.empty(n, dtype=bool)
    for start in range(0, n, batch_size):
        end = start + batch_size
        xb = torch.from_numpy(X[start:end]).to(device)
        yb = torch.from_numpy(y[start:end]).to(device)
        logits = model(xb)
        log_probs = F.log_softmax(logits, dim=1)
        true_log_prob = log_probs.gather(1, yb.unsqueeze(1)).squeeze(1)
        scores[start:end] = (-true_log_prob).cpu().numpy()
        k = min(top_k, logits.shape[1])
        topk = torch.topk(logits, k=k, dim=1).indices
        hit = (topk == yb.unsqueeze(1)).any(dim=1)
        flags[start:end] = (~hit).cpu().numpy()
    return scores, flags


def full_row_arrays(n_rows, target_row_ids, scores, flags):
    row_score = np.zeros(n_rows, dtype=np.float64)
    row_flag = np.zeros(n_rows, dtype=bool)
    row_score[target_row_ids] = scores
    row_flag[target_row_ids] = flags
    return row_score, row_flag


def evaluate_against_injected(df_full, row_score, row_flag, grid_labels_path):
    grid_labels = pd.read_csv(grid_labels_path)
    anomalous_cells = set(zip(grid_labels["node"], grid_labels["window_idx"]))
    injection_ids = sorted(grid_labels["injection_id"].unique())

    df_eval = assign_eval_grid(df_full)
    row_true = pd.Series([k in anomalous_cells for k in df_eval["eval_window_key"]], index=df_eval["row_id"].to_numpy())
    row_true_aligned = df_eval["row_id"].map(row_true).fillna(False)

    row_pred_by_id = pd.Series(row_flag, index=df_full["row_id"].to_numpy())
    lift_metrics, _ = score_detector(df_eval, row_pred_by_id, row_true_aligned, grid_labels, injection_ids)

    row_score_by_id = pd.Series(row_score, index=df_full["row_id"].to_numpy())
    auc_metrics = compute_grid_auc(df_eval, row_score_by_id, row_true_aligned)

    return {**lift_metrics, **auc_metrics}


def evaluate_against_native_labels(df_full, is_train, row_score, row_flag):
    df_test = df_full.loc[~is_train.to_numpy()].reset_index(drop=True)
    df_eval = assign_eval_grid(df_test)

    row_flag_by_id = pd.Series(row_flag, index=df_full["row_id"].to_numpy())
    row_score_by_id = pd.Series(row_score, index=df_full["row_id"].to_numpy())
    row_predicted = df_eval["row_id"].map(row_flag_by_id).fillna(False)
    row_true = df_eval["anomaly"]

    common_unit = evaluate_common_unit(df_eval, row_predicted, row_true)
    grid_flagged_frac = common_unit["n_flagged"] / common_unit["n_eval_cells"] if common_unit["n_eval_cells"] else float("nan")
    lift = common_unit["recall"] / grid_flagged_frac if grid_flagged_frac else float("nan")

    auc_metrics = compute_grid_auc(df_eval, row_score_by_id, row_true)

    return {
        "precision": common_unit["precision"],
        "recall": common_unit["recall"],
        "f1": common_unit["f1"],
        "grid_flagged_frac": grid_flagged_frac,
        "lift": lift,
        **auc_metrics,
    }


def run_invariance_check(model, df_clean, df_injected, h, top_k, device):
    X_clean, y_clean, row_ids_clean = build_windows(df_clean, "template_id", h)
    X_inj, y_inj, row_ids_inj = build_windows(df_injected, "template_id", h)

    n_common = min(len(X_clean), len(X_inj))
    row_ids_aligned = bool(np.array_equal(row_ids_clean[:n_common], row_ids_inj[:n_common]))
    n_input_diff = int((X_clean[:n_common] != X_inj[:n_common]).any(axis=1).sum())
    n_target_diff = int((y_clean[:n_common] != y_inj[:n_common]).sum())

    scores_clean, flags_clean = predict_scores(model, X_clean[:n_common], y_clean[:n_common], device, top_k)
    scores_inj, flags_inj = predict_scores(model, X_inj[:n_common], y_inj[:n_common], device, top_k)
    score_diff = np.abs(scores_clean - scores_inj)
    n_score_diff = int((score_diff > 1e-9).sum())
    n_flag_diff = int((flags_clean != flags_inj).sum())
    max_score_diff = float(score_diff.max()) if n_common else float("nan")

    df_eval_clean = assign_eval_grid(df_clean).set_index("row_id")["eval_window_key"]
    df_eval_inj = assign_eval_grid(df_injected).set_index("row_id")["eval_window_key"]
    common_row_ids = df_eval_clean.index.intersection(df_eval_inj.index)
    n_grid_diff = int((df_eval_clean.loc[common_row_ids] != df_eval_inj.loc[common_row_ids]).sum())

    result = {
        "n_common_windows": n_common,
        "row_id_alignment_identical": row_ids_aligned,
        "n_input_window_mismatches": n_input_diff,
        "n_target_template_mismatches": n_target_diff,
        "n_score_differences": n_score_diff,
        "n_topk_flag_differences": n_flag_diff,
        "max_abs_score_diff": max_score_diff,
        "n_common_rows_for_grid_check": len(common_row_ids),
        "n_grid_cell_reassignments": n_grid_diff,
    }
    print("=== Invariance check ===")
    for k_, v_ in result.items():
        print(f"  {k_}: {v_}")
    return result


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["bgl", "thunderbird"], required=True)
    ap.add_argument("--fault", choices=["stall", "burst"], default=None)
    ap.add_argument("--eval-target", choices=["injected", "native"], default="injected")
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=9)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--subsample-include-injected", type=int, default=None)
    ap.add_argument("--check-invariance", action="store_true")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--save-model", type=Path, default=None)
    ap.add_argument("--load-model", type=Path, default=None)
    args = ap.parse_args()
    if args.eval_target == "injected" and args.fault is None:
        ap.error("--fault is required when --eval-target injected")
    if args.subsample_include_injected is not None and args.eval_target != "injected":
        ap.error("--subsample-include-injected only applies to --eval-target injected")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    config = DATASET_CONFIG[args.dataset]
    t0 = time.time()

    must_include_nodes = None
    if args.subsample_include_injected is not None:
        must_include_nodes = select_injected_nodes(config["injected_loaders"][args.fault], args.subsample_include_injected, args.seed)

    df_clean = load_clean_by_node(config["clean_path"], subsample=args.subsample, must_include_nodes=must_include_nodes)
    is_train, cutoff = config["split_fn"](df_clean)
    train_mask = is_train.to_numpy() & (~df_clean["anomaly"].to_numpy())
    df_train = df_clean.loc[train_mask].copy()

    vocab = build_vocabulary(df_train["event_template"])
    df_clean["template_id"] = encode_templates(df_clean["event_template"], vocab)
    df_train["template_id"] = encode_templates(df_train["event_template"], vocab)

    X_train, y_train, _ = build_windows(df_train, "template_id", args.window_size)
    print(f"train windows: {len(X_train)}, vocab size (incl. OOV): {len(vocab)}, chronological cutoff: {cutoff}")

    model = DeepLog(len(vocab), args.hidden_size, args.num_layers)
    t_train0 = time.time()
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        train_history = None
    else:
        train_history = train_model(model, X_train, y_train, args.epochs, args.batch_size, args.lr, device, args.seed)
    train_time_s = time.time() - t_train0
    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)

    metrics = {
        "dataset": args.dataset,
        "eval_target": args.eval_target,
        "detector": "deeplog",
        "window_size": args.window_size,
        "top_k": args.top_k,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "seed": args.seed,
        "subsample": args.subsample,
        "subsample_include_injected": args.subsample_include_injected,
        "vocab_size": len(vocab),
        "n_train_windows": len(X_train),
        "train_time_s": train_time_s,
        "final_train_loss": train_history[-1] if train_history else None,
    }

    df_injected = None
    if args.eval_target == "injected":
        df_injected = load_by_node(config["injected_loaders"][args.fault](), subsample=args.subsample, must_include_nodes=must_include_nodes)
        df_injected["template_id"] = encode_templates(df_injected["event_template"], vocab)
        oov_rows = int((df_injected["template_id"] == vocab[OOV_TOKEN]).sum())

        t_infer0 = time.time()
        X_infer, y_infer, target_row_ids = build_windows(df_injected, "template_id", args.window_size)
        scores, flags = predict_scores(model, X_infer, y_infer, device, args.top_k)
        infer_time_s = time.time() - t_infer0
        row_score, row_flag = full_row_arrays(len(df_injected), target_row_ids, scores, flags)

        eval_metrics = evaluate_against_injected(df_injected, row_score, row_flag, config["grid_labels_path"][args.fault])
        metrics.update(eval_metrics)
        metrics.update(
            {
                "fault_type": args.fault,
                "oov_rows": oov_rows,
                "oov_frac": oov_rows / len(df_injected),
                "n_infer_windows": len(X_infer),
                "infer_time_s": infer_time_s,
            }
        )
    else:
        oov_rows_test = int((df_clean.loc[~is_train.to_numpy(), "template_id"] == vocab[OOV_TOKEN]).sum())
        n_test_rows = int((~is_train.to_numpy()).sum())

        t_infer0 = time.time()
        X_infer, y_infer, target_row_ids = build_windows(df_clean, "template_id", args.window_size)
        scores, flags = predict_scores(model, X_infer, y_infer, device, args.top_k)
        infer_time_s = time.time() - t_infer0
        row_score, row_flag = full_row_arrays(len(df_clean), target_row_ids, scores, flags)

        eval_metrics = evaluate_against_native_labels(df_clean, is_train, row_score, row_flag)
        metrics.update(eval_metrics)
        metrics.update(
            {
                "fault_type": None,
                "oov_rows": oov_rows_test,
                "oov_frac": oov_rows_test / n_test_rows if n_test_rows else float("nan"),
                "n_infer_windows": len(X_infer),
                "infer_time_s": infer_time_s,
            }
        )

    metrics["total_time_s"] = time.time() - t0

    default_name = f"deeplog_{args.dataset}_{args.fault}.csv" if args.eval_target == "injected" else f"deeplog_{args.dataset}_native.csv"
    out_csv = args.out_csv or Path("results") / default_name
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    print(pd.Series(metrics).to_string())
    print(f"Wrote {out_csv}")

    if args.check_invariance:
        if args.eval_target != "injected":
            raise ValueError("--check-invariance requires --eval-target injected")
        run_invariance_check(model, df_clean, df_injected, args.window_size, args.top_k, device)


if __name__ == "__main__":
    main()
