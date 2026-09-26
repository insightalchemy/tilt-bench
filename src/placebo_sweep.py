"""
Placebo-controlled windowing sweep (paper Section V-D, Table V, Fig. 4).

AUC computed against injected labels mixes a detector's response to the fault with its prior
preference for the windows where faults happen to be placed. Each detector is therefore scored
twice per (fault, scheme) cell:

  AUC_injected: injected-data scores vs injected-span labels.
  AUC_placebo:  clean-data scores vs the same labels, mapped by row identity. Injection is
                row-preserving (only timestamps move), so a clean row is positive iff its
                injected counterpart (same row_id) lies in an injected span. Clean rows are
                aggregated into clean windows under the same scheme.
  delta = AUC_injected - AUC_placebo, the fault-attributable detection.

A cluster bootstrap over injection events gives a 95% interval on delta. It is paired: both AUCs
in a replicate use the same resampled injection set. It uses the searchsorted technique of
src/auc_bootstrap_ci.py, with injection ids derived per window from the node.

Detectors: count_vector_pca, isolation_forest_counts, and the LogAnomaly-style detector (content),
plus the z-score and log-ratio timing detectors. Placebo timing scores are obtained by scoring the
clean frame against the same clean baseline.

Sanity check: under fixed-count windowing, every content detector must have delta exactly 0
(Proposition 1). A nonzero value indicates a timestamp-dependence bug, and the run stops.

Writes:
  results/placebo/placebo_sweep_{dataset}.csv / .md
  results/checkpoints/placebo_{dataset}_{fault}_checkpoint.csv  (rewritten after every scheme)
  data/processed/scores/{dataset}_{fault}_{scheme}.parquet      (window-level clean and injected
                                                                  scores, used by make_figures.py)

Usage:
    python src/placebo_sweep.py --dataset bgl
    python src/placebo_sweep.py --dataset thunderbird
    python src/placebo_sweep.py --dataset openstack --fixed-time-sizes 60 --fixed-count-sizes 50
    # quick end-to-end check on a subsample:
    python src/placebo_sweep.py --dataset bgl --fixed-time-sizes 15 30 --fixed-count-sizes 20 50 \
        --subsample 50000 --subsample-include-injected 5
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.deeplog import load_by_node, load_clean_by_node, select_injected_nodes
from src.detectors.windowing import build_vocabulary, build_windows
from src.loganomaly_invariance import LogAnomalyDetector, build_feature_matrix, build_template2vec
from src.metrics import rows_to_grid, rows_to_grid_max
from src.run_baseline_detectors import THRESHOLD_PERCENTILE
from src.run_baseline_detectors import chronological_split as generic_chronological_split
from src.windowing_sweep import (
    DATASET_CONFIG,
    assign_fixed_count_window,
    build_fixed_count_windows_table,
    build_timing_scores,
    compare_window_scores,
    count_grid_reassignments,
    evaluate_cell,
    fit_count_detectors_on_scheme,
    label_rows_in_injected_span,
    load_pos,
    score_count_detectors,
    scores_by_window_key,
    verify_std_pos_equivalence,
)


def load_spirit_injected(fault):
    path = Path(f"data/processed/spirit_injected_{fault}.parquet")
    df = pd.read_parquet(path)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


SPIRIT_CONFIG = {
    "clean_path": Path("data/processed/spirit_parsed.parquet"),
    "split_fn": generic_chronological_split,
    "injected_loaders": {"stall": lambda: load_spirit_injected("stall"), "burst": lambda: load_spirit_injected("burst")},
    "injected_paths": {
        "stall": Path("data/processed/spirit_injected_stall.parquet"),
        "burst": Path("data/processed/spirit_injected_burst.parquet"),
    },
    "injection_labels_path": {
        "stall": Path("data/processed/spirit_injection_labels_stall.csv"),
        "burst": Path("data/processed/spirit_injection_labels_burst.csv"),
    },
}

from src.multi_dataset_registry import DATASETS as LOGHUB_DATASETS
from src.multi_dataset_registry import make_generic_dataset_config as _make_loghub_config

ALL_DATASET_CONFIG = {
    **DATASET_CONFIG,
    "spirit": SPIRIT_CONFIG,
    **{name: _make_loghub_config(name) for name in LOGHUB_DATASETS},
}


def _load_injected_from_path(path):
    df = pd.read_parquet(path)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def _suffixed_path(path, suffix):
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def apply_injection_suffix(config, suffix):
    """Point a dataset config at injected/label files with `suffix` (e.g. "_n300", matching
    src.injector's naming for non-default --n-injections runs) inserted before each extension.
    The clean data is unchanged. suffix="" returns config unchanged."""
    if not suffix:
        return config
    injected_paths = {fault: _suffixed_path(path, suffix) for fault, path in config["injected_paths"].items()}
    injection_labels_path = {fault: _suffixed_path(path, suffix) for fault, path in config["injection_labels_path"].items()}
    injected_loaders = {fault: (lambda path=path: _load_injected_from_path(path)) for fault, path in injected_paths.items()}
    return {
        **config,
        "injected_paths": injected_paths,
        "injection_labels_path": injection_labels_path,
        "injected_loaders": injected_loaders,
    }

FAULTS = ["stall", "burst"]
CONTENT_DETECTORS = ["count_vector_pca", "isolation_forest_counts", "loganomaly"]
TIMING_DETECTORS = ["z_score_threshold", "log_ratio_threshold"]
ALL_DETECTORS = CONTENT_DETECTORS + TIMING_DETECTORS
FIXED_TIME_SIZES_DEFAULT = [15, 30, 60, 120, 300, 600]
FIXED_COUNT_SIZES_DEFAULT = [20, 50, 100, 200, 500]
LOGANOMALY_SEED = 42
N_BOOT = 1000
BOOT_SEED = 0
FIXED_COUNT_DELTA_TOLERANCE = 1e-9

OUT_DIR = Path("results/placebo")
SCORES_DIR = Path("data/processed/scores")
CHECKPOINT_DIR = Path("results/checkpoints")


def subsample_df(df, subsample, must_include_nodes):
    if subsample is None:
        return df
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    if must_include_nodes:
        included = df[df["node"].isin(must_include_nodes)]
        remainder_budget = max(subsample - len(included), 0)
        remainder = df[~df["node"].isin(must_include_nodes)].head(remainder_budget)
        df = pd.concat([included, remainder]).sort_values(["node", "timestamp"], kind="mergesort")
    else:
        df = df.head(subsample)
    return df.reset_index(drop=True)


def make_run_config(config, subsample, must_include_nodes, tmp_dir):
    if subsample is None:
        return config
    run_config = dict(config)

    clean_df = subsample_df(pd.read_parquet(config["clean_path"]), subsample, must_include_nodes)
    clean_tmp_path = tmp_dir / f"{config['clean_path'].stem}_subsample.parquet"
    clean_df.to_parquet(clean_tmp_path)
    run_config["clean_path"] = clean_tmp_path

    run_config["injected_loaders"] = {
        fault: (lambda loader_fn=loader_fn: subsample_df(loader_fn(), subsample, must_include_nodes))
        for fault, loader_fn in config["injected_loaders"].items()
    }

    injected_paths = {}
    for fault, path in config["injected_paths"].items():
        injected_df = subsample_df(pd.read_parquet(path), subsample, must_include_nodes)
        injected_tmp_path = tmp_dir / f"{path.stem}_subsample.parquet"
        injected_df.to_parquet(injected_tmp_path)
        injected_paths[fault] = injected_tmp_path
    run_config["injected_paths"] = injected_paths

    return run_config


def build_scheme_frames(scheme_kind, size, df_train_std, df_clean_std, df_injected_std, df_train_pos, df_clean_pos, df_injected_pos):
    if scheme_kind == "fixed_time":
        df_train_w, windows_train = build_windows(df_train_std, scheme="fixed_time", size=size)
        df_train_w["eval_window_key"] = df_train_w["window_key"]
        df_clean_w, windows_clean = build_windows(df_clean_std, scheme="fixed_time", size=size)
        df_clean_w["eval_window_key"] = df_clean_w["window_key"]
        df_inj_w, windows_inj = build_windows(df_injected_std, scheme="fixed_time", size=size)
        df_inj_w["eval_window_key"] = df_inj_w["window_key"]
    else:
        df_train_w = assign_fixed_count_window(df_train_pos, size)
        windows_train = build_fixed_count_windows_table(df_train_w)
        df_clean_w = assign_fixed_count_window(df_clean_pos, size)
        windows_clean = build_fixed_count_windows_table(df_clean_w)
        df_inj_w = assign_fixed_count_window(df_injected_pos, size)
        windows_inj = build_fixed_count_windows_table(df_inj_w)
    return df_train_w, windows_train, df_clean_w, windows_clean, df_inj_w, windows_inj


def fit_score_loganomaly(df_train_w, windows_train, df_clean_w, windows_clean, df_inj_w, windows_inj, vocabulary, embeddings):
    X_train = build_feature_matrix(df_train_w, windows_train, vocabulary, embeddings)
    train_normal_mask = ~windows_train["label"].to_numpy()
    detector = LogAnomalyDetector(random_state=LOGANOMALY_SEED).fit(X_train[train_normal_mask])
    threshold = np.percentile(detector.score(X_train[train_normal_mask]), THRESHOLD_PERCENTILE)

    def score_frame(df_w, windows):
        X = build_feature_matrix(df_w, windows, vocabulary, embeddings)
        window_score = pd.Series(detector.score(X), index=windows["window_key"])
        window_flag = window_score > threshold
        row_score = pd.Series(df_w["window_key"].map(window_score).to_numpy(), index=df_w["row_id"].to_numpy())
        row_flag = pd.Series(df_w["window_key"].map(window_flag).to_numpy(), index=df_w["row_id"].to_numpy())
        return row_score, row_flag

    return {"loganomaly": (score_frame(df_clean_w, windows_clean), score_frame(df_inj_w, windows_inj))}


def build_auc_arrays(df_eval, row_score_by_id, row_true_by_id, node_to_injection):
    row_score_aligned = df_eval["row_id"].map(row_score_by_id).fillna(0.0)
    row_true_aligned = df_eval["row_id"].map(row_true_by_id).fillna(False)
    score_grid = rows_to_grid_max(df_eval, row_score_aligned)
    true_grid = rows_to_grid(df_eval, row_true_aligned)
    aligned = pd.concat([true_grid.rename("y_true"), score_grid.rename("score")], axis=1)
    y_true = aligned["y_true"].astype(bool).to_numpy()
    y_score = aligned["score"].to_numpy()
    window_nodes = pd.Series([k[0] for k in aligned.index], index=aligned.index)
    injection_ids = window_nodes.map(node_to_injection).fillna(-1).astype(int).to_numpy()
    return y_true, y_score, injection_ids


def precompute_contrib(y_true, y_score):
    pos_mask = y_true
    neg_scores = np.sort(y_score[~pos_mask])
    n_neg = neg_scores.size
    pos_scores = y_score[pos_mask]
    count_less = np.searchsorted(neg_scores, pos_scores, side="left")
    count_leq = np.searchsorted(neg_scores, pos_scores, side="right")
    base_contrib = count_less + 0.5 * (count_leq - count_less)
    return base_contrib, n_neg


def paired_delta_bootstrap(y_true_inj, y_score_inj, ids_inj, y_true_pla, y_score_pla, ids_pla, n_injections, n_boot, rng):
    contrib_inj, n_neg_inj = precompute_contrib(y_true_inj, y_score_inj)
    contrib_pla, n_neg_pla = precompute_contrib(y_true_pla, y_score_pla)
    pos_ids_inj = ids_inj[y_true_inj]
    pos_ids_pla = ids_pla[y_true_pla]

    point_auc_inj = float(roc_auc_score(y_true_inj, y_score_inj)) if 0 < y_true_inj.sum() < len(y_true_inj) else float("nan")
    point_auc_pla = float(roc_auc_score(y_true_pla, y_score_pla)) if 0 < y_true_pla.sum() < len(y_true_pla) else float("nan")
    point_delta = point_auc_inj - point_auc_pla

    deltas = np.full(n_boot, np.nan)
    for b in range(n_boot):
        draw = rng.integers(0, n_injections, size=n_injections)
        multiplicity = np.bincount(draw, minlength=n_injections)
        w_inj = multiplicity[pos_ids_inj] if len(pos_ids_inj) else np.array([])
        w_pla = multiplicity[pos_ids_pla] if len(pos_ids_pla) else np.array([])
        if w_inj.sum() == 0 or w_pla.sum() == 0:
            continue
        auc_inj_b = np.sum(w_inj * contrib_inj) / (w_inj.sum() * n_neg_inj)
        auc_pla_b = np.sum(w_pla * contrib_pla) / (w_pla.sum() * n_neg_pla)
        deltas[b] = auc_inj_b - auc_pla_b

    n_nan = int(np.isnan(deltas).sum())
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return point_delta, float(lo), float(hi), point_auc_inj, point_auc_pla, n_nan


def run_scheme(dataset, fault, scheme_kind, size, frames, vocabulary, embeddings, timing_scores_injected, timing_scores_placebo, row_true_by_id, node_to_injection, rng):
    df_train_w, windows_train, df_clean_w, windows_clean, df_inj_w, windows_inj = frames

    fitted = fit_count_detectors_on_scheme(df_train_w, windows_train, vocabulary)
    count_scores_clean = score_count_detectors(fitted, df_clean_w, windows_clean, vocabulary)
    count_scores_injected = score_count_detectors(fitted, df_inj_w, windows_inj, vocabulary)
    loganomaly_result = fit_score_loganomaly(df_train_w, windows_train, df_clean_w, windows_clean, df_inj_w, windows_inj, vocabulary, embeddings)

    detector_scores = {}
    for name in ["count_vector_pca", "isolation_forest_counts"]:
        detector_scores[name] = {"clean": count_scores_clean[name], "injected": count_scores_injected[name]}
    detector_scores["loganomaly"] = {"clean": loganomaly_result["loganomaly"][0], "injected": loganomaly_result["loganomaly"][1]}
    for name in TIMING_DETECTORS:
        detector_scores[name] = {"clean": timing_scores_placebo[name], "injected": timing_scores_injected[name]}

    scheme_name = f"{scheme_kind}_{size}"
    sweep_rows = []
    save_frame = None
    n_injections = len(node_to_injection)

    for name in ALL_DETECTORS:
        score_clean, flag_clean = detector_scores[name]["clean"]
        score_inj, flag_inj = detector_scores[name]["injected"]

        metrics_injected = evaluate_cell(df_inj_w, score_inj, flag_inj, row_true_by_id)
        metrics_placebo = evaluate_cell(df_clean_w, score_clean, flag_clean, row_true_by_id)

        y_true_inj, y_score_inj, ids_inj = build_auc_arrays(df_inj_w, score_inj, row_true_by_id, node_to_injection)
        y_true_pla, y_score_pla, ids_pla = build_auc_arrays(df_clean_w, score_clean, row_true_by_id, node_to_injection)
        point_delta, ci_lo, ci_hi, auc_inj_check, auc_pla_check, n_degenerate = paired_delta_bootstrap(
            y_true_inj, y_score_inj, ids_inj, y_true_pla, y_score_pla, ids_pla, n_injections, N_BOOT, rng
        )

        score_clean_by_key = scores_by_window_key(df_clean_w, score_clean)
        score_inj_by_key = scores_by_window_key(df_inj_w, score_inj)
        invariance = compare_window_scores(score_clean_by_key, score_inj_by_key)
        n_grid_reassign, n_common_rows = count_grid_reassignments(df_clean_w, df_inj_w)

        row = {
            "dataset": dataset,
            "fault_type": fault,
            "scheme": scheme_name,
            "scheme_kind": scheme_kind,
            "scheme_size": size,
            "detector": name,
            "auc_injected": metrics_injected["auc_roc"],
            "auc_placebo": metrics_placebo["auc_roc"],
            "delta": metrics_injected["auc_roc"] - metrics_placebo["auc_roc"],
            "delta_ci_lower": ci_lo,
            "delta_ci_upper": ci_hi,
            "n_bootstrap_degenerate": n_degenerate,
            "recall_injected": metrics_injected["recall"],
            "precision_injected": metrics_injected["precision"],
            "lift_injected": metrics_injected["lift"],
            **invariance,
            "n_grid_cell_reassignments": n_grid_reassign,
            "n_common_rows_for_grid_check": n_common_rows,
        }
        sweep_rows.append(row)

        if save_frame is None:
            all_keys = sorted(set(score_clean_by_key.index) | set(score_inj_by_key.index))
            save_frame = pd.DataFrame({"node": [k[0] for k in all_keys], "window_idx": [k[1] for k in all_keys]}, index=all_keys)
            true_by_key = scores_by_window_key(df_inj_w, row_true_by_id)
            save_frame["y_true"] = save_frame.index.map(true_by_key).fillna(False)
            save_frame["injection_id"] = save_frame["node"].map(node_to_injection).fillna(-1).astype(int)
        save_frame[f"score_clean_{name}"] = save_frame.index.map(score_clean_by_key)
        save_frame[f"score_injected_{name}"] = save_frame.index.map(score_inj_by_key)

    save_frame = save_frame.reset_index(drop=True)
    return sweep_rows, save_frame


def run_fault(dataset, config, fault, fixed_time_sizes, fixed_count_sizes, subsample, subsample_include_injected, seed, tmp_dir):
    print(f"=== {dataset} / {fault} ===", flush=True)
    t0 = time.time()

    must_include_nodes = None
    if subsample_include_injected is not None:
        must_include_nodes = set()
        for f in FAULTS:
            must_include_nodes |= select_injected_nodes(config["injected_loaders"][f], subsample_include_injected, seed)

    run_config = make_run_config(config, subsample, must_include_nodes, tmp_dir)

    df_clean_std = load_clean_by_node(run_config["clean_path"])
    is_train_std, _ = config["split_fn"](df_clean_std)
    df_train_std = df_clean_std.loc[is_train_std.to_numpy()].reset_index(drop=True)
    vocabulary = build_vocabulary(df_train_std)
    embeddings = build_template2vec(vocabulary, seed=LOGANOMALY_SEED)
    print(f"  vocab size: {len(vocabulary)}, elapsed: {time.time() - t0:.0f}s", flush=True)

    df_clean_pos = load_pos(run_config["clean_path"])
    eq_clean = verify_std_pos_equivalence(df_clean_std, df_clean_pos, f"{dataset} clean")
    is_train_pos = df_clean_pos["timestamp"] <= df_clean_std.loc[is_train_std.to_numpy(), "timestamp"].max()
    df_train_pos = df_clean_pos.loc[is_train_pos.to_numpy()].reset_index(drop=True)

    df_injected_std = load_by_node(run_config["injected_loaders"][fault]())
    df_injected_pos = load_pos(run_config["injected_paths"][fault])
    eq_injected = verify_std_pos_equivalence(df_injected_std, df_injected_pos, f"{dataset} {fault} injected")
    if not (eq_clean and eq_injected):
        print("CRITICAL: std/pos row-order equivalence failed -- position-based windowing is not trustworthy for this run.", file=sys.stderr)
        sys.exit(1)

    labels_df = pd.read_csv(config["injection_labels_path"][fault], parse_dates=["start", "end"])
    node_to_injection = labels_df.set_index("node")["injection_id"]

    row_true_std = label_rows_in_injected_span(df_injected_std, labels_df)
    row_true_by_id = pd.Series(row_true_std.to_numpy(), index=df_injected_std["row_id"].to_numpy())

    timing_scores_injected = build_timing_scores(dataset, df_clean_std, is_train_std, df_injected_std)
    timing_scores_placebo = build_timing_scores(dataset, df_clean_std, is_train_std, df_clean_std)
    print(f"  timing features + equivalence done, elapsed: {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(seed)
    all_sweep_rows = []
    schemes = [("fixed_time", s) for s in fixed_time_sizes] + [("fixed_count", s) for s in fixed_count_sizes]

    for scheme_kind, size in schemes:
        print(f"  scheme {scheme_kind}_{size}, elapsed: {time.time() - t0:.0f}s", flush=True)
        frames = build_scheme_frames(scheme_kind, size, df_train_std, df_clean_std, df_injected_std, df_train_pos, df_clean_pos, df_injected_pos)
        sweep_rows, save_frame = run_scheme(
            dataset, fault, scheme_kind, size, frames, vocabulary, embeddings,
            timing_scores_injected, timing_scores_placebo, row_true_by_id, node_to_injection, rng,
        )
        all_sweep_rows.extend(sweep_rows)

        if subsample is None:
            SCORES_DIR.mkdir(parents=True, exist_ok=True)
            save_frame.to_parquet(SCORES_DIR / f"{dataset}_{fault}_{scheme_kind}_{size}.parquet", index=False)

        if scheme_kind == "fixed_count":
            bad = [
                r for r in sweep_rows
                if r["detector"] in CONTENT_DETECTORS and abs(r["delta"]) > FIXED_COUNT_DELTA_TOLERANCE
            ]
            if bad:
                print(f"\nCRITICAL: nonzero placebo delta on a fixed-count cell for a content detector -- treat as a bug, not a finding:", file=sys.stderr)
                for r in bad:
                    print(f"  {r['dataset']} {r['fault_type']} {r['scheme']} {r['detector']}: delta={r['delta']}", file=sys.stderr)
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(all_sweep_rows).to_csv(CHECKPOINT_DIR / f"placebo_{dataset}_{fault}_FAILED.csv", index=False)
                sys.exit(1)

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_sweep_rows).to_csv(CHECKPOINT_DIR / f"placebo_{dataset}_{fault}_checkpoint.csv", index=False)
        del frames
        gc.collect()

    print(f"  {dataset}/{fault} done, elapsed: {time.time() - t0:.0f}s", flush=True)
    return all_sweep_rows


def write_markdown(df, path, args):
    lines = [
        f"# Placebo-controlled windowing sweep -- {args.dataset}",
        "",
        f"fixed_time_sizes={args.fixed_time_sizes}, fixed_count_sizes={args.fixed_count_sizes}, "
        f"subsample={args.subsample}, subsample_include_injected={args.subsample_include_injected}, "
        f"n_boot={N_BOOT}, seed={args.seed}",
        "",
    ]
    fixed_count_content = df[(df["scheme_kind"] == "fixed_count") & (df["detector"].isin(CONTENT_DETECTORS))]
    all_zero = bool((fixed_count_content["delta"].abs() < FIXED_COUNT_DELTA_TOLERANCE).all()) if len(fixed_count_content) else None
    lines.append(f"Fixed-count content-detector delta all exactly zero: {all_zero} ({len(fixed_count_content)} cells checked)")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(ALL_DATASET_CONFIG), required=True)
    ap.add_argument("--fault", choices=["stall", "burst"], default=None)
    ap.add_argument("--fixed-time-sizes", type=int, nargs="+", default=None)
    ap.add_argument("--fixed-count-sizes", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--subsample-include-injected", type=int, default=None)
    ap.add_argument("--injection-suffix", default="", help='e.g. "_n300" to score a non-default src/injector.py --n-injections run instead of the n=100 files.')
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    return ap.parse_args()


def main():
    import tempfile
    import shutil

    args = parse_args()
    config = apply_injection_suffix(ALL_DATASET_CONFIG[args.dataset], args.injection_suffix)
    faults = [args.fault] if args.fault else FAULTS

    if args.injection_suffix:
        fixed_time_sizes = args.fixed_time_sizes if args.fixed_time_sizes is not None else [60]
        fixed_count_sizes = args.fixed_count_sizes if args.fixed_count_sizes is not None else [50]
    else:
        fixed_time_sizes = args.fixed_time_sizes if args.fixed_time_sizes is not None else FIXED_TIME_SIZES_DEFAULT
        fixed_count_sizes = args.fixed_count_sizes if args.fixed_count_sizes is not None else FIXED_COUNT_SIZES_DEFAULT
    args.fixed_time_sizes = fixed_time_sizes
    args.fixed_count_sizes = fixed_count_sizes

    tmp_dir = Path(tempfile.mkdtemp(prefix="placebo_sweep_")) if args.subsample is not None else None
    all_rows = []
    try:
        for fault in faults:
            rows = run_fault(
                args.dataset, config, fault, fixed_time_sizes, fixed_count_sizes,
                args.subsample, args.subsample_include_injected, args.seed, tmp_dir,
            )
            all_rows.extend(rows)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    df = pd.DataFrame(all_rows)
    out_csv = args.out_csv or OUT_DIR / f"placebo_sweep_{args.dataset}{args.injection_suffix}.csv"
    out_md = args.out_md or OUT_DIR / f"placebo_sweep_{args.dataset}{args.injection_suffix}.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out_csv}")
    write_markdown(df, out_md, args)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
