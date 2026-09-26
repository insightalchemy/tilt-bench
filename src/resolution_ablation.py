"""
Timestamp-resolution ablation (paper Section IX, "Timestamp resolution").

Rounds a clean dataset's timestamps to whole seconds, applies the same burst injection
(src.injector.plan_burst_injections / apply_burst_injections, same seed) to both the original and
the rounded data, and reports the fraction of injected burst rows on which the z-score (|z| > 3)
and log-ratio detectors fire, together with the number of eligible nodes at each resolution.

Bursts are used because a burst's z-score is bounded regardless of intensity (see
ZScoreThresholdDetector), making it the resolution-sensitive case. Both runs use
require_valid_baseline=True and exclude_zero_from_pooled=True, since rounding creates zero gaps.

--self-test runs the full pipeline once on a small synthetic frame.

Writes results/resolution_ablation_{dataset}.csv / .md

Usage:
    python src/resolution_ablation.py --dataset bgl
    python src/resolution_ablation.py --self-test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.detectors.timing_detector import LogRatioThresholdDetector, ZScoreThresholdDetector, add_log_ratio_feature, build_features
from src.injector import BURST_MIN_NODE_EVENTS, BURST_SEED, MAX_MAD_EFF_S, apply_burst_injections, eligible_nodes, plan_burst_injections
from src.timing_baseline import add_sequence_context, compute_node_baselines

Z_THRESH = 3.0
ROUND_FREQ = "1s"


def clean_path_for(dataset: str) -> Path:
    known = {"bgl": "bgl", "thunderbird": "thunderbird", "spirit": "spirit"}
    name = known.get(dataset, dataset)
    return Path(f"data/processed/{name}_parsed.parquet")


def round_timestamps(df: pd.DataFrame, freq: str = ROUND_FREQ) -> pd.DataFrame:
    """Round timestamps to `freq`. Rounding is monotonic, so per-node order can only become tied,
    never inverted."""
    df = df.copy()
    df["timestamp"] = df["timestamp"].dt.round(freq)
    return df


def inject_and_score(df_clean: pd.DataFrame, label: str) -> dict:
    df_clean = df_clean.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df_clean["anomaly"] = df_clean["anomaly"].fillna(False).astype(bool)
    df_clean["row_id"] = np.arange(len(df_clean))
    df_clean = add_sequence_context(df_clean)

    normal_seq = (
        (~df_clean["anomaly"].fillna(False).astype(bool))
        & (~df_clean["prev_anomaly"].fillna(True).astype(bool))
        & df_clean["gap_prev_s"].notna()
    )
    node_baselines = compute_node_baselines(df_clean, normal_seq, exclude_zero_from_pooled=True)

    n_eligible = len(
        eligible_nodes(
            df_clean, min_events=BURST_MIN_NODE_EVENTS, node_baselines=node_baselines,
            require_valid_baseline=True, max_mad_eff=MAX_MAD_EFF_S,
        )
    )
    n_injections = min(100, n_eligible)
    if n_injections == 0:
        return {"label": label, "n_eligible": 0, "n_injections": 0, "frac_z_gt_3": float("nan"), "frac_log_ratio_flagged": float("nan")}

    plans = plan_burst_injections(
        df_clean, node_baselines, seed=BURST_SEED, n_injections=n_injections,
        min_node_events=BURST_MIN_NODE_EVENTS, require_valid_baseline=True, max_mad_eff=MAX_MAD_EFF_S,
    )
    df_injected, labels_df = apply_burst_injections(df_clean, plans)

    df_injected = df_injected.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df_injected = add_sequence_context(df_injected)
    injected_normal = (~df_injected["injected_row"]) & df_injected["gap_prev_s"].notna()

    features, baselines = build_features(df_injected, baseline_mask=injected_normal)
    features = add_log_ratio_feature(features, baselines)

    z_detector = ZScoreThresholdDetector(z_thresh=Z_THRESH)
    log_detector = LogRatioThresholdDetector()
    log_detector.fit(features.loc[injected_normal])

    z_scores = z_detector.score(features)
    log_scores = log_detector.score(features)

    burst_rows = features["injected_row"].to_numpy()
    n_burst_rows = int(burst_rows.sum())
    frac_z = float((z_scores[burst_rows] > Z_THRESH).mean()) if n_burst_rows else float("nan")
    frac_log = float((log_scores[burst_rows] > log_detector.threshold).mean()) if n_burst_rows else float("nan")

    return {
        "label": label,
        "n_eligible": n_eligible,
        "n_injections": n_injections,
        "n_burst_rows": n_burst_rows,
        "frac_z_gt_3": frac_z,
        "frac_log_ratio_flagged": frac_log,
    }


def run(df_clean: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    result_orig = inject_and_score(df_clean, "original_resolution")
    df_rounded = round_timestamps(df_clean)
    result_rounded = inject_and_score(df_rounded, f"rounded_{ROUND_FREQ}")
    rows = [result_orig, result_rounded]
    for r in rows:
        r["dataset"] = dataset_name
    return pd.DataFrame(rows)


def build_synthetic_df(n_nodes=4, n_per_node=50, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2020-01-01")
    for node_i in range(n_nodes):
        node = f"node-{node_i}"
        t = t0
        for i in range(n_per_node):
            gap = rng.uniform(0.3, 1.2)
            t = t + pd.Timedelta(seconds=gap)
            rows.append({"timestamp": t, "node": node, "anomaly": False, "event_template": f"tmpl{i % 3}", "raw_message": "msg"})
    return pd.DataFrame(rows)


def self_test():
    import src.injector as inj

    inj.BURST_MIN_NODE_EVENTS = 15
    inj.MAX_MAD_EFF_S = 3600.0
    global BURST_MIN_NODE_EVENTS, MAX_MAD_EFF_S
    BURST_MIN_NODE_EVENTS = inj.BURST_MIN_NODE_EVENTS
    MAX_MAD_EFF_S = inj.MAX_MAD_EFF_S

    df = build_synthetic_df()
    result = run(df, "synthetic")
    print(result.to_string(index=False))
    assert len(result) == 2, "expected one row for original and one for rounded resolution"
    print("\nSELF-TEST PASSED: resolution ablation pipeline ran end-to-end on synthetic data with no crash.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="bgl")
    ap.add_argument("--self-test", action="store_true", help="Run on an in-memory synthetic DataFrame; touches no real data.")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    df_clean = pd.read_parquet(clean_path_for(args.dataset))
    result = run(df_clean, args.dataset)

    out_csv = args.out_csv or Path(f"results/resolution_ablation_{args.dataset}.csv")
    out_md = args.out_md or Path(f"results/resolution_ablation_{args.dataset}.md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    print(result.to_string(index=False))
    lines = [
        f"# Resolution ablation: {args.dataset}",
        "",
        f"Rounded to {ROUND_FREQ}. Same burst injection (seed={BURST_SEED}) applied at both resolutions.",
        "",
        result.to_markdown(index=False),
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_csv}, {out_md}")


if __name__ == "__main__":
    main()
