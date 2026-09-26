"""
Physical-plausibility check for injected stalls and bursts (paper Section IV-B): where does each
injected (stall) or compressed (burst) gap fall within its node's natural pre-injection
distribution of normal-to-normal gaps?

Reports, per injection, the percentile of the injected gap within the node's natural gaps and
whether it falls outside the node's natural [min, max] range, in both directions (a stall widens
a gap, a burst compresses it).

Natural gaps use the same normal-to-normal selection as the per-node baselines, read from the clean
parquet. Injected gaps are those on the injected parquet's injected_row=True rows. Gaps are
computed only on the injected nodes' rows; per-node gaps do not depend on other nodes.

Writes:
  results/plausibility/plausibility_{dataset}_{fault}.csv

Usage:
    python src/plausibility.py --dataset bgl --fault stall
    python src/plausibility.py --dataset bgl --fault burst
    python src/plausibility.py --dataset thunderbird --fault stall
    python src/plausibility.py --dataset thunderbird --fault burst
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.timing_baseline import add_sequence_context
from src.windowing_sweep import DATASET_CONFIG

OUT_DIR = Path("results/plausibility")


def natural_gaps_by_node(df_clean, nodes):
    df = df_clean[df_clean["node"].isin(nodes)].copy()
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df = add_sequence_context(df)
    normal = (~df["anomaly"]) & (~df["prev_anomaly"].fillna(True)) & df["gap_prev_s"].notna()
    natural = df.loc[normal, ["node", "gap_prev_s"]]
    return {node: g["gap_prev_s"].to_numpy() for node, g in natural.groupby("node")}


def injected_gaps_by_node(df_injected, nodes):
    df = df_injected[df_injected["node"].isin(nodes)].copy()
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df = add_sequence_context(df)
    injected = df.loc[df["injected_row"], ["node", "gap_prev_s"]]
    return {node: g["gap_prev_s"].to_numpy() for node, g in injected.groupby("node")}


def percentile_within(value, natural):
    if len(natural) == 0 or np.isnan(value):
        return float("nan")
    return 100.0 * float((natural <= value).mean())


def run(dataset, fault, config):
    print(f"=== {dataset} / {fault} : plausibility ===", flush=True)
    labels_df = pd.read_csv(config["injection_labels_path"][fault])
    nodes = labels_df["node"].unique().tolist()

    df_clean = pd.read_parquet(config["clean_path"])
    natural_by_node = natural_gaps_by_node(df_clean, nodes)
    del df_clean

    df_injected = pd.read_parquet(config["injected_paths"][fault])
    injected_by_node = injected_gaps_by_node(df_injected, nodes)
    del df_injected

    rows = []
    for rec in labels_df.to_dict("records"):
        node = rec["node"]
        natural = natural_by_node.get(node, np.array([]))
        gaps = injected_by_node.get(node, np.array([]))

        percentiles = [percentile_within(g, natural) for g in gaps]
        natural_max = float(natural.max()) if len(natural) else float("nan")
        natural_min = float(natural.min()) if len(natural) else float("nan")
        n_exceeds_max = int(np.sum(gaps > natural_max)) if len(natural) else 0
        n_below_min = int(np.sum(gaps < natural_min)) if len(natural) else 0

        rows.append(
            {
                "dataset": dataset,
                "fault_type": fault,
                "injection_id": rec["injection_id"],
                "node": node,
                "intensity": rec["intensity"],
                "n_natural_gaps": len(natural),
                "n_injected_gaps": len(gaps),
                "mean_percentile": float(np.mean(percentiles)) if percentiles else float("nan"),
                "min_percentile": float(np.min(percentiles)) if percentiles else float("nan"),
                "max_percentile": float(np.max(percentiles)) if percentiles else float("nan"),
                "frac_exceeds_natural_max": n_exceeds_max / len(gaps) if len(gaps) else float("nan"),
                "frac_below_natural_min": n_below_min / len(gaps) if len(gaps) else float("nan"),
                "natural_max_s": natural_max,
                "natural_min_s": natural_min,
            }
        )

    return pd.DataFrame(rows)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["bgl", "thunderbird"], required=True)
    ap.add_argument("--fault", choices=["stall", "burst"], required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    config = DATASET_CONFIG[args.dataset]
    df = run(args.dataset, args.fault, config)

    out_csv = args.out_csv or OUT_DIR / f"plausibility_{args.dataset}_{args.fault}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
