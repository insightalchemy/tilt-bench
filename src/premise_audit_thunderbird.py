"""
Premise audit (src/premise_audit.py signatures) on Thunderbird's native labeled anomalies.

Reads data/processed/thunderbird_parsed.parquet (src/parser_thunderbird.py). Writes:
  results/thunderbird_premise_audit.csv -- same signature table as results/premise_audit.csv

Prints a BGL vs Thunderbird comparison (requires src/premise_audit.py to have been run).

Usage:
    python src/premise_audit_thunderbird.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import pandas as pd

from src.premise_audit import (
    flag_pure_order_anomaly,
    flag_template_rarity,
    flag_timing_gap_anomaly,
    gap_zero_diagnostics,
    summarize_signatures,
    timing_rate_by_content,
)
from src.timing_baseline import add_sequence_context

IN_PATH = Path("data/processed/thunderbird_parsed.parquet")
OUT_CSV = Path("results/thunderbird_premise_audit.csv")


def load():
    df = pd.read_parquet(IN_PATH)
    return df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)


def main():
    df = load()
    df = add_sequence_context(df)
    df = flag_template_rarity(df)

    gap_diag = gap_zero_diagnostics(df)

    pure_order, order_diag = flag_pure_order_anomaly(df)
    timing_anomaly, timing_diag = flag_timing_gap_anomaly(df)

    anomalies = df[df["anomaly"]].copy()
    anomalies["pure_order_anomaly"] = pure_order
    anomalies["timing_gap_anomaly"] = timing_anomaly

    content_split = timing_rate_by_content(anomalies)
    summary = summarize_signatures(anomalies, order_diag)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print("=== Thunderbird premise audit ===")
    print(summary.to_string(index=False))

    print("\n=== timing baseline diagnostics ===")
    print(f"Overall fraction of ALL inter-arrival gaps == 0:        {gap_diag['overall_zero_frac']:.4f}")
    print(f"Nodes with >=1 valid gap:                               {gap_diag['n_nodes_with_gaps']:,}")
    print(f"Per-node zero-gap fraction, mean / median:              {gap_diag['node_zero_frac_mean']:.4f} / {gap_diag['node_zero_frac_median']:.4f}")
    print(f"% of nodes that are majority-zero-gap:                  {gap_diag['pct_nodes_majority_zero_gaps']:.2f}%")
    print(f"Anomalous rows' distinct nodes:                         {timing_diag['n_anomaly_nodes']:,}")
    print(f"  -> with a valid per-node baseline:                    {timing_diag['n_anomaly_nodes_valid_baseline']:,}")
    print(f"  -> fell back to pooled (too few normal gaps):         {timing_diag['n_anomaly_nodes_fallback_low_count']:,}")
    print(f"  -> fell back to pooled (per-node MAD == 0):           {timing_diag['n_anomaly_nodes_fallback_zero_mad']:,}")

    print("\n=== timing_gap_anomaly rate by content-novelty status ===")
    print(content_split.to_string(index=False))

    print("\n=== BGL vs Thunderbird ===")
    bgl = pd.read_csv("results/premise_audit.csv")
    bgl_map = dict(zip(bgl["signature"], bgl["pct_of_anomalies"]))
    tb_map = dict(zip(summary["signature"], summary["pct_of_anomalies"]))
    rows = [{"signature": k, "bgl_pct": bgl_map.get(k), "thunderbird_pct": tb_map.get(k)} for k in tb_map]
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False))

    print(f"\nWrote {OUT_CSV}")
    return summary, gap_diag, timing_diag, content_split, comparison


if __name__ == "__main__":
    main()
