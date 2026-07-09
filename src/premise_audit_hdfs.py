"""
Run the SAME premise-audit signatures used for BGL/Thunderbird (src/premise_audit.py) on HDFS's
natively-labeled anomalies, adapted to HDFS's block-session structure. No new signature logic --
imports and reuses flag_template_rarity, flag_pure_order_anomaly, flag_timing_gap_anomaly,
gap_zero_diagnostics, summarize_v2, timing_rate_by_content exactly as validated for BGL/Thunderbird.

Adaptation: pure_order_anomaly and timing_gap_anomaly both key off whatever column is literally
named "node" (per-node bigram sets, per-node timing baselines). HDFS has no reliable per-row node
field (only 15.48% of rows carry an extractable DataNode IP -- see results/hdfs_setup_notes.md
Part 2). Its natural, complete, always-present grouping key is block_id instead. So this script
builds a working copy with `node := block_id` and passes that into the existing functions
unmodified -- reusing the per-node machinery as per-BLOCK machinery, which is exactly candidate
(a) ("within-block") from the Part 2 design analysis: mechanically complete (every row belongs to
exactly one block, unlike node coverage), even though Part 2 found only 14.24% of blocks have a
non-degenerate baseline. This audit reports the same 4-signature table anyway, with that caveat
attached, for direct comparability with BGL/Thunderbird's tables.

Reads data/processed/hdfs_parsed.parquet. Writes:
  results/hdfs_premise_audit.csv

Usage:
    python src/premise_audit_hdfs.py
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
    summarize_v2,
    timing_rate_by_content,
)
from src.timing_baseline import add_sequence_context

IN_PATH = Path("data/processed/hdfs_parsed.parquet")
OUT_CSV = Path("results/hdfs_premise_audit.csv")


def load():
    df = pd.read_parquet(IN_PATH)
    # Alias block_id as "node" -- HDFS's complete, always-present grouping key, reused as-is by
    # the shared per-node signature functions (see module docstring).
    df["node"] = df["block_id"]
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
    summary = summarize_v2(anomalies, order_diag)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print("=== HDFS premise audit (node := block_id) ===")
    print(summary.to_string(index=False))

    print("\n=== timing baseline diagnostics (within-block, default pooled fallback -- unfixed) ===")
    print(f"Overall fraction of ALL within-block gaps == 0:         {gap_diag['overall_zero_frac']:.4f}")
    print(f"Blocks with >=1 valid gap:                              {gap_diag['n_nodes_with_gaps']:,}")
    print(f"Per-block zero-gap fraction, mean / median:             {gap_diag['node_zero_frac_mean']:.4f} / {gap_diag['node_zero_frac_median']:.4f}")
    print(f"% of blocks that are majority-zero-gap:                 {gap_diag['pct_nodes_majority_zero_gaps']:.2f}%")
    print(f"Anomalous rows' distinct blocks:                        {timing_diag['n_anomaly_nodes']:,}")
    print(f"  -> with a valid per-block baseline:                   {timing_diag['n_anomaly_nodes_valid_baseline']:,}")
    print(f"  -> fell back to pooled (too few normal gaps):         {timing_diag['n_anomaly_nodes_fallback_low_count']:,}")
    print(f"  -> fell back to pooled (per-block MAD == 0):          {timing_diag['n_anomaly_nodes_fallback_zero_mad']:,}")

    print("\n=== timing_gap_anomaly rate by content-novelty status ===")
    print(content_split.to_string(index=False))

    print("\n=== BGL vs Thunderbird vs HDFS (pct_of_anomalies) ===")
    bgl = pd.read_csv("results/premise_audit_v2.csv")
    tb = pd.read_csv("results/thunderbird_premise_audit_v2.csv")
    bgl_map = dict(zip(bgl["signature"], bgl["pct_of_anomalies"]))
    tb_map = dict(zip(tb["signature"], tb["pct_of_anomalies"]))
    hdfs_map = dict(zip(summary["signature"], summary["pct_of_anomalies"]))
    comparison = pd.DataFrame(
        [{"signature": k, "bgl_pct": bgl_map.get(k), "thunderbird_pct": tb_map.get(k), "hdfs_pct": hdfs_map.get(k)} for k in hdfs_map]
    )
    print(comparison.to_string(index=False))

    print(f"\nWrote {OUT_CSV}")
    return summary, gap_diag, timing_diag, content_split, comparison


if __name__ == "__main__":
    main()
