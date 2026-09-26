"""
Premise audit (paper Section III, Table I): how do BGL's native labeled anomalies manifest?

For every labeled-anomalous row:
  1. new_or_rare_template  -- its event_template is absent from, or rare in, the normal-only data.
  2. pure_order_anomaly    -- evaluated only on rows whose own and neighboring templates all occur
                              in that node's normal traffic: the local bigram (prev->this or
                              this->next) never occurs in that node's normal-only sequence.
  3. timing_gap_anomaly    -- the gap to the node's previous or next event is an outlier
                              (|z| > Z_THRESH) against the node's normal-gap baseline.
  4. timing_only           -- flagged by (3) and conclusively not by (1) or (2).

The restriction in (2) is essential: a novel template makes every bigram touching it novel, so an
unrestricted "novel bigram" test would mostly restate template novelty.

Also reports the fraction of inter-arrival gaps that are exactly zero (timestamp resolution) and
how many nodes use the pooled timing baseline. Deterministic (fixed thresholds, no randomness).

Reads data/processed/bgl_parsed.parquet (src/parser.py). Writes:
  results/premise_audit.csv            -- signature summary table
  results/premise_audit_summary.md     -- short text summary
  results/premise_audit_breakdown.png  -- bar chart of signature rates

Usage:
    python src/premise_audit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.timing_baseline import add_sequence_context, compute_node_baselines, score_gap_zscore

IN_PATH = Path("data/processed/bgl_parsed.parquet")
OUT_CSV = Path("results/premise_audit.csv")
OUT_MD = Path("results/premise_audit_summary.md")
OUT_PNG = Path("results/premise_audit_breakdown.png")

RARE_TEMPLATE_FREQ = 1e-4  # "rare" = under 0.01% of normal-only rows
Z_THRESH = 3.0  # |z| above this on inter-arrival gap = timing-gap anomaly


def load():
    df = pd.read_parquet(IN_PATH)
    return df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)


def flag_template_rarity(df):
    normal = df.loc[~df["anomaly"]]
    counts = normal["event_template"].value_counts()
    freq = counts / counts.sum()
    df["normal_template_freq"] = df["event_template"].map(freq).fillna(0.0)
    df["new_template"] = df["normal_template_freq"] == 0.0
    df["rare_template"] = (df["normal_template_freq"] > 0.0) & (df["normal_template_freq"] < RARE_TEMPLATE_FREQ)
    df["new_or_rare_template"] = df["new_template"] | df["rare_template"]
    return df


def flag_pure_order_anomaly(df, subset_mask=None):
    """Order signal restricted to rows where content novelty cannot be the explanation.

    A row is eligible only if its own template and any prev/next template it has were all seen
    in that node's normal-only traffic. Among eligible rows, flag if a local bigram never occurred
    in that node's normal-only sequence. Ineligible rows and rows with no neighbor are NaN
    (excluded, neither flagged nor unflagged).

    subset_mask selects the rows to evaluate (default: df["anomaly"]). The normal reference
    traffic is always ~df["anomaly"], so injected rows can be evaluated via subset_mask without
    being treated as normal traffic.
    """
    if subset_mask is None:
        subset_mask = df["anomaly"]

    normal = df[~df["anomaly"]]
    node_normal_templates = normal.groupby("node")["event_template"].apply(lambda s: frozenset(s.unique()))

    normal_seq = (~df["anomaly"]) & (~df["prev_anomaly"].fillna(True)) & df["prev_template"].notna()
    normal_node_bigrams = set(
        zip(df.loc[normal_seq, "node"], df.loc[normal_seq, "prev_template"], df.loc[normal_seq, "event_template"])
    )

    anomalies = df[subset_mask].copy()
    node_templ = anomalies["node"].map(node_normal_templates).apply(
        lambda s: s if isinstance(s, frozenset) else frozenset()
    )

    this_known = pd.Series(
        [t in s for t, s in zip(anomalies["event_template"], node_templ)], index=anomalies.index
    )
    prev_known = pd.Series(
        [pd.isna(p) or (p in s) for p, s in zip(anomalies["prev_template"], node_templ)], index=anomalies.index
    )
    next_known = pd.Series(
        [pd.isna(n) or (n in s) for n, s in zip(anomalies["next_template"], node_templ)], index=anomalies.index
    )
    eligible = this_known & prev_known & next_known
    has_context = anomalies["prev_template"].notna() | anomalies["next_template"].notna()

    def bigram_seen(node, prev, cur):
        return [
            (nd, p, c) in normal_node_bigrams if (pd.notna(p) and pd.notna(c)) else np.nan
            for nd, p, c in zip(node, prev, cur)
        ]

    in_seen = pd.Series(
        bigram_seen(anomalies["node"], anomalies["prev_template"], anomalies["event_template"]),
        index=anomalies.index,
    )
    out_seen = pd.Series(
        bigram_seen(anomalies["node"], anomalies["event_template"], anomalies["next_template"]),
        index=anomalies.index,
    )
    either_flagged = (in_seen == False) | (out_seen == False)  # noqa: E712

    evaluated = eligible & has_context
    pure_order_anomaly = pd.Series(np.where(evaluated, either_flagged, np.nan), index=anomalies.index)

    diag = {
        "n_ineligible_unknown_template": int((~eligible).sum()),
        "n_insufficient_context": int((eligible & ~has_context).sum()),
        "n_evaluated": int(evaluated.sum()),
    }
    return pure_order_anomaly, diag


def gap_zero_diagnostics(df):
    """Fraction of same-node consecutive timestamps that collide exactly (resolution floor),
    over all rows."""
    gaps = df["gap_prev_s"].dropna()
    per_node_zero_frac = (
        df.dropna(subset=["gap_prev_s"]).groupby("node")["gap_prev_s"].apply(lambda s: (s == 0).mean())
    )
    return {
        "overall_zero_frac": float((gaps == 0).mean()),
        "n_nodes_with_gaps": int(len(per_node_zero_frac)),
        "node_zero_frac_mean": float(per_node_zero_frac.mean()),
        "node_zero_frac_median": float(per_node_zero_frac.median()),
        "pct_nodes_majority_zero_gaps": float(100 * (per_node_zero_frac > 0.5).mean()),
    }


def flag_timing_gap_anomaly(df, subset_mask=None):
    """subset_mask: which rows to evaluate (defaults to df["anomaly"]). The baseline is always
    built from genuine normal-to-normal gaps (~df["anomaly"]), regardless of subset_mask."""
    if subset_mask is None:
        subset_mask = df["anomaly"]

    normal_seq = (~df["anomaly"]) & (~df["prev_anomaly"].fillna(True)) & df["gap_prev_s"].notna()
    baselines = compute_node_baselines(df, normal_seq)

    anomalies = df[subset_mask].copy()
    z_prev = score_gap_zscore(anomalies["gap_prev_s"], anomalies["node"], baselines)
    z_next = score_gap_zscore(anomalies["gap_next_s"], anomalies["node"], baselines)

    flagged = (z_prev.abs() > Z_THRESH) | (z_next.abs() > Z_THRESH)
    has_context = anomalies["gap_prev_s"].notna() | anomalies["gap_next_s"].notna()
    timing_anomaly = pd.Series(np.where(has_context, flagged, np.nan), index=anomalies.index)

    anomaly_nodes = anomalies["node"].unique()
    diag = {
        "n_anomaly_nodes": len(anomaly_nodes),
        "n_anomaly_nodes_valid_baseline": int(pd.Index(anomaly_nodes).isin(baselines["valid_nodes"]).sum()),
        "n_anomaly_nodes_fallback_low_count": int(
            pd.Index(anomaly_nodes).isin(baselines["fallback_low_count_nodes"]).sum()
        ),
        "n_anomaly_nodes_fallback_zero_mad": int(
            pd.Index(anomaly_nodes).isin(baselines["fallback_zero_mad_nodes"]).sum()
        ),
        "n_nodes_fallback_low_count_total": len(baselines["fallback_low_count_nodes"]),
        "n_nodes_fallback_zero_mad_total": len(baselines["fallback_zero_mad_nodes"]),
    }
    return timing_anomaly, diag


def timing_rate_by_content(anomalies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for has_content, group in anomalies.groupby("new_or_rare_template"):
        n = len(group)
        flagged = int((group["timing_gap_anomaly"] == True).sum())  # noqa: E712
        rows.append(
            {
                "new_or_rare_template": bool(has_content),
                "n_anomalies": n,
                "n_timing_flagged": flagged,
                "pct_timing_flagged": round(100 * flagged / n, 2) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_signatures(anomalies: pd.DataFrame, order_diag: dict) -> pd.DataFrame:
    n = len(anomalies)

    def row(name, flagged_mask, insufficient=0, excluded_unknown=0):
        flagged = int((flagged_mask == True).sum())  # noqa: E712
        return {
            "signature": name,
            "count_flagged": flagged,
            "pct_of_anomalies": round(100 * flagged / n, 2),
            "count_insufficient_context": insufficient,
            "count_excluded_unknown_template": excluded_unknown,
        }

    rows = [
        row("new_or_rare_template", anomalies["new_or_rare_template"]),
        row(
            "pure_order_anomaly",
            anomalies["pure_order_anomaly"],
            insufficient=order_diag["n_insufficient_context"],
            excluded_unknown=order_diag["n_ineligible_unknown_template"],
        ),
        row(
            "timing_gap_anomaly",
            anomalies["timing_gap_anomaly"],
            insufficient=int(anomalies["timing_gap_anomaly"].isna().sum()),
        ),
    ]

    # timing_only: flagged on timing and conclusively not on template or pure order. NaN
    # (ineligible) order rows are excluded because their order status is undetermined.
    timing_only = (
        (anomalies["new_or_rare_template"] == False)  # noqa: E712
        & (anomalies["pure_order_anomaly"] == False)  # noqa: E712  (NaN != False, so this excludes NaN rows)
        & (anomalies["timing_gap_anomaly"] == True)  # noqa: E712
    )
    rows.append(row("timing_only", timing_only))

    return pd.DataFrame(rows)


def plot_breakdown(summary: pd.DataFrame):
    core = summary[summary["signature"].isin(["new_or_rare_template", "pure_order_anomaly", "timing_gap_anomaly"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(core["signature"], core["pct_of_anomalies"], color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("% of anomalous rows flagged")
    ax.set_title("BGL native anomalies: which signature do they show?")
    ax.set_ylim(0, 100)
    for i, v in enumerate(core["pct_of_anomalies"]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def write_markdown_summary(summary: pd.DataFrame, n_anomalies: int, gap_diag: dict, timing_diag: dict, content_split: pd.DataFrame):
    pct = {r["signature"]: r["pct_of_anomalies"] for r in summary.to_dict("records")}
    counts = {r["signature"]: r["count_flagged"] for r in summary.to_dict("records")}
    order_row = summary[summary["signature"] == "pure_order_anomaly"].iloc[0]

    timing_given_content = content_split[content_split["new_or_rare_template"]].iloc[0]
    timing_given_no_content = content_split[~content_split["new_or_rare_template"]].iloc[0]

    text = (
        "# Premise audit summary\n\n"
        f"Labeled-anomalous rows: {n_anomalies:,}\n\n"
        f"- new_or_rare_template: {counts['new_or_rare_template']:,} ({pct['new_or_rare_template']:.2f}%)\n"
        f"- pure_order_anomaly: {order_row['count_flagged']:,} ({pct['pure_order_anomaly']:.2f}%); "
        f"{order_row['count_excluded_unknown_template']:,} rows excluded for involving a template the node "
        "never produced normally\n"
        f"- timing_gap_anomaly: {counts['timing_gap_anomaly']:,} ({pct['timing_gap_anomaly']:.2f}%); "
        f"{timing_given_content['pct_timing_flagged']:.1f}% of the {timing_given_content['n_anomalies']:,} rows "
        f"with a new/rare template vs {timing_given_no_content['pct_timing_flagged']:.1f}% of the "
        f"{timing_given_no_content['n_anomalies']:,} rows without\n"
        f"- timing_only: {counts['timing_only']:,} ({pct['timing_only']:.2f}%)\n\n"
        "Timing baseline diagnostics:\n\n"
        f"- inter-arrival gaps exactly zero: {100 * gap_diag['overall_zero_frac']:.2f}% "
        f"(median per-node {100 * gap_diag['node_zero_frac_median']:.2f}%, "
        f"{gap_diag['pct_nodes_majority_zero_gaps']:.1f}% of nodes majority-zero-gap)\n"
        f"- nodes on the pooled baseline: {timing_diag['n_nodes_fallback_low_count_total']:,} (too few normal gaps), "
        f"{timing_diag['n_nodes_fallback_zero_mad_total']:,} (zero MAD)\n"
    )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(text)


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
    plot_breakdown(summary)
    write_markdown_summary(summary, len(anomalies), gap_diag, timing_diag, content_split)

    print("=== timing baseline diagnostics ===")
    print(f"Overall fraction of ALL inter-arrival gaps == 0:        {gap_diag['overall_zero_frac']:.4f}")
    print(f"Nodes with >=1 valid gap:                               {gap_diag['n_nodes_with_gaps']:,}")
    print(f"Per-node zero-gap fraction, mean / median:              {gap_diag['node_zero_frac_mean']:.4f} / {gap_diag['node_zero_frac_median']:.4f}")
    print(f"% of nodes that are majority-zero-gap:                  {gap_diag['pct_nodes_majority_zero_gaps']:.2f}%")
    print(f"Anomalous rows' distinct nodes:                         {timing_diag['n_anomaly_nodes']:,}")
    print(f"  -> with a valid per-node baseline:                    {timing_diag['n_anomaly_nodes_valid_baseline']:,}")
    print(f"  -> fell back to pooled (too few normal gaps):         {timing_diag['n_anomaly_nodes_fallback_low_count']:,}")
    print(f"  -> fell back to pooled (per-node MAD == 0):           {timing_diag['n_anomaly_nodes_fallback_zero_mad']:,}")
    print(f"(corpus-wide: {timing_diag['n_nodes_fallback_low_count_total']:,} nodes fall back for low count, "
          f"{timing_diag['n_nodes_fallback_zero_mad_total']:,} for zero MAD)")

    print("\n=== timing_gap_anomaly rate by content-novelty status ===")
    print(content_split.to_string(index=False))

    print(f"\nWrote {OUT_CSV}, {OUT_MD}, {OUT_PNG}")


if __name__ == "__main__":
    main()
