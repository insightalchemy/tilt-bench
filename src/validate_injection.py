"""
The two validations CLAUDE.md requires before trusting any injector metric. Works for both
injection types (--type stall|burst).

1. VISUAL + ORDERING: for a few affected nodes, plot inter-arrival times before vs after
   injection, and run an automated zero-ordering-violation check across the WHOLE injected
   dataset (every node, not just injected ones).
2. INSTRUMENT VALIDATION: run the existing premise-audit signatures (new/rare template,
   pure_order_anomaly, timing_gap_anomaly) on the injected rows. Since injection only touches
   timestamps, the expectation is: timing fires, content/order signatures stay silent. If the
   audit instrument can't see our own injected timing fault, it can't be trusted to measure
   the real phenomenon either -- so this check validates the injector AND the audit together.

Writes:
  figures/{type}_injection_check.png -- before/after IAT plot for 3 affected nodes

Reads data/processed/bgl_injected_{type}.parquet + injection_labels_{type}.csv (from src/injector.py).

Usage:
    python src/validate_injection.py --type stall   # default
    python src/validate_injection.py --type burst
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.premise_audit import flag_pure_order_anomaly, flag_template_rarity, flag_timing_gap_anomaly
from src.timing_baseline import add_sequence_context

CLEAN_PATH = Path("data/processed/bgl_parsed.parquet")

N_PLOT_NODES = 3
COLOR_CLEAN = "#2a78d6"  # blue
COLOR_INJECTED = "#e34948"  # red

# column used to pick the most dramatic examples to plot, per injection type
SORT_COL_BY_TYPE = {"stall": "delta_seconds", "burst": "total_time_saved_s"}


def load_clean_sorted():
    df = pd.read_parquet(CLEAN_PATH)
    return df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)


def check_ordering(df_injected: pd.DataFrame) -> dict:
    """Zero-ordering-violation check across the WHOLE injected dataset, not just injected nodes."""
    df = df_injected.sort_values(["node", "timestamp"], kind="mergesort")
    diffs = df.groupby("node", sort=False)["timestamp"].diff().dt.total_seconds()
    violations = diffs < 0
    return {
        "n_rows": len(df),
        "n_violations": int(violations.sum()),
        "violation_node_count": int(df.loc[violations.fillna(False), "node"].nunique()) if violations.any() else 0,
    }


def plot_before_after(df_clean: pd.DataFrame, df_injected: pd.DataFrame, labels: pd.DataFrame, nodes: list, inj_type: str, fig_path: Path):
    fig, axes = plt.subplots(len(nodes), 1, figsize=(9, 3.2 * len(nodes)), squeeze=False)
    axes = axes[:, 0]

    for ax, node in zip(axes, nodes):
        clean_node = df_clean[df_clean["node"] == node].sort_values("timestamp")
        inj_node = df_injected[df_injected["node"] == node].sort_values("timestamp")

        clean_gaps = clean_node["timestamp"].diff().dt.total_seconds()
        inj_gaps = inj_node["timestamp"].diff().dt.total_seconds()

        ax.plot(np.arange(len(clean_gaps)), clean_gaps, color=COLOR_CLEAN, linewidth=1.5, label="clean (before)", alpha=0.85)
        ax.plot(np.arange(len(inj_gaps)), inj_gaps, color=COLOR_INJECTED, linewidth=1.2, linestyle="--", label="injected (after)", alpha=0.85)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylabel("inter-arrival gap (s, symlog)")
        ax.set_title(f"node {node}")
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("event index (within node)")
    fig.suptitle(f"{inj_type.capitalize()} injection: inter-arrival time before vs after, per affected node", y=1.00)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def instrument_validation(df_injected: pd.DataFrame) -> pd.DataFrame:
    """Run the premise-audit signatures on the injected rows (the specific row right after each
    stall) and report the rate at which each signature fires."""
    df = df_injected.drop(columns=["injected_row"]).copy()
    df = add_sequence_context(df)
    df = flag_template_rarity(df)

    injected_mask = df_injected["injected_row"].to_numpy()
    pure_order, _ = flag_pure_order_anomaly(df, subset_mask=pd.Series(injected_mask, index=df.index))
    timing_anomaly, _ = flag_timing_gap_anomaly(df, subset_mask=pd.Series(injected_mask, index=df.index))

    injected_rows = df.loc[injected_mask].copy()
    injected_rows["pure_order_anomaly"] = pure_order
    injected_rows["timing_gap_anomaly"] = timing_anomaly

    n = len(injected_rows)
    rows = []
    for name, col in [
        ("new_or_rare_template", injected_rows["new_or_rare_template"]),
        ("pure_order_anomaly", injected_rows["pure_order_anomaly"]),
        ("timing_gap_anomaly", injected_rows["timing_gap_anomaly"]),
    ]:
        flagged = int((col == True).sum())  # noqa: E712
        insufficient = int(col.isna().sum())
        rows.append(
            {
                "signature": name,
                "count_flagged": flagged,
                "pct_of_injections": round(100 * flagged / n, 2) if n else float("nan"),
                "count_insufficient_context": insufficient,
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=["stall", "burst"], default="stall")
    args = ap.parse_args()
    inj_type = args.type

    injected_path = Path(f"data/processed/bgl_injected_{inj_type}.parquet")
    labels_path = Path(f"data/processed/injection_labels_{inj_type}.csv")
    fig_path = Path(f"figures/{inj_type}_injection_check.png")
    sort_col = SORT_COL_BY_TYPE[inj_type]

    df_clean = load_clean_sorted()
    df_injected = pd.read_parquet(injected_path)
    labels = pd.read_csv(labels_path, parse_dates=["start", "end"])

    ordering = check_ordering(df_injected)
    print(f"=== Validation 1 ({inj_type}): ordering check (whole injected dataset) ===")
    print(f"Rows checked: {ordering['n_rows']:,}")
    print(f"Ordering violations (negative gaps): {ordering['n_violations']:,}")
    print(f"Nodes with a violation: {ordering['violation_node_count']:,}")
    print("PASS -- zero ordering violations" if ordering["n_violations"] == 0 else "FAIL -- ordering violated")

    plot_nodes = labels.sort_values(sort_col, ascending=False)["node"].head(N_PLOT_NODES).tolist()
    plot_before_after(df_clean, df_injected, labels, plot_nodes, inj_type, fig_path)
    print(f"\nWrote {fig_path} for nodes: {plot_nodes}")

    print(f"\n=== Validation 2 ({inj_type}): premise-audit signatures on injected rows ===")
    audit_result = instrument_validation(df_injected)
    print(audit_result.to_string(index=False))

    timing_rate = audit_result.loc[audit_result["signature"] == "timing_gap_anomaly", "pct_of_injections"].iloc[0]
    template_rate = audit_result.loc[audit_result["signature"] == "new_or_rare_template", "pct_of_injections"].iloc[0]
    order_rate = audit_result.loc[audit_result["signature"] == "pure_order_anomaly", "pct_of_injections"].iloc[0]
    print(
        f"\nExpected: timing fires, content/order stay quiet. "
        f"Observed: timing={timing_rate:.1f}%, template={template_rate:.1f}%, pure_order={order_rate:.1f}%."
    )


if __name__ == "__main__":
    main()
