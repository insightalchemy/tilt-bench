"""
Injector validation (ordering check, before/after plot, instrument validation) for Spirit,
using the functions of src/validate_injection.py with Spirit paths.

Usage:
    python src/validate_injection_spirit.py --type stall   # default
    python src/validate_injection_spirit.py --type burst
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.validate_injection import N_PLOT_NODES, SORT_COL_BY_TYPE, check_ordering, instrument_validation, plot_before_after

CLEAN_PATH = Path("data/processed/spirit_parsed.parquet")


def load_clean_sorted():
    df = pd.read_parquet(CLEAN_PATH)
    return df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=["stall", "burst"], default="stall")
    args = ap.parse_args()
    inj_type = args.type

    injected_path = Path(f"data/processed/spirit_injected_{inj_type}.parquet")
    labels_path = Path(f"data/processed/spirit_injection_labels_{inj_type}.csv")
    fig_path = Path(f"figures/spirit_{inj_type}_injection_check.png")
    sort_col = SORT_COL_BY_TYPE[inj_type]

    df_clean = load_clean_sorted()
    df_injected = pd.read_parquet(injected_path)
    labels = pd.read_csv(labels_path, parse_dates=["start", "end"])

    ordering = check_ordering(df_injected)
    print(f"=== Validation 1 (spirit, {inj_type}): ordering check (whole injected dataset) ===")
    print(f"Rows checked: {ordering['n_rows']:,}")
    print(f"Ordering violations (negative gaps): {ordering['n_violations']:,}")
    print(f"Nodes with a violation: {ordering['violation_node_count']:,}")
    print("PASS -- zero ordering violations" if ordering["n_violations"] == 0 else "FAIL -- ordering violated")

    plot_nodes = labels.sort_values(sort_col, ascending=False)["node"].head(N_PLOT_NODES).tolist()
    plot_before_after(df_clean, df_injected, labels, plot_nodes, f"spirit {inj_type}", fig_path)
    print(f"\nWrote {fig_path} for nodes: {plot_nodes}")

    print(f"\n=== Validation 2 (spirit, {inj_type}): premise-audit signatures on injected rows ===")
    audit_result = instrument_validation(df_injected)
    print(audit_result.to_string(index=False))

    timing_rate = audit_result.loc[audit_result["signature"] == "timing_gap_anomaly", "pct_of_injections"].iloc[0]
    template_rate = audit_result.loc[audit_result["signature"] == "new_or_rare_template", "pct_of_injections"].iloc[0]
    order_rate = audit_result.loc[audit_result["signature"] == "pure_order_anomaly", "pct_of_injections"].iloc[0]
    insufficient = audit_result.loc[audit_result["signature"] == "timing_gap_anomaly", "count_insufficient_context"].iloc[0]
    print(
        f"\nExpected: timing fires, content/order stay quiet. "
        f"Observed: timing={timing_rate:.1f}%, template={template_rate:.1f}%, pure_order={order_rate:.1f}% "
        f"(timing signature had {insufficient} rows with insufficient context out of the injected set)."
    )


if __name__ == "__main__":
    main()
