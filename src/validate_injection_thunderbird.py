"""
Port of src/validate_injection.py to Thunderbird -- reuses check_ordering, plot_before_after, and
instrument_validation UNMODIFIED (they already take df/paths as parameters, no BGL-specific logic
inside); only the path construction differs.

Note on the timing_gap_anomaly check specifically: it recomputes a fresh baseline from the
injected dataset (via src.premise_audit.flag_timing_gap_anomaly, unmodified). This does NOT need
the exclude_zero_from_pooled fix from Part 1, because our injected rows all live on nodes that
were specifically selected for having a VALID per-node baseline (require_valid_baseline=True in
src/injector_thunderbird.py) -- so `node in baselines["valid_nodes"]` is true for them regardless
of whether the pooled fallback is fixed, and the pooled fallback is never actually used for these
specific rows.

Usage:
    python src/validate_injection_thunderbird.py --type stall   # default
    python src/validate_injection_thunderbird.py --type burst
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import pandas as pd

from src.validate_injection import SORT_COL_BY_TYPE, N_PLOT_NODES, check_ordering, instrument_validation, plot_before_after

CLEAN_PATH = Path("data/processed/thunderbird_parsed.parquet")


def load_clean_sorted():
    df = pd.read_parquet(CLEAN_PATH)
    return df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=["stall", "burst"], default="stall")
    args = ap.parse_args()
    inj_type = args.type

    injected_path = Path(f"data/processed/thunderbird_injected_{inj_type}.parquet")
    labels_path = Path(f"data/processed/thunderbird_injection_labels_{inj_type}.csv")
    fig_path = Path(f"figures/thunderbird_{inj_type}_injection_check.png")
    sort_col = SORT_COL_BY_TYPE[inj_type]

    df_clean = load_clean_sorted()
    df_injected = pd.read_parquet(injected_path)
    labels = pd.read_csv(labels_path, parse_dates=["start", "end"])

    ordering = check_ordering(df_injected)
    print(f"=== Validation 1 (thunderbird, {inj_type}): ordering check (whole injected dataset) ===")
    print(f"Rows checked: {ordering['n_rows']:,}")
    print(f"Ordering violations (negative gaps): {ordering['n_violations']:,}")
    print(f"Nodes with a violation: {ordering['violation_node_count']:,}")
    print("PASS -- zero ordering violations" if ordering["n_violations"] == 0 else "FAIL -- ordering violated")

    plot_nodes = labels.sort_values(sort_col, ascending=False)["node"].head(N_PLOT_NODES).tolist()
    plot_before_after(df_clean, df_injected, labels, plot_nodes, f"thunderbird {inj_type}", fig_path)
    print(f"\nWrote {fig_path} for nodes: {plot_nodes}")

    print(f"\n=== Validation 2 (thunderbird, {inj_type}): premise-audit signatures on injected rows ===")
    audit_result = instrument_validation(df_injected)
    print(audit_result.to_string(index=False))

    timing_rate = audit_result.loc[audit_result["signature"] == "timing_gap_anomaly", "pct_of_injections"].iloc[0]
    template_rate = audit_result.loc[audit_result["signature"] == "new_or_rare_template", "pct_of_injections"].iloc[0]
    order_rate = audit_result.loc[audit_result["signature"] == "pure_order_anomaly", "pct_of_injections"].iloc[0]
    insufficient = audit_result.loc[audit_result["signature"] == "timing_gap_anomaly", "count_insufficient_context"].iloc[0]
    n = audit_result.loc[audit_result["signature"] == "timing_gap_anomaly", "count_flagged"].iloc[0]
    print(
        f"\nExpected: timing fires, content/order stay quiet. "
        f"Observed: timing={timing_rate:.1f}%, template={template_rate:.1f}%, pure_order={order_rate:.1f}% "
        f"(timing signature had {insufficient} rows with insufficient context out of the injected set)."
    )


if __name__ == "__main__":
    main()
