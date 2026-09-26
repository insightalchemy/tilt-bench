"""
Per-stream timing-baseline viability gate for the additional Loghub datasets.

Computes per-stream (median, MAD) baselines of inter-arrival gaps (src/timing_baseline.py) and
reports the fraction of streams with a valid, non-degenerate baseline. A dataset is VIABLE at
>= VIABLE_THRESHOLD, CONDITIONALLY VIABLE (inject only into valid-baseline streams) at
>= CONDITIONAL_THRESHOLD, and NOT VIABLE otherwise (exit code 2).

These datasets have no anomaly labels, so every row with a preceding gap enters the baseline.

Writes:
  results/multi_dataset/<name>/viability_gate.md

Usage:
    python src/multi_dataset_gate.py --dataset openstack
    python src/multi_dataset_gate.py --dataset openstack --parsed-path /tmp/openstack_probe.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.multi_dataset_registry import (
    CONDITIONAL_THRESHOLD,
    VIABLE_THRESHOLD,
    DATASETS,
    parsed_path,
    require_known,
    results_dir,
)
from src.timing_baseline import add_sequence_context, compute_node_baselines


def run_viability_gate(name: str, parsed_path_: Path):
    if not parsed_path_.exists():
        print(f"ERROR: {parsed_path_} does not exist. Run src/parser_{name}.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[viability-gate] loading {parsed_path_}", flush=True)
    df = pd.read_parquet(parsed_path_)
    df["anomaly"] = df["anomaly"].fillna(False).astype(bool)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df = add_sequence_context(df)
    normal_seq = (~df["prev_anomaly"].fillna(True).astype(bool)) & df["gap_prev_s"].notna()

    baselines_raw = compute_node_baselines(df, normal_seq, exclude_zero_from_pooled=False)
    baselines_fixed = compute_node_baselines(df, normal_seq, exclude_zero_from_pooled=True)

    total_nodes = df["node"].nunique()
    n_valid = len(baselines_raw["valid_nodes"])
    n_low_count = len(baselines_raw["fallback_low_count_nodes"])
    n_zero_mad = len(baselines_raw["fallback_zero_mad_nodes"])
    valid_frac = n_valid / total_nodes if total_nodes else float("nan")

    lines = [f"# {name} viability gate: per-node timing baseline", ""]
    lines.append(f"Total rows: {len(df):,}")
    lines.append(f"Total streams ('node'): {total_nodes:,}")
    lines.append(f"Valid (non-degenerate) baseline streams: {n_valid:,} ({valid_frac:.2%})")
    lines.append(f"  -- too few normal-to-normal gaps: {n_low_count:,} ({n_low_count / total_nodes:.2%})" if total_nodes else "  -- N/A (no streams)")
    lines.append(f"  -- zero per-stream MAD: {n_zero_mad:,} ({n_zero_mad / total_nodes:.2%})" if total_nodes else "  -- N/A (no streams)")
    lines.append(f"Pooled fallback (all gaps): median={baselines_raw['global_median']:.6f}s mad={baselines_raw['global_mad']:.6f}s")
    lines.append(f"Pooled fallback (nonzero gaps only): median={baselines_fixed['global_median']:.6f}s mad={baselines_fixed['global_mad']:.6f}s")
    pooled_raw_degenerate = baselines_raw["global_mad"] == 0
    if pooled_raw_degenerate:
        lines.append(
            "Pooled (all-gaps) fallback IS degenerate (MAD=0) -- use exclude_zero_from_pooled=True and "
            "require_valid_baseline=True if injecting."
        )

    if valid_frac >= VIABLE_THRESHOLD:
        verdict = "VIABLE -- per-stream inter-arrival injection should work directly, as on BGL."
        supports = True
    elif valid_frac >= CONDITIONAL_THRESHOLD:
        verdict = (
            "CONDITIONALLY VIABLE -- restrict injector eligibility to the valid-baseline subset "
            "(require_valid_baseline=True)."
        )
        supports = True
    else:
        verdict = "NOT VIABLE -- too few streams have a usable timing baseline for per-stream inter-arrival injection."
        supports = False

    lines.append("")
    lines.append(f"VERDICT: {name} {'DOES' if supports else 'DOES NOT'} support the per-stream inter-arrival injection model as currently designed.")
    lines.append(f"  {verdict}")

    for line in lines:
        print(line)

    return supports, "\n".join(lines) + "\n", valid_frac


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--parsed-path", type=Path, default=None)
    args = ap.parse_args()
    require_known(args.dataset)

    parsed = args.parsed_path or parsed_path(args.dataset)
    supports, report, valid_frac = run_viability_gate(args.dataset, parsed)

    out_dir = results_dir(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viability_gate.md").write_text(report)
    print(f"\nWrote {out_dir / 'viability_gate.md'}")
    if not supports:
        sys.exit(2)


if __name__ == "__main__":
    main()
