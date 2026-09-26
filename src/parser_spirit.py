"""
Parse data/raw/Spirit.log (the published 5M-line Spirit subset; see src/spirit_setup.py) into the
same per-line schema as src/parser.py (BGL) and src/parser_thunderbird.py.

Raw line format (same as Thunderbird; 8 fixed fields + free-text content):
    <label> <unix_ts> <date> <node> <month> <day> <time> <user@host> <content...>

`label` is "-" for normal lines, or an alert code for anomalies. `node` (field 4) is the
per-node grouping key and `unix_ts` (field 2) the timestamp (second resolution). The parser
tracks the first malformed line and warns if the skip rate exceeds SKIP_WARN_FRACTION.

Output columns: timestamp, node, label, anomaly, event_template, raw_message.

Usage (after `python src/spirit_setup.py download`):
    python src/parser_spirit.py --sample-raw          # print 20 raw lines, no parsing
    python src/parser_spirit.py --test                # first 5000 lines -> stdout preview
    python src/parser_spirit.py                       # full slice -> data/processed/spirit_parsed.parquet
    python src/parser_spirit.py --limit 500000 --out /tmp/spirit_probe.parquet
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

RAW_LOG_PATH = Path("data/raw/Spirit.log")
DEFAULT_OUT_PATH = Path("data/processed/spirit_parsed.parquet")
CHUNK_SIZE = 200_000
SKIP_WARN_FRACTION = 0.01


def build_template_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def parse_line(line: str):
    parts = line.rstrip("\n").split(None, 8)
    if len(parts) < 8:
        return None
    label, unix_ts, _date, node, _month, _day, _time, _user_host = parts[:8]
    if not unix_ts.lstrip("-").isdigit():
        return None
    content = parts[8] if len(parts) == 9 else ""
    return label, node, unix_ts, content


def parse_chunk(lines, template_miner: TemplateMiner):
    labels, nodes, timestamps, anomalies, templates, raw_messages = [], [], [], [], [], []
    skipped = 0
    sample_skipped_line = None
    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            skipped += 1
            if sample_skipped_line is None:
                sample_skipped_line = line.rstrip("\n")
            continue
        label, node, unix_ts, content = parsed
        result = template_miner.add_log_message(content)
        labels.append(label)
        nodes.append(node)
        timestamps.append(int(unix_ts))
        anomalies.append(label != "-")
        templates.append(result["template_mined"])
        raw_messages.append(content)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pd.array(timestamps, dtype="int64"), unit="s"),
            "node": nodes,
            "label": labels,
            "anomaly": anomalies,
            "event_template": templates,
            "raw_message": raw_messages,
        }
    )
    return df, skipped, sample_skipped_line


def sample_raw_lines(path: Path, n: int):
    if not path.exists():
        print(f"ERROR: {path} does not exist. Run 'python src/spirit_setup.py download' first.", file=sys.stderr)
        sys.exit(1)
    print(f"=== First {n} raw lines of {path} ===")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(f"{i:>4}: {line.rstrip(chr(10))}")


def parse_file(limit: int | None = None):
    if not RAW_LOG_PATH.exists():
        print(
            f"ERROR: {RAW_LOG_PATH} does not exist. Run 'python src/spirit_setup.py download' "
            "first to produce this file.",
            file=sys.stderr,
        )
        sys.exit(1)

    template_miner = build_template_miner()
    chunk_dfs = []
    lines_seen = 0
    skipped_total = 0
    first_sample_skipped = None
    start = time.time()

    try:
        with open(RAW_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            buffer = []
            for line in f:
                buffer.append(line)
                lines_seen += 1
                if limit is not None and lines_seen >= limit:
                    break
                if len(buffer) >= CHUNK_SIZE:
                    df, skipped, sample_skipped = parse_chunk(buffer, template_miner)
                    skipped_total += skipped
                    if first_sample_skipped is None:
                        first_sample_skipped = sample_skipped
                    chunk_dfs.append(df)
                    buffer = []
                    elapsed = time.time() - start
                    print(
                        f"  ...{lines_seen:,} lines processed ({elapsed:.0f}s elapsed, "
                        f"{skipped_total:,} skipped so far, this-chunk skip rate {skipped / CHUNK_SIZE:.2%})",
                        file=sys.stderr,
                        flush=True,
                    )
                    if lines_seen >= CHUNK_SIZE and (skipped_total / lines_seen) > SKIP_WARN_FRACTION:
                        print(
                            f"  WARNING: cumulative skip rate {skipped_total / lines_seen:.2%} exceeds "
                            f"{SKIP_WARN_FRACTION:.0%} -- the raw format may not match the expected "
                            f"8-field layout. Sample skipped line: {first_sample_skipped!r}",
                            file=sys.stderr,
                            flush=True,
                        )
            if buffer:
                df, skipped, sample_skipped = parse_chunk(buffer, template_miner)
                skipped_total += skipped
                if first_sample_skipped is None:
                    first_sample_skipped = sample_skipped
                chunk_dfs.append(df)
    except Exception:
        print(
            f"FATAL: parse_file crashed after {lines_seen:,} lines seen, {skipped_total:,} skipped, "
            f"{time.time() - start:.0f}s elapsed. Full traceback follows.",
            file=sys.stderr,
        )
        traceback.print_exc()
        raise

    full_df = pd.concat(chunk_dfs, ignore_index=True) if chunk_dfs else pd.DataFrame(
        columns=["timestamp", "node", "label", "anomaly", "event_template", "raw_message"]
    )
    return full_df, skipped_total, lines_seen, first_sample_skipped


def compute_zero_gap_stats(df: pd.DataFrame):
    if df.empty:
        return float("nan"), float("nan")
    df_sorted = df.sort_values(["node", "timestamp"], kind="mergesort")
    gaps = df_sorted.groupby("node")["timestamp"].diff().dt.total_seconds()
    valid = gaps.notna()
    if not valid.any():
        return float("nan"), float("nan")
    overall_zero_frac = (gaps[valid] == 0).mean()
    per_node_zero_frac = gaps[valid].groupby(df_sorted.loc[valid, "node"]).apply(lambda s: (s == 0).mean())
    majority_zero_node_frac = (per_node_zero_frac > 0.5).mean() if len(per_node_zero_frac) else float("nan")
    return overall_zero_frac, majority_zero_node_frac


def print_summary(df: pd.DataFrame, skipped: int, lines_seen: int, first_sample_skipped):
    n = len(df)
    pct_anomalous = 100 * df["anomaly"].mean() if n else float("nan")
    skip_frac = skipped / lines_seen if lines_seen else float("nan")
    overall_zero_frac, majority_zero_node_frac = compute_zero_gap_stats(df)

    print("\n=== Summary ===")
    print(f"Total lines read:        {lines_seen:,}")
    print(f"Rows parsed:             {n:,}")
    print(f"Skipped (malformed):     {skipped:,}  ({skip_frac:.2%} of lines read)")
    if skip_frac > SKIP_WARN_FRACTION:
        print(
            f"WARNING: skip rate {skip_frac:.2%} exceeds {SKIP_WARN_FRACTION:.0%} -- do NOT trust this "
            f"parse. Sample skipped line: {first_sample_skipped!r}. Re-check the raw format assumption "
            "in this module's docstring against real Spirit lines (--sample-raw)."
        )
    print(f"% anomalous:             {pct_anomalous:.2f}%")
    print(f"Unique nodes:            {df['node'].nunique() if n else 0:,}")
    print(f"Unique event templates:  {df['event_template'].nunique() if n else 0:,}")
    if n:
        print(f"Timestamp range:         {df['timestamp'].min()} -> {df['timestamp'].max()}")
    else:
        print("Timestamp range:         N/A (no rows parsed)")
    print(f"Timestamp dtype:         {df['timestamp'].dtype if n else 'N/A'}")
    print(f"Fraction of per-node inter-arrival gaps exactly 0s (overall):        {overall_zero_frac:.4f}")
    print(f"Fraction of nodes that are majority-zero-gap (>50% of their gaps==0): {majority_zero_node_frac:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="Preview first 5000 lines to stdout, don't write output.")
    ap.add_argument("--sample-raw", nargs="?", type=int, const=20, default=None, help="Print N raw lines (default 20) unparsed, then exit. No template mining, no output written.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N lines.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Output parquet path.")
    args = ap.parse_args()

    if args.sample_raw is not None:
        sample_raw_lines(RAW_LOG_PATH, args.sample_raw)
        return

    limit = args.limit
    if args.test and limit is None:
        limit = 5000

    df, skipped, lines_seen, first_sample_skipped = parse_file(limit=limit)

    if args.test:
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 160)
        print(df.head(20).to_string())
        print("\ndtypes:")
        print(df.dtypes)
        print_summary(df, skipped, lines_seen, first_sample_skipped)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print_summary(df, skipped, lines_seen, first_sample_skipped)


if __name__ == "__main__":
    main()
