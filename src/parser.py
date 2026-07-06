"""
Parse data/raw/BGL.log into a tidy table for TILT-Bench.

Raw BGL line format (whitespace-separated, 10 fields):
    <label> <unix_ts> <date> <node> <precise_time> <node_repeat> <type> <component> <level> <content...>
e.g.
    - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected

`label` is "-" for normal lines, or an alert code (e.g. "APPREAD") for anomalies.
`precise_time` carries microsecond resolution and is used for the timestamp column
(the unix_ts field is second-resolution only).

Output columns: timestamp, node, label, anomaly, event_template, raw_message.

Usage:
    python src/parser.py --test                  # first 5000 lines -> stdout preview, no file written
    python src/parser.py                         # full file -> data/processed/bgl_parsed.parquet
    python src/parser.py --limit 5000 --out data/processed/bgl_parsed_test.parquet
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

RAW_LOG_PATH = Path("data/raw/BGL.log")
DEFAULT_OUT_PATH = Path("data/processed/bgl_parsed.parquet")
CHUNK_SIZE = 200_000
TIME_FORMAT = "%Y-%m-%d-%H.%M.%S.%f"


def build_template_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def parse_line(line: str):
    """Split one raw BGL line into its fields. Returns None for malformed lines.

    Some lines (e.g. "RAS KERNEL FATAL") have no content after the Level field --
    that's a valid empty message, not a malformed line, so we only require 9 fields.
    """
    parts = line.rstrip("\n").split(None, 9)
    if len(parts) < 9:
        return None
    label, _unix_ts, _date, node, precise_time, _node_repeat, _type, _component, _level = parts[:9]
    content = parts[9] if len(parts) == 10 else ""
    return label, node, precise_time, content


def parse_chunk(lines, template_miner: TemplateMiner):
    labels, nodes, timestamps, anomalies, templates, raw_messages = [], [], [], [], [], []
    skipped = 0
    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            skipped += 1
            continue
        label, node, precise_time, content = parsed
        result = template_miner.add_log_message(content)
        labels.append(label)
        nodes.append(node)
        timestamps.append(precise_time)
        anomalies.append(label != "-")
        templates.append(result["template_mined"])
        raw_messages.append(content)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, format=TIME_FORMAT),
            "node": nodes,
            "label": labels,
            "anomaly": anomalies,
            "event_template": templates,
            "raw_message": raw_messages,
        }
    )
    return df, skipped


def parse_file(limit: int | None = None):
    """Read RAW_LOG_PATH in chunks, mining templates incrementally with a single shared
    TemplateMiner so templates stay consistent across chunks. Returns (df, skipped, lines_seen)."""
    template_miner = build_template_miner()
    chunk_dfs = []
    lines_seen = 0
    skipped_total = 0
    start = time.time()

    with open(RAW_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        buffer = []
        for line in f:
            buffer.append(line)
            lines_seen += 1
            if limit is not None and lines_seen >= limit:
                break
            if len(buffer) >= CHUNK_SIZE:
                df, skipped = parse_chunk(buffer, template_miner)
                skipped_total += skipped
                chunk_dfs.append(df)
                buffer = []
                elapsed = time.time() - start
                print(f"  ...{lines_seen:,} lines processed ({elapsed:.0f}s elapsed)", file=sys.stderr)
        if buffer:
            df, skipped = parse_chunk(buffer, template_miner)
            skipped_total += skipped
            chunk_dfs.append(df)

    full_df = pd.concat(chunk_dfs, ignore_index=True)
    return full_df, skipped_total, lines_seen


def print_summary(df: pd.DataFrame, skipped: int, lines_seen: int):
    n = len(df)
    pct_anomalous = 100 * df["anomaly"].mean() if n else 0.0
    print("\n=== Summary ===")
    print(f"Total lines read:        {lines_seen:,}")
    print(f"Rows parsed:             {n:,}")
    print(f"Skipped (malformed):     {skipped:,}")
    print(f"% anomalous:             {pct_anomalous:.2f}%")
    print(f"Unique nodes:            {df['node'].nunique():,}")
    print(f"Unique event templates:  {df['event_template'].nunique():,}")
    print(f"Timestamp range:         {df['timestamp'].min()} -> {df['timestamp'].max()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="Preview first 5000 lines to stdout, don't write output.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N lines.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Output parquet path.")
    args = ap.parse_args()

    limit = args.limit
    if args.test and limit is None:
        limit = 5000

    df, skipped, lines_seen = parse_file(limit=limit)

    if args.test:
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 160)
        print(df.head(20).to_string())
        print("\ndtypes:")
        print(df.dtypes)
        print_summary(df, skipped, lines_seen)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print_summary(df, skipped, lines_seen)


if __name__ == "__main__":
    main()
