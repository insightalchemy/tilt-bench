"""
Parse data/raw/Thunderbird.log (a 5M-line contiguous head slice -- see
results/thunderbird_setup_notes.md) into the SAME tidy schema as src/parser.py's BGL output,
reusing the identical Drain3-based approach (chunked reading, one shared TemplateMiner, same
output columns). Only the raw line field-splitting differs, because Thunderbird's raw format is
genuinely different from BGL's -- this is not a redesign of the parsing approach.

Raw Thunderbird line format (whitespace-separated, 8 fixed fields + free-text content):
    <label> <unix_ts> <date> <node> <month> <day> <time> <user@host> <content...>
e.g.
    - 1131523501 2005.11.09 aadmin1 Nov 10 00:05:01 src@aadmin1 in.tftpd[14620]: tftp: client does not accept options

`label` is "-" for normal lines, or an alert code (CPU / ECC / EXT_FS / VAPI observed in this
slice) for anomalies. `node` (field 4) is the originating hostname, the per-node grouping key.

IMPORTANT timestamp finding: unlike BGL, Thunderbird has NO sub-second timestamp field anywhere --
both `unix_ts` and the self-reported `<month> <day> <time>` fields are second-resolution only.
Worse, the self-reported field is not always reliable (spot-checked: 4/5 sampled lines showed a
consistent -8h offset from unix_ts, consistent with a fixed Pacific-time convention like BGL's, but
1/5 showed a +16h "offset" that is really a one-day rollover error in the self-reported month/day --
i.e. real per-node clock drift in the source data, not a parsing bug). `unix_ts` is therefore used
as the authoritative timestamp here, not the self-reported fields.

Output columns: timestamp, node, label, anomaly, event_template, raw_message (identical to BGL).

Usage:
    python src/parser_thunderbird.py --test                  # first 5000 lines -> stdout preview
    python src/parser_thunderbird.py                         # full slice -> data/processed/thunderbird_parsed.parquet
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

RAW_LOG_PATH = Path("data/raw/Thunderbird.log")
DEFAULT_OUT_PATH = Path("data/processed/thunderbird_parsed.parquet")
CHUNK_SIZE = 200_000


def build_template_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def parse_line(line: str):
    """Split one raw Thunderbird line into its fields. Returns None for malformed lines."""
    parts = line.rstrip("\n").split(None, 8)
    if len(parts) < 8:
        return None
    label, unix_ts, _date, node, _month, _day, _time, _user_host = parts[:8]
    content = parts[8] if len(parts) == 9 else ""
    return label, node, unix_ts, content


def parse_chunk(lines, template_miner: TemplateMiner):
    labels, nodes, timestamps, anomalies, templates, raw_messages = [], [], [], [], [], []
    skipped = 0
    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            skipped += 1
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
    return df, skipped


def parse_file(limit: int | None = None):
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
    gaps = df.sort_values(["node", "timestamp"]).groupby("node")["timestamp"].diff().dt.total_seconds().dropna()
    print("\n=== Summary ===")
    print(f"Total lines read:        {lines_seen:,}")
    print(f"Rows parsed:             {n:,}")
    print(f"Skipped (malformed):     {skipped:,}")
    print(f"% anomalous:             {pct_anomalous:.2f}%")
    print(f"Unique nodes:            {df['node'].nunique():,}")
    print(f"Unique event templates:  {df['event_template'].nunique():,}")
    print(f"Timestamp range:         {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Timestamp dtype:         {df['timestamp'].dtype}")
    print(f"Fraction of per-node inter-arrival gaps exactly 0s: {(gaps == 0).mean():.4f}  (resolution check)")


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
