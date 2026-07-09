"""
Parse data/raw/HDFS.log into the SAME tidy schema as src/parser.py's BGL output, reusing the
identical Drain3-based approach (chunked reading, one shared TemplateMiner). HDFS's raw format and
labeling are both structurally different from BGL/Thunderbird -- this is not a redesign of the
parsing approach, just the necessary adaptation:

Raw HDFS line format (whitespace-separated, 4 fixed fields + free-text content):
    <date YYMMDD> <time HHMMSS> <pid> <level> <component>: <content...>
e.g.
    081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010

No sub-second field anywhere (same coarse-resolution situation as Thunderbird, but HDFS doesn't
even have a second timestamp source like Thunderbird's unix_ts -- date+time (whole seconds) is
the ONLY timestamp available).

HDFS has NO first-token alert label. Instead, EVERY line references exactly one block_id
(regex `blk_-?\\d+`, confirmed present in 100% of lines via direct check), and a SEPARATE file
(data/raw/anomaly_label.csv) gives a Normal/Anomaly label PER BLOCK SESSION, not per line. This
parser extracts block_id from each line, then LEFT-JOINS the per-block label onto every
constituent line after the main parse (so all lines belonging to an anomalous block get
anomaly=True, matching BGL/Thunderbird's per-row `anomaly` column convention, even though the
label's true granularity is the whole block, not the individual line).

`node` is a best-effort proxy, NOT a reliable per-machine field like BGL/Thunderbird had: the
DataNode's own IP only appears in some line types (e.g. "dest: /10.x.x.x:50010" on
DataXceiver/Receiving-block lines), not universally. Extracted here as the `dest:` IP when
present, else null. See results/hdfs_setup_notes.md for how much of the data this actually covers
and whether it's usable for a per-node timing analysis (Part 2 of that report).

Output columns: timestamp, node, block_id, label, anomaly, event_template, raw_message.

Usage:
    python src/parser_hdfs.py --test                  # first 5000 lines -> stdout preview
    python src/parser_hdfs.py                         # full file -> data/processed/hdfs_parsed.parquet
"""

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

RAW_LOG_PATH = Path("data/raw/HDFS.log")
LABEL_PATH = Path("data/raw/anomaly_label.csv")
DEFAULT_OUT_PATH = Path("data/processed/hdfs_parsed.parquet")
CHUNK_SIZE = 200_000

BLOCK_RE = re.compile(r"blk_-?\d+")
DEST_IP_RE = re.compile(r"dest:\s*/(\d+\.\d+\.\d+\.\d+)")


def build_template_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def parse_line(line: str):
    """Split one raw HDFS line into its fields. Returns None for malformed lines."""
    parts = line.rstrip("\n").split(None, 4)
    if len(parts) < 5:
        return None
    date, time_str, _pid, _level, content = parts
    block_match = BLOCK_RE.search(content)
    if block_match is None:
        return None
    block_id = block_match.group(0)
    dest_match = DEST_IP_RE.search(content)
    node = dest_match.group(1) if dest_match else None
    return date, time_str, block_id, node, content


def parse_chunk(lines, template_miner: TemplateMiner):
    dates, times, block_ids, nodes, templates, raw_messages = [], [], [], [], [], []
    skipped = 0
    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            skipped += 1
            continue
        date, time_str, block_id, node, content = parsed
        result = template_miner.add_log_message(content)
        dates.append(date)
        times.append(time_str)
        block_ids.append(block_id)
        nodes.append(node)
        templates.append(result["template_mined"])
        raw_messages.append(content)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([d + t for d, t in zip(dates, times)], format="%y%m%d%H%M%S"),
            "node": nodes,
            "block_id": block_ids,
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


def join_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One vectorized join at the end, rather than a per-line lookup inside the hot parsing loop."""
    labels = pd.read_csv(LABEL_PATH)
    label_map = dict(zip(labels["BlockId"], labels["Label"]))
    df["label"] = df["block_id"].map(label_map)
    unmatched = df["label"].isna().sum()
    df["anomaly"] = df["label"] == "Anomaly"
    return df, unmatched, len(labels)


def print_summary(df: pd.DataFrame, skipped: int, lines_seen: int, unmatched: int, n_blocks_in_label_file: int):
    n = len(df)
    n_blocks = df["block_id"].nunique()
    n_anom_blocks = df.loc[df["anomaly"], "block_id"].nunique()
    pct_anomalous_lines = 100 * df["anomaly"].mean() if n else 0.0
    pct_anomalous_blocks = 100 * n_anom_blocks / n_blocks if n_blocks else 0.0
    node_coverage = df["node"].notna().mean()
    gaps_within_block = df.sort_values(["block_id", "timestamp"]).groupby("block_id")["timestamp"].diff().dt.total_seconds().dropna()
    print("\n=== Summary ===")
    print(f"Total lines read:            {lines_seen:,}")
    print(f"Rows parsed:                 {n:,}")
    print(f"Skipped (malformed/no blk):  {skipped:,}")
    print(f"Rows with unmatched block_id (no label found): {unmatched:,}")
    print(f"Block sessions in label file: {n_blocks_in_label_file:,}")
    print(f"Distinct block sessions seen in log: {n_blocks:,}")
    print(f"% anomalous LINES:           {pct_anomalous_lines:.2f}%")
    print(f"% anomalous BLOCK SESSIONS:  {pct_anomalous_blocks:.2f}%")
    print(f"Unique event templates:      {df['event_template'].nunique():,}")
    print(f"Node (dest IP) coverage:     {100*node_coverage:.2f}% of rows have an extractable node")
    print(f"Unique node (dest IP) values: {df['node'].nunique():,}")
    print(f"Timestamp range:             {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Timestamp dtype:             {df['timestamp'].dtype}")
    print(f"Fraction of WITHIN-BLOCK inter-arrival gaps exactly 0s: {(gaps_within_block == 0).mean():.4f}")


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
    df, unmatched, n_blocks_in_label_file = join_labels(df)
    df = df[["timestamp", "node", "block_id", "label", "anomaly", "event_template", "raw_message"]]

    if args.test:
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 160)
        print(df.head(20).to_string())
        print("\ndtypes:")
        print(df.dtypes)
        print_summary(df, skipped, lines_seen, unmatched, n_blocks_in_label_file)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print_summary(df, skipped, lines_seen, unmatched, n_blocks_in_label_file)


if __name__ == "__main__":
    main()
