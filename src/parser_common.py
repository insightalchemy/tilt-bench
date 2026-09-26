"""
Chunked Drain3 parsing shared by the additional Loghub datasets (OpenStack, Hadoop, Zookeeper,
Android, Spark, Windows; see src/multi_dataset_registry.py). Each src/parser_<name>.py supplies
only a `parse_line(line, relpath)` field split and a bulk timestamp builder.

A Loghub archive may extract to a single file or to a directory tree of per-component files;
`iter_raw_lines` walks either in sorted-path order. Datasets without a stream-key field in the
line can use the relative file path (passed to every parse_line call) as the stream key.

These datasets have no per-line anomaly labels. The invariance experiments only need timestamps
and a stream key, so `label` is always "-" and `anomaly` always False, keeping the schema
identical to BGL/Thunderbird/Spirit.

Several of these archives are not time-ordered within a stream (their per-container or per-thread
files interleave). Downstream code assumes the parquet's row order is per-node chronological, so
`parse_file` stable-sorts by (node, timestamp, original file position) before returning.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

CHUNK_SIZE = 200_000
SKIP_WARN_FRACTION = 0.05  # warn if more than 5% of lines fail to parse
DEFAULT_SAMPLE_LINES = 20


def build_template_miner(sim_th: float | None = None) -> TemplateMiner:
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    if sim_th is not None:
        config.drain_sim_th = sim_th
    return TemplateMiner(config=config)


def iter_raw_lines(raw_root: Path):
    """Yields (relpath: str, line: str) for every line of every file under raw_root, in a
    deterministic (sorted-path) order. raw_root may be a single file or a directory tree."""
    if not raw_root.exists():
        raise FileNotFoundError(f"{raw_root} does not exist -- run the acquisition step first.")
    if raw_root.is_file():
        with open(raw_root, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield raw_root.name, line
        return
    paths = sorted(p for p in raw_root.rglob("*") if p.is_file())
    for path in paths:
        relpath = str(path.relative_to(raw_root))
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield relpath, line


def sample_raw_lines(raw_root: Path, n: int):
    if not raw_root.exists():
        print(f"ERROR: {raw_root} does not exist. Run the acquisition step first.", file=sys.stderr)
        sys.exit(1)
    print(f"=== First {n} raw lines under {raw_root} ===")
    for i, (relpath, line) in enumerate(iter_raw_lines(raw_root)):
        if i >= n:
            break
        print(f"{i:>4} [{relpath}]: {line.rstrip(chr(10))}")


def parse_chunk(lines_with_paths, parse_line_fn, build_timestamps_fn, template_miner):
    stream_keys, ts_tokens, anomalies, templates, raw_messages = [], [], [], [], []
    skipped = 0
    sample_skipped_line = None
    for relpath, line in lines_with_paths:
        parsed = parse_line_fn(line, relpath)
        if parsed is None:
            skipped += 1
            if sample_skipped_line is None:
                sample_skipped_line = line.rstrip("\n")
            continue
        stream_key, ts_token, content = parsed
        result = template_miner.add_log_message(content)
        stream_keys.append(stream_key)
        ts_tokens.append(ts_token)
        anomalies.append(False)
        templates.append(result["template_mined"])
        raw_messages.append(content)

    df = pd.DataFrame(
        {
            "timestamp": build_timestamps_fn(ts_tokens),
            "node": stream_keys,
            "label": "-",
            "anomaly": pd.array(anomalies, dtype=bool),
            "event_template": templates,
            "raw_message": raw_messages,
        }
    )
    return df, skipped, sample_skipped_line


def parse_file(raw_root: Path, parse_line_fn, build_timestamps_fn, sim_th=None, limit=None):
    template_miner = build_template_miner(sim_th=sim_th)
    chunk_dfs = []
    lines_seen = 0
    skipped_total = 0
    first_sample_skipped = None
    start = time.time()

    try:
        buffer = []
        for relpath, line in iter_raw_lines(raw_root):
            buffer.append((relpath, line))
            lines_seen += 1
            if limit is not None and lines_seen >= limit:
                break
            if len(buffer) >= CHUNK_SIZE:
                df, skipped, sample_skipped = parse_chunk(buffer, parse_line_fn, build_timestamps_fn, template_miner)
                skipped_total += skipped
                if first_sample_skipped is None:
                    first_sample_skipped = sample_skipped
                chunk_dfs.append(df)
                buffer = []
                elapsed = time.time() - start
                print(
                    f"  ...{lines_seen:,} lines processed ({elapsed:.0f}s elapsed, {skipped_total:,} skipped so far, "
                    f"this-chunk skip rate {skipped / CHUNK_SIZE:.2%})",
                    file=sys.stderr,
                    flush=True,
                )
                if lines_seen >= CHUNK_SIZE and (skipped_total / lines_seen) > SKIP_WARN_FRACTION:
                    print(
                        f"  WARNING: cumulative skip rate {skipped_total / lines_seen:.2%} exceeds "
                        f"{SKIP_WARN_FRACTION:.0%} -- this dataset's raw format may not match the assumed "
                        f"layout in this parser's module docstring. Sample skipped line: {first_sample_skipped!r}",
                        file=sys.stderr,
                        flush=True,
                    )
        if buffer:
            df, skipped, sample_skipped = parse_chunk(buffer, parse_line_fn, build_timestamps_fn, template_miner)
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

    full_df = (
        pd.concat(chunk_dfs, ignore_index=True)
        if chunk_dfs
        else pd.DataFrame(
            {
                "timestamp": pd.array([], dtype="datetime64[us]"),
                "node": pd.array([], dtype="object"),
                "label": pd.array([], dtype="object"),
                "anomaly": pd.array([], dtype=bool),
                "event_template": pd.array([], dtype="object"),
                "raw_message": pd.array([], dtype="object"),
            }
        )
    )
    full_df["anomaly"] = full_df["anomaly"].astype(bool)
    full_df = sort_chronologically(full_df)
    return full_df, skipped_total, lines_seen, first_sample_skipped


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Stable-sorts by (node, timestamp), tie-broken by original file position, and resets the
    index so file order equals per-node chronological order."""
    df = df.reset_index(drop=True)
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(["node", "timestamp", "_orig_pos"], kind="mergesort").reset_index(drop=True)
    return df.drop(columns=["_orig_pos"])


def print_summary(df: pd.DataFrame, skipped: int, lines_seen: int, first_sample_skipped, name: str):
    n = len(df)
    skip_frac = skipped / lines_seen if lines_seen else float("nan")
    print("\n=== Summary ===")
    print(f"Dataset:                 {name}")
    print(f"Total lines read:        {lines_seen:,}")
    print(f"Rows parsed:             {n:,}")
    print(f"Skipped (malformed):     {skipped:,}  ({skip_frac:.2%} of lines read)")
    if skip_frac > SKIP_WARN_FRACTION:
        print(
            f"WARNING: skip rate {skip_frac:.2%} exceeds {SKIP_WARN_FRACTION:.0%} -- do NOT trust this parse. "
            f"Sample skipped line: {first_sample_skipped!r}. Re-check the raw format assumption in this "
            "parser's module docstring against real lines (--sample-raw)."
        )
    print(f"Unique streams ('node'): {df['node'].nunique() if n else 0:,}")
    print(f"Unique event templates:  {df['event_template'].nunique() if n else 0:,}")
    if n:
        print(f"Timestamp range:         {df['timestamp'].min()} -> {df['timestamp'].max()}")
        print(f"Timestamp dtype:         {df['timestamp'].dtype}")
    else:
        print("Timestamp range:         N/A (no rows parsed)")


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="Preview first 5000 lines to stdout, don't write output.")
    ap.add_argument(
        "--sample-raw", nargs="?", type=int, const=DEFAULT_SAMPLE_LINES, default=None,
        help="Print N raw lines (default 20), prefixed by their source path, then exit. No parsing, no output.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N lines.")
    ap.add_argument("--out", type=Path, default=None, help="Output parquet path (defaults to this parser's registry path).")
    ap.add_argument("--sim-th", type=float, default=None, help="Override Drain3's similarity threshold (default 0.4).")
    return ap


def run_parser_main(name: str, raw_root: Path, out_path: Path, parse_line_fn, build_timestamps_fn, description: str):
    """Shared entry point every src/parser_<name>.py calls from `if __name__ == '__main__'`."""
    ap = build_arg_parser(description)
    args = ap.parse_args()

    if args.sample_raw is not None:
        sample_raw_lines(raw_root, args.sample_raw)
        return

    out_path = args.out or out_path
    limit = args.limit
    if args.test and limit is None:
        limit = 5000

    df, skipped, lines_seen, first_sample_skipped = parse_file(
        raw_root, parse_line_fn, build_timestamps_fn, sim_th=args.sim_th, limit=limit
    )

    if args.test:
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 160)
        print(df.head(20).to_string())
        print("\ndtypes:")
        print(df.dtypes)
        print_summary(df, skipped, lines_seen, first_sample_skipped, name)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print_summary(df, skipped, lines_seen, first_sample_skipped, name)
