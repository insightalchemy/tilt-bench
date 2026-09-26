"""
Parse the Loghub Windows log (Zhu et al., ISSRE 2023) into the common per-line schema via
src/parser_common.py.

Raw line format:
    <date:YYYY-MM-DD> <time:HH:mm:ss>, <level>  <component>  <content...>
e.g.
    2016-09-28 04:30:31, Info                  CBS    Loaded Servicing Stack v6.1.7601.23505 with Core: ...

This is a single-machine event log, so the subsystem field (`component`, e.g. "CBS", "DISM") is
used as the stream key ("node"): each subsystem's event cadence is one timing stream.

No anomaly labels are used: label is always "-", anomaly always False.

The full release exceeds 100M lines; --max-lines (default 10,000,000) caps how much is parsed.
--limit overrides it, and --test parses 5000 lines.

Usage:
    python src/parser_windows.py [--max-lines N] [--sample-raw N] [--test] [--limit N] [--out PATH] [--sim-th X]
"""

import pandas as pd

from src import parser_common
from src.multi_dataset_registry import parsed_path, raw_root

NAME = "windows"
RAW_ROOT = raw_root(NAME)
OUT_PATH = parsed_path(NAME)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_LINES = 10_000_000


def parse_line(line: str, relpath: str):
    parts = line.rstrip("\n").split(None, 3)
    if len(parts) < 4:
        return None
    date, time_comma, level, rest = parts
    if not (len(date) == 10 and date[4] == "-" and date[7] == "-"):
        return None
    time_ = time_comma.rstrip(",")
    rest_parts = rest.split(None, 1)
    if not rest_parts:
        return None
    component = rest_parts[0]
    content = rest_parts[1] if len(rest_parts) > 1 else ""
    return component, f"{date} {time_}", f"{level} {content}"


def build_timestamps(tokens):
    return pd.to_datetime(tokens, format=TIME_FORMAT, errors="coerce")


def main():
    ap = parser_common.build_arg_parser(__doc__)
    ap.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    args = ap.parse_args()

    if args.sample_raw is not None:
        parser_common.sample_raw_lines(RAW_ROOT, args.sample_raw)
        return

    out_path = args.out or OUT_PATH
    limit = args.limit if args.limit is not None else args.max_lines
    if args.test:
        limit = 5000

    df, skipped, lines_seen, first_sample_skipped = parser_common.parse_file(
        RAW_ROOT, parse_line, build_timestamps, sim_th=args.sim_th, limit=limit
    )

    if args.test:
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 160)
        print(df.head(20).to_string())
        print("\ndtypes:")
        print(df.dtypes)
        parser_common.print_summary(df, skipped, lines_seen, first_sample_skipped, NAME)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    parser_common.print_summary(df, skipped, lines_seen, first_sample_skipped, NAME)


if __name__ == "__main__":
    main()
