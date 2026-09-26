"""
Parse the Loghub Android log (Zhu et al., ISSRE 2023) into the common per-line schema via
src/parser_common.py.

Raw line format (logcat):
    <date:MM-DD> <time:HH:mm:ss.SSS> <pid> <tid> <level> <component>: <content...>
e.g.
    03-17 16:13:38.811  1702  2395 D PowerManagerService: acquireWakeLock: lock=..., flags=0x1, tag="..."

The date has no year. YEAR_ASSUMPTION fills in a fixed year so timestamps are orderable; this
does not change ordering or gap sizes unless a capture crosses a year boundary.

This is a single-device log, so the logcat tag (`component`, e.g. "PowerManagerService") is used
as the stream key ("node"): each subsystem's event cadence is one timing stream.

No anomaly labels are used: label is always "-", anomaly always False.

Usage:
    python src/parser_android.py [--sample-raw N] [--test] [--limit N] [--out PATH] [--sim-th X]
"""

import pandas as pd

from src import parser_common
from src.multi_dataset_registry import parsed_path, raw_root

NAME = "android"
RAW_ROOT = raw_root(NAME)
OUT_PATH = parsed_path(NAME)
YEAR_ASSUMPTION = 2016  # logcat dates carry no year
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def parse_line(line: str, relpath: str):
    parts = line.rstrip("\n").split(None, 5)
    if len(parts) < 6:
        return None
    date, time_, pid, tid, level, rest = parts
    if not (len(date) == 5 and date[2] == "-"):
        return None
    rest_parts = rest.split(None, 1)
    if not rest_parts:
        return None
    component = rest_parts[0].rstrip(":")
    content = rest_parts[1] if len(rest_parts) > 1 else ""
    ts_token = f"{YEAR_ASSUMPTION}-{date} {time_}"
    return component, ts_token, f"{level} {pid} {tid} {content}"


def build_timestamps(tokens):
    return pd.to_datetime(tokens, format=TIME_FORMAT, errors="coerce")


if __name__ == "__main__":
    parser_common.run_parser_main(NAME, RAW_ROOT, OUT_PATH, parse_line, build_timestamps, __doc__)
