"""
Parse the Loghub Hadoop log (Zhu et al., ISSRE 2023) into the common per-line schema via
src/parser_common.py.

Raw line format:
    <date:YYYY-MM-DD> <time:HH:mm:ss,SSS> <level> [<thread>] <component>: <content...>
e.g.
    2015-10-18 18:01:47,978 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Created MRAppMaster for application appattempt_1445062781478_0011_000002

Only date, time, and level are split out; the rest is left to Drain3 to template.

The release has one directory per application with one file per container, so the stream key
("node") is the parent directory of the source file (from `relpath`). If the archive were a single
flat file, every row would share one stream, which the viability gate would report.

No anomaly labels are used: label is always "-", anomaly always False.

Usage:
    python src/parser_hadoop.py [--sample-raw N] [--test] [--limit N] [--out PATH] [--sim-th X]
"""

from pathlib import Path, PurePosixPath

import pandas as pd

from src import parser_common
from src.multi_dataset_registry import parsed_path, raw_root

NAME = "hadoop"
RAW_ROOT = raw_root(NAME)
OUT_PATH = parsed_path(NAME)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def _stream_key(relpath: str) -> str:
    parent = str(PurePosixPath(relpath.replace("\\", "/")).parent)
    return parent if parent not in ("", ".") else relpath


def parse_line(line: str, relpath: str):
    parts = line.rstrip("\n").split(None, 3)
    if len(parts) < 4:
        return None
    date, time_, level, rest = parts
    if not (len(date) == 10 and date[4] == "-" and date[7] == "-"):
        return None
    time_normalized = time_.replace(",", ".")
    return _stream_key(relpath), f"{date} {time_normalized}", f"{level} {rest}"


def build_timestamps(tokens):
    return pd.to_datetime(tokens, format=TIME_FORMAT, errors="coerce")


if __name__ == "__main__":
    parser_common.run_parser_main(NAME, RAW_ROOT, OUT_PATH, parse_line, build_timestamps, __doc__)
