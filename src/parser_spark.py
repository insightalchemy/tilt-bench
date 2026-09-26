"""
Parse the Loghub Spark log (Zhu et al., ISSRE 2023) into the common per-line schema via
src/parser_common.py.

Raw line format:
    <date:yy/MM/dd> <time:HH:mm:ss> <level> <component>: <content...>
e.g.
    17/06/09 20:10:40 INFO executor.CoarseGrainedExecutorBackend: Registered signal handlers for [TERM, HUP, INT]

Lines carry no executor or host id. The release has one file per application/executor, so the
stream key ("node") is the file's path relative to the archive root (`relpath`).

No anomaly labels are used: label is always "-", anomaly always False.

Usage:
    python src/parser_spark.py [--sample-raw N] [--test] [--limit N] [--out PATH] [--sim-th X]
"""

from pathlib import Path

import pandas as pd

from src import parser_common
from src.multi_dataset_registry import parsed_path, raw_root

NAME = "spark"
RAW_ROOT = raw_root(NAME)
OUT_PATH = parsed_path(NAME)
TIME_FORMAT = "%y/%m/%d %H:%M:%S"


def parse_line(line: str, relpath: str):
    parts = line.rstrip("\n").split(None, 3)
    if len(parts) < 4:
        return None
    date, time_, level, rest = parts
    if not (len(date) == 8 and date[2] == "/" and date[5] == "/"):
        return None
    return relpath, f"{date} {time_}", f"{level} {rest}"


def build_timestamps(tokens):
    return pd.to_datetime(tokens, format=TIME_FORMAT, errors="coerce")


if __name__ == "__main__":
    parser_common.run_parser_main(NAME, RAW_ROOT, OUT_PATH, parse_line, build_timestamps, __doc__)
