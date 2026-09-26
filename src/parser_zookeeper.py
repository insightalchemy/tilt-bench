"""
Parse the Loghub Zookeeper log (Zhu et al., ISSRE 2023) into the common per-line schema via
src/parser_common.py.

Raw line format:
    <date:YYYY-MM-DD> <time:HH:mm:ss,SSS> - <level>  [<thread>:<component>@<line>] - <content...>
e.g.
    2015-07-29 17:41:41,536 - INFO  [main:QuorumPeerConfig@103] - Reading configuration from: /...

The log has no host field, so the thread name (the token before ':' inside the brackets, e.g.
"main", "NIOServerCxn.Factory") is used as the stream key ("node"): one timing stream per
execution context rather than per server.

No anomaly labels are used: label is always "-", anomaly always False.

Usage:
    python src/parser_zookeeper.py [--sample-raw N] [--test] [--limit N] [--out PATH] [--sim-th X]
"""

import re

import pandas as pd

from src import parser_common
from src.multi_dataset_registry import parsed_path, raw_root

NAME = "zookeeper"
RAW_ROOT = raw_root(NAME)
OUT_PATH = parsed_path(NAME)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

_BRACKET_RE = re.compile(r"\[([^\]:]+):")


def parse_line(line: str, relpath: str):
    parts = line.rstrip("\n").split(None, 5)
    if len(parts) < 6:
        return None
    date, time_comma, dash, level, bracket, rest = parts
    if not (len(date) == 10 and date[4] == "-" and date[7] == "-"):
        return None
    if dash != "-":
        return None
    m = _BRACKET_RE.match(bracket)
    thread = m.group(1) if m else bracket.strip("[]")
    time_normalized = time_comma.replace(",", ".")
    content = rest.lstrip("- ").strip()
    return thread, f"{date} {time_normalized}", f"{level} {content}"


def build_timestamps(tokens):
    return pd.to_datetime(tokens, format=TIME_FORMAT, errors="coerce")


if __name__ == "__main__":
    parser_common.run_parser_main(NAME, RAW_ROOT, OUT_PATH, parse_line, build_timestamps, __doc__)
