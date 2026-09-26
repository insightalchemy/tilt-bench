"""
Parse the Loghub OpenStack log (Zhu et al., ISSRE 2023) into the common per-line schema via
src/parser_common.py.

Raw line format:
    <logrecord> <date:YYYY-MM-DD> <time:HH:mm:ss.SSS> <pid> <level> <component> <content...>
e.g.
    nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:00:00.008 25746 INFO nova.osapi_compute.wsgi.server [req-38101a0b-2096-447d-96ea-a692162415ae] <client-ip> "GET /v2/... HTTP/1.1" status: 200 len: 1893 time: 0.2477829

The release concatenates several services into one file; `logrecord` (field 1) names the
originating per-service log file and is used as the stream key ("node"). The component field is
left inside the content for Drain3 to template on.

No anomaly labels are used: label is always "-", anomaly always False.

Usage:
    python src/parser_openstack.py [--sample-raw N] [--test] [--limit N] [--out PATH] [--sim-th X]
"""

import pandas as pd

from src import parser_common
from src.multi_dataset_registry import parsed_path, raw_root

NAME = "openstack"
RAW_ROOT = raw_root(NAME)
OUT_PATH = parsed_path(NAME)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def parse_line(line: str, relpath: str):
    parts = line.rstrip("\n").split(None, 5)
    if len(parts) < 6:
        return None
    logrecord, date, time_, pid, level, rest = parts
    if not (len(date) == 10 and date[4] == "-" and date[7] == "-"):
        return None
    return logrecord, f"{date} {time_}", f"{pid} {level} {rest}"


def build_timestamps(tokens):
    return pd.to_datetime(tokens, format=TIME_FORMAT, errors="coerce")


if __name__ == "__main__":
    parser_common.run_parser_main(NAME, RAW_ROOT, OUT_PATH, parse_line, build_timestamps, __doc__)
