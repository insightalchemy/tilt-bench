"""
Acquisition, diversity report, and per-node timing-baseline viability gate for Spirit
(Oliner & Stearley, DSN 2007).

Uses the published 5,000,000-line Spirit subset from the replication package of Wu, Li, and
Khomh, "On the effectiveness of log representation for log-based anomaly detection" (EMSE 2023):
    https://zenodo.org/records/7851024/files/spirit2_5m.tar.gz?download=1
The archive holds one file, spirit2_5m.log, whose line format is the same as Thunderbird's:
    <label> <unix_ts> <date> <node> <month> <day> <time> <user@host> <content...>

Subcommands:
  download        Download, md5-verify, and extract to data/raw/Spirit.log.
  sample-raw      Print the first N raw lines.
  diversity       Bucket anomalies by line window and report label/node diversity and top-node
                  share, to detect a slice dominated by a single incident.
  viability-gate  Report the fraction of nodes with a valid per-node timing baseline and a
                  VIABLE / CONDITIONALLY VIABLE / NOT VIABLE verdict (exit code 2 if not viable).

Usage:
    python src/spirit_setup.py download
    python src/spirit_setup.py sample-raw --lines 20
    python src/spirit_setup.py diversity --bucket-size 1000000
    python src/parser_spirit.py
    python src/spirit_setup.py viability-gate
"""

import argparse
import hashlib
import shutil
import sys
import tarfile
import time
import traceback
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.timing_baseline import add_sequence_context, compute_node_baselines

SOURCE_URL = "https://zenodo.org/records/7851024/files/spirit2_5m.tar.gz?download=1"
SOURCE_MD5 = "e51053a7f78971e91b083d3caa4062aa"
RAW_LOG_PATH = Path("data/raw/Spirit.log")
PARSED_PATH = Path("data/processed/spirit_parsed.parquet")
DEFAULT_SAMPLE_LINES = 20
DEFAULT_BUCKET_SIZE = 1_000_000
VIABLE_THRESHOLD = 0.5
CONDITIONAL_THRESHOLD = 0.2
DOWNLOAD_CHUNK = 1 << 20


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(DOWNLOAD_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_download(args):
    tmp_dir = Path("data/raw/.spirit_download_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_dir / "spirit2_5m.tar.gz"

    if RAW_LOG_PATH.exists() and not args.force:
        print(f"{RAW_LOG_PATH} already exists -- skipping download (pass --force to redo).")
        return

    print(f"[download] fetching {args.source_url}", flush=True)
    t0 = time.time()
    try:
        with urllib.request.urlopen(args.source_url, timeout=120) as response, open(archive_path, "wb") as out:
            shutil.copyfileobj(response, out, length=DOWNLOAD_CHUNK)
    except Exception:
        print("FATAL: download failed.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    print(f"[download] wrote {archive_path} ({archive_path.stat().st_size:,} bytes) in {time.time() - t0:.0f}s", flush=True)

    print("[download] verifying md5...", flush=True)
    actual_md5 = md5_of(archive_path)
    if actual_md5 != args.expected_md5:
        print(f"FATAL: md5 mismatch -- expected {args.expected_md5}, got {actual_md5}. Not extracting.", file=sys.stderr)
        sys.exit(1)
    print(f"[download] md5 OK ({actual_md5})", flush=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        if len(members) != 1:
            print(f"FATAL: expected exactly one file in the archive, found {len(members)}: {[m.name for m in members]}", file=sys.stderr)
            sys.exit(1)
        member = members[0]
        print(f"[download] extracting {member.name} ({member.size:,} bytes)...", flush=True)
        tar.extract(member, path=tmp_dir)
        extracted_path = tmp_dir / member.name

    RAW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted_path), str(RAW_LOG_PATH))
    archive_path.unlink()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    with open(RAW_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        n_lines = sum(1 for _ in f)
    print(f"[download] wrote {RAW_LOG_PATH} ({RAW_LOG_PATH.stat().st_size:,} bytes, {n_lines:,} lines)")


def cmd_sample_raw(args):
    if not RAW_LOG_PATH.exists():
        print(f"ERROR: {RAW_LOG_PATH} does not exist. Run 'python src/spirit_setup.py download' first.", file=sys.stderr)
        sys.exit(1)
    print(f"=== First {args.lines} raw lines of {RAW_LOG_PATH} ===")
    with open(RAW_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= args.lines:
                break
            print(f"{i:>4}: {line.rstrip(chr(10))}")


def parse_diversity_line(line):
    parts = line.split(None, 8)
    if len(parts) < 8:
        return None
    return parts[0], parts[3]


def cmd_diversity(args):
    if not RAW_LOG_PATH.exists():
        print(f"ERROR: {RAW_LOG_PATH} does not exist. Run 'python src/spirit_setup.py download' first.", file=sys.stderr)
        sys.exit(1)

    buckets = defaultdict(lambda: {"n_anomalies": 0, "labels": Counter(), "nodes": Counter()})
    n_lines = 0
    n_malformed = 0
    overall_labels = Counter()
    overall_nodes = Counter()
    t0 = time.time()

    with open(RAW_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            parsed = parse_diversity_line(line)
            if parsed is None:
                n_malformed += 1
                continue
            label, node = parsed
            if label != "-":
                bucket = buckets[(n_lines - 1) // args.bucket_size]
                bucket["n_anomalies"] += 1
                bucket["labels"][label] += 1
                bucket["nodes"][node] += 1
                overall_labels[label] += 1
                overall_nodes[node] += 1

    print(f"[diversity] {n_lines:,} lines, {n_malformed:,} malformed, elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"\n=== Diversity report (bucket_size={args.bucket_size:,}) ===")
    print(f"{'window':>23} {'anomalies':>10} {'labels':>7} {'nodes':>7} {'top_node_share':>15}")
    for idx in sorted(buckets):
        bucket = buckets[idx]
        start_line = idx * args.bucket_size + 1
        end_line = start_line + args.bucket_size - 1
        n_anom = bucket["n_anomalies"]
        n_labels = len(bucket["labels"])
        n_nodes = len(bucket["nodes"])
        top_share = bucket["nodes"].most_common(1)[0][1] / n_anom if n_anom else float("nan")
        print(f"{start_line:>11,}-{end_line:>10,} {n_anom:>10,} {n_labels:>7} {n_nodes:>7} {top_share:>14.1%}")

    total_anomalies = sum(overall_labels.values())
    overall_top_share = overall_nodes.most_common(1)[0][1] / total_anomalies if total_anomalies else float("nan")
    print(f"\nOverall: {total_anomalies:,} anomalies, {len(overall_labels)} distinct labels, {len(overall_nodes)} distinct anomalous nodes, top-node share {overall_top_share:.1%}")
    if overall_top_share > 0.5:
        print(
            f"NOTE: a single node accounts for {overall_top_share:.1%} of all anomalies in this slice -- "
            "the premise audit on this slice largely characterizes one incident. Injection experiments "
            "use normal traffic only and are unaffected."
        )
    else:
        print(f"No single node dominates (top-node share {overall_top_share:.1%} <= 50%) -- no obvious single-incident artifact in this slice.")


def run_viability_gate(parsed_path: Path):
    if not parsed_path.exists():
        print(f"ERROR: {parsed_path} does not exist. Run src/parser_spirit.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[viability-gate] loading {parsed_path}", flush=True)
    df = pd.read_parquet(parsed_path)
    df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
    df = add_sequence_context(df)
    normal_seq = (~df["anomaly"]) & (~df["prev_anomaly"].fillna(True)) & df["gap_prev_s"].notna()

    baselines_raw = compute_node_baselines(df, normal_seq, exclude_zero_from_pooled=False)
    baselines_fixed = compute_node_baselines(df, normal_seq, exclude_zero_from_pooled=True)

    total_nodes = df["node"].nunique()
    n_valid = len(baselines_raw["valid_nodes"])
    n_low_count = len(baselines_raw["fallback_low_count_nodes"])
    n_zero_mad = len(baselines_raw["fallback_zero_mad_nodes"])
    valid_frac = n_valid / total_nodes if total_nodes else float("nan")

    lines = ["# Spirit viability gate: per-node timing baseline", ""]
    lines.append(f"Total nodes: {total_nodes:,}")
    lines.append(f"Valid (non-degenerate) baseline nodes: {n_valid:,} ({valid_frac:.2%})")
    lines.append(f"  -- too few normal-to-normal gaps: {n_low_count:,} ({n_low_count / total_nodes:.2%})")
    lines.append(f"  -- zero per-node MAD: {n_zero_mad:,} ({n_zero_mad / total_nodes:.2%})")
    lines.append(f"Pooled fallback (all gaps): median={baselines_raw['global_median']:.6f}s mad={baselines_raw['global_mad']:.6f}s")
    lines.append(f"Pooled fallback (nonzero gaps only): median={baselines_fixed['global_median']:.6f}s mad={baselines_fixed['global_mad']:.6f}s")
    pooled_raw_degenerate = baselines_raw["global_mad"] == 0
    if pooled_raw_degenerate:
        lines.append(
            "Pooled (all-gaps) fallback IS degenerate (MAD=0) -- "
            "use exclude_zero_from_pooled=True and require_valid_baseline=True if injecting."
        )

    if valid_frac >= VIABLE_THRESHOLD:
        verdict = "VIABLE -- per-node inter-arrival injection should work directly, as on BGL."
        supports = True
    elif valid_frac >= CONDITIONAL_THRESHOLD:
        verdict = "CONDITIONALLY VIABLE -- restrict injector node eligibility to the valid-baseline subset (require_valid_baseline=True)."
        supports = True
    else:
        verdict = "NOT VIABLE -- too few nodes have a usable timing baseline for per-node inter-arrival injection."
        supports = False

    lines.append("")
    lines.append(f"VERDICT: Spirit {'DOES' if supports else 'DOES NOT'} support the per-node inter-arrival injection model as currently designed.")
    lines.append(f"  {verdict}")

    for line in lines:
        print(line)

    return supports, "\n".join(lines) + "\n", valid_frac


def cmd_viability_gate(args):
    supports, report, valid_frac = run_viability_gate(args.parsed_path)
    out_dir = Path("results/spirit")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viability_gate.md").write_text(report)
    print(f"\nWrote {out_dir / 'viability_gate.md'}")
    if not supports:
        sys.exit(2)


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p_download = sub.add_parser("download", help="Download, md5-verify, and extract the Spirit subset to data/raw/Spirit.log.")
    p_download.add_argument("--source-url", default=SOURCE_URL)
    p_download.add_argument("--expected-md5", default=SOURCE_MD5)
    p_download.add_argument("--force", action="store_true")
    p_download.set_defaults(func=cmd_download)

    p_sample = sub.add_parser("sample-raw", help="Print the first N raw lines to confirm field layout.")
    p_sample.add_argument("--lines", type=int, default=DEFAULT_SAMPLE_LINES)
    p_sample.set_defaults(func=cmd_sample_raw)

    p_diversity = sub.add_parser("diversity", help="Bucket anomalies by line-window; report label/node diversity and top-node share.")
    p_diversity.add_argument("--bucket-size", type=int, default=DEFAULT_BUCKET_SIZE)
    p_diversity.set_defaults(func=cmd_diversity)

    p_gate = sub.add_parser("viability-gate", help="Report per-node timing-baseline validity fraction and an explicit injection-viability verdict.")
    p_gate.add_argument("--parsed-path", type=Path, default=PARSED_PATH)
    p_gate.set_defaults(func=cmd_viability_gate)

    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
