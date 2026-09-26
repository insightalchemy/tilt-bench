"""
Re-parse data/raw/Spirit.log with Drain3 masking of variable fields (numbers, hex values, IP and
MAC addresses, filesystem paths), for the parser-robustness variant of the invariance check.
The line parsing is src.parser_spirit.parse_chunk; only the TemplateMinerConfig differs.

Masking rules (drain3.masking.MaskingInstruction, applied in this order so specific patterns are
not pre-empted by general ones):
  IP address        \\b\\d{1,3}(?:\\.\\d{1,3}){3}\\b            -> <IP>
  MAC / hex-colon    \\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\\b  -> <MAC>
  hex literal        \\b0x[0-9a-fA-F]+\\b                        -> <HEX>
  Unix path          (?:/[\\w.#-]+){2,}                          -> <PATH>
  bracketed number    \\[\\d+\\]                                  -> [<NUM>]
  bare number         \\d+                                        -> <NUM>

Writes:
  data/processed/spirit_parsed_masked.parquet
  results/spirit/parser_masked_spirit.md -- unmasked vs masked template counts, skip rate

Usage:
    python src/parser_spirit_masked.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

from src.parser_spirit import CHUNK_SIZE, RAW_LOG_PATH, compute_zero_gap_stats, parse_chunk

OUT_PATH = Path("data/processed/spirit_parsed_masked.parquet")
OUT_MD = Path("results/spirit/parser_masked_spirit.md")

MASKING_RULES = [
    (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "IP"),
    (r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b", "MAC"),
    (r"\b0x[0-9a-fA-F]+\b", "HEX"),
    (r"(?:/[\w.#-]+){2,}", "PATH"),
    (r"\[\d+\]", "[<NUM>]"),
    (r"\d+", "NUM"),
]


def build_masked_template_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.masking_instructions = [MaskingInstruction(pattern, mask_with) for pattern, mask_with in MASKING_RULES]
    return TemplateMiner(config=config)


def parse_file_masked():
    template_miner = build_masked_template_miner()
    chunk_dfs = []
    lines_seen = 0
    skipped_total = 0
    start = time.time()

    with open(RAW_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        buffer = []
        for line in f:
            buffer.append(line)
            lines_seen += 1
            if len(buffer) >= CHUNK_SIZE:
                df, skipped, _ = parse_chunk(buffer, template_miner)
                skipped_total += skipped
                chunk_dfs.append(df)
                buffer = []
                print(f"  ...{lines_seen:,} lines processed ({time.time() - start:.0f}s elapsed)", file=sys.stderr, flush=True)
        if buffer:
            df, skipped, _ = parse_chunk(buffer, template_miner)
            skipped_total += skipped
            chunk_dfs.append(df)

    full_df = pd.concat(chunk_dfs, ignore_index=True)
    return full_df, skipped_total, lines_seen


def main():
    if not RAW_LOG_PATH.exists():
        print(f"ERROR: {RAW_LOG_PATH} does not exist.", file=sys.stderr)
        sys.exit(1)

    df, skipped, lines_seen = parse_file_masked()
    n_templates_masked = df["event_template"].nunique()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    overall_zero_frac, majority_zero_node_frac = compute_zero_gap_stats(df)

    old_path = Path("data/processed/spirit_parsed.parquet")
    n_templates_old = None
    if old_path.exists():
        old_df = pd.read_parquet(old_path, columns=["event_template"])
        n_templates_old = old_df["event_template"].nunique()
        del old_df

    lines = [
        "# Spirit parser robustness: Drain masking for numbers/hex/IP/paths",
        "",
        f"Masking rules applied: {[m for _, m in MASKING_RULES]}",
        "",
        f"Lines parsed: {lines_seen:,}, skipped: {skipped:,}",
        f"Unmasked template count (src/parser_spirit.py): {n_templates_old if n_templates_old is not None else 'N/A'}",
        f"Masked template count (this run): {n_templates_masked:,}",
    ]
    if n_templates_old:
        reduction = 100 * (1 - n_templates_masked / n_templates_old)
        lines.append(f"Reduction: {reduction:.1f}%")
    lines.append(f"Fraction of per-node inter-arrival gaps exactly 0s (overall): {overall_zero_frac:.4f}")
    lines.append(f"Fraction of nodes majority-zero-gap: {majority_zero_node_frac:.4f}")
    lines.append(f"Wrote {OUT_PATH}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
