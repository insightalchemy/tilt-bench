#!/usr/bin/env bash
# Spirit pipeline: download -> diversity report -> parse -> premise audit -> viability gate ->
# stall/burst injection -> injector validation -> placebo sweep, plus the masked re-parse used for
# the parser-robustness invariance check.
#
# Usage (from the repository root):
#   bash scripts/run_spirit.sh
#
# Outputs go to results/spirit/, results/placebo/, results/parser_robustness/, figures/, and
# data/processed/. Exits nonzero on any failure, including a NOT VIABLE gate verdict.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
OUT_DIR="results/spirit"
mkdir -p "$OUT_DIR"

"$PYTHON" src/spirit_setup.py download 2>&1 | tee "$OUT_DIR/download.log"
"$PYTHON" src/spirit_setup.py sample-raw --lines 20 2>&1 | tee "$OUT_DIR/sample_raw.txt"
"$PYTHON" src/spirit_setup.py diversity --bucket-size 1000000 2>&1 | tee "$OUT_DIR/diversity.txt"
"$PYTHON" src/parser_spirit.py 2>&1 | tee "$OUT_DIR/parse.log"
"$PYTHON" src/premise_audit_spirit.py 2>&1 | tee "$OUT_DIR/premise_audit.log"
"$PYTHON" src/spirit_setup.py viability-gate 2>&1 | tee "$OUT_DIR/viability_gate.log"

"$PYTHON" src/injector_spirit.py --type stall 2>&1 | tee "$OUT_DIR/injector_stall.log"
"$PYTHON" src/injector_spirit.py --type burst 2>&1 | tee "$OUT_DIR/injector_burst.log"
"$PYTHON" src/validate_injection_spirit.py --type stall 2>&1 | tee "$OUT_DIR/validate_stall.log"
"$PYTHON" src/validate_injection_spirit.py --type burst 2>&1 | tee "$OUT_DIR/validate_burst.log"
"$PYTHON" src/injector_slowdown.py --dataset spirit 2>&1 | tee "$OUT_DIR/injector_slowdown.log"

"$PYTHON" src/placebo_sweep.py --dataset spirit 2>&1 | tee "$OUT_DIR/placebo_sweep.log"

"$PYTHON" src/parser_spirit_masked.py 2>&1 | tee "$OUT_DIR/parse_masked.log"
"$PYTHON" src/parser_invariance.py --dataset spirit_masked 2>&1 | tee "$OUT_DIR/parser_invariance_masked.log"

echo "Spirit pipeline complete."
