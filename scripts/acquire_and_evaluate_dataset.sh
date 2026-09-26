#!/usr/bin/env bash
# Full pipeline for one additional Loghub dataset (see src/multi_dataset_registry.py):
# download -> parse -> viability gate -> stall/burst injection -> fixed-count invariance check
# (LogAnomaly-style detector, N = 20/50/100) -> placebo sweep (fixed-time 60 s, fixed-count 50).
#
# Usage (from the repository root):
#   bash scripts/acquire_and_evaluate_dataset.sh <openstack|hadoop|zookeeper|android|spark|windows> [--gate-only]
#
# --gate-only stops after the viability gate and treats a NOT VIABLE verdict as a result rather than
# an error (used for Spark and Windows, which do not support per-stream injection).
#
# Outputs go to results/multi_dataset/<name>/ and data/processed/. Exits nonzero on any failure,
# including a parser skip rate above PARSE_SKIP_RATE_MAX_PCT.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"

NAME="${1:?Usage: $0 <openstack|hadoop|zookeeper|android|spark|windows> [--gate-only]}"
GATE_ONLY="${2:-}"
PARSE_SKIP_RATE_MAX_PCT=5
OUT_DIR="results/multi_dataset/${NAME}"
mkdir -p "$OUT_DIR"

"$PYTHON" src/multi_dataset_acquire.py verify-url --dataset "$NAME" 2>&1 | tee "$OUT_DIR/verify_url.log"
"$PYTHON" src/multi_dataset_acquire.py download --dataset "$NAME" 2>&1 | tee "$OUT_DIR/download.log"
"$PYTHON" src/multi_dataset_acquire.py sample-raw --dataset "$NAME" --lines 20 2>&1 | tee "$OUT_DIR/sample_raw.txt"
"$PYTHON" "src/parser_${NAME}.py" 2>&1 | tee "$OUT_DIR/parse.log"

SKIP_PCT=$(grep "Skipped (malformed):" "$OUT_DIR/parse.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+%' | tr -d '%' || true)
if [ -z "$SKIP_PCT" ]; then
  echo "${NAME}: no skip-rate line found in $OUT_DIR/parse.log" >&2
  exit 1
fi
if awk -v p="$SKIP_PCT" -v max="$PARSE_SKIP_RATE_MAX_PCT" 'BEGIN { exit !(p > max) }'; then
  echo "${NAME}: parser skip rate ${SKIP_PCT}% exceeds ${PARSE_SKIP_RATE_MAX_PCT}%; the raw format does not match the parser." >&2
  exit 1
fi

set +e
"$PYTHON" src/multi_dataset_gate.py --dataset "$NAME" 2>&1 | tee "$OUT_DIR/viability_gate.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e
if [ "$GATE_ONLY" = "--gate-only" ]; then
  if [ "$GATE_STATUS" -ne 0 ] && [ "$GATE_STATUS" -ne 2 ]; then
    echo "${NAME}: viability gate crashed (exit ${GATE_STATUS})." >&2
    exit 1
  fi
  echo "${NAME}: gate-only run complete (see $OUT_DIR/viability_gate.md)."
  exit 0
fi
if [ "$GATE_STATUS" -ne 0 ]; then
  echo "${NAME}: viability gate did not pass (exit ${GATE_STATUS}); see $OUT_DIR/viability_gate.md." >&2
  exit 1
fi

"$PYTHON" src/injector_multi.py --dataset "$NAME" --type stall 2>&1 | tee "$OUT_DIR/injector_stall.log"
"$PYTHON" src/injector_multi.py --dataset "$NAME" --type burst 2>&1 | tee "$OUT_DIR/injector_burst.log"

"$PYTHON" src/loganomaly_invariance.py --dataset "$NAME" --window-sizes 20 50 100 \
  --out-csv "$OUT_DIR/fixed_count_invariance.csv" --out-md "$OUT_DIR/fixed_count_invariance.md" \
  2>&1 | tee "$OUT_DIR/fixed_count_invariance.log"

"$PYTHON" src/placebo_sweep.py --dataset "$NAME" --fixed-time-sizes 60 --fixed-count-sizes 50 \
  --out-csv "$OUT_DIR/placebo_sweep.csv" --out-md "$OUT_DIR/placebo_sweep.md" \
  2>&1 | tee "$OUT_DIR/placebo_sweep.log"

echo "${NAME}: complete."
