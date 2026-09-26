#!/usr/bin/env bash
# End-to-end reproduction driver for the TILT-Bench paper.
#
# Usage (from the repository root, with the raw datasets in data/raw/, see README.md):
#   bash scripts/reproduce_all.sh <stage> [<stage> ...]
#   bash scripts/reproduce_all.sh all
#
# Stages, in dependency order:
#   bgl          parse BGL, premise audit, stall/burst/slowdown injection, validation, plausibility
#   thunderbird  parse Thunderbird, premise audit, stall/burst injection, validation, plausibility
#   spirit       full Spirit pipeline (scripts/run_spirit.sh)
#   hdfs         parse HDFS and run its premise audit (scope boundary, no injection)
#   loghub       OpenStack, Hadoop, Zookeeper, Android (full pipeline); Spark, Windows (gate only)
#   invariance   windowing sweeps, LogAnomaly-style, sliding-window, slowdown, parser-robustness checks
#   placebo      placebo-controlled sweeps for BGL and Thunderbird
#   auc          fixed-time AUC with bootstrap CIs, instrument validation, five-seed replication
#   boundary     time-aware isolation forest, time-aware DeepLog, resolution ablation
#   deep         DeepLog (native and injected) and the small LogBERT-style transformer
#   figures      paper figures from the placebo sweep outputs
#   all          every stage above, in order
#
# Each command's stdout/stderr is written to logs/<step>.log. The script stops with a nonzero exit
# status at the first failing command. The deep and boundary stages train neural models and take
# many hours on CPU; the invariance, placebo, and auc stages need roughly 25-31 GB of RAM at full scale.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cpu}"
mkdir -p logs results

run() {
  local step="$1"
  shift
  echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${step}: $* ==="
  if ! "$@" > "logs/${step}.log" 2>&1; then
    echo "FAILED: ${step} (see logs/${step}.log)" >&2
    tail -n 20 "logs/${step}.log" >&2
    exit 1
  fi
}

stage_bgl() {
  run bgl_parse "$PYTHON" src/parser.py
  run bgl_premise_audit "$PYTHON" src/premise_audit.py
  run bgl_inject_stall "$PYTHON" src/injector.py --type stall
  run bgl_inject_burst "$PYTHON" src/injector.py --type burst
  run bgl_inject_slowdown "$PYTHON" src/injector_slowdown.py --dataset bgl
  run bgl_validate_stall "$PYTHON" src/validate_injection.py --type stall
  run bgl_validate_burst "$PYTHON" src/validate_injection.py --type burst
  run bgl_plausibility_stall "$PYTHON" src/plausibility.py --dataset bgl --fault stall
  run bgl_plausibility_burst "$PYTHON" src/plausibility.py --dataset bgl --fault burst
}

stage_thunderbird() {
  run tb_parse "$PYTHON" src/parser_thunderbird.py
  run tb_premise_audit "$PYTHON" src/premise_audit_thunderbird.py
  run tb_inject_stall "$PYTHON" src/injector_thunderbird.py --type stall
  run tb_inject_burst "$PYTHON" src/injector_thunderbird.py --type burst
  run tb_validate_stall "$PYTHON" src/validate_injection_thunderbird.py --type stall
  run tb_validate_burst "$PYTHON" src/validate_injection_thunderbird.py --type burst
  run tb_plausibility_stall "$PYTHON" src/plausibility.py --dataset thunderbird --fault stall
  run tb_plausibility_burst "$PYTHON" src/plausibility.py --dataset thunderbird --fault burst
}

stage_spirit() {
  run spirit bash scripts/run_spirit.sh
}

stage_hdfs() {
  run hdfs_parse "$PYTHON" src/parser_hdfs.py
  run hdfs_premise_audit "$PYTHON" src/premise_audit_hdfs.py
}

stage_loghub() {
  for name in openstack hadoop zookeeper android; do
    run "loghub_${name}" bash scripts/acquire_and_evaluate_dataset.sh "$name"
  done
  for name in spark windows; do
    run "loghub_${name}_gate" bash scripts/acquire_and_evaluate_dataset.sh "$name" --gate-only
  done
}

stage_invariance() {
  run windowing_sweep "$PYTHON" src/windowing_sweep.py bgl thunderbird
  run windowing_sweep_extended_bgl "$PYTHON" src/windowing_sweep_extended.py --dataset bgl
  run loganomaly_invariance_bgl "$PYTHON" src/loganomaly_invariance.py --dataset bgl \
    --out-csv results/loganomaly_invariance_bgl.csv --out-md results/loganomaly_invariance_bgl.md
  run loganomaly_invariance_thunderbird "$PYTHON" src/loganomaly_invariance.py --dataset thunderbird \
    --out-csv results/loganomaly_invariance_thunderbird.csv --out-md results/loganomaly_invariance_thunderbird.md
  for dataset in bgl spirit; do
    for fault in stall burst; do
      run "sliding_window_${dataset}_${fault}" "$PYTHON" src/sliding_window_invariance.py --dataset "$dataset" --fault "$fault"
    done
    run "slowdown_${dataset}" "$PYTHON" src/run_slowdown_evaluation.py --dataset "$dataset"
  done
  run bgl_parse_simth0.5 "$PYTHON" src/parser.py --sim-th 0.5 --out data/processed/bgl_parsed_simth0.5.parquet
  run bgl_parse_simth0.7 "$PYTHON" src/parser.py --sim-th 0.7 --out data/processed/bgl_parsed_simth0.7.parquet
  run parser_invariance_bgl_simth0.5 "$PYTHON" src/parser_invariance.py --dataset bgl --clean-parsed data/processed/bgl_parsed_simth0.5.parquet
  run parser_invariance_bgl_simth0.7 "$PYTHON" src/parser_invariance.py --dataset bgl --clean-parsed data/processed/bgl_parsed_simth0.7.parquet
}

stage_placebo() {
  run placebo_bgl "$PYTHON" src/placebo_sweep.py --dataset bgl
  run placebo_thunderbird "$PYTHON" src/placebo_sweep.py --dataset thunderbird
}

stage_auc() {
  run core_metrics "$PYTHON" src/core_metrics.py
  run multiseed_auc "$PYTHON" src/multiseed_auc.py --n-seeds 5 --dataset both --fault both
}

stage_boundary() {
  run timeaware_control_bgl "$PYTHON" src/timeaware_control.py --dataset bgl
  run timeaware_control_thunderbird "$PYTHON" src/timeaware_control.py --dataset thunderbird
  run deeplog_timeaware_bgl "$PYTHON" src/deeplog_timeaware.py --time-aware --epochs 20 --device "$DEVICE"
  run resolution_ablation_bgl "$PYTHON" src/resolution_ablation.py --dataset bgl
}

stage_deep() {
  # The BGL model is trained once (native target) and reused for the injected target, so both
  # evaluations use the same trained instance.
  run deeplog_bgl_native "$PYTHON" src/deeplog.py --dataset bgl --eval-target native --device "$DEVICE" \
    --save-model results/models/deeplog_bgl.pt
  run deeplog_bgl_injected_stall "$PYTHON" src/deeplog.py --dataset bgl --fault stall --eval-target injected \
    --device "$DEVICE" --load-model results/models/deeplog_bgl.pt --check-invariance
  run deeplog_thunderbird_native "$PYTHON" src/deeplog.py --dataset thunderbird --eval-target native --device "$DEVICE"
  run logbert_bgl_native "$PYTHON" src/logbert_small.py --dataset bgl --epochs 20 --eval-target native --device "$DEVICE"
  run logbert_bgl_injected_stall "$PYTHON" src/logbert_small.py --dataset bgl --fault stall --epochs 20 --eval-target injected --device "$DEVICE"
}

stage_figures() {
  run figures "$PYTHON" src/make_figures.py --figure all \
    --placebo-csv results/placebo/placebo_sweep_bgl.csv results/placebo/placebo_sweep_thunderbird.csv
}

ALL_STAGES="bgl thunderbird spirit hdfs loghub invariance placebo auc boundary deep figures"

if [ "$#" -eq 0 ]; then
  sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

STAGES="$*"
if [ "$STAGES" = "all" ]; then
  STAGES="$ALL_STAGES"
fi

for stage in $STAGES; do
  case "$stage" in
    bgl|thunderbird|spirit|hdfs|loghub|invariance|placebo|auc|boundary|deep|figures) "stage_${stage}" ;;
    *) echo "Unknown stage: ${stage}" >&2; exit 1 ;;
  esac
done

echo "Done: ${STAGES}"
