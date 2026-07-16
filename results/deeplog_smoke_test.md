# DeepLog smoke test

Read-only local verification of `src/deeplog.py`. No full training was run locally, per instruction
— this MacBook Air is not used for real DeepLog training. This document reports the smoke-test
result, the invariance check, and the exact commands to run on a GPU server.

Command run (the only local execution of this script, run twice — once for output, once
instrumented for peak memory; both produced bit-identical results):

```
python src/deeplog.py --dataset bgl --fault stall --subsample 50000 --epochs 1 --device cpu --check-invariance --out-csv /tmp/deeplog_smoke_bgl_stall.csv
```

## Outcome: PASSED

The pipeline ran end-to-end without error, on a 50,000-row chronological prefix of BGL, one
training epoch, CPU only, and produced output in the exact schema `auc_metrics.csv` /
`FINAL_results.csv` use (lift, detection_rate, base rate, AUC-PR, no-skill baseline, AUC-PR ratio,
AUC-ROC), plus DeepLog-specific columns (vocab size, OOV count, window/top-k/hidden-size
hyperparameters, timing).

```
train windows: 34809, vocab size (incl. OOV): 20, chronological cutoff: 2005-09-20 12:06:50.122528
precision                     0.0
recall                        0.0
f1                            0.0
grid_flagged_frac        0.879545
lift                          0.0
detection_rate                0.0
n_injections_detected           0
n_injections_total            100
auc_pr                        0.0
no_skill_baseline             0.0
auc_pr_ratio                  NaN
auc_roc                       NaN
n_eval_cells                  440
n_positive_cells                0
oov_rows                      189
oov_frac                  0.00378
n_train_windows             34809
n_infer_windows             49980
train_time_s                 1.70
infer_time_s                 0.34
final_train_loss           2.5195
```

**`n_positive_cells = 0` is expected and correctly handled, not a bug.** The smoke-test subsample
(the chronological first 50,000 rows of BGL) selects a small, alphabetically-early slice of nodes;
none of the 100 randomly-seeded stall-injection target nodes happened to fall inside it. With zero
true positives in the eval grid, lift/AUC-PR/AUC-ROC are structurally undefined — the script
returns `0.0`/`NaN` rather than crashing (confirmed against `sklearn`'s own "no positive class
found" warning, handled gracefully), which is itself a useful robustness check, but this run does
**not** exercise the "does DeepLog actually score a real injected fault" path. That can only be
verified in a real GPU run against the full dataset, where all 100 injected nodes are present.

`grid_flagged_frac = 0.88` (DeepLog's own top-k miss rate on this vocab) is expected to be high on
a 1-epoch, 20-template, barely-trained model — not evidence of detector quality, only evidence the
pipeline computes and threads the number through correctly.

## Invariance check: IDENTICAL — DeepLog is blind to these timing faults by construction

```
=== Invariance check ===
  n_common_windows: 49980
  row_id_alignment_identical: True
  n_input_window_mismatches: 0
  n_target_template_mismatches: 0
  n_score_differences: 0
  n_topk_flag_differences: 0
  max_abs_score_diff: 0.0
  n_common_rows_for_grid_check: 50000
  n_grid_cell_reassignments: 0
```

Ran the trained model on clean BGL and on `bgl_injected_stall` (same subsample, same row
alignment). **Every one of the 49,980 comparable prediction windows produced bit-identical scores
and top-k flags between clean and injected data** (`max_abs_score_diff = 0.0`, not merely small —
exactly zero). Input windows and target templates also matched exactly (`n_input_window_mismatches
= 0`, `n_target_template_mismatches = 0`), confirming the sequence-construction is template-content-
driven, not timestamp-value-driven, as intended.

**Stated plainly, as instructed**: this confirms the theoretical expectation was correct. The
injector modifies only timestamps; DeepLog consumes only the template-ID sequence. A row-driven
model that never looks at a timestamp cannot distinguish clean from injected data, by construction
— not because it failed to learn the difference, but because the difference is literally not
present in its input. This is architectural, not empirical: **no amount of training or
hyperparameter tuning can make row/sequence-only DeepLog detect a timing-only fault.** Any real
detection DeepLog shows on the full injected datasets would have to come from a secondary channel
this smoke test doesn't probe — e.g. a stall's absence of intervening rows changing which template
IDs are adjacent in the surviving sequence (only possible if the stall removes rows, which it does
not) — so the honest expectation for the full GPU run is that DeepLog will show at-or-near-chance
performance on both fault types, on both datasets, and that would be the *correct*, *expected*
result, not a failure of the implementation.

**Grid-cell reassignment caveat, reported honestly rather than overclaimed**: `n_grid_cell_
reassignments: 0` is consistent with the hypothesis (shifted timestamps can remap a row to a
different 60s cell even when its prediction is unchanged) but is **not a meaningful test of it
here** — since this subsample contains zero actually-injected nodes (see above), none of its rows'
timestamps were touched by injection at all, so trivially their grid cells can't have moved either.
This specific sub-check needs to be rerun on the full dataset (or a subsample deliberately including
injected nodes) to be informative. The row-level score/prediction invariance above, by contrast,
was tested across all 49,980 common windows and is a complete, meaningful confirmation.

## Runtime and memory

| | value |
|---|---|
| rows (subsample) | 50,000 |
| epochs | 1 |
| vocab size (incl. OOV) | 20 |
| train windows | 34,809 |
| total wall time | 8.3s (uninstrumented) / 11.5s (instrumented, `/usr/bin/time -l`) |
| peak RSS | 2.04 GB |
| peak memory footprint | 2.71 GB |

**Honest extrapolation to a full run (order-of-magnitude only, not a guarantee)**: per-epoch cost
scales roughly with `n_windows × vocab_size` (the one-hot LSTM input and the final linear layer
both scale with vocabulary size). Calibrating off this smoke test (`1.7s ÷ (34,809 × 20) ≈
2.4×10⁻⁶ s` per window-vocab-unit):

- **BGL full**: ~3M train windows (70% chronological split, native-normal only, ~2,000–2,400
  templates per the premise audit) → roughly **4–5 hours per epoch on this machine's CPU**. At the
  default 100 epochs, that's **on the order of 3 weeks — confirms local full training is not
  viable**, exactly as instructed.
- **Thunderbird full**: ~7M train windows, but a much larger vocabulary (**9,874 unique templates**
  per `results/thunderbird_setup_notes_v2.md`) — the one-hot input/output layers alone are ~5×
  BGL's, so expect Thunderbird to be the more expensive of the two datasets despite similar row
  counts, likely **10+ hours/epoch on CPU**.
- **GPU estimate**: LSTM + one-hot workloads at this hidden size (64) typically see a 10–30×
  speedup on a modern GPU from batched matrix parallelism, giving a rough **10–30 minutes/epoch for
  BGL, 20–60 minutes/epoch for Thunderbird** on a single mid-range GPU (untested — this is
  extrapolation, not measurement). At 100 epochs that's still multi-day; **recommend starting with
  a much smaller epoch count (10–20) and checking convergence via `final_train_loss` before
  committing to the full 100**, and consider capping the vocabulary (analogous to the existing
  count-vector detectors' `TOP_K_TEMPLATES=300`) if Thunderbird's one-hot memory footprint proves
  too large for the GPU's memory (batch_size × window_size × vocab_size one-hot floats — at
  `batch_size=2048`, `window_size=10`, `vocab_size≈10,000`, that one-hot tensor alone is
  `2048×10×10000×4` bytes ≈ **820 MB per batch**, before any hidden-state activations — reduce
  `--batch-size` first if you hit an out-of-memory error on Thunderbird).

## Exact server run commands

```
python src/deeplog.py --dataset bgl --fault stall --device cuda
python src/deeplog.py --dataset bgl --fault burst --device cuda
python src/deeplog.py --dataset thunderbird --fault stall --device cuda
python src/deeplog.py --dataset thunderbird --fault burst --device cuda
```

Each writes its own `results/deeplog_{dataset}_{fault}.csv` (default `--out-csv`, override if you
want a different location). Add `--check-invariance` to any of these to re-run the invariance check
against the full dataset (recommended once, on `bgl stall`, as a final correctness gate before
trusting the full-scale numbers) — expect the identical zero-mismatch result seen here, now
genuinely exercising injected nodes since the full dataset includes all 100 of them.

If 100 epochs proves too slow even on GPU, reduce with `--epochs 20` (or lower) and inspect
`final_train_loss` in the output CSV before deciding whether more epochs are worth the wall time.

## Files needed for the GPU server transfer

Transitive imports pull in most of `src/` (the reused evaluation harness, its own dependents, and
the loaders each detector script defines) — safest to transfer the whole `src/` directory rather
than hand-pick files and risk a missing transitive import mid-run. Specifically exercised by
`deeplog.py`:

```
src/deeplog.py
src/metrics.py
src/auc_metrics.py
src/core_result_stall.py
src/core_result_stall_final.py
src/core_result_burst.py
src/core_result_thunderbird.py
src/core_result_v2_symmetric.py      (imported transitively by auc_metrics.py)
src/run_baseline_detectors.py
src/timing_baseline.py
src/detectors/__init__.py
src/detectors/windowing.py
src/detectors/count_pca.py
src/detectors/isolation_forest_counts.py
src/detectors/timing_detector.py
```

Data files (read-only, none modified by this task):

```
data/processed/bgl_parsed.parquet
data/processed/thunderbird_parsed.parquet
data/processed/bgl_injected_stall.parquet
data/processed/bgl_injected_burst.parquet
data/processed/thunderbird_injected_stall.parquet
data/processed/thunderbird_injected_burst.parquet
data/processed/injection_grid_labels_stall.csv
data/processed/injection_grid_labels_burst.csv
data/processed/thunderbird_injection_grid_labels_stall.csv
data/processed/thunderbird_injection_grid_labels_burst.csv
```

Python dependencies beyond this project's existing `requirements.txt`: `torch` (install the CUDA
build matching the server's CUDA version, e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu121`). Everything else DeepLog needs (`pandas`, `numpy`,
`scikit-learn`, `pyarrow`) is already in `requirements.txt`.

## What was NOT done, per explicit instruction

- No full training was run locally, on any dataset.
- No existing result file (`FINAL_results.csv`, `FINDINGS.md`, `auc_metrics.csv`, or any
  `results/core_result_*` / `results/thunderbird_*` file) was read, modified, or regenerated.
  `deeplog.py` writes only to its own `results/deeplog_{dataset}_{fault}.csv` path (or `--out-csv`
  if overridden — the smoke test explicitly redirected output to `/tmp`, not `results/`, to avoid
  leaving a misleading garbage-numbers file in the results directory).
- DeepLog's numbers are not consolidated into any project-level table. That happens only after a
  real run completes on the GPU server, per instruction.
