# DeepLog smoke test

Read-only local verification of `src/deeplog.py`. No full training was run locally, per
instruction — this MacBook Air is not used for real DeepLog training. This document reports both
smoke tests (injected-fault scoring and native-label scoring), the invariance check, and the exact
commands to run on a GPU server.

**The critical framing for anything written up from these results**: row-level DeepLog predictions
are provably identical between clean and injected data (`max_abs_score_diff = 0.0`, confirmed
below, including with actually-injected nodes present). **Any nonzero DeepLog lift on injected data
therefore arises solely from timestamp shifts remapping rows into different 60s grid cells — not
from detection.** DeepLog cannot detect these timing faults; a nonzero lift number on the injected
path is a labeling-grid artifact, not evidence of partial detection capability, and must be
reported as such.

## Smoke test 1: injected-fault scoring, with guaranteed positive cells

The first version of this smoke test (50,000-row chronological prefix, no targeted node inclusion)
passed structurally but happened to contain zero of the 100 injected nodes, so it never exercised
scoring against a real positive cell. Fixed via `--subsample-include-injected N`, which forces N
of the actual injected nodes into the subsample (all of that node's rows, chosen via the same seed
as everything else) before padding out to `--subsample` with the usual chronological-prefix rows.

```
python src/deeplog.py --dataset bgl --fault stall --eval-target injected --subsample 50000 --subsample-include-injected 5 --epochs 1 --device cpu --check-invariance --out-csv /tmp/deeplog_smoke2_injected.csv
```

### Outcome: PASSED, and now genuinely exercises the scoring path

```
train windows: 34754, vocab size (incl. OOV): 41, chronological cutoff: 2005-09-20 12:06:48.659526
precision                      0.010169
recall                         0.666667
f1                             0.020033
grid_flagged_frac              0.721271
lift                           0.924294
detection_rate                     0.04
n_injections_detected                 4
n_injections_total                  100
auc_pr                         0.013653
no_skill_baseline              0.011002
auc_pr_ratio                   1.240908
auc_roc                        0.387378
n_eval_cells                        818
n_positive_cells                      9
fault_type                        stall
oov_rows                            404
oov_frac                        0.00808
n_train_windows                   34754
n_infer_windows                   49930
train_time_s                   1.672359
infer_time_s                   0.352886
final_train_loss               3.370508
```

`n_positive_cells = 9` (was 0 before the fix) — AUC-PR, AUC-ROC, and the no-skill baseline all
compute as real numbers, not `NaN`. `auc_roc = 0.387` is *below* random — on this thin, 1-epoch,
barely-trained slice that's expected noise, not a finding; the number that matters here is that the
column computes correctly at all, which it now does. Schema still matches `auc_metrics.csv` exactly
on the shared columns.

### Invariance check: still IDENTICAL, now with real injected nodes present

```
=== Invariance check ===
  n_common_windows: 49930
  row_id_alignment_identical: True
  n_input_window_mismatches: 0
  n_target_template_mismatches: 0
  n_score_differences: 0
  n_topk_flag_differences: 0
  max_abs_score_diff: 0.0
  n_common_rows_for_grid_check: 50000
  n_grid_cell_reassignments: 331
```

This is the meaningful version of the check the previous smoke test couldn't run: 5 of the 100
actually-injected nodes are now guaranteed present, and **every one of the 49,930 comparable
prediction windows is still bit-identical between clean and injected data** — same as before, now
against real timing-perturbed rows rather than untouched ones. This is the strong confirmation:
DeepLog's predictions do not change when timestamps are perturbed, even on the specific rows whose
timestamps *were* perturbed.

**`n_grid_cell_reassignments = 331` is nonzero this time**, and this is the other half of the
picture. 331 of the 50,000 common rows (0.66%) landed in a *different* 60s grid cell between the
clean view and the injected view, purely because their timestamp moved — while their DeepLog
prediction for that exact row stayed byte-for-byte the same. This directly demonstrates the
mechanism stated at the top of this document: DeepLog's row-level output is invariant, but the
*grid* it gets scored against is timestamp-dependent, so a nonzero lift number on the injected
dataset reflects grid-cell reshuffling, not detection.

## Smoke test 2: native-label scoring (the competence check)

New evaluation path, `--eval-target native`. Trains identically (chronological split, native-normal
train rows only); evaluates on the TEST-period rows of the same clean dataset, against BGL's own
native alert labels, restricted to the test period specifically to avoid leakage (no evaluation on
rows the model could have influenced via training). Ground truth and grid-level scoring reuse
`src.metrics.evaluate_common_unit` and `src.auc_metrics.compute_grid_auc` directly — no new scoring
logic was written, matching the existing harness exactly, including for the injected path
(`src.core_result_stall_final.score_detector`).

```
python src/deeplog.py --dataset bgl --eval-target native --subsample 200000 --epochs 1 --device cpu --out-csv /tmp/deeplog_smoke2_native.csv
```

### Outcome: PASSED, positive cells present without any special handling

```
train windows: 119918, vocab size (incl. OOV): 177, chronological cutoff: 2005-09-20 12:09:07.784020
precision                     0.099757
recall                             1.0
f1                            0.181417
grid_flagged_frac             0.902878
lift                          1.107569
auc_pr                        0.112152
no_skill_baseline             0.090069
auc_pr_ratio                  1.245189
auc_roc                       0.591161
n_eval_cells                     10492
n_positive_cells                   945
fault_type                       None
oov_rows                          12334
oov_frac                      0.205567
n_infer_windows                  186190
infer_time_s                   1.649134
final_train_loss              3.174218
```

`n_positive_cells = 945` — a 200,000-row chronological-prefix subsample already contains enough of
BGL's native alerts (7.34% base rate project-wide) without needing a targeted-inclusion mechanism.
Full shared column set present (precision/recall/f1/grid_flagged_frac/lift/auc_pr/no_skill_
baseline/auc_pr_ratio/auc_roc/n_eval_cells/n_positive_cells), matching the injected path and
`auc_metrics.csv` exactly. `recall = 1.0` / `grid_flagged_frac = 0.90` (near-blanket flagging) is
expected from a 1-epoch, 177-template, barely-converged model (`final_train_loss = 3.17`, still
close to the random-guess entropy for this vocab) — this run is not a claim about DeepLog's real
competence, only a confirmation the native-label scoring path runs correctly end-to-end and
produces a genuine, non-degenerate `auc_roc = 0.591`. `oov_frac = 20.6%` on native-labeled test rows
is markedly higher than the injected path's 0.8% — consistent with, not contradicting, the
project's own premise-audit finding that 99.31% of BGL's native anomalies carry a new/rare
template: anomalous rows are disproportionately the ones introducing templates the (small,
subsampled) training vocabulary never saw.

**This is the number that must sit next to the injected-path result when the DeepLog finding is
written up.** A real full-scale run producing a materially-above-chance native-label AUC-ROC
alongside a near-random injected-fault AUC-ROC is what would establish "DeepLog is competent but
structurally blind to timing faults" rather than "DeepLog doesn't work." A real run producing a
near-random result on *both* would instead mean the model itself is undertrained/miscalibrated and
the blindness claim isn't isolable from that — the native-label number is the control this finding
needs.

## Runtime and memory (both smoke tests)

| | injected (fixed) | native |
|---|---|---|
| rows (subsample) | 50,000 (+5 forced nodes) | 200,000 |
| epochs | 1 | 1 |
| vocab size (incl. OOV) | 41 | 177 |
| train windows | 34,754 | 119,918 |
| total wall time | ~10.2s | ~6.3s (reported `total_time_s`, wall-clock; not independently re-timed with `/usr/bin/time -l` this pass) |

Peak memory was independently measured on the original (unfixed) 50,000-row smoke test via
`/usr/bin/time -l`: **2.04 GB peak RSS / 2.71 GB peak memory footprint**. The two runs above are
close enough in scale (rows and vocab both still small) that this figure remains the right
order-of-magnitude reference; it was not re-measured per run this pass.

**Full-run extrapolation is unchanged from the previous pass of this document** (still an honest
order-of-magnitude estimate, not a guarantee): **BGL ≈ 4–5 hours/epoch on this machine's CPU (~3
weeks at 100 epochs — confirms local full training is not viable)**; **Thunderbird likely 10+
hours/epoch on CPU** given its much larger native vocabulary (9,874 unique templates vs BGL's
~2,000–2,400, per `results/thunderbird_setup_notes_v2.md`); **GPU should bring BGL to roughly
10–30 minutes/epoch**, Thunderbird 20–60 minutes/epoch, both untested. Start with 10–20 epochs and
check `final_train_loss` before committing to a full 100-epoch run; watch Thunderbird's one-hot
batch memory in particular (`batch_size × window_size × vocab_size` floats — ~820 MB per batch at
`batch_size=2048`, `window_size=10`, `vocab_size≈10,000` — reduce `--batch-size` first if this
overflows GPU memory).

## Exact server run commands

Both eval targets, both datasets — four required runs plus one invariance re-check:

```
python src/deeplog.py --dataset bgl --fault stall --eval-target injected --device cuda
python src/deeplog.py --dataset bgl --fault burst --eval-target injected --device cuda
python src/deeplog.py --dataset thunderbird --fault stall --eval-target injected --device cuda
python src/deeplog.py --dataset thunderbird --fault burst --eval-target injected --device cuda

python src/deeplog.py --dataset bgl --eval-target native --device cuda
python src/deeplog.py --dataset thunderbird --eval-target native --device cuda

python src/deeplog.py --dataset bgl --fault stall --eval-target injected --device cuda --check-invariance
```

The last line re-runs the invariance check against the full (non-subsampled) BGL dataset as a final
correctness gate before trusting the full-scale numbers — expect the identical
zero-score-difference result seen in both smoke tests, now against all 100 real injected nodes at
full scale.

`--eval-target injected` writes `results/deeplog_{dataset}_{fault}.csv`; `--eval-target native`
writes `results/deeplog_{dataset}_native.csv` (both overridable via `--out-csv`). `--fault` is
*required* when `--eval-target injected` (validated at parse time — omitting it is an error) and
simply unused when `--eval-target native` (harmless if passed, ignored; omit it for clarity, as in
the commands above).

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

`--eval-target native` needs only the two `*_parsed.parquet` files (no injected parquets or grid
labels) — but transferring all ten is simplest since the injected-eval runs need the rest anyway.

Python dependencies beyond this project's existing `requirements.txt`: `torch` (install the CUDA
build matching the server's CUDA version, e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu121`). Everything else DeepLog needs (`pandas`, `numpy`,
`scikit-learn`, `pyarrow`) is already in `requirements.txt`.

## What was NOT done, per explicit instruction

- No full training was run locally, on any dataset.
- No existing result file (`FINAL_results.csv`, `FINDINGS.md`, `auc_metrics.csv`, or any
  `results/core_result_*` / `results/thunderbird_*` file) was read, modified, or regenerated.
  `deeplog.py` writes only to its own `results/deeplog_{dataset}_{fault}.csv` /
  `results/deeplog_{dataset}_native.csv` paths (or `--out-csv` if overridden — both smoke tests
  this pass explicitly redirected output to `/tmp`, not `results/`, to avoid leaving misleading
  garbage-numbers files in the results directory).
- DeepLog's numbers are not consolidated into any project-level table. That happens only after a
  real run completes on the GPU server, per instruction.
