# Windowing sweep notes

Companion to `results/windowing_sweep.csv` (detection-metric sweep) and
`results/windowing_sweep_invariance.csv` (clean-vs-injected invariance check), produced by
`src/windowing_sweep.py`. Both CSVs mark every row's completion state in a `status` column
(`complete` / `pending`); pending rows carry the literal string `PENDING` in every metric column —
never a blank, `NaN`, or `0` — so a pending cell can never be misread as a null or zero result.

## Status at a glance

| dataset | scheme kind | status |
|---|---|---|
| BGL | fixed-time (30/60/120s) + fixed-count (N=20/50/100) | **complete** — full 48-row sweep, both fault types, all 4 detectors |
| Thunderbird | fixed-count (N=20/50/100) | **complete** — 24-row sweep, both fault types, all 4 detectors |
| Thunderbird | fixed-time (30/60/120s) | **PENDING** — OOM-killed on this 8GB machine (see below); scheduled to run on a lab server |

## 1. Invariance check: fixed-count windowing makes timing-only faults invisible to content/count detectors

For `count_vector_pca` and `isolation_forest_counts`, comparing clean vs. injected data under
fixed-count windowing (N=20/50/100), on **both datasets, both fault types**:

**`max_abs_score_diff = 0.0`, `n_differing_windows = 0`, `n_grid_cell_reassignments = 0`.**
24 cells total (2 datasets × 2 fault types × 3 window sizes × 2 detectors), **no exceptions.**

| dataset | fault | scheme | detector | max_abs_score_diff | n_differing_windows | n_grid_cell_reassignments |
|---|---|---|---|---|---|---|
| BGL | stall/burst | N=20/50/100 | PCA, isoforest | 0.0 | 0 | 0 |
| Thunderbird | stall/burst | N=20/50/100 | PCA, isoforest | 0.0 | 0 | 0 |

This is an exact structural result, not a near-zero approximation: under fixed-count windowing, a
window is defined purely by an event's position in its node's sequence, never by its timestamp.
Shifting timestamps (stall/burst injection) changes *when* a row occurs but never *which* window it
falls into or *what* the count-vector/isolation-forest score for that window is — so clean and
injected data produce byte-identical windows, byte-identical scores, and zero grid-cell
reassignments. This holds independent of dataset scale, node population, or fault type.

## 2. Contrast: fixed-time windowing does make the perturbation visible (BGL)

Under fixed-time windowing (30s/60s/120s), the same two detectors show real, substantial
differences on BGL:

- Score differences: **12.9–348** (max_abs_score_diff, across schemes/fault types).
- Hundreds of differing windows per cell.
- Thousands of grid-cell reassignments per cell (BGL: 5,245–9,473, out of 4,747,963 rows checked).

This is the entire point of the sweep: the *same two detectors*, run on the *same injected data*,
go from perfectly blind (fixed-count) to partially sighted (fixed-time) purely because of how
windows are drawn. Nothing about the detector or the injection changed between the two columns —
only the windowing scheme.

Thunderbird's fixed-time numbers are **PENDING** (§4) — expected to show the same qualitative
pattern (real, nonzero differences) once run, but not yet measured.

## 3. DeepLog (BGL, sequence windowing): a third, independent confirmation

From `results/deeplog_smoke_test.md`'s invariance check (5 of the 100 actually-injected stall
nodes forced into a 50,000-row subsample, 1-epoch smoke-test model — **not** a full training run,
see §4):

**`max_abs_score_diff = 0.0` across 49,930 comparable prediction windows; `n_grid_cell_reassignments
= 331`.**

DeepLog's row-level predictions are provably byte-identical between clean and injected data — a
sequence-based deep model is exactly as structurally blind to timing-only perturbation as the
count-vector detectors are, for the same underlying reason (its input is the sequence of template
IDs, which a timestamp shift never changes). The nonzero `n_grid_cell_reassignments = 331` shows
the *other* half of the mechanism directly: 331 of 50,000 common rows (0.66%) landed in a different
60-second grid cell purely because their timestamp moved, while the DeepLog prediction attached to
that exact row stayed byte-for-byte the same. Any nonzero DeepLog lift measured against injected
data is therefore a grid-cell-reshuffling artifact of the evaluation grid, not evidence that DeepLog
detected anything.

## Conclusion

Content and sequence detectors are **provably blind** to timing-only faults — this is now
demonstrated three independent ways (count-vector/PCA, isolation-forest-on-counts, and DeepLog's
LSTM sequence model), across two datasets for the count detectors. Any apparent timing-fault
"detection" that shows up under fixed-time windowing does not come from the detector gaining
sensitivity to timing; it comes from the windowing scheme itself converting a timing perturbation
into a **count** perturbation (rows moving between fixed-time buckets), which a count-based detector
can then pick up as an ordinary count anomaly. Windowing choice, not detector architecture, is what
determines whether timing faults are observable at all — the central claim this sweep was built to
test.

## 4. What is still pending

- **Thunderbird fixed-time windowing (30s/60s/120s), all 4 detectors, both fault types** (24 sweep
  rows + 12 invariance rows). Attempted locally and **OOM-killed** by the macOS kernel
  (`memorystatus: killing largest compressed process Python [69770] 23625 MB`) — confirmed via the
  unified system log, not a repeat of the earlier fileproviderd/iCloud stall issue this project had
  previously traced and fixed. Root cause: `src/detectors/windowing.py`'s dense
  `(n_windows × 300)` float32 count matrix, held concurrently across train/clean/injected copies
  plus sklearn's own internal copies during `fit()`, peaks in the multi-GB range for Thunderbird's
  ~2.6–3.0M fixed-time windows — workable on a machine with sufficient RAM, but this development
  machine has only 8GB total, shared with other running applications. **Scheduled to run on a lab
  server; no code changes have been made to the detector/windowing logic.**
- **DeepLog full training run**, both datasets, both eval targets (injected + native). Only a
  1-epoch, subsampled smoke test has been run locally, per explicit instruction that this machine
  is not used for real DeepLog training (`results/deeplog_smoke_test.md`). The smoke test's
  invariance numbers (§3) are trustworthy as a structural/mechanism result (they don't depend on
  training quality), but no full-scale DeepLog detection numbers exist yet.
