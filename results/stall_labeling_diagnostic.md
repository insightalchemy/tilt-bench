# Stall labeling diagnostic: are zero-row grid cells depressing stall AUC?

Read-only diagnostic. No injections or detectors were re-run; this only counts grid cells from data already on disk (n=100 injected parquets + grid labels + `results/auc_metrics.csv`).

## Hypothesis

`src.injector.label_spans_on_grid` labels every 60s grid cell whose TIME RANGE overlaps an injected span as anomalous, computed purely from timestamps, not from which rows actually exist. A stall's entire mechanism is a long gap with no events in it, so some of a stall's labeled cells can contain zero rows -- structurally undetectable by any row-driven detector, since a detector can only score a cell that has at least one row to score. If those cells were being counted as false negatives, every stall AUC/recall number would be mechanically penalized regardless of detector quality. Bursts compress events into a shorter span rather than emptying it, so were *expected* to have few or no such cells -- **this expectation turned out to be wrong for BGL, see §5.**

## 1-3. Cell counts (n=100, all four dataset x fault_type combinations)

| dataset | fault | injections | labeled anomalous cells | zero-row (undetectable) | detectable positives | % undetectable | cells/injection |
|---|---|---|---|---|---|---|---|
| BGL | stall | 100 | 114671 | 114475 | 196 | 99.8% | 1146.71 |
| BGL | burst | 100 | 141326 | 140546 | 780 | 99.4% | 1413.26 |
| Thunderbird | stall | 100 | 34472 | 34276 | 196 | 99.4% | 344.72 |
| Thunderbird | burst | 100 | 1203 | 674 | 529 | 56.0% | 12.03 |

## 4. Recomputed AUC-PR / AUC-ROC restricted to detectable positives only

**No recomputation was necessary, and none was performed** (per instruction: no re-running detectors) -- because the *existing* `results/auc_metrics.csv` numbers are **already** computed over detectable-positives-only, as a structural consequence of how the grid aggregation works, not by explicit design.

`src.metrics.rows_to_grid` / `rows_to_grid_max` both aggregate via `groupby(df_eval["eval_window_key"])`, where `df_eval` has exactly one row per **log event**. A grid cell can only appear in the resulting Series' index if at least one row maps to it -- so the true-label grid (`true_grid`) and every detector's score grid can *only ever* be defined over populated cells. A zero-row anomalous cell is never counted as a false negative; it simply never enters the evaluation at all, exactly as if it had been explicitly excluded.

This is verified empirically, not just argued: the `n_detectable_positive_cells` column above (computed independently, directly from the on-disk parquets and grid-label CSVs, with zero reference to any detector) is cross-checked against `results/auc_metrics.csv`'s existing `n_positive_cells` column (which the AUC computation actually used):

| dataset | fault | n_detectable_positive_cells (this diagnostic) | n_positive_cells (existing auc_metrics.csv) | match |
|---|---|---|---|---|
| BGL | stall | 196 | 196 | YES |
| BGL | burst | 780 | 780 | YES |
| Thunderbird | stall | 196 | 196 | YES |
| Thunderbird | burst | 529 | 529 | YES |

**All four rows match exactly.** This confirms the existing AUC-PR/AUC-ROC numbers in `results/auc_metrics.csv` and `results/FINAL_results.csv` already reflect detectable-positives-only scoring. There is nothing to recompute: the side-by-side comparison the task asked for is between the existing numbers and themselves.

## 5. Does this change anything?

**No — the hypothesis is real (zero-row cells genuinely exist in the stall ground truth) but it does not depress the current numbers, because those cells were already structurally excluded from evaluation, not silently counted as misses.** Specifically:

- **The zero-row "quiet middle" cells are real and substantial on THREE of the four combinations, not just stall.** BGL-stall: 99.8% undetectable (114475/114671). Thunderbird-stall: 99.4% (34276/34472). **BGL-burst: 99.4% (140546/141326) — essentially the same magnitude as BGL-stall**, contradicting the hypothesis's own assumption that bursts wouldn't have this problem. Only Thunderbird-burst is meaningfully less affected, at 56.0% (674/1203) — still over half, not "none."
- **Why BGL-burst was wrong to assume safe**: burst eligibility (`BURST_MIN_NODE_EVENTS=55`) is a minimum *event count*, not a minimum *event rate* — it says nothing about how much wall-clock time those 55+ events span. Some eligible nodes are so sparse that their events are spread across the dataset's *entire* multi-month observation window. Inspecting the actual injected spans directly: the largest BGL-burst injection (node `R00-M0-N4-C:J15-U11`) compressed a run of 25 gaps by 10x and still left a `compressed_span_s` of 1,214,495 seconds (~14 days, 20,243 grid cells) — a 10-30x compression factor cannot shrink a multi-week gap down to something a 60-second grid can represent as densely populated. The median BGL-burst injection spans 290.5 cells; the top few span 6,000-20,000. This is the *same* underlying mechanism as stall's quiet middle (a large real-world time gap on a sparse node), not a different one — burst just makes it somewhat less universal because compression *can* work when the pre-existing gaps aren't already enormous, which is more often true on Thunderbird's burst-eligible nodes than BGL's in this specific n=100 draw.
- **The z_score-stall AUC-ROC does not move from 0.537 (BGL) / 0.587 (Thunderbird)** — there is no undercounted-false-negative correction to apply; those numbers already are the detectable-positives-only numbers. The same now applies to burst too: burst AUC-ROC numbers (e.g. BGL-burst log_ratio 0.673) are equally already computed on detectable-only cells, so they don't move either.
- **The stall/burst asymmetry does not shrink** for the same reason — it isn't caused by this labeling artifact (on any fault type), so removing the (already-absent) artifact can't close it.
- **A related, genuinely structural (not artifactual) point worth carrying into the writeup**: burst still generates more *detectable* positive cells per injection than stall on both datasets (BGL: 780 detectable / 100 injections = 7.80 avg vs stall's 196/100 = 1.96 — a 3.98x ratio; Thunderbird: 529/100 = 5.29 vs 1.96 — a 2.70x ratio), despite the raw *labeled* cell counts being dominated by sparse-node artifacts on both fault types. This detectable-cell richness is a real, structural burst-vs-stall difference (a compressed run of up to 25 events straightforwardly produces more populated cells than a stall's single inserted gap, WHEN the node isn't pathologically sparse) and can independently contribute to burst's stronger AUC-PR-ratio numbers on top of (not instead of) the real log_ratio feature-geometry effect documented in FINDINGS.md §5.

