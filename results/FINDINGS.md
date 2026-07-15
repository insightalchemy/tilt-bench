# TILT-Bench: Findings Summary

Plain-language consolidation of the project to date, at **n=100 injections per fault type per
dataset** (locked after this pass — see "Honest open items" for why n was scaled 15 → 40 → 100
before settling). Every number here is already computed and saved in `results/`; see
`results/FINAL_results.csv` (BGL) and `results/thunderbird_vs_bgl_comparison.csv` (both datasets)
for the master tables.

## 1. The premise

Standard log-anomaly-detection benchmarks are evaluated almost entirely against **content**-based
anomalies (new/rare event templates, broken sequence order). We asked: do these datasets' own
native anomaly labels already contain timing-only faults, or is timing a genuine blind spot in how
these benchmarks are built? Checked on two datasets with genuine per-line native labels (BGL,
Thunderbird); HDFS's block-level labeling breaks the premise-audit signatures structurally and is
handled separately in §7.

| signature | BGL (348,460 anomalous rows) | Thunderbird (170,422 anomalous rows) |
|---|---|---|
| new_or_rare_template | 99.31% | 99.99% |
| pure_order_anomaly | 0.52% | **38.97%** |
| timing_gap_anomaly | 46.97% | 49.84% |
| timing-only (no content/order riding along) | 0.00% (3/348,460) | 0.00% |

Content-novelty is near-universal on both datasets, confirming the premise: native anomaly labels
are overwhelmingly content-defined, and **zero** rows on either dataset are purely timing-anomalous.
This motivated building a controlled timing-fault injector rather than relying on native anomaly
mix to evaluate timing detection.

One real cross-dataset difference surfaced along the way, not assumed away: Thunderbird's
pure-order rate (38.97%) is **75× BGL's** (0.52%) — checked against a representative, diverse
50M–60M-line slice (not a single-incident artifact; see `results/thunderbird_setup_notes_v2.md`).
Thunderbird's native anomalies are more likely to reuse content a node already produces normally
while landing in an unusual sequence position — a structurally different anomaly mechanism from
BGL's, though it doesn't change the core timing-blind-spot conclusion (timing-only is still 0% on
both).

## 2. The injector and its validation

`src/injector.py` (BGL) / `src/injector_thunderbird.py` (Thunderbird, same functions reused
unmodified) implement two fault types, both per-node/per-sequence and both provably
order-preserving (a stall shifts every downstream timestamp by a constant; a burst compresses a run
of gaps then shifts everything downstream by the exact time saved — both are single-constant shifts
to a whole tail of the sequence, which can never invert order).

- **Stall**: insert a large gap = `intensity × node_scale` (additive), where `node_scale` is the
  per-node MAD-derived scale.
- **Burst**: compress `L` consecutive gaps by `new_gap = orig_gap / intensity` (multiplicative),
  then shift everything downstream backward by the total time saved.

**Thunderbird-specific fix, applied before any Thunderbird injection**: Thunderbird's pooled
fallback baseline is itself degenerate (median=MAD=0, since >50% of *all* gaps are exactly 0 —
second-level timestamp resolution). Fixed via two additive, opt-in parameters (BGL's own invocation
is provably unaffected — regression-diffed): `require_valid_baseline=True` restricts injection
eligibility to nodes with a genuine non-degenerate per-node baseline (no node is ever scaled against
a broken fallback), and `exclude_zero_from_pooled=True` makes the pooled fallback itself
non-degenerate for the content/order detectors that still use it.

**Node-eligibility check at n=100** (the injection count approaches what's available on
Thunderbird, so this was verified *before* injecting, not after):

| dataset | fault | eligible nodes | injections drawn | distinct nodes used | used_pooled_fallback |
|---|---|---|---|---|---|
| BGL | stall | 37,765 | 100 | 100 | 0 (checked directly) |
| BGL | burst | 30,889 | 100 | 100 | n/a (BGL doesn't require it; not a gate) |
| Thunderbird | stall | 3,433 | 100 | 100 | 0 (structurally guaranteed — `require_valid_baseline=True`) |
| Thunderbird | burst | 3,418 | 100 | 100 | 0 (structurally guaranteed) |

All four combinations clear 100 eligible nodes by a wide margin (34×–378× the requested count) —
no marginal/low-quality-baseline nodes were needed.

**Two validations were run before trusting any detection result, at n=100, all four
dataset×fault-type combinations:**

- **Ordering**: automated check across the *entire* injected dataset (not just injected nodes) —
  **0 violations** in every case: BGL 4,747,963 rows (×2 fault types), Thunderbird 10,000,000 rows
  (×2 fault types).
- **Instrument validation**: re-ran the premise-audit signatures on the injected rows themselves.
  Expectation: timing fires, content/order stay silent.

| dataset | fault | timing_gap_anomaly | new_or_rare_template | pure_order_anomaly |
|---|---|---|---|---|
| BGL | stall | **99.0%** | 3.0% | 0.0% |
| BGL | burst | 20.4% | 0.2% | 0.0% |
| Thunderbird | stall | **100.0%** | 0.0% | 0.0% |
| Thunderbird | burst | 4.4% | 0.6% | 0.0% |

Content/order stay essentially silent everywhere (≤3%), confirming clean injections on both
datasets. The lower timing-fire rate on bursts (20.4% BGL, 4.4% Thunderbird — both far below
stall's ~100%) is not an injector defect; it's the first visible sign of the feature-geometry issue
in §5 (an additive z-score is structurally bounded against multiplicative compression).

## 3. The stall result

Fit on clean (pre-injection) train-period data, detectors run against the 100 injected stalls per
dataset, scored on the injected-span ground truth (not native labels), on a common 60-second
per-node grid.

| dataset | detector | detection rate (of 100) | lift |
|---|---|---|---|
| BGL | z_score_threshold (timing) | 91/100 (91%) | **1.33** |
| BGL | count_vector_pca (content) | 30/100 (30%) | 0.72 (below chance) |
| BGL | isolation_forest_counts (dead baseline) | 12/100 (12%) | 1.53 |
| Thunderbird | z_score_threshold (timing) | 93/100 (93%) | **1.39** |
| Thunderbird | count_vector_pca (content) | 6/100 (6%) | 1.98 |
| Thunderbird | isolation_forest_counts (dead baseline) | 9/100 (9%) | 6.70 |

**Lift** = recall ÷ the detector's own indiscriminate flagging rate. z_score shows real,
above-chance detection on stalls on **both** datasets, holding up cleanly from n=15 through n=100
(§5 has the full stability table). Thunderbird's PCA lift (1.98) is nominally above chance despite
a very low absolute detection rate (6/100) — treat with caution; see §8.

## 4. The burst result

Same setup, burst-injected data:

| dataset | detector | detection rate (of 100) | lift |
|---|---|---|---|
| BGL | log_ratio_threshold (timing, symmetric) | 68/100 (68%) | **7.29** |
| BGL | count_vector_pca (content) | 46/100 (46%) | 0.52 (below chance) |
| BGL | z_score_threshold (timing, additive) | 63/100 (63%) | 0.50 (below chance) |
| BGL | isolation_forest_counts (dead baseline) | 24/100 (24%) | 0.99 (≈chance) |
| Thunderbird | log_ratio_threshold (timing, symmetric) | 59/100 (59%) | **2.75** |
| Thunderbird | count_vector_pca (content) | 11/100 (11%) | 1.01 (≈chance) |
| Thunderbird | z_score_threshold (timing, additive) | 34/100 (34%) | 0.35 (below chance) |
| Thunderbird | isolation_forest_counts (dead baseline) | 43/100 (43%) | 11.96 |

The additive z-score, which works cleanly on stalls on both datasets, **fails on bursts on both
datasets** — not assumed to be "bursts are just harder," traced to a specific mechanism (§5). The
symmetric `log_ratio_threshold` fixes this on both datasets, with real above-chance lift.

## 5. The feature-geometry / complementarity finding

`z_score = (gap − node_median) / node_scale` is **additive**. A stall adds `intensity × node_scale`
directly, so its z-score is unbounded by construction. A burst instead **compresses** the gap
(`new_gap = orig_gap / intensity`); its z-score is bounded by roughly `−(node_median / node_scale)`
as intensity → ∞, regardless of compression severity — structurally far below the `|z|>3` threshold
for gap distributions with a positive median/MAD ratio, which both BGL and Thunderbird have. This
single mechanism explains both datasets' low burst instrument-fire rate (§2) and failed burst
z_score detection (§4).

Fix tested: `log_ratio = log(gap / node_median)` is **symmetric** — a stall gives a large positive
value, a burst gives a large negative value of comparable magnitude.

**Lift by detector × fault × dataset, at n=100 (bold = above chance, i.e. lift > 1):**

| dataset | fault | PCA (content) | z_score (timing, additive) | log_ratio (timing, symmetric) |
|---|---|---|---|---|
| BGL | stall | 0.72 | **1.33** | 0.95 |
| BGL | burst | 0.52 | 0.50 | **7.29** |
| Thunderbird | stall | **1.98** | **1.39** | **1.42** |
| Thunderbird | burst | ≈1.01 | 0.35 | **2.75** |

**Does the pattern hold?**
- **z_score is cleanly stall-only, on both datasets, with no exception**: above chance on both
  stall cells (1.33, 1.39), below chance on both burst cells (0.50, 0.35).
- **log_ratio is above chance on burst on both datasets** (7.29, 2.75) — the fix works as designed
  everywhere it was meant to.
- **log_ratio on stall is dataset-dependent, not a clean split.** Above chance on Thunderbird
  (1.42) but **not decisively below chance on BGL** (0.95) — this specific cell has moved close
  enough to the chance line (1.0) that it can no longer be reported as a confident negative result
  (see the stability table below).
- **PCA is not uniformly at/below chance on timing faults** — it fails as expected on 3 of 4 cells,
  but Thunderbird-stall PCA (1.98) is a persistent, repeated exception across every n tested (0/15,
  5.04/40, 1.98/100 — see below). Content detection is not structurally blind to timing faults on
  every dataset; this should be stated as a qualified finding, not a universal one.

**Stability across n=15 → n=40 → n=100** (the reason n was scaled before locking):

| dataset | fault | detector | n=15 | n=40 | n=100 | stable? |
|---|---|---|---|---|---|---|
| BGL | stall | PCA | 1.09 | 0.60 | 0.72 | below chance from n=40 on |
| BGL | stall | z_score | 1.55 | 1.27 | 1.33 | **stable, above chance throughout** |
| BGL | stall | log_ratio | 2.25 | 0.68 | 0.95 | **not stable — see caveat below** |
| BGL | burst | PCA | 0.73 | 0.49 | 0.52 | stable, below chance |
| BGL | burst | z_score | 0.46 | 0.37 | 0.50 | **stable, below chance throughout** |
| BGL | burst | log_ratio | 7.33 | 8.12 | 7.29 | **stable, above chance throughout** |
| Thunderbird | stall | PCA | 0.00 (0/15) | 5.04 (5/40) | 1.98 (6/100) | **not stable — thin-sample artifact** |
| Thunderbird | stall | z_score | 1.56 | 1.39 | 1.39 | **stable, above chance throughout** |
| Thunderbird | stall | log_ratio | 1.59 | 1.55 | 1.42 | stable, above chance |
| Thunderbird | burst | PCA | 0.87 | 0.71 | 1.01 | **borderline — hovering at chance across all three n** |
| Thunderbird | burst | z_score | 0.26 | 0.49 | 0.35 | stable, below chance |
| Thunderbird | burst | log_ratio | 1.84 | 1.64 | 2.75 | stable, above chance |

**Two cells are explicitly flagged as not decisive, and should not be used to support a strong
claim in either direction:**
1. **BGL-stall log_ratio** (2.25 → 0.68 → 0.95): non-monotonic across all three sample sizes, and
   at n=100 sits at lift=0.9488 with only 8/100 detections — indistinguishable from chance given
   sampling noise at this base rate. The honest statement is "log_ratio does not show a reliable
   signal on BGL stalls," not "log_ratio fails on BGL stalls."
2. **Thunderbird-stall PCA** (0.00 → 5.04 → 1.98) and **Thunderbird-burst PCA** (0.87 → 0.71 →
   1.01): both driven by single-digit-to-low-double-digit absolute detection counts (0/15, 5/40,
   6/100 and 6/15, 18/40 [scaled], 11/100 respectively) and both hover on or near the chance line
   across every n tested rather than settling. Treat Thunderbird-PCA numbers as noisy, not as
   evidence either way.

**What does hold cleanly, at every n tested, on both datasets, with no exceptions**: z_score
detects stalls above chance and fails bursts below chance; log_ratio detects bursts above chance.
That is the paper's core, load-bearing claim — feature geometry must match fault geometry — and it
is not sensitive to the log_ratio/PCA instability above.

## 6. Threshold-independent evaluation (AUC-PR / AUC-ROC)

Every result above is **lift**, which depends on each detector's own threshold — and those
thresholds are not matched: `z_score_threshold` uses a fixed `|z|>3` rule that flags ~49% of
everything, while `log_ratio_threshold` and both ML detectors are calibrated to their own
train-normal 95th percentile (~5–13% base rate, see §8). A detector with a permissive threshold can
post a lift>1 by flagging so much that it's bound to catch a fair share of the (rare) positives,
without its underlying score actually ranking anomalous cells much higher than normal ones. AUC-PR
and AUC-ROC (`src/auc_metrics.py`, `results/auc_metrics.csv`) score each detector's raw continuous
score (PCA reconstruction error, `|z_score|`, `|log_ratio|`, isolation-forest anomaly score) with no
threshold at all, aggregated onto the identical 60s grid via max-score-per-cell. Because positives
are rare (~100 injections over ~2.8–2.9M grid cells per run), AUC-PR is reported alongside its
no-skill baseline (positive-class prevalence) and their ratio, so the numbers are readable.

| dataset | fault | detector | lift | AUC-PR ratio | AUC-ROC |
|---|---|---|---|---|---|
| BGL | stall | PCA | 0.72 | 0.85 | 0.407 |
| BGL | stall | z_score | **1.33** | 1.04 | 0.537 |
| BGL | stall | log_ratio | 0.95 | 0.91 | 0.465 |
| BGL | stall | isoforest | 1.53 | 1.10 | 0.467 |
| BGL | burst | PCA | 0.52 | 0.80 | 0.331 |
| BGL | burst | z_score | 0.50 | 0.80 | 0.428 |
| BGL | burst | log_ratio | **7.29** | **16.10** | **0.673** |
| BGL | burst | isoforest | 0.99 | 0.86 | 0.408 |
| Thunderbird | stall | PCA | 1.98 | 1.59 | **0.613** |
| Thunderbird | stall | z_score | 1.39 | 1.44 | 0.587 |
| Thunderbird | stall | log_ratio | 1.42 | 1.17 | 0.520 |
| Thunderbird | stall | isoforest | 6.70 | 3.27 | 0.545 |
| Thunderbird | burst | PCA | 1.01 | 1.28 | 0.569 |
| Thunderbird | burst | z_score | 0.35 | 0.80 | 0.294 |
| Thunderbird | burst | log_ratio | **2.75** | 1.71 | **0.637** |
| Thunderbird | burst | isoforest | 11.96 | 5.22 | 0.574 |

**Verdict: PARTIALLY CONFIRMED — the core claim survives, but not unscathed.** Three specific
disagreements between lift and AUC, none smoothed over:

1. **BGL-stall `z_score`'s apparent strength was partly a calibration artifact.** Lift=1.33 with
   91/100 detected reads as decisive; AUC-ROC=0.537 shows the underlying continuous score barely
   ranks anomalous cells above normal ones — barely better than a coin flip. The permissive `|z|>3`
   threshold buys high recall cheaply. The stall-vs-burst *direction* still holds (z_score's AUC-ROC
   is higher on both stall cells than both burst cells), but "strong on stalls" should be downgraded
   to "a real but modest positive signal."
2. **`isolation_forest_counts`' high-lift cells are confirmed unreliable, not just suspected.**
   Every cell where it posted a large lift (BGL-stall 1.53, Thunderbird-stall 6.70, Thunderbird-burst
   11.96) has an AUC-ROC of 0.47–0.57 — near or barely above random. This is independent
   confirmation of the "dead baseline, zero credibility" conclusion already in §8, not a new problem.
3. **`log_ratio` on stall is weak under AUC-ROC on *both* datasets** (0.465 BGL, 0.520 Thunderbird)
   regardless of what lift said in either direction (0.95 BGL "near chance", 1.42 Thunderbird "above
   chance"). This *resolves* rather than deepens the §5 ambiguity: log_ratio has no reliable ranking
   signal on stalls, full stop, on either dataset.
4. **Thunderbird-stall PCA (AUC-ROC=0.613) is one of the strongest rankings in the entire table** —
   this outright contradicts "content detectors are structurally blind to timing faults" as a
   general claim. That framing is **BGL-specific**, not universal; Thunderbird's PCA has genuine,
   threshold-independent ranking power on timing-injected data that BGL's does not (BGL PCA AUC-ROC:
   0.407 stall, 0.331 burst — both clearly below random).

**What survives everything — lift, AUC-PR, AUC-ROC, both datasets**: `log_ratio_threshold` detects
bursts with real, robust, above-chance ranking power (AUC-ROC 0.673 BGL / 0.637 Thunderbird, AUC-PR
ratio 16.1 / 1.71). This is the one finding in the whole project that every metric agrees on — the
strongest candidate for the paper's headline claim.

## 7. HDFS: scope boundary (characterization only)

HDFS was investigated for premise-audit characterization **only** — no injector, no detectors, by
explicit design decision. This is not an oversight; it reflects a structural finding that HDFS
doesn't fit this project's injection design as-is.

| signature | BGL | Thunderbird | HDFS |
|---|---|---|---|
| new_or_rare_template | 99.31% | 99.99% | **4.53%** |
| pure_order_anomaly | 0.52% | 38.97% | 0.00%* |
| timing_gap_anomaly | 46.97% | 49.84% | 59.27%** |

\* Not meaningfully evaluable: HDFS labels are uniform per block (an anomalous block has zero
normal rows by construction, confirmed directly), so the signature's eligibility check — which
requires the same node/block to have also produced the template *normally* — structurally excludes
99.9% of anomalous rows. 0% here is an artifact of block-level labeling, not evidence of clean
ordering.

\** Not trustworthy as a real statistical signal, for the same root cause: 100% of anomalous blocks
have zero normal-sequence rows, so all of them fall back to the pooled baseline, which is itself
degenerate (median=MAD=0) and floored only by an arbitrary constant — at this floor, almost any
nonzero gap trivially "fires."

**The one clean, robust HDFS number**: new_or_rare_template at 4.53% — the near-total *inverse* of
BGL/Thunderbird. HDFS's native anomaly labels come from block write-pipeline failures (missing
acknowledgments, wrong replica counts, incomplete completion sequences), a fundamentally different
anomaly *mechanism*: about the shape/completeness of a block's event sequence, not any individual
line's content or timing.

**Why no injector was built for HDFS**: the same per-node continuous-stream design used for
BGL/Thunderbird doesn't transfer — HDFS is block-session structured, only 14.24% of blocks have a
valid, non-degenerate per-block timing baseline even restricting to within-block injection, and the
premise-audit signatures used to validate injection quality on the other two datasets hit the same
structural walls described above when applied to HDFS's labeling scheme. HDFS's actual blind spot,
if any, looks like a **third category — sequence-completeness / missing-event anomalies** —
distinct from both the content detectors and the timing detectors this project built. Full detail
in `results/hdfs_setup_notes.md`; this section is the scope boundary, not a plan to extend into
HDFS injection under the current design.

## 8. Honest open items

- **BGL-stall log_ratio and both Thunderbird-PCA cells are not decisive** — see the stability table
  in §5. Do not cite these three cells as either a positive or negative finding without the caveat.
- **log-ratio vs z-score calibration is not apples-to-apples — now partially addressed by §6.**
  `z_score_threshold` uses a fixed `|z|>3` rule that flags ~49% of everything on both datasets
  (heavy-tailed inter-arrival distributions). `log_ratio_threshold` is calibrated from the
  train-normal 95th percentile, giving a disciplined ~6–13% base rate. This was flagged as a reason
  lift comparisons might not be apples-to-apples — §6's AUC-PR/AUC-ROC pass (threshold-free by
  construction) confirms the concern was real: BGL-stall z_score's lift advantage over log_ratio
  (1.33 vs 0.95) shrinks to near-parity under AUC-ROC (0.537 vs 0.465), and z_score's own AUC-ROC is
  barely above random despite the favorable lift. AUC-PR/ROC doesn't fully replace a matched-base-rate
  head-to-head (still not done — see Future Work), but it independently corroborates that some of
  z_score's lift-based advantage was a calibration effect, not a ranking-quality effect.
- **The isolation-forest-wrapped timing detector was tried and abandoned.** Feature-scaling was
  diagnosed and fixed, but it still substantially underperforms the trivial `z_score_threshold` on
  stalls. Not pursued further — `z_score_threshold` and `log_ratio_threshold` are the timing
  detectors actually used in every reported result.
- **`isolation_forest_counts` (the "dead baseline") has zero credibility as a working detector**
  (`results/iforest_diagnosis.md`: AUC=0.507, chance-level ranking) regardless of what its lift
  number says in any given table — including BGL-burst's 0.99 (essentially exactly 1.0, consistent
  with pure noise averaging out as n grows) and Thunderbird's noticeably higher numbers (6.70
  stall, 11.96 burst), which have **not** been mechanistically checked the way BGL's was at n=15
  (`results/iforest_diagnosis.md` predates the n-scaling and Thunderbird work) — treat both as
  reported-for-completeness, not as evidence, until re-audited.
- **n=100 is now locked** for both datasets and both fault types, per explicit decision after
  observing that n=15 and even n=40 were too small to distinguish real signal from sampling noise
  on several cells (§5). Any future re-scaling should re-run the full stability check, not assume
  n=100 is automatically sufficient.

## Exact configuration

- **Datasets**:
  - **BGL** (Blue Gene/L), `data/raw/BGL.log`, 4,747,963 lines, 348,460 (7.34%) natively labeled
    anomalous. Parsed with Drain3 → `data/processed/bgl_parsed.parquet`.
  - **Thunderbird**, representative 10,000,000-line slice (lines 50,000,001–60,000,000 of the
    source archive, chosen for label/node diversity — see `results/thunderbird_setup_notes_v2.md`),
    170,422 (1.70%) natively labeled anomalous, 4,722 nodes. Parsed with the same Drain3 pipeline →
    `data/processed/thunderbird_parsed.parquet`.
- **Chronological split**: first 70% of rows by timestamp = train (fitting only), never random, on
  both datasets.
- **Common evaluation grid**: fixed-time, **60 seconds**, per node (`src/metrics.py`). Stability-
  checked at 30s/60s/120s on BGL before certification (`results/window_size_stability.csv`).
- **Per-node timing baseline**: median + `1.4826×MAD` per node. BGL: nodes with <10 normal-to-normal
  gaps fall back to a pooled/global baseline (healthy, ~33% of nodes). Thunderbird: pooled fallback
  is degenerate (>50% of all gaps are exactly 0), so injection eligibility is instead restricted to
  the 73.06% of nodes with a valid non-degenerate per-node baseline (`require_valid_baseline=True`,
  `exclude_zero_from_pooled=True`).
- **Stall injection**: `SEED=42`, **100 injections**, one per distinct eligible node, `intensity ∈
  {10,15,20,25,30}` (multiples of node scale, additive), node eligibility ≥30 events (+ valid
  baseline on Thunderbird), injection point restricted to non-outlier pre-existing gaps (`|z|≤3`).
- **Burst injection**: `SEED=43`, **100 injections**, one per distinct eligible node, `intensity ∈
  {10,15,20,25,30}` (compression factor, multiplicative), `burst_length ∈ {10,15,20,25}` consecutive
  gaps, node eligibility ≥55 events (+ valid baseline on Thunderbird), same non-outlier eligibility
  filter applied to all gaps in the burst window.
- **Detector fitting**: PCA (`svd_solver="full"`, `random_state=0`) and IsolationForest
  (`n_estimators=100`, `random_state=0`) both fit on clean train-period, native-label-normal data
  only; count-vector vocabulary = top-300 templates from train. Decision thresholds: 95th percentile
  of train-normal scores (ML detectors), fixed `Z_THRESH=3` (`z_score_threshold`), 95th-percentile of
  train-normal `|log_ratio|` (`log_ratio_threshold`).

## Reproducibility

Every result in `results/` and `figures/` regenerates from raw data via this sequence (run from the
repo root, with `.venv` activated):

```
# BGL
python src/parser.py
python src/premise_audit.py
python src/run_baseline_detectors.py
python src/rescore_common_unit.py
python src/window_size_stability.py
python src/diagnose_iforest.py
python src/injector.py --type stall
python src/injector.py --type burst
python src/validate_injection.py --type stall
python src/validate_injection.py --type burst
python src/core_result_stall.py
python src/core_result_stall_final.py
python src/core_result_burst.py
python src/core_result_v2_symmetric.py

# Thunderbird
python src/parser_thunderbird.py
python src/premise_audit_thunderbird.py
python src/injector_thunderbird.py --type stall
python src/injector_thunderbird.py --type burst
python src/validate_injection_thunderbird.py --type stall
python src/validate_injection_thunderbird.py --type burst
python src/core_result_thunderbird.py

# Threshold-independent metrics (both datasets, both fault types -- see §6)
python src/auc_metrics.py

# Consolidation (both datasets; consolidate_final_results.py merges in auc_metrics.csv)
python src/consolidate_final_results.py
python src/compare_bgl_thunderbird.py

# HDFS (characterization only — no injector/detectors, per §7)
python src/parser_hdfs.py
python src/premise_audit_hdfs.py
```

All injection is fully seeded (`SEED=42` stall, `BURST_SEED=43` burst, same on both datasets) —
identical output on rerun, verified directly this pass by rerunning three of the detector scripts
fresh mid-session and confirming bit-identical lift values.

**Manual/frozen exceptions** (intentional, not bugs):
- `results/premise_audit.csv` (v1) and `results/thunderbird_premise_audit.csv` (v1) are **frozen
  historical snapshots** (pre-fix and pre-representative-slice respectively) — preserved for
  before/after comparison, not regenerated by current scripts.
- `results/core_result_stall.csv` (v1) is a **frozen snapshot** from before the timing-detector
  feature-scaling fix — preserved for comparison, not regenerated.
- Dependencies (`drain3`, `scikit-learn`, `matplotlib`, `pandas`, `pyarrow`) must be installed in
  `.venv` first; no other manual steps.

## Future work (not yet done)

- Additional fault types: slowdown (gradual stretch, distinct from stall's step-function gap) and
  jitter (added noise without a mean shift).
- A fair, matched-calibration head-to-head between `z_score_threshold` and `log_ratio_threshold`
  (same base-rate target) to isolate the geometry effect from the calibration effect noted in §8 —
  §6's AUC-PR/AUC-ROC pass is a threshold-free proxy for this, not a substitute.
- Re-audit `isolation_forest_counts` on Thunderbird the way `results/iforest_diagnosis.md` did for
  BGL — §6 already shows near-random AUC-ROC on its high-lift cells, but a full mechanistic
  diagnosis (the kind done for BGL) hasn't been done.
- Realism calibration against LO2 / AnoMod or similar reference injectors.
- HDFS: either a sequence-completeness detector family, or a narrower within-block timing-injection
  design restricted to the 14.24% valid-baseline subset — both open, per §7.
