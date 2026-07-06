# TILT-Bench: Findings Summary

Plain-language consolidation of the project to date. No new analysis — every number here is
already computed and saved in `results/`; see `results/FINAL_results.csv` for the master table.

## 1. The premise

Standard log-anomaly-detection benchmarks are evaluated almost entirely against **content**-based
anomalies (new/rare event templates, broken sequence order). We asked: does BGL's own native
anomaly labeling already contain timing-only faults, or is timing a genuine blind spot in how
these benchmarks are built?

The premise audit (`src/premise_audit.py`, `results/premise_audit_v2.csv`,
`results/premise_audit_summary.md`) checked every one of BGL's 348,460 natively-labeled anomalous
rows for three independent signatures: new/rare template, "pure" order anomaly (unusual sequence
with otherwise-known content), and timing-gap anomaly. Result: **99.3%** show a new/rare template,
only **0.52%** show a pure order anomaly once content novelty is properly controlled for (a v1 of
this audit had conflated the two — order novelty and content novelty were almost the same 348k
rows; the v2 fix isolated them), and **47.0%** show a timing-gap anomaly — but timing almost never
appears without content novelty riding along (only 3 of 348,460 rows are timing-only). BGL's native
labels are overwhelmingly content-defined. This motivated building a controlled timing-fault
injector rather than relying on BGL's native anomaly mix to evaluate timing detection.

## 2. The injector and its validation

`src/injector.py` implements two fault types, both operating per-node (BGL is a continuous
per-node stream) and both provably order-preserving (a stall shifts every downstream timestamp by
a constant; a burst compresses a run of gaps then shifts everything downstream by the exact time
saved — both are single-constant shifts to a whole tail of the sequence, which can never invert
order).

- **Stall**: insert a large gap = `intensity × node_scale` (additive), where `node_scale` is the
  same per-node MAD-derived scale (with pooled fallback for low-data nodes) used throughout the
  audit and detectors.
- **Burst**: compress `L` consecutive gaps by `new_gap = orig_gap / intensity` (multiplicative),
  then shift everything downstream backward by the total time saved.

Two validations were run before trusting any detection result:
- **Ordering**: automated check across the *entire* injected dataset (not just injected nodes) —
  **0 violations**, for both stall and burst, out of 4,747,963 rows.
- **Instrument validation**: re-ran the premise-audit signatures on the injected rows themselves.
  Stall: timing fired on **100%** of injections, content/order stayed silent (6.7% / 0%, both
  explained as pre-existing coincidence, not injector leakage — see below). Burst: timing fired on
  only **15.1%** of injected rows, content/order silent (0% / 0%) — this lower timing-fire rate on
  bursts turned out to be the first sign of the feature-geometry issue described in §5, not a
  problem with the injector itself.

One coincidence was investigated rather than assumed benign: 1 of 15 stall injections tripped the
template-rarity signature. Checked directly — the injected row's own (untouched) content was a
real, pre-existing "Lustre mount FAILED" incident already present in BGL's native labels at that
exact node/time. Confirmed as coincidence, not a labeling leak, before proceeding.

## 3. The stall result

Fit on clean (pre-injection) train-period data, detectors were run against the 15 injected stalls,
scored on the injected-span ground truth (not BGL's native labels), on a common 60-second per-node
grid so every detector is comparable.

| detector | detection rate (of 15) | lift |
|---|---|---|
| z_score_threshold (timing) | **14/15 (93.3%)** | **1.55** |
| count_vector_pca (content) | 7/15 (46.7%) | 1.09 (≈chance) |
| isolation_forest_counts (dead baseline) | 1/15 (6.7%) | 0.87 (below chance) |

**Lift** = recall ÷ the detector's own indiscriminate flagging rate — the metric that separates a
real signal from "flags so much of everything that it hits small spans by chance." The timing
detector shows real, above-chance detection; the content detector does not (lift ≈1 means its
apparent recall is no better than its own base flagging rate would already predict). Combining
content + timing (Jaccard 0.25) reaches 90% fusion recall — genuine complementarity.

Note: an isolation-forest-wrapped version of the timing detector was also built and tuned (feature
scaling was diagnosed and fixed after it showed lift <1 despite a strong raw signal), but even
after the fix it substantially underperformed `z_score_threshold` (see §6). `z_score_threshold` —
a trivial `|z|>3` rule, no ML — is the timing detector actually used in every reported result.

## 4. The burst result

Same setup, burst-injected data:

| detector | detection rate (of 15) | lift |
|---|---|---|
| z_score_threshold (timing, additive) | 6/15 (40.0%) | **0.46 (below chance)** |
| count_vector_pca (content) | 6/15 (40.0%) | 0.73 (≈chance) |
| isolation_forest_counts (dead baseline) | 7/15 (46.7%) | 3.15 (checked, likely sampling noise — see §6) |

The z-score timing detector, which worked cleanly on stalls, **failed on bursts** — this was not
assumed to be "bursts are just harder" and left at that; it was traced to a specific mechanism
(§5).

## 5. The feature-geometry finding

`z_score = (gap − node_median) / node_scale` is **additive**. A stall adds `intensity × node_scale`
directly, so its z-score is unbounded by construction — bigger intensity, bigger z-score, always
detectable given enough intensity. A burst instead **compresses** the gap
(`new_gap = orig_gap / intensity`); its z-score is bounded by roughly `−(node_median / node_scale)`
as intensity → ∞, regardless of how aggressive the compression is. Measured directly: BGL nodes
have a strikingly consistent `median/MAD ≈ 0.675`, meaning no compression, however extreme, can
push a typical gap's z-score much past ≈ −0.68 — structurally far below the `|z|>3` threshold. This
explains both the burst instrument-validation rate (15.1%, §2) and the failed burst detection
(§4) with the same single mechanism.

Fix tested: `log_ratio = log(gap / node_median)` is **symmetric** — a stall gives a large positive
value, a burst gives a large negative value of comparable magnitude, since "N× larger" and "N×
smaller" are equal-magnitude opposite-sign deviations in log space.

| fault | z_score_threshold lift | log_ratio_threshold lift |
|---|---|---|
| stall | 1.55 | 2.25 |
| burst | **0.46** | **7.33** |

**The symmetric feature catches both fault types with real, above-chance lift; the additive
z-score only worked in one direction.** Content detectors (PCA) stayed at/near chance on both
fault types (1.09 stall, 0.73 burst) — the blind spot is structural to content detection, not an
artifact of fault type. Feature geometry must match fault geometry: this is the headline
methodological finding of the project so far.

## 6. Honest open items

- **log-ratio vs z-score calibration is not apples-to-apples.** `z_score_threshold` uses a fixed
  `|z|>3` rule that turns out to flag ~49% of everything (BGL's inter-arrival distribution is
  heavy-tailed enough that a fixed 3-sigma-style cutoff isn't well-controlled). `log_ratio_threshold`
  is calibrated from the train-normal 95th percentile, giving a genuinely disciplined ~5-6% base
  rate. This is why log-ratio shows *higher lift* on stalls (2.25 vs 1.55) while *detecting fewer*
  of them in absolute terms (3/15 vs 14/15) — it's stricter, not worse. A fair head-to-head would
  calibrate both the same way; this hasn't been done.
- **The isolation-forest-wrapped timing detector is still weak.** Feature scaling was diagnosed and
  fixed (raw-seconds features were drowning the properly-normalized z-score), which measurably
  improved its lift (0.83 → 1.67 on stalls), but it still catches far fewer stalls (2/15) than the
  trivial `z_score_threshold` (14/15) it's supposed to generalize. Not pursued further per explicit
  scope decision — `z_score_threshold` is the working timing detector going forward.
- **isolation_forest_counts' burst lift (3.15) is likely noise, not signal.** It was checked (not
  taken at face value): the hypothesized "windowing cramming" mechanism doesn't hold up
  (burst-affected grid cells average 1.08 distinct native windows vs 1.03 for normal cells — not
  enough to explain a 3x effect), and the true-positive sample is tiny (~83 grid cells total). This
  detector has zero credibility as a working baseline (`results/iforest_diagnosis.md`: AUC=0.507,
  chance-level ranking) — its burst number is reported for completeness, not as evidence.
- **Only one dataset.** Everything here is BGL. No cross-dataset check yet (see Future Work).

## Exact configuration

- **Dataset**: BGL (Blue Gene/L), `data/raw/BGL.log`, 4,747,963 lines, 348,460 (7.34%) natively
  labeled anomalous. Parsed with Drain3 → `data/processed/bgl_parsed.parquet`.
- **Chronological split**: first 70% of rows by timestamp = train (fitting only), never random.
- **Common evaluation grid**: fixed-time, **60 seconds**, per node (`src/metrics.py`:
  `EVAL_WINDOW_SCHEME="fixed_time"`, `EVAL_WINDOW_SIZE=60`). Stability-checked at 30s/60s/120s
  before being certified (`results/window_size_stability.csv`) — ordering and qualitative story
  held at all three; 60s was adopted as the reference.
- **Per-node timing baseline**: median + `1.4826×MAD` per node; nodes with <10 normal-to-normal
  gaps fall back to a pooled/global baseline (~33% of nodes, confirmed in the premise audit and
  reused identically in the injector and every timing detector).
- **Stall injection**: `SEED=42`, 15 injections, one per distinct node, `intensity ∈
  {10,15,20,25,30}` (multiples of node scale, additive), node eligibility ≥30 events, injection
  point restricted to non-outlier pre-existing gaps (`|z|≤3`).
- **Burst injection**: `SEED=43`, 15 injections, one per distinct node, `intensity ∈
  {10,15,20,25,30}` (compression factor, multiplicative), `burst_length ∈ {10,15,20,25}` consecutive
  gaps, node eligibility ≥55 events, same non-outlier eligibility filter applied to all gaps in the
  burst window.
- **Detector fitting**: PCA (`svd_solver="full"`, `random_state=0`) and IsolationForest
  (`n_estimators=100`, `random_state=0`) both fit on clean train-period, BGL-label-normal data only;
  count-vector vocabulary = top-300 templates from train. Decision thresholds: 95th percentile of
  train-normal scores (ML detectors), fixed `Z_THRESH=3` (`z_score_threshold`), 95th-percentile of
  train-normal `|log_ratio|` (`log_ratio_threshold`).

## Reproducibility

Every result in `results/` and `figures/` regenerates from raw BGL via this sequence (run from the
repo root, with `.venv` activated):

```
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
python src/core_result_v2_symmetric.py
python src/consolidate_final_results.py
```

**Manual/frozen exceptions** (intentional, not bugs):
- `results/premise_audit.csv` (v1) is a **frozen historical snapshot** from before the
  `pure_order_anomaly` fix. The current `src/premise_audit.py` only produces v2
  (`premise_audit_v2.csv`); v1 is preserved on disk for the before/after comparison and is not
  regenerated by any script.
- `results/core_result_stall.csv` (v1) is similarly a **frozen snapshot** from before the
  timing-detector feature-scaling fix. Running `src/core_result_stall.py` today produces v2
  (`core_result_stall_v2.csv` + the feature diagnostic) — v1 is preserved for comparison, not
  regenerated.
- `results/core_result_burst.csv` is an **intermediate, superseded artifact** from the
  `z_score_threshold`-only burst experiment. It's still reproducible by running
  `src/core_result_burst.py` directly, but the final pipeline above regenerates the equivalent
  (and more complete) numbers via `core_result_v2_symmetric.py`, so that separate script is not
  part of the required sequence.
- Dependencies (`drain3`, `scikit-learn`, `matplotlib`, `pandas`, `pyarrow`) must be installed in
  `.venv` first; no other manual steps.

## Future work (not yet done)

- Additional fault types: slowdown (gradual stretch, distinct from stall's step-function gap) and
  jitter (added noise without a mean shift).
- A second dataset (Thunderbird or HDFS) to check whether the content-blind-spot / feature-geometry
  findings generalize beyond BGL's specific gap-distribution shape (the `median/MAD ≈ 0.675`
  property that drove the z-score/log-ratio asymmetry is itself a BGL-specific measurement).
- Realism calibration against LO2 / AnoMod or similar reference injectors, to characterize how
  representative these synthetic intensities are of real-world timing faults.
- A fair, matched-calibration head-to-head between `z_score_threshold` and `log_ratio_threshold`
  (same base-rate target) to isolate the geometry effect from the calibration effect noted in §6.
