# TILT-Bench: Findings Summary

Plain-language consolidation of the project to date, at **n=100 injections per fault type per
dataset**, injected under a **rate-gated eligibility rule** (§2b) that replaced a count-only rule
after a design flaw was diagnosed. Every number in this document reflects the rate-gated
injections; see `results/FINAL_results.csv` (BGL) and `results/thunderbird_vs_bgl_comparison.csv`
(both datasets) for the master tables, and `results/stall_labeling_diagnostic.md` for the diagnostic
that preceded this fix.

**Numbers changed substantially in this pass — several findings from the count-only-eligibility
version of this document no longer hold, and some flipped direction.** §2b documents the fix and
what it did and did not accomplish; §5 documents which parts of the feature-geometry story survived
and which didn't.

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

**Node-eligibility check at n=100**, under the current rate-gated rule (§2b; supersedes an earlier
count-only version of this table):

| dataset | fault | eligible nodes | injections drawn | distinct nodes used | used_pooled_fallback |
|---|---|---|---|---|---|
| BGL | stall | 7,379 | 100 | 100 | 0 (checked directly) |
| BGL | burst | 6,469 | 100 | 100 | n/a (BGL doesn't require it; not a gate) |
| Thunderbird | stall | 741 | 100 | 100 | 0 (structurally guaranteed — `require_valid_baseline=True`) |
| Thunderbird | burst | 727 | 100 | 100 | 0 (structurally guaranteed) |

All four combinations still clear 100 eligible nodes with comfortable headroom (7.3×–74× the
requested count) even after the stricter rate gate — no marginal/low-quality-baseline nodes were
needed.

**Two validations were run before trusting any detection result, at n=100, all four
dataset×fault-type combinations:**

- **Ordering**: automated check across the *entire* injected dataset (not just injected nodes) —
  **0 violations** in every case: BGL 4,747,963 rows (×2 fault types), Thunderbird 10,000,000 rows
  (×2 fault types).
- **Instrument validation**: re-ran the premise-audit signatures on the injected rows themselves.
  Expectation: timing fires, content/order stay silent.

| dataset | fault | timing_gap_anomaly | new_or_rare_template | pure_order_anomaly |
|---|---|---|---|---|
| BGL | stall | **97.0%** | 3.0% | 0.0% |
| BGL | burst | 12.3% | 1.1% | 0.0% |
| Thunderbird | stall | **100.0%** | 1.0% | 0.0% |
| Thunderbird | burst | 8.9% | 2.2% | 0.0% |

Content/order stay essentially silent everywhere (≤3%), confirming clean injections on both
datasets. The lower timing-fire rate on bursts (12.3% BGL, 8.9% Thunderbird — both far below
stall's ~100%) is not an injector defect; it's the first visible sign of the feature-geometry issue
in §5 (an additive z-score is structurally bounded against multiplicative compression).

## 2b. Eligibility fix: from event-count to event-count + duration gating

**The problem.** Node eligibility originally gated on event *count* only (`MIN_NODE_EVENTS≥30` /
`BURST_MIN_NODE_EVENTS≥55`), not event *rate*. A node could clear the count threshold with its
events spread across weeks or months. Since `delta_seconds = intensity × mad_eff` (`mad_eff` = the
node's own MAD-derived baseline scale), a sparse node with a large `mad_eff`, hit with the max
drawn intensity (30), produced faults spanning weeks of wall-clock time — not physically plausible
as a transient "timing anomaly," and diagnosed (`results/stall_labeling_diagnostic.md`) as
concentrating almost all injected-span ground truth into a handful of grid cells per injection
(BGL-stall: worst case 20,243 grid cells / ~14 days from one injection).

**The fix.** Added a rate/density gate: node eligibility now additionally requires
`mad_eff ≤ MAX_MAD_EFF_S = 120s`. This value is derived, not tuned to results: chosen from
`MAX_PLAUSIBLE_FAULT_DURATION_S = 3600s` (1 hour — a domain judgment that a *transient* timing
fault on an actively-monitored node, as opposed to an extended outage, should resolve on this
timescale) divided by `max(INTENSITY_CHOICES) = 30`. This guarantees, by construction, that
**stall** duration can never exceed 1 hour: `delta_seconds = intensity × mad_eff ≤ 30 × 120s =
3600s`. Headroom was checked *before* implementing (§2's table) — all four combinations cleared 100
nodes by 7×–74× before the fix was written.

**Duration distribution, before → after** (cells/injection at 60s/cell; 8,938 cells ≈ 149 hours):

| dataset | fault | median cells/injection (old → new) | max cells/injection (old → new) |
|---|---|---|---|
| BGL | stall | 1,146.71 → **20** | ~20,243 → **55** |
| BGL | burst | 290.50 → **2** | 20,243 → **8,938** (see caveat below) |
| Thunderbird | stall | 344.72 → **5.5** | (large) → **45** |
| Thunderbird | burst | 5 → **1** | 169 → **159** |

**Stall's guarantee held exactly**: worst-case duration dropped from ~14 days to 55 minutes,
comfortably under the 1-hour cap, on both datasets. **Burst's guarantee is only approximate, and
the gap shows up in the data**: burst compresses *observed* local gaps, not `mad_eff` itself, so a
node with low dispersion (small `mad_eff`, passes the gate) can still have a large *median* gap
(infrequent but very regular ticking) — compressing 25 such gaps by only 10× can still produce a
multi-hour-to-multi-day span. BGL-burst's max case is still 8,938 cells (~149 hours) post-fix, down
from ~20,243 (~14 days) but nowhere near the 1-hour target. This was flagged as a known limitation
in the fix's own design comment before implementation, not discovered after the fact — `mad_eff`
(dispersion) and node *median* gap (density/rate) are different quantities, and gating on the
former does not exactly bound the latter's contribution to burst duration. **Not fixed as of this
pass** — see Future Work.

**Did detectable positive-cell counts rise, as hypothesized?** No — this is the central,
counterintuitive finding of this fix, reported plainly per instruction not to smooth it over:

| dataset | fault | detectable positive cells (old → new) | change |
|---|---|---|---|
| BGL | stall | 196 → **187** | flat (−4.6%) |
| BGL | burst | 780 → **250** | **−68%** |
| Thunderbird | stall | 196 → **181** | flat (−7.7%) |
| Thunderbird | burst | 529 → **251** | **−53%** |

The motivating hypothesis — that pathologically long stall spans were *suppressing* detectable
signal by burying it in a sea of unlabeled quiet-middle cells — does not hold. `results/
stall_labeling_diagnostic.md` had already shown those zero-row cells are structurally excluded from
every metric (they can't be scored, so they were never counted as false negatives either). What
*does* explain the flat stall numbers: a stall's detectable signal is bounded by roughly "one grid
cell on each side of the gap," a property of the injection mechanism itself, independent of how
long the gap lasts — fixing duration doesn't add detectable cells because duration was never what
determined that count. Burst's *drop* is a genuine side effect of the gate, not noise: excluding
high-`mad_eff` nodes removes some nodes whose sparse-but-irregular gaps, once compressed, happened
to land many rows inside the compressed window; the newly-eligible low-`mad_eff` (regular) nodes
produce tighter, smaller compressed windows with fewer rows in them. **The rate gate solved the
physical-plausibility problem it was built for. It did not solve, and was never going to solve, a
different problem (detectable-cell scarcity) that turned out not to share the same root cause.**

## 3. The stall result

Fit on clean (pre-injection) train-period data, detectors run against the 100 injected stalls per
dataset (rate-gated eligibility, §2b), scored on the injected-span ground truth (not native
labels), on a common 60-second per-node grid.

| dataset | detector | detection rate (of 100) | lift |
|---|---|---|---|
| BGL | count_vector_pca (content) | 51/100 (51%) | **1.28** |
| BGL | z_score_threshold (timing) | 54/100 (54%) | 0.76 (below chance) |
| BGL | isolation_forest_counts (dead baseline) | 17/100 (17%) | **1.88** |
| Thunderbird | count_vector_pca (content) | 35/100 (35%) | **12.32** |
| Thunderbird | z_score_threshold (timing) | 60/100 (60%) | 1.07 (≈chance) |
| Thunderbird | isolation_forest_counts (dead baseline) | 30/100 (30%) | **22.22** |

**This table looks nothing like the count-only-eligibility version of this document, and the
reversal is real, not noise — confirmed under threshold-independent AUC-ROC scoring too (§6).**
`z_score_threshold`'s stall advantage, previously the paper's cleanest single result (1.33 lift,
91/100 detected), **collapses on BGL** (0.76, below chance) once injections land on genuinely
active nodes instead of pathologically sparse ones — its AUC-ROC crosses from 0.537 to 0.464,
confirming this is a real loss of ranking power, not a threshold artifact. It holds up only weakly
on Thunderbird (1.07, essentially chance). Meanwhile `count_vector_pca`, previously weak-to-mixed on
stalls, is now clearly above chance on **both** datasets. See §5 for the mechanism.

## 4. The burst result

Same setup, burst-injected data:

| dataset | detector | detection rate (of 100) | lift |
|---|---|---|---|
| BGL | log_ratio_threshold (timing, symmetric) | 47/100 (47%) | **5.14** |
| BGL | count_vector_pca (content) | 63/100 (63%) | **1.48** |
| BGL | z_score_threshold (timing, additive) | 34/100 (34%) | 0.75 (below chance) |
| BGL | isolation_forest_counts (dead baseline) | 32/100 (32%) | **2.72** |
| Thunderbird | log_ratio_threshold (timing, symmetric) | 1/100 (1%) | 0.03 (near-zero — see §6, threshold artifact) |
| Thunderbird | count_vector_pca (content) | 50/100 (50%) | **11.79** |
| Thunderbird | z_score_threshold (timing, additive) | 41/100 (41%) | **1.20** |
| Thunderbird | isolation_forest_counts (dead baseline) | 48/100 (48%) | **17.44** |

`log_ratio_threshold` still detects bursts on BGL with strong, above-chance lift (5.14, down from
7.29 but decisively real) — the paper's most robust single finding survives. **Thunderbird's
log_ratio-burst lift collapsed to near-zero (0.03, 1/100 detected) — but §6's AUC-ROC shows this is
a threshold-calibration artifact, not a loss of signal** (AUC-ROC 0.598, comfortably above random,
essentially unchanged from 0.637 pre-fix). The fixed 95th-percentile threshold, calibrated once
from the whole clean dataset, no longer matches the smaller (but still real) magnitude of
compression on now-regular, low-`mad_eff` nodes.

**The clean "additive z-score fails on bursts on both datasets" claim from the earlier pass no
longer holds as stated.** `z_score_threshold`-burst is still below chance on BGL (0.75) but has
flipped to **above chance on Thunderbird** (1.20, up from 0.35) — confirmed by AUC-ROC too (0.294 →
0.594). This directly contradicts part of §5's original mechanism claim and is investigated (not
yet resolved) in §5 below.

## 5. The feature-geometry / complementarity finding

`z_score = (gap − node_median) / node_scale` is **additive**. A stall adds `intensity × node_scale`
directly, so its z-score is unbounded by construction. A burst instead **compresses** the gap
(`new_gap = orig_gap / intensity`); its z-score is bounded by roughly `−(node_median / node_scale)`
as intensity → ∞, regardless of compression severity. This mechanism is unchanged and still
correct as a description of the math. **What changed is which nodes get injected, and that turned
out to matter more than the mechanism itself predicted.**

Fix tested: `log_ratio = log(gap / node_median)` is **symmetric** — a stall gives a large positive
value, a burst gives a large negative value of comparable magnitude.

**Lift by detector × fault × dataset, at n=100, rate-gated eligibility (bold = above chance):**

| dataset | fault | PCA (content) | z_score (timing, additive) | log_ratio (timing, symmetric) |
|---|---|---|---|---|
| BGL | stall | **1.28** | 0.76 | 0.27 (well below chance) |
| BGL | burst | **1.48** | 0.75 | **5.14** |
| Thunderbird | stall | **12.32** | 1.07 | 0.09 (near-zero — see AUC-ROC caveat below) |
| Thunderbird | burst | **11.79** | **1.20** | 0.03 (threshold artifact — AUC-ROC 0.598, see §6) |

**This is a fundamentally different table from the count-only-eligibility version. Reporting
plainly, per instruction, rather than smoothing over what broke:**

1. **"z_score is cleanly stall-only" no longer holds.** Previously above chance on both stall cells
   with no exception; now below chance on BGL-stall (0.76) and barely at chance on
   Thunderbird-stall (1.07). Confirmed as a real effect, not noise, by AUC-ROC (§6: BGL-stall
   z_score AUC-ROC crossed 0.537 → 0.464, below random). **The original "z_score detects stalls"
   finding was substantially an artifact of injecting onto pathologically sparse nodes**, where the
   same fixed `intensity × mad_eff` delta produced enormous, trivially-detectable z-scores because
   `mad_eff` itself was enormous. On genuinely active nodes (small, gated `mad_eff`), the same
   mechanism produces a smaller absolute jump that a fixed `|z|>3` rule catches far less reliably.
2. **"log_ratio detects bursts on both datasets" is now BGL-only by lift, but survives on both
   datasets by AUC-ROC.** BGL-burst log_ratio remains strong (5.14 lift, AUC-ROC 0.663) — the most
   robust result in the project, unchanged in kind. Thunderbird-burst log_ratio's lift collapsed
   to near-zero, but its AUC-ROC (0.598) shows the underlying ranking signal is still real and
   essentially unchanged from before the fix (0.637) — this is a threshold-calibration failure, not
   a mechanism failure (§6 has the full argument). **The honest framing: log_ratio's burst-detection
   *capability* holds on both datasets; its currently-deployed *fixed threshold* only works on one.**
3. **z_score-burst partially contradicts its own mechanism claim.** The additive-bias argument
   above predicts z_score should fail on bursts on any dataset with a positive median/MAD ratio —
   and it still does on BGL (0.75, below chance). But it now flips *above* chance on Thunderbird
   (1.20 lift, 0.594 AUC-ROC, up from 0.35/0.294). This is not yet mechanistically explained and is
   flagged as unresolved, not papered over — see Future Work. One plausible direction: Thunderbird's
   eligible (low-`mad_eff`) node population may have a systematically different median/MAD ratio
   than BGL's, changing where the theoretical z-score floor actually sits; this hasn't been checked.
4. **PCA and isolation_forest_counts improved substantially and consistently, nearly across the
   board** (PCA: 0.72→1.28 BGL-stall, 0.52→1.48 BGL-burst, 1.98→12.32 TB-stall, 1.01→11.79 TB-burst;
   isoforest similarly). Content/count detectors benefit from active nodes because the contrast
   between "a normal busy window" and "an anomalous empty/compressed window" is far starker than on
   an already-sparse node where windows look thin even under normal conditions — a clean, sensible
   mechanism, though isoforest's high numbers should still be read through the §8 credibility
   caveat (its raw score barely outranks random even where its lift looks strong).

**A secondary, unverified hypothesis worth flagging for anyone extending stall's recall analysis**:
stall's grid-level recall dropped substantially even though the injected row's own z-score should
still clear `|z|>3` regardless of node density (delta ≈ intensity × mad_eff either way). One
plausible explanation not yet confirmed: on dense/active nodes, a stall's labeled span (now 20-55
grid cells at the extremes) likely contains genuinely-normal populated cells adjacent to the actual
injected gap, which a correct detector rightly does not flag — and each one counts as a false
negative under grid-cell recall even though it was never truly anomalous. This is distinct from
the "quiet middle" issue already resolved in `results/stall_labeling_diagnostic.md` and has not
been separately investigated.

**The n=15→40→100 stability analysis from the count-only-eligibility pass is retired by this fix**
(it characterized a node population — sparse, pathological — that injections no longer land on).
Re-running an n-stability check under the new eligibility rule has not been done; treat every
number in this section as an n=100, single-seed result until it has.

## 6. Threshold-independent evaluation (AUC-PR / AUC-ROC)

Every lift number above depends on each detector's own threshold — and those thresholds are not
matched: `z_score_threshold` uses a fixed `|z|>3` rule that flags ~49% of everything, while
`log_ratio_threshold` and both ML detectors are calibrated to their own train-normal 95th
percentile (~5–13% base rate, see §8). AUC-PR and AUC-ROC (`src/auc_metrics.py`, `results/
auc_metrics.csv`) score each detector's raw continuous score with no threshold at all, aggregated
onto the identical 60s grid via max-score-per-cell. AUC-PR is reported alongside its no-skill
baseline (positive-class prevalence) and their ratio, since raw AUC-PR is only interpretable
relative to that baseline. **Recomputed on the rate-gated injections; this is the decisive check on
whether the dramatic §3–§5 lift reversals are real signal-quality changes or threshold artifacts.**

| dataset | fault | detector | lift | AUC-PR ratio | AUC-ROC |
|---|---|---|---|---|---|
| BGL | stall | PCA | 1.28 | 1.20 | **0.603** |
| BGL | stall | z_score | 0.76 | 0.94 | 0.464 |
| BGL | stall | log_ratio | 0.27 | 0.96 | 0.482 |
| BGL | stall | isoforest | 1.88 | 2.92 | **0.786** |
| BGL | burst | PCA | 1.48 | 1.51 | **0.659** |
| BGL | burst | z_score | 0.75 | 1.67 | 0.533 |
| BGL | burst | log_ratio | **5.14** | **8.27** | **0.663** |
| BGL | burst | isoforest | 2.72 | 4.02 | **0.782** |
| Thunderbird | stall | PCA | 12.32 | 4.21 | 0.561 |
| Thunderbird | stall | z_score | 1.07 | 1.76 | 0.555 |
| Thunderbird | stall | log_ratio | 0.09 | 1.33 | **0.598** |
| Thunderbird | stall | isoforest | 22.22 | 47.88 | **0.738** |
| Thunderbird | burst | PCA | 11.79 | 3.01 | 0.454 |
| Thunderbird | burst | z_score | 1.20 | 5.33 | 0.594 |
| Thunderbird | burst | log_ratio | 0.03 | 1.23 | 0.598 |
| Thunderbird | burst | isoforest | 15.71 | 17.44 | **0.618** |

**This settles the single most important open question from §3–§5: is log_ratio's Thunderbird
lift-collapse (1.42→0.09 stall, 2.75→0.03 burst — essentially zero detections) a real loss of
signal, or a threshold-calibration artifact?** AUC-ROC answers decisively: **artifact.** Both
Thunderbird log_ratio cells *held or improved* under AUC-ROC (stall: 0.520→0.598; burst:
0.637→0.598) — both comfortably above random. The raw continuous `|log_ratio|` score still ranks
anomalous cells above normal ones about as well as it did before the fix; the fixed 95th-percentile
threshold, calibrated once from the whole clean dataset, simply no longer intersects the smaller
(but still real) magnitude range that compression on now-regular, low-`mad_eff` nodes produces.
**The detector still works. The specific threshold picked for it, in this specific deployment,
does not.**

**BGL-stall `z_score`'s collapse is different in kind — genuinely real, not an artifact.** Its
AUC-ROC crossed from 0.537 (barely-above-random) to 0.464 (below random) — confirming, under a
threshold-free metric, that its previous apparent stall-detection strength was substantially an
artifact of injecting onto pathologically sparse nodes, not a property of the detector that
survives onto genuinely active ones.

**PCA and isolation_forest_counts improved consistently and often dramatically** — isoforest in
particular now clears 0.6-0.79 AUC-ROC in every cell, a real jump from the previous 0.41-0.55 range,
though its lift numbers (up to 22.2) remain far more extreme than its AUC-ROC (up to 0.79) would
suggest, consistent with the ongoing "high lift, more modest but real ranking power" caveat in §8 —
this detector should not be cited by its lift number alone in either the old or new numbers.

**What survives everything, at every stage of this whole project (lift, AUC-PR, AUC-ROC; sparse-node
and rate-gated eligibility; both datasets)**: `log_ratio_threshold` has real, above-chance ranking
power on burst (AUC-ROC 0.663 BGL / 0.598 Thunderbird). This is the one finding that has never once
reversed under any metric or eligibility rule tried in this project — the strongest candidate for
the paper's headline claim, though its usable *lift* now requires dataset-specific threshold
recalibration to actually realize on Thunderbird (see Future Work).

**AUC-PR ratio is not comparable across fault types, and the gap between them just shrank.**
Burst still yields more detectable positive cells per injection than stall (§2b), so its
lower no-skill baseline mechanically inflates AUC-PR ratios relative to stall even at matched
AUC-ROC — compare AUC-ROC across fault types, not AUC-PR ratio. The detectable-cell ratio itself
dropped substantially under the new gate: BGL burst/stall was 780/196 = 3.98× before, now
250/187 = **1.34×**; Thunderbird was 529/196 = 2.70× before, now 251/181 = **1.39×**. Burst and
stall are now much closer to comparable in raw positive-class size than they were, though still not
equal.

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

- **The eligibility rate-gate (§2b) fixed physical plausibility but exposed how much of the
  original stall/burst story depended on WHICH nodes got injected, not just the injection
  mechanism.** Treat every pre-§2b number in older commits/notes as describing a different,
  now-superseded experiment (pathologically sparse nodes), not a smaller-n version of the current
  one.
- **z_score-burst's flip to above-chance on Thunderbird (0.294→0.594 AUC-ROC) is unexplained.** It
  contradicts the additive-bias mechanism in §5 as stated. Needs a direct check of whether
  Thunderbird's rate-gated node population has a different median/`mad_eff` ratio than BGL's before
  trusting either the flip or the original mechanism claim at face value.
- **log_ratio's Thunderbird thresholds need dataset-specific recalibration.** §6 shows the ranking
  signal (AUC-ROC) is intact on Thunderbird for both fault types; the fixed 95th-percentile
  threshold computed once from the whole clean dataset is what's failing, not the feature. A
  dataset-specific (or node-population-specific) threshold would likely restore usable lift without
  changing the underlying detector.
- **The stall grid-recall-dilution hypothesis (§5) is unverified.** Plausible mechanism, not
  confirmed: dense nodes' stall spans may now contain populated-but-genuinely-normal cells that
  correctly-non-flagging detectors get penalized for. Would explain the recall drop despite the
  injected row's own z-score being unaffected by node density; not yet checked directly.
- **log-ratio vs z-score calibration is not apples-to-apples — partially addressed by §6, same as
  before the eligibility fix.** `z_score_threshold` uses a fixed `|z|>3` rule; `log_ratio_threshold`
  is calibrated from the train-normal 95th percentile. §6's AUC-PR/AUC-ROC pass is threshold-free by
  construction and remains the best available cross-check until a genuine matched-base-rate
  head-to-head is built (Future Work).
- **The isolation-forest-wrapped timing detector was tried and abandoned** (pre-dates §2b, still
  true). Feature-scaling was diagnosed and fixed, but it still substantially underperforms
  `z_score_threshold`/`log_ratio_threshold` on the metric that matters. Not pursued further.
- **`isolation_forest_counts` (the "dead baseline") remains untrustworthy despite improved AUC-ROC
  post-fix.** Its AUC-ROC now clears 0.6–0.79 in every cell (up from 0.41–0.55) — a real
  improvement — but its lift numbers (up to 22.2) are still wildly disproportionate to that AUC-ROC,
  and the original BGL mechanistic diagnosis (`results/iforest_diagnosis.md`: AUC=0.507 on native
  labels, chance-level) has not been rerun on the rate-gated injections for either dataset. Treat
  its numbers as directionally informative, not as evidence of a working detector.
- **n=100 is locked** for both datasets and both fault types under the rate-gated eligibility rule.
  The n=15→40→100 stability analysis that justified this under the OLD (count-only) eligibility rule
  no longer applies to the current node population (§5) — a fresh stability check has not been run
  under the new rule. Given how much the numbers moved from one eligibility-rule change alone, this
  is a real gap, not a formality.

## 9. Windowing sweep: timing faults are structurally invisible under fixed-count windowing

`src/windowing_sweep.py` directly tests CLAUDE.md's central windowing claim: does the *windowing
scheme itself*, independent of detector family, determine whether a timing-only fault is
observable at all? Same detectors, same injected data, only the window-assignment rule varies
(fixed-count: N events per node; fixed-time: fixed wall-clock seconds per node). Full results in
`results/windowing_sweep.csv` (detection metrics) and `results/windowing_sweep_invariance.csv`
(clean-vs-injected invariance check), with narrative detail in `results/windowing_sweep_notes.md`.
**This section documents a partially-complete sweep — see "Still pending" below before citing any
number as final.**

**Invariance result (complete on both datasets):** for `count_vector_pca` and
`isolation_forest_counts` under fixed-count windowing (N=20/50/100), clean and injected data
produce **exactly identical** windows and scores — `max_abs_score_diff = 0.0`,
`n_differing_windows = 0`, `n_grid_cell_reassignments = 0`, across both fault types and all three
window sizes, on **both BGL and Thunderbird** (24 cells total, no exceptions). This is exact, not
approximate: fixed-count window membership depends only on an event's position in its node's
sequence, never on its timestamp, so a timestamp-shifting fault cannot change which window a row
lands in or what a count-based detector scores it as.

**Contrast (BGL only, complete):** under fixed-time windowing (30s/60s/120s), the same two
detectors show real, substantial differences on the same injected data — score diffs of 12.9–348,
hundreds of differing windows, and thousands of grid-cell reassignments (5,245–9,473 per cell, out
of 4,747,963 rows checked).

**Independent third confirmation (BGL, DeepLog smoke test):** `results/deeplog_smoke_test.md`'s
invariance check on a subsampled, 1-epoch DeepLog model shows the identical pattern at the
sequence-model level — `max_abs_score_diff = 0.0` across 49,930 comparable prediction windows, with
`n_grid_cell_reassignments = 331` (0.66% of 50,000 common rows landed in a different 60s grid cell
purely from the timestamp shift, while each row's DeepLog prediction stayed byte-for-byte
unchanged). A sequence-based deep model is exactly as structurally blind to timing-only faults as
the count-vector detectors, for the same underlying reason: its input (the template-ID sequence)
never changes under a timestamp shift.

**Conclusion this supports:** content and sequence detectors are provably blind to timing-only
faults — demonstrated three independent ways (count-vector/PCA, isolation-forest-on-counts,
DeepLog), on two datasets for the count detectors. Any apparent timing-fault "detection" that shows
up under fixed-time windowing does not come from a detector gaining sensitivity to timing; it comes
from the windowing scheme converting a timing perturbation into a **count** perturbation (rows
shifting between fixed-time buckets), which a count-based detector then picks up as an ordinary
count anomaly. Windowing choice, not detector architecture, determines whether timing faults are
observable at all.

**Still pending — do not treat the sweep as complete:**
- **Thunderbird fixed-time windowing (30s/60s/120s)**, all 4 detectors, both fault types (24 sweep
  rows + 12 invariance rows, marked `PENDING` in both CSVs, not zero/blank). Attempted locally and
  **OOM-killed** by the macOS kernel (confirmed via the unified system log: `memorystatus: killing
  largest compressed process Python [69770] 23625 MB`) — a genuine memory-capacity failure on this
  8GB development machine, not a repeat of the project's earlier fileproviderd/iCloud stall issue
  and not a correctness bug in the injector or windowing logic. Scheduled to run on a lab server;
  no code changes have been made in response.
- **DeepLog full training run**, both datasets, both eval targets. Only the 1-epoch, subsampled
  smoke test above has been run locally, per explicit instruction that this machine is not used for
  real DeepLog training. Its invariance result is trustworthy as a structural finding (it doesn't
  depend on training quality), but no full-scale DeepLog detection numbers exist yet.

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
  {10,15,20,25,30}` (multiples of node scale, additive), node eligibility ≥30 events AND
  `mad_eff ≤ 120s` (+ valid baseline on Thunderbird), injection point restricted to non-outlier
  pre-existing gaps (`|z|≤3`).
- **Burst injection**: `SEED=43`, **100 injections**, one per distinct eligible node, `intensity ∈
  {10,15,20,25,30}` (compression factor, multiplicative), `burst_length ∈ {10,15,20,25}` consecutive
  gaps, node eligibility ≥55 events AND `mad_eff ≤ 120s` (+ valid baseline on Thunderbird), same
  non-outlier eligibility filter applied to all gaps in the burst window.
- **Rate/density gate** (§2b): `mad_eff = max(node_mad × 1.4826, MAD_FLOOR_SEC) ≤ 120s`, derived
  from `MAX_PLAUSIBLE_FAULT_DURATION_S=3600s ÷ max(INTENSITY_CHOICES)=30`. Exact guarantee for
  stall duration; approximate (not exact) for burst duration — see §2b's caveat.
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

# Windowing sweep (fixed-count vs fixed-time, both datasets -- see §9)
# `python src/windowing_sweep.py` with no arguments runs both datasets end-to-end and regenerates
# results/windowing_sweep.csv / results/windowing_sweep_invariance.csv in full -- only viable on a
# machine with enough RAM for Thunderbird's fixed-time schemes (OOM-killed on the 8GB dev machine,
# see §9). The current results/windowing_sweep.csv was assembled by hand from three partial runs
# instead, pending that full re-run:
python src/windowing_sweep.py bgl            # already complete; do not re-run, do not overwrite
# Thunderbird fixed-count schemes only (low memory footprint, ~1-3GB peak) -- complete:
#   scratch driver that calls windowing_sweep.process_count_scheme() directly, skipping the
#   memory-heavy fixed-time schemes; see results/windowing_sweep_notes.md §4.
# Thunderbird fixed-time schemes (30/60/120s) -- PENDING, run on a lab server with more RAM:
python src/windowing_sweep.py thunderbird

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

- **Fix burst's duration guarantee properly**: `mad_eff` bounds stall duration exactly but only
  approximately bounds burst duration (§2b — BGL-burst's worst case is still ~149 hours post-fix).
  A gate on the node's *observed local gap magnitude at the candidate injection point* (rather than
  the node-level `mad_eff` summary) would close this gap directly.
- **Recalibrate `log_ratio_threshold`'s decision threshold per dataset (or per node population)**
  rather than using one fixed 95th-percentile cutoff computed from the whole clean dataset — §6
  shows this alone would likely restore most of Thunderbird's burst/stall lift without touching the
  underlying feature.
- **Investigate z_score-burst's Thunderbird flip** (§5, §8) — check whether Thunderbird's rate-gated
  eligible-node population has a different median/`mad_eff` ratio than BGL's, which would explain
  (or rule out) a dataset-specific shift in the theoretical z-score floor.
- **Verify the stall grid-recall-dilution hypothesis** (§5, §8) directly — check whether populated,
  non-injected-row cells inside a stall's labeled span on dense nodes are being penalized as false
  negatives by detectors that correctly identify them as normal.
- **A fair, matched-calibration head-to-head** between `z_score_threshold` and `log_ratio_threshold`
  (same base-rate target) to isolate the geometry effect from the calibration effect noted in §8 —
  §6's AUC-PR/AUC-ROC pass is a threshold-free proxy for this, not a substitute.
- **Re-run the n=15→40→100 stability analysis under the rate-gated eligibility rule** — the version
  in an earlier pass characterized a now-superseded (sparse) node population (§5, §8).
- Re-audit `isolation_forest_counts` on both datasets under the rate-gated injections the way
  `results/iforest_diagnosis.md` did for BGL pre-fix — §6 shows improved but still lift-disproportionate
  AUC-ROC; a full mechanistic diagnosis hasn't been redone.
- Additional fault types: slowdown (gradual stretch, distinct from stall's step-function gap) and
  jitter (added noise without a mean shift).
- Realism calibration against LO2 / AnoMod or similar reference injectors.
- HDFS: either a sequence-completeness detector family, or a narrower within-block timing-injection
  design restricted to the 14.24% valid-baseline subset — both open, per §7.
