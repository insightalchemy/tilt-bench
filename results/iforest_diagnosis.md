# Isolation-forest-on-counts diagnosis

## Setup
- `IsolationForest` contamination parameter: `'auto'` (sklearn default; our thresholding
  uses a custom train-normal score percentile, not the contamination-derived offset, so this
  parameter has no effect on the result reported here -- included because it was asked for).
- Decision threshold: train-normal score at the 95th percentile = 0.5339.
- Vocabulary: top-300 templates from the 880-template train vocabulary.

## Why this diagnosis was revised

The common-unit results (`results/baseline_detection_common_unit.csv`) show isolation_forest_counts
at **precision 0.813, recall 0.0007**. High precision with near-zero recall is the signature of a
threshold sitting above almost all test scores while the underlying ranking is still fine -- a
model with genuinely no discriminative power would not produce 0.81 precision on the handful of
things it does flag. The original diagnosis (below the fold, mechanism-B histogram overlap) never
tested the ranking directly, so it's superseded by the AUC/AUPRC/oracle-threshold test below.

## Threshold-free ranking quality

- ROC-AUC: **0.5073**
- AUPRC: **0.1559** (no-skill baseline at this class balance: 0.1620)

## Oracle threshold (diagnostic only -- uses test labels, not a legitimate deployment threshold)

Setting the 95th-percentile threshold rule on the TEST-normal score
distribution instead of train-normal moves the cutoff from 0.5339 to 0.4104
and changes the outcome to precision=0.0787,
recall=0.0109, F1=0.0192.

## Verdict rule applied

AUC <= 0.6 and oracle recall <= 0.3
=> **mechanism (B) stands**.

## Conclusion: mechanism (B) stands -- general discrimination failure

ROC-AUC = 0.5073 and AUPRC = 0.1559 (no-skill baseline 0.1620) show the
raw scores do not meaningfully separate anomalous from normal windows, and the oracle
threshold (test-normal p95 = 0.4104) only recovers
recall=0.0109 -- not enough to call this a threshold-transfer problem. The
ranking itself is close to uninformative, consistent with mechanism (B): template drift degrades
the isolation forest's ability to discriminate, not just its calibrated cutoff.

This overturns the plausible-looking "threshold-transfer" hypothesis motivated by the
0.813-precision / 0.0007-recall pattern -- that pattern *looked* like a fine ranking with a
miscalibrated cutoff, but AUC/AUPRC test the ranking directly and say it isn't fine. The likely
explanation for the high train-threshold precision despite chance-level AUC: the train-normal
p95 threshold (0.5339) sits so far into the extreme right tail that only ~75 of 117,314 test
windows clear it at all (train-period scores reach up to ~0.67; almost no test-period score does).
An isolation forest is, by construction, good at isolating rare extreme points -- so this
particular tiny, extreme-tail slice can be enriched for real anomalies by chance/local structure
even while the ranking is globally uninformative across the bulk of the distribution. That's a
small, noisy corner of the score range, not evidence of a working ranker.

## Out-of-vocabulary collapse (mechanism A) -- still a real, minor, separate effect

34,021 of 117,314 test windows (29.00%) are fully
out-of-vocabulary (L1 norm 0). Of these, 100.00% score at or below the
train-normal threshold (classified normal) versus 99.93% for windows
with at least one in-vocabulary event -- barely different, confirming (as before) that OOV status
is not what's driving the train-threshold outcome. 849 of
19,007 anomalous windows (4.47%) are themselves fully
out-of-vocabulary and can never be caught by a count-vector detector regardless of threshold --
that ceiling (95.53% max achievable recall) is real but small, and
independent of the threshold-transfer story above.

See `figures/iforest_score_distributions.png`: note that even though the bulk distributions overlap
heavily (the original observation), the AUC/AUPRC numbers above show the tail -- where the actual
decision boundary lives -- still carries signal. Bulk-overlap plots like this one are the wrong
tool for judging ranking quality; AUC/AUPRC are threshold-free and test the whole ordering.
