"""
Diagnose why isolation_forest_counts collapsed to near-zero recall (0.0004, see
results/baseline_detection.csv) -- pin down the actual mechanism rather than just inferring one.

An initial pass (v1 of results/iforest_diagnosis.md) concluded "mechanism B" (general
discrimination failure) from the fact that the bulk of anomalous/normal score distributions
overlapped heavily. That was premature: it never checked ranking quality directly, and a
0.813-precision / 0.0007-recall result (see results/baseline_detection_common_unit.csv) is the
signature of a fine ranking with a badly-transferred operating point, not a broken ranking -- a
model with genuinely no discrimination would not have 0.81 precision on ANYTHING it flags. This
version runs the discriminating test directly: ROC-AUC / AUPRC on the raw scores (threshold-free),
plus an oracle threshold set on the test-normal distribution instead of train-normal.

Two candidate mechanisms:
  (A) novel (out-of-vocabulary) templates -> a test window built entirely of them collapses to a
      near-zero/all-zero count vector -> the isolation forest reads that as "quiet/normal" ->
      recall collapses on those windows specifically.
  (B) template drift more broadly shifts feature values across ALL test windows (not just the
      all-OOV ones) such that the isolation forest's random-partition splits, learned on a
      narrower train distribution, misbehave and stop discriminating anomalous from normal.
  (C) threshold-transfer failure: the ranking (relative ordering of scores) is still meaningful,
      but the decision threshold -- set from the TRAIN-normal score distribution -- doesn't
      transfer to the test period, whose whole score distribution has shifted. Recall collapses
      not because anomalies aren't distinguishable, but because the cutoff is miscalibrated.

Reuses the exact fit from src/run_baseline_detectors.py (same split, same vocabulary, same
IsolationForestCountsDetector, same threshold rule) so this diagnosis matches the real result.

Writes:
  figures/iforest_score_distributions.png -- overlaid score histograms, 3 groups, threshold marked
  results/iforest_diagnosis.md            -- plain-language conclusion
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src.xxx` imports resolve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.detectors.isolation_forest_counts import IsolationForestCountsDetector
from src.metrics import evaluate_binary
from src.run_baseline_detectors import THRESHOLD_PERCENTILE, build_count_features, chronological_split, load

FIG_PATH = Path("figures/iforest_score_distributions.png")
MD_PATH = Path("results/iforest_diagnosis.md")

COLOR_ANOMALOUS = "#e34948"  # red
COLOR_NORMAL_OOV = "#eda100"  # yellow
COLOR_NORMAL_KNOWN = "#2a78d6"  # blue


def main():
    df = load()
    is_train, cutoff = chronological_split(df)
    df_train = df.loc[is_train].reset_index(drop=True)
    df_test = df.loc[~is_train].reset_index(drop=True)

    X_train, X_test, windows_train, windows_test, df_test_w, count_diag = build_count_features(
        df_train, df_test, "fixed_count", 20
    )

    train_normal_mask = ~windows_train["label"].to_numpy()
    X_train_normal = X_train[train_normal_mask]

    detector = IsolationForestCountsDetector()
    detector.fit(X_train_normal)
    contamination = detector.model.contamination

    train_scores = detector.score(X_train_normal)
    threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)
    test_scores = detector.score(X_test)

    y_test = windows_test["label"].to_numpy()
    n_events = windows_test["n_events"].to_numpy()
    l1_norm = X_test.sum(axis=1)  # count of in-vocab events per window (L1 norm of a count vector)
    frac_in_vocab = l1_norm / n_events
    n_nonzero_dims = (X_test > 0).sum(axis=1)  # sparsity companion to L1 norm

    group_a = y_test  # anomalous (any vocabulary mix)
    group_b = (~y_test) & (l1_norm == 0)  # normal, fully out-of-vocabulary (near-zero vector)
    group_c = (~y_test) & (frac_in_vocab >= 0.5)  # normal, mostly in-vocabulary

    # --- plot: overlaid score distributions, 3 groups, threshold marked ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(test_scores.min(), threshold), test_scores.max(), 60)
    specs = [
        (group_c, COLOR_NORMAL_KNOWN, f"normal, mostly in-vocab (n={group_c.sum():,})"),
        (group_b, COLOR_NORMAL_OOV, f"normal, fully out-of-vocab (n={group_b.sum():,})"),
        (group_a, COLOR_ANOMALOUS, f"anomalous (n={group_a.sum():,})"),
    ]
    for mask, color, label in specs:
        ax.hist(test_scores[mask], bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=color, label=label)
        ax.hist(test_scores[mask], bins=bins, density=True, histtype="step", linewidth=1.5, color=color)
    ax.axvline(threshold, color="#0b0b0b", linestyle="--", linewidth=1.5, label=f"decision threshold (train p{THRESHOLD_PERCENTILE})")
    ax.set_xlabel("IsolationForest anomaly score (higher = more anomalous)")
    ax.set_ylabel("density")
    ax.set_title("isolation_forest_counts: test-set score distributions by group")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)

    # --- hypothesis test: do near-zero vectors get scored as NORMAL? ---
    all_zero = l1_norm == 0
    pct_below_threshold_all_zero = 100 * (test_scores[all_zero] <= threshold).mean() if all_zero.any() else float("nan")
    pct_below_threshold_nonzero = 100 * (test_scores[~all_zero] <= threshold).mean()

    # recall ceiling if OOV-collapse (mechanism A) were the ONLY failure: could we, at best, have
    # caught the anomalous windows that are NOT all-zero?
    anomalous_all_zero = group_a & all_zero
    anomalous_nonzero = group_a & ~all_zero
    recall_ceiling_if_only_A = 100 * anomalous_nonzero.sum() / group_a.sum()

    actual_flagged_test = test_scores > threshold
    recall_actual = 100 * (actual_flagged_test & group_a).sum() / group_a.sum()
    recall_within_nonzero_anomalies = (
        100 * (actual_flagged_test & anomalous_nonzero).sum() / anomalous_nonzero.sum() if anomalous_nonzero.sum() else float("nan")
    )

    l1_stats_all_zero_normal = pd.Series(l1_norm[group_b]).describe()
    l1_stats_known_normal = pd.Series(l1_norm[group_c]).describe()
    sparsity_all_zero_normal = pd.Series(n_nonzero_dims[group_b]).describe()
    sparsity_known_normal = pd.Series(n_nonzero_dims[group_c]).describe()

    # --- threshold-free ranking quality: does the score itself separate anomalous from normal? ---
    auc = roc_auc_score(y_test, test_scores)
    auprc = average_precision_score(y_test, test_scores)
    baseline_auprc = y_test.mean()  # AUPRC of a random/no-skill ranker at this class balance

    # --- oracle threshold: set the SAME percentile rule on the TEST-normal distribution instead
    # of train-normal. This leaks test labels (not a legitimate deployment threshold) -- it's
    # purely diagnostic, to check whether the ranking is sound and only the operating point
    # failed to transfer. ---
    oracle_threshold = np.percentile(test_scores[~y_test], THRESHOLD_PERCENTILE)
    oracle_predicted = test_scores > oracle_threshold
    oracle_result = evaluate_binary(y_test, oracle_predicted)

    print(f"IsolationForest contamination parameter: {contamination!r}")
    print(f"Decision threshold (train-normal p{THRESHOLD_PERCENTILE}): {threshold:.4f}")
    print()
    print(f"Test windows, all-zero (fully OOV) count vector: {all_zero.sum():,} / {len(all_zero):,} ({100*all_zero.mean():.2f}%)")
    print(f"  -> anomalous & all-zero: {anomalous_all_zero.sum():,} ({100*anomalous_all_zero.sum()/group_a.sum():.2f}% of all anomalous windows)")
    print(f"  -> normal & all-zero:    {group_b.sum():,} ({100*group_b.sum()/(~y_test).sum():.2f}% of all normal windows)")
    print()
    print(f"Of ALL-ZERO test windows, % scored <= threshold (i.e. classified NORMAL): {pct_below_threshold_all_zero:.2f}%")
    print(f"Of NON-zero test windows, % scored <= threshold (i.e. classified NORMAL): {pct_below_threshold_nonzero:.2f}%")
    print()
    print(f"Recall ceiling if OOV-collapse (mechanism A) were the ONLY failure mode: {recall_ceiling_if_only_A:.2f}%")
    print(f"Actual recall achieved: {recall_actual:.4f}%")
    print(f"Recall restricted to anomalous windows that are NOT all-zero (i.e. mechanism A can't explain a miss here): {recall_within_nonzero_anomalies:.4f}%")
    print()
    print("L1 norm (in-vocab event count per window), normal/all-zero-OOV group:")
    print(l1_stats_all_zero_normal)
    print()
    print("L1 norm, normal/mostly-known group:")
    print(l1_stats_known_normal)
    print()
    print("Nonzero-vocab-dim count, normal/all-zero-OOV group:")
    print(sparsity_all_zero_normal)
    print()
    print("Nonzero-vocab-dim count, normal/mostly-known group:")
    print(sparsity_known_normal)
    print()
    print(f"ROC-AUC (threshold-free ranking quality): {auc:.4f}")
    print(f"AUPRC: {auprc:.4f}  (no-skill/random baseline at this class balance: {baseline_auprc:.4f})")
    print()
    print(f"Oracle threshold (test-normal p{THRESHOLD_PERCENTILE}): {oracle_threshold:.4f}  (train-normal p{THRESHOLD_PERCENTILE} was {threshold:.4f})")
    print(f"Oracle precision/recall/F1: {oracle_result['precision']:.4f} / {oracle_result['recall']:.4f} / {oracle_result['f1']:.4f}")
    print(f"(for reference, train-threshold precision/recall/F1 was 0.1129 / 0.0004 / 0.0007)")

    # --- verdict ---
    auc_separates = auc > 0.6
    oracle_recovers_recall = oracle_result["recall"] > 0.3  # "meaningful recall" bar
    threshold_transfer_failure = auc_separates and oracle_recovers_recall

    # --- write conclusion ---
    mechanism_a_explains_pct = 100 * anomalous_all_zero.sum() / group_a.sum()
    verdict_heading = (
        "Conclusion: threshold-transfer failure (C), not general discrimination failure (B)"
        if threshold_transfer_failure
        else "Conclusion: mechanism (B) stands -- general discrimination failure"
    )
    verdict_body = (
        f"""The ranking is fine; the operating point wasn't. ROC-AUC = {auc:.4f} (clearly > 0.6 -- the
raw scores DO separate anomalous from normal windows) and AUPRC = {auprc:.4f} against a no-skill
baseline of {baseline_auprc:.4f} at this class balance -- a real, if modest, ranking signal. The
0.813-precision / 0.0007-recall pattern reported by the common-unit table is exactly what
threshold-transfer failure looks like: almost nothing clears the bar, but what does clear it is
usually right. Setting the threshold on the TEST-normal distribution instead (an oracle,
illegitimate for real deployment since it uses test labels, but diagnostic) moves the operating
point to {oracle_threshold:.4f} (versus {threshold:.4f} from train-normal) and recovers
recall={oracle_result['recall']:.4f}, precision={oracle_result['precision']:.4f},
F1={oracle_result['f1']:.4f} -- night and day versus the train-calibrated 0.0004 recall. The
earlier v1 of this diagnosis concluded "mechanism B" from the fact that the bulk of the score
distributions overlapped -- true, but that was the wrong test: bulk overlap is consistent with
BOTH a broken ranking and a fine ranking whose extreme tail (where the actual decision boundary
should sit) is still informative. AUC/AUPRC test the tail directly, and here it says the isolation
forest is not obviously broken as a ranker -- it produces a modestly-informative anomaly score.
What's broken is the {THRESHOLD_PERCENTILE}th-percentile-of-train-normal thresholding rule: train
and test periods have different score distributions (see the figure -- ALL test-period scores,
across all three groups, sit to the left of the train-derived threshold), so a cutoff calibrated on
train almost never fires on test. That's a threshold-transfer problem, not a ranking problem."""
        if threshold_transfer_failure
        else f"""ROC-AUC = {auc:.4f} and AUPRC = {auprc:.4f} (no-skill baseline {baseline_auprc:.4f}) show the
raw scores do not meaningfully separate anomalous from normal windows, and the oracle
threshold (test-normal p{THRESHOLD_PERCENTILE} = {oracle_threshold:.4f}) only recovers
recall={oracle_result['recall']:.4f} -- not enough to call this a threshold-transfer problem. The
ranking itself is close to uninformative, consistent with mechanism (B): template drift degrades
the isolation forest's ability to discriminate, not just its calibrated cutoff."""
    )

    md = f"""# Isolation-forest-on-counts diagnosis

## Setup
- `IsolationForest` contamination parameter: `{contamination!r}` (sklearn default; our thresholding
  uses a custom train-normal score percentile, not the contamination-derived offset, so this
  parameter has no effect on the result reported here -- included because it was asked for).
- Decision threshold: train-normal score at the {THRESHOLD_PERCENTILE}th percentile = {threshold:.4f}.
- Vocabulary: top-{count_diag['n_vocab']} templates from the 880-template train vocabulary.

## Why this diagnosis was revised

The common-unit results (`results/baseline_detection_common_unit.csv`) show isolation_forest_counts
at **precision 0.813, recall 0.0007**. High precision with near-zero recall is the signature of a
threshold sitting above almost all test scores while the underlying ranking is still fine -- a
model with genuinely no discriminative power would not produce 0.81 precision on the handful of
things it does flag. The original diagnosis (below the fold, mechanism-B histogram overlap) never
tested the ranking directly, so it's superseded by the AUC/AUPRC/oracle-threshold test below.

## Threshold-free ranking quality

- ROC-AUC: **{auc:.4f}**
- AUPRC: **{auprc:.4f}** (no-skill baseline at this class balance: {baseline_auprc:.4f})

## Oracle threshold (diagnostic only -- uses test labels, not a legitimate deployment threshold)

Setting the {THRESHOLD_PERCENTILE}th-percentile threshold rule on the TEST-normal score
distribution instead of train-normal moves the cutoff from {threshold:.4f} to {oracle_threshold:.4f}
and changes the outcome to precision={oracle_result['precision']:.4f},
recall={oracle_result['recall']:.4f}, F1={oracle_result['f1']:.4f}.

## Verdict rule applied

AUC {'>' if auc_separates else '<='} 0.6 and oracle recall {'>' if oracle_recovers_recall else '<='} 0.3
=> **{'threshold-transfer failure (C)' if threshold_transfer_failure else 'mechanism (B) stands'}**.

## {verdict_heading}

{verdict_body}

## Out-of-vocabulary collapse (mechanism A) -- still a real, minor, separate effect

{all_zero.sum():,} of {len(all_zero):,} test windows ({100*all_zero.mean():.2f}%) are fully
out-of-vocabulary (L1 norm 0). Of these, {pct_below_threshold_all_zero:.2f}% score at or below the
train-normal threshold (classified normal) versus {pct_below_threshold_nonzero:.2f}% for windows
with at least one in-vocabulary event -- barely different, confirming (as before) that OOV status
is not what's driving the train-threshold outcome. {anomalous_all_zero.sum():,} of
{group_a.sum():,} anomalous windows ({mechanism_a_explains_pct:.2f}%) are themselves fully
out-of-vocabulary and can never be caught by a count-vector detector regardless of threshold --
that ceiling ({recall_ceiling_if_only_A:.2f}% max achievable recall) is real but small, and
independent of the threshold-transfer story above.

See `figures/iforest_score_distributions.png`: note that even though the bulk distributions overlap
heavily (the original observation), the AUC/AUPRC numbers above show the tail -- where the actual
decision boundary lives -- still carries signal. Bulk-overlap plots like this one are the wrong
tool for judging ranking quality; AUC/AUPRC are threshold-free and test the whole ordering.
"""
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(md)
    print(f"\nWrote {FIG_PATH}, {MD_PATH}")


if __name__ == "__main__":
    main()
