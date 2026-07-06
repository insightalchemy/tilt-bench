# CLAUDE.md — TILT-Bench

## What this project is
Building a benchmark that injects **timing-only anomalies** into public log datasets to show that
current log-anomaly-detection (AD) evaluation has a blind spot. Standard benchmarks (HDFS, BGL) are
saturated by content/sequence detectors because they contain almost no timing faults. We inject
controlled timing faults (stalls, slowdowns, bursts, jitter), keep ground-truth labels, and measure
what existing detectors miss versus a lightweight timing detector.

This is an **evaluation-methodology** paper, NOT a new-detector paper. The contribution is the
benchmark + the finding that benchmark design (especially windowing) determines which anomaly classes
are observable. Do not drift toward inventing novel detector architectures.

## Central scientific claim (keep every task tied to this)
- **Observation:** standard datasets contain few/no timing anomalies.
- **Hypothesis:** because they're missing, current evaluations underestimate the limits of
  content/sequence detectors.
- **Mechanism:** windowing choice is *why* timing faults are invisible — it is the explanation, not
  the headline contribution.
- **Goal:** a calibrated injection framework + evidence of detector complementarity.

## Implementation order (do NOT skip ahead)
Follow this sequence. Prove the premise before building infrastructure.
1. **Parse BGL + premise audit.** Parse raw BGL, then characterize how the *existing* labeled
   anomalies manifest (new/rare templates, order change, sequence length, timing gaps). Produce a
   breakdown table. This validates or kills the hypothesis. Highest priority.
2. **Minimal injector — stall only, BGL only.** One anomaly type, correct downstream propagation,
   seeded, labeled. Visually verify in inter-arrival plots before trusting any metric.
3. **Representative detectors.** count-vector+PCA (statistical), isolation forest on counts (ML),
   lightweight inter-arrival-time detector (timing). Run on clean + stall-injected BGL.
4. **Core result.** Per-detector detection rate on injected stalls + Jaccard overlap between timing
   and content detectors (this is the complementarity result).
5. **Expand only after step 4 works:** add burst (2nd type), an intensity sweep (detectability
   frontier), then optionally one deep model (DeepLog) and calibration vs LO2/AnoMod.

Do not build all four anomaly types or all detectors up front. Stall first, prove the claim, expand.

## Critical technical rules
- **Never mutate raw data.** Scripts read from `data/raw/BGL.log`, write to `data/processed/`.
- **Propagation is mandatory.** A stall/slowdown delays *everything downstream* from that component:
  shift all subsequent timestamps for that component forward, preserving order. NEVER widen a single
  gap in isolation (that creates impossible orderings). Slowdown = gradual stretch; burst = compress
  downstream gaps.
- **BGL is a continuous per-node stream.** Define inter-arrival time per node. (HDFS is session-based
  by block_id and comes later — different injection mode.)
- **Timestamp resolution is microsecond, not second-level.** The raw line's 5th field
  (`YYYY-MM-DD-HH.MM.SS.ffffff`) carries real microsecond precision — verified against the parsed
  data: 0.00% of inter-arrival gaps are exactly zero, diffs are non-uniform (median ~26ms, IQR
  18–38ms, not a fixed-step counter), and the field is globally monotonic across all 4.7M rows with
  zero reversals. It also agrees with the coarser Unix-epoch field (2nd column) exactly modulo a
  timezone offset that correctly flips UTC-7/UTC-8 with US Daylight Saving Time, which rules out it
  being a parser artifact. Use this field (not the Unix-epoch one) for all timing features. The
  practical constraint isn't resolution — it's baseline coverage: ~33% of nodes (22,749 / 69,156)
  have too few normal-only gaps to build a reliable per-node timing baseline and fall back to a
  pooled/global baseline instead (see `src/premise_audit.py`).
- **Windowing is a first-class variable.** Report results by anomaly type × windowing scheme:
  - session/fixed-*count* windows: pure timing shift is invisible (event multiset/order unchanged).
  - fixed-*time* windows (e.g. 60s): bursts/stalls change per-window counts, so partially visible.
- **Intensity is relative.** Parameterize perturbations relative to the component's baseline
  inter-arrival variability, so intensity is comparable across nodes/datasets.

## Dataset
- **BGL**, full raw log, from Loghub / Zenodo record 3227177 (`BGL.log`). ~4,747,963 lines; 348,460
  (7.34%) labeled anomalous.
- **Label column:** first token per line. `-` = normal; anything else = an alert (anomalous).
- Use the FULL `BGL.log`, not the 2k demo sample (`BGL_2k.log` is only for parser sanity checks).
- Parse with Drain (logparser/Loghub tooling).
- Cite Zhu et al. (Loghub, ISSRE 2023) and Oliner & Stearley (DSN 2007) in any output.

## Detectors (one or two per family — do not add more)
- Statistical: count-vector + PCA.
- Classical ML: isolation forest on count vectors.
- Timing: lightweight inter-arrival features (rolling mean/std, EWMA residual, z-score) → isolation
  forest / threshold.
- Deep (optional, step 5 only): DeepLog via an existing toolkit, inference-scale. One deep model max.
- Also include Landauer et al.'s released inter-arrival detector as a reference.

## Metrics
- Per-anomaly-type, per-windowing detection rate / precision / recall / F1.
- Complementarity: Jaccard overlap of detected sets across detector families; OR-fusion recall +
  precision cost.
- Detectability frontier: detection probability vs intensity-relative-to-baseline-variability.
- **Avoid point-adjusted F1** (it inflates time-series AD scores); use range/event-aware scoring.
- Chronological train/test split — never random split (avoids leakage).

## Directory structure
```
tiltbench/
├── CLAUDE.md
├── data/raw/BGL.log         # pristine, read-only in practice
├── data/processed/          # parsed CSVs, injected variants, labels
├── src/                     # parser.py, injector.py, detectors/, metrics.py
├── results/                 # tables, plots, metric outputs
└── notebooks/               # visual sanity checks (IAT plots)
```

## Coding conventions
- Python 3, pandas / numpy / scikit-learn / matplotlib. No heavy frameworks unless needed.
- **Everything seeded and reproducible.** All randomness (injection, detectors) takes an explicit
  seed; record it. Same seed → same output.
- Injections must be regenerable from a config + seed; save the config alongside outputs.
- Ground-truth labels recorded as (start, end, node, type, intensity, seed).
- Prefer small, single-purpose scripts over one monolith. Keep functions testable.
- Save intermediate artifacts (parsed CSV, injected CSV) so steps don't rerun from scratch.

## Correctness discipline (this is research, not just shipping code)
- "It runs" ≠ "it's correct." For the injector propagation, the labeling, and the metric
  computation, the *logic* must be verified, not just the absence of errors — a silent bug there
  corrupts the headline numbers.
- After building the injector, always produce the inter-arrival plot and confirm the injected fault
  is visible and ordering is preserved before running metrics.
- If a result looks too clean or too good, suspect leakage or a labeling/metric bug first.

## Scope / what NOT to do
- Don't add datasets beyond BGL until the BGL core result is done (Thunderbird → HDFS come later).
- Don't add anomaly types beyond stall until step 4 works; burst is the second, not the first.
- Don't build a deep-model zoo; the paper is about evaluation, not architectures.
- Don't optimize/refactor prematurely; get correct core results first.