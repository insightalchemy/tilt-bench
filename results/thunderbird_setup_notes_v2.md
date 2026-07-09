# Thunderbird setup notes v2 — representative slice (exploratory prep, injector NOT yet run)

Supersedes `results/thunderbird_setup_notes.md`. The original 5M-line head slice was confirmed
99.99% dominated by a single InfiniBand (VAPI) incident on 6 nodes — its premise-audit numbers
described that one incident, not Thunderbird's anomaly behavior. This pass replaces it with a
genuinely diverse, representative slice, before deciding whether to proceed to injection.

## Part 1 — Finding a representative slice

**Survey method**: streamed the first 60,000,000 lines of the source archive (never downloading
or decompressing the full 29.6 GB file — same `curl | tar -xzO | head` approach as before, this
time piped through `awk` to emit only a lightweight `(line_number, label, node)` record per
anomalous row, discarding normal rows entirely, so the survey itself never materialized more than
a few hundred MB on disk).

**What the survey found**: bucketing the 1,804,157 anomalies found in those 60M lines into 5M-line
windows showed wildly uneven concentration:

| window | anomalies | distinct labels | distinct nodes | top node share |
|---|---|---|---|---|
| 0M–5M (the original slice) | 226,095 | 4 | 15 | 55.8% |
| 5M–10M | 127,699 | 5 | 23 | 86.3% |
| 15M–20M | 395,485 | 5 | 13 | 77.4% |
| **50M–60M** | **170,422** | **8** | **47** | **15.3%** |

Every early window was dominated by one or two nodes producing a storm of identical alerts (top-node
share 55–86%). The 50M–60M window was the clear outlier in the right direction: full label
diversity (all 8 alert types seen anywhere in the 60M-line survey: `CPU, ECC, EXT_FS, MPT,
PBS_BFD, PBS_CON, SCSI, VAPI`), 47 distinct anomalous nodes (vs 15 in the original slice), and no
single node above 15.3% (top-3 nodes combined: 42.2%, vs the original slice's single node at
55.8%). Nearby sub-windows were checked too (45M–55M, 50M–58M, 52M–60M, 55M–60M) — 50M–60M gave
the best combination of volume and diversity among all candidates.

**Chosen window**: lines 50,000,001–60,000,000 of the source file (a full 10M-line contiguous
chronological slice — within the "5-10M, laptop-sized" range, at the upper end because that's
where the diversity was). Extracted via the same streaming pipeline (`tail -n +50000001 | head -n
10000000`), confirmed clean (well-formed first/last lines, label counts matching the survey
exactly: 170,422 total anomalies, both from the independent survey pass and from a direct count on
the extracted file). This **replaces** `data/raw/Thunderbird.log` (the old biased slice was deleted
— nothing of value in it once superseded, and disk is tight).

**Zero-gap timestamp fraction for this slice**: **55.07%** of per-node inter-arrival gaps are
exactly 0 seconds (26.67% of nodes majority-zero-gap). This is essentially unchanged from the
original slice's 62.11% — confirming, as expected, that the coarse (second-only) timestamp
resolution problem is a property of the *dataset*, not an artifact of which slice was chosen. See
Part 3 for why this matters concretely.

## Part 2 — Parsing + premise audit, representative slice

Same parser (`src/parser_thunderbird.py`, unmodified), same schema.

**Parsing summary** (`data/processed/thunderbird_parsed.parquet`, now the representative slice):
- 10,000,000 lines, 0 skipped
- % anomalous: **1.70%** (170,422 rows — matches the survey exactly)
- Unique nodes: 4,722
- Unique event templates: **9,874** (vs the biased slice's 1,703 — much richer content diversity,
  as expected from a non-single-incident sample)
- Timestamp range: 2006-01-17 01:42:47 → 2006-02-08 07:01:01 (~3 weeks)
- Timestamp resolution: `datetime64[s]`, confirmed still second-level only (unchanged finding)

**Premise audit** (`src/premise_audit_thunderbird.py`, unmodified logic, now pointed at the
representative slice): `results/thunderbird_premise_audit_v2.csv`.

| signature | BGL | Thunderbird v1 (biased, 1 incident) | **Thunderbird v2 (representative)** |
|---|---|---|---|
| new_or_rare_template | 99.31% | 100.00% | **99.99%** |
| pure_order_anomaly | 0.52% | 48.09% | **38.97%** |
| timing_gap_anomaly | 46.97% | 50.13% | **49.84%** |
| timing_only | 0.00% | 0.00% | **0.00%** |

**Content-novelty holds identically to BGL** — 99.99%, matching both BGL and the biased slice.

**Pure-order anomaly is confirmed as a genuine, robust difference from BGL — not a single-incident
artifact.** Moving from the biased slice (48.09%, 1 dominant incident) to a genuinely diverse
47-node, 8-label sample only brought the number down to 38.97% — still **75x BGL's rate**. This
rules out "it was just the one VAPI storm" as the explanation. The eligibility gate that filters
which anomalies can even be evaluated for pure-order also shifted in an informative direction:
13,639 of 170,422 anomalies (8.0%) were excluded here for touching a node-locally-unfamiliar
template, vs essentially none (24 of 226,095, 0.01%) in the biased slice — a more diverse sample
surfaces more node-local content novelty too, as expected, but even among the 92% that remain
eligible, order disruption is common. Thunderbird's anomalies are structurally different from
BGL's: more likely to reuse content the node already produces normally, while still landing in an
unusual local sequence position.

**Timing-gap anomaly is consistent across all three views (~47-50%)**, but see Part 3 for why this
number needs to be read cautiously on Thunderbird specifically.

**Plain statement**: the premise (native labels are content-driven) holds for the content
signature specifically (99.99%, matching BGL). It does **not** hold the same way for order — this
is a real, checked, mechanism-backed structural difference from BGL, present at a genuinely
diverse sample, not an artifact of slice selection. Timing-gap-anomaly rate is similar in magnitude
to BGL's, but its reliability is a separate question, addressed next.

## Part 3 — Gate check: is per-node timing detection viable on Thunderbird?

This is the load-bearing question before injecting anything. Computed directly (not estimated):

- **4,722 total nodes** in the representative slice.
- **3,450 nodes (73.06%) have a valid, non-degenerate per-node timing baseline** (≥10 normal
  normal-to-normal gaps, MAD > 0) — for these, the existing per-node MAD-based methodology (used
  unchanged throughout the BGL work) applies directly and should work.
- **1,270 nodes (26.94%) do not**: 12 (0.25%) have too few normal gaps, and **1,258 (26.64%) have
  an exactly-zero per-node MAD** — a direct consequence of the 55% global zero-gap rate: a node
  whose normal gaps are mostly-or-all 0 has no measurable dispersion at all.
- **The pooled fallback itself is broken on this dataset.** For BGL, nodes with an unusable
  per-node baseline fell back to a healthy pooled (global) median/MAD. Here, the pooled fallback
  computed the same way — **median = 0.0000s, MAD = 0.0000s** — because more than half of *all*
  nodes' gaps are exactly zero, so the zero value dominates the global distribution too. Any node
  routed to this fallback would have its timing z-score/log-ratio effectively computed against a
  scale of zero (floored only by the arbitrary `MAD_FLOOR_SEC` constant), which is not a
  meaningful "typical variability" reference the way BGL's pooled fallback was.

**Verdict: conditionally viable, not a clean yes.** Per-node timing detection works as-is for the
73% majority of nodes that have real per-node dispersion. It does **not** currently work for the
other 27% — not because those nodes are unusable in principle, but because the *fallback
mechanism* that was supposed to cover them is itself degenerate on this dataset. This is a
different failure mode than "timing detection doesn't work on Thunderbird" — it's "the existing
low-data-node safety net needs a Thunderbird-specific fix before it can be trusted here." Two
concrete fix directions (not implemented, per instruction to stop): (a) restrict injector node
eligibility to the 3,450 valid-baseline nodes only, skipping the pooled-fallback path entirely for
this dataset, or (b) recompute the pooled fallback from non-zero gaps only, which would likely
restore a meaningful (if coarse) global scale. Either is a small, targeted change, not a redesign.

**This does change how Thunderbird should be used relative to BGL**: content-detector behavior
(PCA, isolation-forest-on-counts) can be evaluated on Thunderbird exactly as on BGL, no caveats.
Timing-fault injection needs the fallback fix above decided and applied first, and even after that,
27% of nodes carry a structurally different (zero-heavy) timing signal than BGL's nodes did — worth
tracking separately in any cross-dataset comparison, not averaged away.

## Files produced this pass

- `data/raw/Thunderbird.log` — **replaced**: now the 10M-line representative slice (lines
  50M-60M of the source), not the original biased 5M-line head slice (deleted)
- `data/processed/thunderbird_parsed.parquet` — re-parsed from the representative slice
- `results/thunderbird_premise_audit_v2.csv` — the 4-signature summary table, representative slice
- `results/thunderbird_setup_notes_v2.md` — this file
- `results/thunderbird_premise_audit.csv`, `results/thunderbird_setup_notes.md` — **v1, frozen**,
  preserved for the before/after comparison, not regenerated

No injector or detector code was run against Thunderbird this pass, per instruction. Stopping here
pending the injection prompt.
