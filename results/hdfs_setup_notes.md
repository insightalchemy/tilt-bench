# HDFS setup notes (design + audit only — NO injector, NO detectors run this pass)

HDFS is structurally unlike BGL/Thunderbird: block-session grouping instead of a continuous
per-node stream, and block-level (not line-level) anomaly labels. This pass exists specifically to
determine whether timing-fault injection is even well-defined here before building anything.

## Part 1 — Acquisition and parsing

Source: `https://zenodo.org/records/8196385/files/HDFS_v1.zip` (186.6 MB compressed — small enough
to download directly, unlike BGL/Thunderbird's multi-GB archives; no streaming tricks needed).
Contains `HDFS.log` (1.58 GB) and `preprocessed/anomaly_label.csv` (per-block Normal/Anomaly
labels, 575,061 rows) — both extracted.

**Raw format** (fixed 4 fields + free-text content, no first-token alert label like BGL/Thunderbird):
```
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
```
`<date YYMMDD> <time HHMMSS> <pid> <level> <component>: <content>`. Every line references exactly
one `blk_-?\d+` block ID (confirmed: 0 lines lack one). A new parser (`src/parser_hdfs.py`) reuses
the same Drain3/chunked architecture, extracts block_id via regex, and does a single vectorized
join of `anomaly_label.csv` onto every line by block_id at the end (not per-line inside the hot
loop) — 100% join coverage, 0 unmatched block_ids.

**Summary** (`data/processed/hdfs_parsed.parquet`):
- 11,175,629 lines, 0 skipped
- 575,061 distinct block sessions (matches the label file exactly)
- % anomalous LINES: 2.58% / % anomalous BLOCK SESSIONS: 2.93% (16,838 of 575,061)
- Unique event templates: **114** (vs BGL's 2,388, Thunderbird's 9,874 — HDFS logging is far more
  repetitive/structured; this alone is a hint of what Part 3 confirms)
- **Node coverage: only 15.48%** of rows carry an extractable DataNode IP (the `dest:` address on
  `DataXceiver`/`Receiving block` lines specifically) — most HDFS log line types (PacketResponder,
  FSNamesystem bookkeeping, block-scanner verification) never mention an IP at all. 203 distinct
  DataNode IPs found.
- Timestamp range: 2008-11-09 20:35:18 → 2008-11-11 11:16:28 (**~38.7 hours** — far shorter than
  BGL's 7 months or Thunderbird's 3 weeks; this is a short, intense benchmark run, not a
  long-deployment log)
- Timestamp resolution: `datetime64[us]` internally, but the SOURCE data is whole-seconds only —
  no sub-second field exists anywhere in the raw format (same situation as Thunderbird, arguably
  worse: Thunderbird at least had a redundant self-reported field to cross-check; HDFS has exactly
  one second-resolution timestamp per line and nothing else)
- **Zero-gap fraction (within-block consecutive events): 62.79%** — essentially the same severity
  as Thunderbird's 55%.

## Part 2 — Defining "inter-arrival time" on HDFS (the hard part)

HDFS has no field analogous to BGL/Thunderbird's per-node identity attached to every line. Two
candidate groupings were evaluated, with real numbers, not assumptions:

### (a) Within-block: gaps between consecutive events of the same block_id

- **Coverage is excellent**: block session size is remarkably consistent (median 19 events, IQR
  19–20, min 2, max 298) — 98.92% of blocks have ≥10 events, 0% have only 1 event (every block
  supports at least one gap).
- **Baseline variability is not**: of normal-labeled blocks, **100% have ≥10 within-block gaps,
  but only 14.24% (79,498 of 558,223) have a non-degenerate (MAD > 0) baseline.** The other 85.76%
  have a per-block MAD of exactly zero — a block's ~19-event write pipeline typically completes so
  fast, at whole-second resolution, that most or all of its gaps collapse to the same value.
- **The pooled fallback is also compromised, in a new way**: zero-inclusive pooled median/MAD are
  both 0 (as expected, mirroring Thunderbird). But even the *zero-excluded* pooled fallback gives
  median=39s, MAD=38s — suspiciously large for what should be a fast write-pipeline signal. Likely
  explanation: `block_id` "sessions" aren't strictly one bounded write burst — a block can be
  re-mentioned much later (e.g. `DataBlockScanner: Verification succeeded for blk_X`, observed
  directly in the raw sample) for reasons unrelated to the original write, injecting rare
  long-delay gaps into the pooled population. The within-block gap sequence is not always a single
  coherent episode.
- **Mechanically**: a stall/burst injection (shift-everything-downstream-in-the-same-session) is
  well-defined here — a block session is bounded and complete (unlike candidate b below), so
  "shift every subsequent same-block event forward" has no missing-event problem. It would just be
  operating on a short (~19-event) session rather than a long per-node stream, and would need to
  be restricted to the 14.24% of blocks with usable baseline variability — the same
  "restrict eligibility to valid-baseline units" fix already applied for Thunderbird, just far
  more aggressive here (14% eligible vs Thunderbird's 73%).

### (b) Per-DataNode: gaps between consecutive events sharing the same extracted `dest:` IP, across blocks

- **Numerically, this is excellent** — better than either within-block-HDFS or Thunderbird:
  despite only 15.48% row coverage, that's still spread across just 203 distinct DataNodes, so
  each one accumulates thousands of events (mean 8,524, median 8,644, min 121) across the full
  38.7-hour window. **100% of the 203 nodes have ≥10 normal-sequence gaps AND non-degenerate MAD —
  every single one is "valid."** Zero-gap fraction here is also the lowest of any candidate/dataset
  checked: 28.4%.
- **But two problems make this NOT a sound basis for injection, despite the clean numbers**:
  1. **Incomplete propagation.** Only the 15.48% of a DataNode's log lines that happen to mention
     its IP would be visible to "this node's stream" — the other 84.5% of that node's true logged
     activity (PacketResponder completions, block-scanner events, etc.) would be invisible to and
     unaffected by an injection on this stream. CLAUDE.md's propagation rule ("delays *everything*
     downstream from that component") would be violated by construction: a "stalled" DataNode
     would still show plenty of untouched, normally-timed activity in the other 84.5% of its rows,
     because those rows aren't linked to the IP-bearing subset at all.
  2. **Not one coherent process.** A DataNode participates in many concurrent, causally-independent
     block transfers simultaneously (that's the point of a distributed filesystem). The per-node
     gap sequence built here is an interleaving of unrelated block operations, not one process's
     pacing the way a BGL node's RAS log was. "Stalling this DataNode" doesn't have a clean causal
     meaning the way "stalling this compute node" did on BGL.

### Verdict

**Neither candidate is a clean port of the BGL/Thunderbird per-node injection model.**
Within-block (a) is mechanically sound (complete, bounded, no missing-event problem) but
statistically thin (14.24% of blocks usable) and its pooled fallback is contaminated by rare
much-later block re-mentions. Per-DataNode (b) is statistically excellent but mechanically
unsound — it would inject into an incomplete, non-causally-coherent projection of a node's
activity, breaking the propagation guarantee this project has held to throughout.

If HDFS injection is pursued, **(a) within-block, restricted to the 14.24% valid-baseline subset**
is the more defensible starting point — same "restrict eligibility" pattern already proven on
Thunderbird, just with a much smaller eligible pool and a genuine open question about whether 15
injections (BGL/Thunderbird's count) can even be drawn from a pool this size without exhausting
distinct blocks' content diversity. This is a design decision for the next pass, not resolved here.

## Part 3 — Premise audit, adapted to block-session structure

Ran the exact same signatures (`src/premise_audit.py`, unmodified) via `src/premise_audit_hdfs.py`,
using `node := block_id` (HDFS's complete, always-present grouping key — reusing the per-node
machinery as per-block machinery, consistent with candidate (a) above).

| signature | BGL | Thunderbird | **HDFS** |
|---|---|---|---|
| new_or_rare_template | 99.31% | 99.99% | **4.53%** |
| pure_order_anomaly | 0.52% | 38.97% | **0.00%\*** |
| timing_gap_anomaly | 46.97% | 49.84% | **59.27%\*\*** |

**\*pure_order_anomaly is not meaningfully evaluable here, and 0.00% should not be read as "clean."**
Its eligibility check requires the SAME node(=block) to have ALSO produced the row's template(s)
*normally*. But HDFS labels are uniform per block (confirmed: 0 blocks have rows disagreeing on
`anomaly`) — an anomalous block has zero normal rows by construction, so it can never supply its
own "normal content" reference. Directly confirmed: **0 of 16,838 anomalous blocks have any
representation in the per-block baseline at all.** 288,250 of 288,522 anomalous rows (99.9%) were
excluded from evaluation for exactly this reason. The signature is structurally inapplicable under
block-level labeling, not evidence that HDFS's anomalies preserve normal ordering.

**\*\*timing_gap_anomaly's 59.27% is measured but not trustworthy as a real statistical signal.**
Because anomalous blocks have zero normal-sequence rows (same root cause as above), **100% of the
16,838 anomalous blocks fall back to the pooled baseline** — never their own history. And the
pooled baseline is the degenerate zero-inclusive one (median=MAD=0, per Part 2), floored only by
the arbitrary `MAD_FLOOR_SEC` constant. With that floor, any gap over ~3ms already exceeds
`Z_THRESH=3` — at whole-second resolution, this means almost any nonzero gap trivially "fires."
The 59.27% figure is closer to "fraction of anomalous rows with a nonzero within-block gap" than a
genuine measure of timing unusualness relative to a real baseline.

**The one number that IS a clean, robust, reliable finding**: new_or_rare_template at **4.53%** —
the near-total *inverse* of BGL and Thunderbird. This was foreshadowed directly in the first parsed
sample (an `Anomaly`-labeled row shared the exact same Drain template as surrounding normal rows)
and confirmed at scale here. HDFS's anomaly labels were constructed from block write-pipeline
failures (missing acknowledgments, wrong replica counts, incomplete completion sequences) — a
fundamentally different anomaly *mechanism* than BGL/Thunderbird's largely content-driven native
alerts. HDFS anomalies are about the SHAPE of a block's event sequence (which events happened, how
many, in what completeness), not about any individual line's content being unusual.

## Plain statement: does the premise hold on HDFS?

**No — not in the form established for BGL/Thunderbird, and the reason is itself the interesting
result.** The premise ("native anomalies are content-driven, timing is a blind spot") was built
around datasets where anomalies manifest as unusual log *content*. HDFS's anomalies manifest as
unusual *event-sequence shape* at the block-session level — neither cleanly "content novelty" (the
lines look normal) nor cleanly "timing" in the inter-arrival sense audited here (the pure-order and
timing signatures both hit structural walls specific to block-level labeling). This suggests HDFS's
actual blind spot, if any, may be a **third category — sequence-completeness / missing-event
anomalies** — distinct from both content detectors and the timing detectors built for BGL/Thunderbird.
Neither is currently designed to catch "this block was missing its third replica acknowledgment."

## Overall recommendation

HDFS is workable as a dataset, but **not as a drop-in port of the BGL/Thunderbird timing-injection
design**. Before committing further: (1) decide whether to pursue within-block injection on the
14.24% valid-baseline subset (a real but narrow design, and untested at what sample size 15
injections would still be meaningful), (2) treat the premise-audit's pure-order and timing-gap
numbers for HDFS as unreliable until/unless a block-appropriate alternative signature is designed,
and (3) consider whether HDFS is better used to test a sequence-completeness detector family
instead of (or alongside) the timing-fault injector this project has built so far. Stopping here
per instruction — no injector, no detectors.

## Files produced this pass

- `data/raw/HDFS.log`, `data/raw/anomaly_label.csv` — raw data (untouched further)
- `data/processed/hdfs_parsed.parquet` — parsed tidy table (timestamp, node, block_id, label,
  anomaly, event_template, raw_message)
- `results/hdfs_premise_audit.csv` — the 4-signature summary table (with the caveats above)
- `results/hdfs_setup_notes.md` — this file
- `src/parser_hdfs.py`, `src/premise_audit_hdfs.py` — the two adapter scripts (reuse BGL's
  Drain3/premise-audit logic unmodified; block_id extraction, IP extraction, and the label join
  are the only new logic)
