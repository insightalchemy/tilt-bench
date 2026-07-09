# Thunderbird setup notes (exploratory prep — injector/detectors NOT yet run)

## Part 1 — Acquisition

Source: `https://zenodo.org/records/8196385/files/Thunderbird.tar.gz` (2.02 GB compressed, single
entry `Thunderbird.log`, **29.6 GB decompressed** — confirmed via a small ranged download + `tar -tzv`
before committing to anything, since available disk was only 17 GB).

Given the size mismatch (29.6 GB vs 17 GB free), the full archive was never downloaded or
decompressed to disk. Instead: `curl -sL <url> | tar -xzO | head -n 5000000 > data/raw/Thunderbird.log`
— a single streaming pipeline. `head` closes its input once it has read 5,000,000 lines, which
raises `SIGPIPE` back through `tar` and `curl`, terminating the download early (confirmed in stderr:
`Write error: Broken pipe`). Only ~2.4% of the compressed stream needed to be fetched. No
intermediate full-archive file was ever written, so there was nothing to delete afterward — the
868,147,617-byte `data/raw/Thunderbird.log` is the only artifact, and it *is* the working slice.

- **Lines used**: 5,000,000 (contiguous head slice — the file's first 5M lines in original
  chronological order, not a random sample, so timing structure is intact).
- **File size**: 868,147,617 bytes (≈828 MiB), comparable in scale to BGL's 743 MB / 4.75M lines.
- **Contiguity confirmed**: first/last lines inspected directly — the last line
  (`dn30 ... Nov 18 01:09:39 ... FATAL state`) is a complete, well-formed line, not truncated
  mid-record; no null bytes or binary corruption (`file` reports plain ASCII text).
- Placed at `data/raw/Thunderbird.log`, following the same `data/raw/<Dataset>.log` convention as
  `data/raw/BGL.log`. **Note**: unlike `BGL.log`, this is deliberately a 5M-line slice of a much
  larger source file, not the complete dataset — flagged here so it isn't mistaken for the full
  Thunderbird corpus (211M+ lines) in any future work.

## Part 2 — Parsing

Raw format is structurally different from BGL (expected — different system, different syslog
convention), parsed with a new field-splitting adapter (`src/parser_thunderbird.py`) that reuses
the *same* Drain3-based approach, chunked reading, and output schema as `src/parser.py` — no
redesign of the pipeline, just a different 8-field-plus-content split:

```
<label> <unix_ts> <date> <node> <month> <day> <time HH:MM:SS> <user@host> <content...>
```

vs BGL's 9-field format with a distinct microsecond-precision field. Output columns are identical
to BGL's: `timestamp, node, label, anomaly, event_template, raw_message`.

**Summary** (`data/processed/thunderbird_parsed.parquet`):
- Total lines: 5,000,000, all parsed (0 skipped/malformed)
- % anomalous: **4.52%** (226,095 rows) — vs BGL's 7.34%
- Unique nodes: 4,533 (BGL: 69,252 — Thunderbird's slice covers far fewer distinct hosts)
- Unique event templates: 1,703 (BGL: 2,388)
- Timestamp range: 2005-11-09 08:05:01 → 2005-11-18 09:09:39 (9 days)

**Timestamp resolution — a meaningful difference from BGL, checked directly, not assumed:**
- `dtype` is `datetime64[s]` — **second-level only**. There is no sub-second field anywhere in
  Thunderbird's raw format (BGL had a dedicated microsecond field).
- Consequence, measured directly: **62.11%** of all per-node inter-arrival gaps are exactly 0
  seconds (BGL: ~0.00%). 37.2% of nodes are majority-zero-gap.
- The self-reported `<month> <day> <time>` field was spot-checked against `unix_ts` and found
  *not fully reliable*: 4 of 5 sampled lines showed a consistent −8h offset (a plausible fixed
  Pacific-time convention, same idea as BGL's timezone offset), but 1 of 5 showed a +16h
  "offset" that is actually a one-day rollover error in the self-reported field — real per-node
  clock drift in the source data. `unix_ts` was used as the authoritative timestamp for this
  reason, but the underlying resolution limit (whole seconds) applies to both fields equally.

**This affects everything downstream**: any timing-based analysis on Thunderbird (baseline
inter-arrival variability, z-score/log-ratio features, the whole point of TILT-Bench) will be
working with a much coarser, more collision-prone signal than BGL provided. This should be treated
as a first-order consideration before deciding whether Thunderbird is a good second dataset for the
injector, not a minor footnote.

## Part 3 — Premise audit

Ran the exact same signatures used for BGL (`src/premise_audit.py`'s `flag_template_rarity`,
`flag_pure_order_anomaly`, `flag_timing_gap_anomaly`, unmodified) via
`src/premise_audit_thunderbird.py`. Result: `results/thunderbird_premise_audit.csv`.

| signature | BGL | Thunderbird |
|---|---|---|
| new_or_rare_template | 99.31% | **100.00%** |
| pure_order_anomaly | 0.52% | **48.09%** |
| timing_gap_anomaly | 46.97% | 50.13% |
| timing_only | 0.00% | 0.00% |

**Content-novelty holds identically** — Thunderbird's anomalies are, if anything, even more
uniformly content-novel than BGL's (100% vs 99.3%).

**Pure-order anomaly does NOT hold the same way — this is a real, meaningful difference, not
noise.** BGL's anomalies almost never showed order disruption once content novelty was controlled
for (0.52%); Thunderbird's show it **48% of the time**. This is not simply "Thunderbird is
different" in a vague sense — it traces to a specific, checkable mechanism: for BGL, 99.5% of
anomalies were *excluded* from the pure-order check because they used at least one template the
node had never produced normally (content and order novelty were entangled). For Thunderbird, only
**0.01%** of anomalies were excluded that way (24 of 226,095) — the overwhelming majority of
Thunderbird's anomalous rows use templates that specific node *has already produced normally*, so
the order-novelty check is actually evaluable at scale, and a large fraction of it comes back
"unusual." Content familiarity + order disruption is a structurally different anomaly shape than
BGL's content-novelty-dominated one.

**Critical caveat, surfaced rather than smoothed over: this 5M-line slice's anomaly signal is
almost entirely one incident.** 226,071 of 226,095 anomalies (99.99%) carry the same label
(`VAPI`, an InfiniBand fault code), concentrated on just 6 nodes (`dn30` alone: 126,258
occurrences). The other three label types observed (`CPU`, `EXT_FS`, `ECC`) total 24 rows combined.
**The pure-order (48%) and timing-gap (50%) numbers above are therefore substantially a
description of one repeating VAPI-storm incident, not a diverse sample of Thunderbird anomaly
behavior.** The timing-gap number is further weakened by small-sample baseline degradation: only
15 distinct nodes have any anomalous rows at all, and of those, 10 fell back to the pooled
baseline specifically because their per-node MAD was exactly zero (consistent with the severe
zero-gap collision problem above) — meaning two-thirds of the anomalous nodes' timing signature is
being judged against a coarse pooled reference, not their own baseline.

**Plain statement**: the premise (native labels are content-driven, not timing-driven) holds for
Thunderbird's content signature, but the pure-order finding is a genuine structural difference from
BGL that should not be waved away as sampling noise — it has a clear mechanism (content
familiarity at the node level) — while simultaneously the *magnitude* of that finding, and the
timing-gap number, are both heavily colored by this slice being dominated by a single incident
type on a handful of nodes. Before treating Thunderbird as a confirmed second dataset, a
larger and/or differently-sampled slice (or explicit stratification across incident types) would
be needed to know whether 48% pure-order is a real Thunderbird-wide property or an artifact of
which 5M lines happened to be first in the file.

## Files produced this pass

- `data/raw/Thunderbird.log` — 5M-line contiguous head slice (raw, untouched further)
- `data/processed/thunderbird_parsed.parquet` — parsed tidy table
- `results/thunderbird_premise_audit.csv` — the 4-signature summary table
- `results/thunderbird_setup_notes.md` — this file
- `src/parser_thunderbird.py`, `src/premise_audit_thunderbird.py` — the two adapter scripts (reuse
  BGL's Drain3/premise-audit logic unmodified; only the raw-line field-splitting is new)

No injector or detector code was run against Thunderbird, per instruction. Stopping here.
