# Multi-seed AUC-ROC stability

**SMOKE-TEST OUTPUT -- subsample=300000, n_injections_override=10. Numbers below are NOT representative of real detector performance; this run only verifies the harness executes end-to-end and the schema/aggregation are correct. Re-run without `--subsample`/`--n-injections` on the server for real numbers.**

Seeds used: [42, 43]

Server run command (full 5-seed run, both datasets, both fault types):

```
python src/multiseed_auc.py --n-seeds 5 --dataset both --fault both
```

Seeds completed so far: [42, 43], elapsed 78.0s

| dataset | fault | detector | mean AUC-ROC | std | min | max | n seeds | crosses chance (0.5)? |
|---|---|---|---|---|---|---|---|---|
| BGL | burst | count_vector_pca | 0.9104 | 0.0678 | 0.8624 | 0.9583 | 2 | no |
| BGL | burst | isolation_forest_counts | 0.9592 | 0.0234 | 0.9426 | 0.9757 | 2 | no |
| BGL | burst | log_ratio_threshold | 0.9500 | 0.0170 | 0.9379 | 0.9620 | 2 | no |
| BGL | burst | z_score_threshold | 0.9789 | 0.0184 | 0.9658 | 0.9919 | 2 | no |
| BGL | stall | count_vector_pca | 0.9630 | 0.0011 | 0.9622 | 0.9637 | 2 | no |
| BGL | stall | isolation_forest_counts | 0.9013 | 0.0337 | 0.8775 | 0.9252 | 2 | no |
| BGL | stall | log_ratio_threshold | 0.8424 | 0.0335 | 0.8187 | 0.8661 | 2 | no |
| BGL | stall | z_score_threshold | 0.9436 | 0.0032 | 0.9414 | 0.9459 | 2 | no |
| Thunderbird | burst | count_vector_pca | 0.2589 | 0.0831 | 0.2001 | 0.3177 | 2 | no |
| Thunderbird | burst | isolation_forest_counts | 0.6501 | 0.0556 | 0.6108 | 0.6895 | 2 | no |
| Thunderbird | burst | log_ratio_threshold | 0.6478 | 0.0359 | 0.6224 | 0.6731 | 2 | no |
| Thunderbird | burst | z_score_threshold | 0.4363 | 0.0280 | 0.4166 | 0.4561 | 2 | no |
| Thunderbird | stall | count_vector_pca | 0.6900 | 0.0766 | 0.6359 | 0.7441 | 2 | no |
| Thunderbird | stall | isolation_forest_counts | 0.4949 | 0.0578 | 0.4540 | 0.5357 | 2 | **YES** |
| Thunderbird | stall | log_ratio_threshold | 0.6644 | 0.0754 | 0.6111 | 0.7177 | 2 | no |
| Thunderbird | stall | z_score_threshold | 0.4796 | 0.1197 | 0.3949 | 0.5643 | 2 | **YES** |

Does NOT overwrite results/auc_metrics.csv. Raw per-seed values: results/multiseed_auc.csv.

## Estimated cost of the full 5-seed server run (not measured -- extrapolated from the smoke test)

Smoke-test measurements (this machine, 2 seeds, both datasets subsampled to 300,000 rows each,
n_injections=10): **79.9s wall clock, 3.80GB peak memory footprint (1.88GB max RSS)**.

**Time.** The smoke test's per-(dataset,fault) cost scales with row count (rolling-feature
computation, sorting, count-matrix construction), not injection count. Full BGL is ~4.7M rows
(15.7x the 300k-row subsample); full Thunderbird is ~10M rows (33x the subsample, though the
subsample's chronological-prefix slice likely under-represents Thunderbird's true node diversity,
so this ratio is probably an underestimate for Thunderbird specifically). Linear extrapolation:
roughly **4-5 min per (dataset, fault) per seed for BGL** and **8-10 min for Thunderbird**, i.e.
very roughly **35-45 min for all BGL seeds/faults + 80-100 min for all Thunderbird seeds/faults +
a few minutes of context loading ~= 2-2.5 hours total, sequential, on comparable CPU hardware.** A
faster/more-parallel server should only improve this.

**Memory.** Both dataset contexts (raw + node-sorted/context-augmented copies) are held in memory
for the entire run, with only the per-seed transient scoring objects freed (`del` + `gc.collect()`)
between iterations -- so peak memory should stay roughly flat across seeds rather than growing,
but the baseline "both full datasets resident" cost is large. Naive linear extrapolation from the
smoke test (variable cost only, backing out a ~0.5GB fixed interpreter/library floor) gives a very
rough **60-90GB peak** for the full 14.7M-row combined dataset. This is a coarse order-of-magnitude
estimate, not a measurement -- **budget for a server with at least 64GB RAM, ideally 128GB for
margin.** If the server has less, run `--dataset bgl` and `--dataset thunderbird` as two separate
invocations (roughly halving peak memory) and concatenate the two resulting
`results/multiseed_auc.csv` files by hand -- the script does not currently append across runs, it
overwrites both output files with only the current invocation's rows.