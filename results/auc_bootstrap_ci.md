# Bootstrap 95% CIs for AUC-ROC (n=100 injections per dataset x fault type)

Cluster bootstrap over the 100 injection events (1000 resamples, seed=0), negatives held fixed as the background population. Point estimates recomputed here match `results/auc_metrics.csv` to within 1.8e-02 (read-only cross-check; that file was not modified). See `src/auc_bootstrap_ci.py` module docstring for the full method.

| Dataset | Fault | Detector | AUC-ROC | 95% CI lower | 95% CI upper | Crosses chance (0.5)? |
|---|---|---|---|---|---|---|
| BGL | stall | count_vector_pca | 0.6025 | 0.5699 | 0.6346 | no |
| BGL | stall | z_score_threshold | 0.4668 | 0.4341 | 0.5047 | **yes** |
| BGL | stall | log_ratio_threshold | 0.4830 | 0.4324 | 0.5311 | **yes** |
| BGL | stall | isolation_forest_counts | 0.7860 | 0.7453 | 0.8232 | no |
| BGL | burst | count_vector_pca | 0.6590 | 0.6177 | 0.7014 | no |
| BGL | burst | z_score_threshold | 0.5429 | 0.4798 | 0.5968 | **yes** |
| BGL | burst | log_ratio_threshold | 0.6648 | 0.6137 | 0.7235 | no |
| BGL | burst | isolation_forest_counts | 0.7833 | 0.6814 | 0.8623 | no |
| Thunderbird | stall | count_vector_pca | 0.5610 | 0.4841 | 0.6418 | **yes** |
| Thunderbird | stall | z_score_threshold | 0.5580 | 0.5041 | 0.6059 | no |
| Thunderbird | stall | log_ratio_threshold | 0.6014 | 0.5633 | 0.6383 | no |
| Thunderbird | stall | isolation_forest_counts | 0.7334 | 0.6836 | 0.7800 | no |
| Thunderbird | burst | count_vector_pca | 0.4544 | 0.3596 | 0.5687 | **yes** |
| Thunderbird | burst | z_score_threshold | 0.5988 | 0.4971 | 0.6900 | **yes** |
| Thunderbird | burst | log_ratio_threshold | 0.6031 | 0.5614 | 0.6581 | no |
| Thunderbird | burst | isolation_forest_counts | 0.6362 | 0.5468 | 0.7384 | no |

**6 of 16 cells have a 95% CI that crosses 0.5** -- for these, "above chance" or "below chance" is not statistically defensible at n=100 injections and the paper's wording should be softened accordingly (e.g. "not distinguishable from chance at this sample size" rather than asserting a direction).

Cells crossing chance:
- BGL / stall / z_score_threshold: AUC-ROC=0.4668, CI=[0.4341, 0.5047]
- BGL / stall / log_ratio_threshold: AUC-ROC=0.4830, CI=[0.4324, 0.5311]
- BGL / burst / z_score_threshold: AUC-ROC=0.5429, CI=[0.4798, 0.5968]
- Thunderbird / stall / count_vector_pca: AUC-ROC=0.5610, CI=[0.4841, 0.6418]
- Thunderbird / burst / count_vector_pca: AUC-ROC=0.4544, CI=[0.3596, 0.5687]
- Thunderbird / burst / z_score_threshold: AUC-ROC=0.5988, CI=[0.4971, 0.6900]
