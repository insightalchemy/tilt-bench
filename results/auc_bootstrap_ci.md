# Bootstrap 95% CIs for AUC-ROC (n=100 injections per dataset x fault type)

Cluster bootstrap over the 100 injection events (1000 resamples, seed=0), negatives held fixed as the background population. Point estimates recomputed here match `results/auc_metrics.csv` to within 5.6e-17 (read-only cross-check; that file was not modified). See `src/auc_bootstrap_ci.py` module docstring for the full method.

| Dataset | Fault | Detector | AUC-ROC | 95% CI lower | 95% CI upper | Crosses chance (0.5)? |
|---|---|---|---|---|---|---|
| BGL | stall | count_vector_pca | 0.6026 | 0.5704 | 0.6347 | no |
| BGL | stall | z_score_threshold | 0.4645 | 0.4333 | 0.5010 | **yes** |
| BGL | stall | log_ratio_threshold | 0.4821 | 0.4309 | 0.5298 | **yes** |
| BGL | stall | isolation_forest_counts | 0.7863 | 0.7459 | 0.8232 | no |
| BGL | burst | count_vector_pca | 0.6590 | 0.6178 | 0.7014 | no |
| BGL | burst | z_score_threshold | 0.5328 | 0.4706 | 0.5878 | **yes** |
| BGL | burst | log_ratio_threshold | 0.6631 | 0.6083 | 0.7244 | no |
| BGL | burst | isolation_forest_counts | 0.7817 | 0.6798 | 0.8612 | no |
| Thunderbird | stall | count_vector_pca | 0.5610 | 0.4842 | 0.6418 | **yes** |
| Thunderbird | stall | z_score_threshold | 0.5552 | 0.5020 | 0.6019 | no |
| Thunderbird | stall | log_ratio_threshold | 0.5977 | 0.5597 | 0.6353 | no |
| Thunderbird | stall | isolation_forest_counts | 0.7381 | 0.6868 | 0.7851 | no |
| Thunderbird | burst | count_vector_pca | 0.4543 | 0.3595 | 0.5686 | **yes** |
| Thunderbird | burst | z_score_threshold | 0.5941 | 0.4916 | 0.6863 | **yes** |
| Thunderbird | burst | log_ratio_threshold | 0.5979 | 0.5536 | 0.6554 | no |
| Thunderbird | burst | isolation_forest_counts | 0.6181 | 0.4931 | 0.7551 | **yes** |

**7 of 16 cells have a 95% CI that crosses 0.5** -- for these, "above chance" or "below chance" is not statistically defensible at n=100 injections and the paper's wording should be softened accordingly (e.g. "not distinguishable from chance at this sample size" rather than asserting a direction).

Cells crossing chance:
- BGL / stall / z_score_threshold: AUC-ROC=0.4645, CI=[0.4333, 0.5010]
- BGL / stall / log_ratio_threshold: AUC-ROC=0.4821, CI=[0.4309, 0.5298]
- BGL / burst / z_score_threshold: AUC-ROC=0.5328, CI=[0.4706, 0.5878]
- Thunderbird / stall / count_vector_pca: AUC-ROC=0.5610, CI=[0.4842, 0.6418]
- Thunderbird / burst / count_vector_pca: AUC-ROC=0.4543, CI=[0.3595, 0.5686]
- Thunderbird / burst / z_score_threshold: AUC-ROC=0.5941, CI=[0.4916, 0.6863]
- Thunderbird / burst / isolation_forest_counts: AUC-ROC=0.6181, CI=[0.4931, 0.7551]
