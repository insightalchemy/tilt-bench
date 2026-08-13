# Multi-seed AUC-ROC stability

Seeds used: [42, 43, 44, 45, 46]

Server run command (full 5-seed run, both datasets, both fault types):

```
python src/multiseed_auc.py --n-seeds 5 --dataset both --fault both
```

Seeds completed so far: [42, 43, 44, 45, 46], elapsed 5949.6s

| dataset | fault | detector | mean AUC-ROC | std | min | max | n seeds | crosses chance (0.5)? |
|---|---|---|---|---|---|---|---|---|
| BGL | burst | count_vector_pca | 0.6293 | 0.0250 | 0.5934 | 0.6593 | 5 | no |
| BGL | burst | isolation_forest_counts | 0.8023 | 0.0213 | 0.7719 | 0.8292 | 5 | no |
| BGL | burst | log_ratio_threshold | 0.6864 | 0.0263 | 0.6626 | 0.7299 | 5 | no |
| BGL | burst | z_score_threshold | 0.4848 | 0.0232 | 0.4541 | 0.5081 | 5 | **YES** |
| BGL | stall | count_vector_pca | 0.6041 | 0.0125 | 0.5883 | 0.6191 | 5 | no |
| BGL | stall | isolation_forest_counts | 0.7799 | 0.0190 | 0.7532 | 0.8008 | 5 | no |
| BGL | stall | log_ratio_threshold | 0.4694 | 0.0238 | 0.4340 | 0.4931 | 5 | no |
| BGL | stall | z_score_threshold | 0.4650 | 0.0196 | 0.4431 | 0.4900 | 5 | no |
| Thunderbird | burst | count_vector_pca | 0.5045 | 0.0595 | 0.4318 | 0.5847 | 5 | **YES** |
| Thunderbird | burst | isolation_forest_counts | 0.6182 | 0.0729 | 0.5045 | 0.7037 | 5 | no |
| Thunderbird | burst | log_ratio_threshold | 0.5927 | 0.0315 | 0.5607 | 0.6392 | 5 | no |
| Thunderbird | burst | z_score_threshold | 0.6102 | 0.0743 | 0.5200 | 0.7222 | 5 | no |
| Thunderbird | stall | count_vector_pca | 0.5947 | 0.0265 | 0.5560 | 0.6196 | 5 | no |
| Thunderbird | stall | isolation_forest_counts | 0.6791 | 0.0285 | 0.6390 | 0.7101 | 5 | no |
| Thunderbird | stall | log_ratio_threshold | 0.5853 | 0.0288 | 0.5396 | 0.6133 | 5 | no |
| Thunderbird | stall | z_score_threshold | 0.5800 | 0.0224 | 0.5481 | 0.6078 | 5 | no |

Does NOT overwrite results/auc_metrics.csv. Raw per-seed values: results/multiseed_auc.csv.