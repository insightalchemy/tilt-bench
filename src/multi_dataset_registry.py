"""
Registry and path conventions for the additional Loghub datasets used in the multi-dataset
invariance and placebo experiments (Zhu et al., "Loghub: A Large Collection of System Log Datasets
for AI-driven Log Analytics", ISSRE 2023; Zenodo record 3227177).

These are component/application logs. Their labels are not used: the invariance proposition only
needs timestamps and a stream key, so anomaly is always False and label always "-" in the parsed
output. Each src/parser_<name>.py documents which raw field serves as the stream key ("node").
All downstream code groups only by that column.

Path conventions (all derived from `name`):
  data/raw/<name>_raw/                                   -- extracted archive contents (file or dir)
  data/processed/<name>_parsed.parquet                   -- src/parser_<name>.py output
  data/processed/<name>_injected_stall.parquet / _burst.parquet
  data/processed/<name>_injection_labels_stall.csv / _burst.csv
  data/processed/<name>_injection_config_stall.json / _burst.json
  data/processed/<name>_injection_grid_labels_stall.csv / _burst.csv
  results/multi_dataset/<name>/                          -- reports for this dataset
"""

from pathlib import Path

ZENODO_RECORD_URL_TEMPLATE = "https://zenodo.org/records/3227177/files/{display_name}.tar.gz?download=1"

# name used throughout the code -> Loghub's display name (used in the URL only)
DATASETS = {
    "spark": "Spark",
    "hadoop": "Hadoop",
    "openstack": "OpenStack",
    "windows": "Windows",
    "zookeeper": "Zookeeper",
    "android": "Android",
}

# Viability-gate thresholds on the fraction of streams with a valid (non-degenerate) baseline.
VIABLE_THRESHOLD = 0.5
CONDITIONAL_THRESHOLD = 0.2


def dataset_url(name: str) -> str:
    return ZENODO_RECORD_URL_TEMPLATE.format(display_name=DATASETS[name])


def raw_root(name: str) -> Path:
    return Path(f"data/raw/{name}_raw")


def parsed_path(name: str) -> Path:
    return Path(f"data/processed/{name}_parsed.parquet")


def injected_path(name: str, fault: str) -> Path:
    return Path(f"data/processed/{name}_injected_{fault}.parquet")


def injection_labels_path(name: str, fault: str) -> Path:
    return Path(f"data/processed/{name}_injection_labels_{fault}.csv")


def injection_config_path(name: str, fault: str) -> Path:
    return Path(f"data/processed/{name}_injection_config_{fault}.json")


def injection_grid_labels_path(name: str, fault: str) -> Path:
    return Path(f"data/processed/{name}_injection_grid_labels_{fault}.csv")


def results_dir(name: str) -> Path:
    return Path(f"results/multi_dataset/{name}")


def require_known(name: str):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known datasets: {sorted(DATASETS)}")


def make_generic_dataset_config(name: str) -> dict:
    """Dataset config in the shape used by src/placebo_sweep.py and src/loganomaly_invariance.py,
    built from the path conventions above."""
    require_known(name)
    from src.run_baseline_detectors import chronological_split as generic_chronological_split

    import numpy as np
    import pandas as pd

    def _load_injected(fault):
        df = pd.read_parquet(injected_path(name, fault))
        df = df.sort_values(["node", "timestamp"], kind="mergesort").reset_index(drop=True)
        df["row_id"] = np.arange(len(df))
        return df

    return {
        "clean_path": parsed_path(name),
        "split_fn": generic_chronological_split,
        "injected_loaders": {"stall": lambda: _load_injected("stall"), "burst": lambda: _load_injected("burst")},
        "injected_paths": {"stall": injected_path(name, "stall"), "burst": injected_path(name, "burst")},
        "injection_labels_path": {"stall": injection_labels_path(name, "stall"), "burst": injection_labels_path(name, "burst")},
    }
