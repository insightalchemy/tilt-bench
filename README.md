# TILT-Bench

TILT-Bench injects three types of timing faults, stalls, bursts, and slowdowns, into public log anomaly detection benchmarks by changing only event timestamps. The event content and per-node event order stay the same. Fault intensity is scaled to each node’s own inter-arrival variability, and every injection is seeded and saved with its ground-truth label. The repository also includes our audit of native benchmark anomalies, checks comparing detector scores on clean and injected data under fixed-count and fixed-time windows, the detectors used in the paper, and a placebo-controlled evaluation that tests whether a detector is responding to the fault itself or simply to where the fault falls within the window.

## Repository structure

```
tilt-bench-release/
├── src/                 parsers, injectors, detectors, evaluation and figure scripts
│   └── detectors/       count-vector PCA, isolation forest on counts, timing detectors, windowing
├── scripts/             shell drivers for the per-dataset pipelines and full reproduction
├── requirements.txt     Python dependencies
├── LICENSE              MIT license
└── README.md            this file
```

All scripts read raw logs from `data/raw/`, write intermediate files to `data/processed/`, and
write outputs to `results/`, `figures/`, and `logs/`, relative to the directory they are run from.
None of these directories are part of the repository.

## Requirements

- Python 3.10 or later (tested with Python 3.12).
- Install dependencies with:

  ```
  pip install -r requirements.txt
  ```

- Run every command from the repository root with the repository on `PYTHONPATH`:

  ```
  export PYTHONPATH="$(pwd)"
  ```

  The shell drivers in `scripts/` set this themselves.

Hardware. No GPU is required, and the full-scale BGL and Thunderbird windowing and placebo sweeps
need roughly 25 to 31 GB of RAM. Training the DeepLog models takes many hours on CPU for BGL and
considerably longer for Thunderbird. The time-aware DeepLog variant and the LogBERT-style
transformer also take several hours each. Most experiment scripts accept `--subsample` and
`--subsample-include-injected` for a quick end-to-end check on a small subset of the data.

## Getting the data

We do not redistribute any dataset. Place the raw files under `data/raw/` as listed below.

| Dataset | Expected file(s) | Source |
|---|---|---|
| BGL | `data/raw/BGL.log` | Loghub (Zhu et al., ISSRE 2023), Zenodo record 3227177, `BGL.tar.gz` [SOURCE TO CONFIRM] |
| Thunderbird | `data/raw/Thunderbird.log` | Loghub `Thunderbird.tar.gz`, https://zenodo.org/records/8196385/files/Thunderbird.tar.gz [SOURCE TO CONFIRM] |
| Spirit | `data/raw/Spirit.log` | 5M-line subset from the replication package of Wu, Li, and Khomh (EMSE 2023), https://zenodo.org/records/7851024/files/spirit2_5m.tar.gz?download=1 (downloaded and md5-checked by `python src/spirit_setup.py download`) |
| HDFS | `data/raw/HDFS.log`, `data/raw/anomaly_label.csv` | Loghub `HDFS_v1.zip`, https://zenodo.org/records/8196385/files/HDFS_v1.zip [SOURCE TO CONFIRM] |
| OpenStack, Hadoop, Zookeeper, Android, Spark, Windows | `data/raw/<name>_raw/` | Loghub, Zenodo record 3227177, `https://zenodo.org/records/3227177/files/<Name>.tar.gz?download=1` (downloaded and extracted by `python src/multi_dataset_acquire.py download --dataset <name>`) |

Thunderbird slice. The experiments use a 10,000,000-line contiguous slice of the full
Thunderbird log, lines 50,000,001 to 60,000,000, chosen for anomaly diversity. From the extracted
full log:

```
tail -n +50000001 Thunderbird.log | head -n 10000000 > data/raw/Thunderbird.log
```

Windows. `src/parser_windows.py` parses only the first 10,000,000 lines by default
(`--max-lines`).

## Usage

The examples use BGL; the other datasets have analogous scripts (`*_thunderbird.py`,
`*_spirit.py`, and `injector_multi.py` / `multi_dataset_*.py` for the additional Loghub datasets).

Parse a dataset with Drain:

```
python src/parser.py                       # BGL -> data/processed/bgl_parsed.parquet
python src/parser_thunderbird.py
python src/parser_spirit.py
python src/parser_hdfs.py
python src/parser_openstack.py             # likewise hadoop, zookeeper, android, spark, windows
```

Run the premise audit of native anomalies:

```
python src/premise_audit.py
```

Inject timing faults and validate the injection (ordering check, inter-arrival plots, and the
premise-audit signatures re-run on injected rows):

```
python src/injector.py --type stall
python src/injector.py --type burst
python src/injector_slowdown.py --dataset bgl
python src/validate_injection.py --type stall
python src/validate_injection.py --type burst
python src/plausibility.py --dataset bgl --fault stall
```

Run the invariance check (clean vs injected detector scores under each windowing scheme):

```
python src/windowing_sweep.py bgl thunderbird
python src/windowing_sweep_extended.py --dataset bgl
python src/loganomaly_invariance.py --dataset bgl
python src/sliding_window_invariance.py --dataset bgl --fault stall
python src/run_slowdown_evaluation.py --dataset bgl
python src/parser_invariance.py --dataset bgl --clean-parsed data/processed/bgl_parsed_simth0.5.parquet
```

Run the detectors (count-vector PCA, isolation forest on counts, z-score, and log-ratio) on the
shared 60 s evaluation grid, with bootstrap confidence intervals and five-seed replication:

```
python src/core_metrics.py
python src/multiseed_auc.py --n-seeds 5 --dataset both --fault both
python src/deeplog.py --dataset bgl --eval-target native
python src/logbert_small.py --dataset bgl --fault stall --epochs 20 --eval-target injected
```

Run the placebo-controlled evaluation (delta = AUC on injected data minus AUC on clean data
against the same labels, with a cluster bootstrap over injection events):

```
python src/placebo_sweep.py --dataset bgl
python src/placebo_sweep.py --dataset thunderbird
```

Each script documents its arguments and outputs in its module docstring (`python src/<script>.py --help`
for scripts with command-line options).

## Reproducing the paper

`scripts/reproduce_all.sh` runs every step below in dependency order; individual stages can be
run on their own, for example `bash scripts/reproduce_all.sh bgl thunderbird placebo`.

| Result | Script | Command |
|---|---|---|
| Premise audit table (BGL, Thunderbird, Spirit, HDFS) | `premise_audit.py`, `premise_audit_thunderbird.py`, `premise_audit_spirit.py`, `premise_audit_hdfs.py` | `bash scripts/reproduce_all.sh bgl thunderbird spirit hdfs` |
| Spirit masked-template count | `parser_spirit_masked.py` | `python src/parser_spirit_masked.py` |
| HDFS, Spark, and Windows scope-boundary characterization | `parser_hdfs.py`, `multi_dataset_gate.py` | `python src/parser_hdfs.py`; `bash scripts/acquire_and_evaluate_dataset.sh spark --gate-only` (likewise `windows`) |
| Eligible-node counts and per-node baseline coverage | `injector*.py`, `spirit_setup.py viability-gate` | printed by the injectors and the gate |
| Ordering check and instrument validation table | `core_metrics.py` (BGL, Thunderbird), `validate_injection_spirit.py` | `python src/core_metrics.py`; `bash scripts/run_spirit.sh` |
| Physical plausibility of injected gaps | `plausibility.py` | `python src/plausibility.py --dataset {bgl,thunderbird} --fault {stall,burst}` |
| Slowdown plausibility (injections exceeding the one-hour bound) | `injector_slowdown.py` | `python src/injector_slowdown.py --dataset {bgl,spirit}` |
| Invariance check table: fixed-count and fixed-time, PCA and isolation forest, BGL and Thunderbird | `windowing_sweep.py`, `windowing_sweep_extended.py` | `python src/windowing_sweep.py bgl thunderbird`; `python src/windowing_sweep_extended.py --dataset bgl` |
| Invariance check table: LogAnomaly-style detector | `loganomaly_invariance.py` | `python src/loganomaly_invariance.py --dataset {bgl,thunderbird,openstack,hadoop,zookeeper,android}` |
| Invariance check table: Spirit and OpenStack PCA / isolation forest rows | `placebo_sweep.py` (invariance columns) | `python src/placebo_sweep.py --dataset spirit`; `bash scripts/acquire_and_evaluate_dataset.sh openstack` |
| Invariance check table: sliding windows | `sliding_window_invariance.py` | `python src/sliding_window_invariance.py --dataset {bgl,spirit} --fault {stall,burst}` |
| Invariance check table: slowdowns | `run_slowdown_evaluation.py` | `python src/run_slowdown_evaluation.py --dataset {bgl,spirit}` |
| Invariance check table: parser configurations | `parser_invariance.py` | `bash scripts/reproduce_all.sh invariance` (masked Spirit via `scripts/run_spirit.sh`) |
| Invariance check table: transformer | `logbert_small.py` | `python src/logbert_small.py --dataset bgl --fault stall --epochs 20 --eval-target injected` |
| DeepLog table (native and injected) and DeepLog invariance check | `deeplog.py` | `bash scripts/reproduce_all.sh deep` |
| Placebo table (fault-attributable detection under fixed-time windowing) | `placebo_sweep.py` | `python src/placebo_sweep.py --dataset {bgl,thunderbird,spirit}`; other datasets via `scripts/acquire_and_evaluate_dataset.sh` |
| Time-aware isolation forest (boundary of the proposition) | `timeaware_control.py` | `python src/timeaware_control.py --dataset {bgl,thunderbird}` |
| Time-aware DeepLog | `deeplog_timeaware.py` | `python src/deeplog_timeaware.py --time-aware --epochs 20` |
| Bootstrap AUC table | `core_metrics.py` | `python src/core_metrics.py` |
| Five-seed table | `multiseed_auc.py` | `python src/multiseed_auc.py --n-seeds 5 --dataset both --fault both` |
| Timestamp-resolution ablation | `resolution_ablation.py` | `python src/resolution_ablation.py --dataset bgl` |
| `windowing_mechanism.png` | `make_figures.py` | `python src/make_figures.py --figure windowing_mechanism` |
| `invariance_vs_window.png` | `make_figures.py` | `python src/make_figures.py --figure invariance_vs_window` |
| `placebo_delta.png` | `make_figures.py` | `python src/make_figures.py --figure placebo_delta` |
| Injection inter-arrival plots (`{stall,burst}_injection_check.png`) | `validate_injection*.py` | `python src/validate_injection.py --type {stall,burst}` |

## License

MIT. See `LICENSE`.
