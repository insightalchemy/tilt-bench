"""
Figures from saved results (no detector fitting or scoring here). 300 DPI, Okabe-Ito palette.

  windowing_mechanism   schematic of fixed-time vs fixed-count windowing under a stall (paper Fig. 3).
  invariance_vs_window  max_abs_score_diff vs window size per detector, fixed-time and fixed-count
                        side by side, on a symlog axis so exact zeros are visible (paper Fig. 2).
                        Reads results/placebo/placebo_sweep_{dataset}.csv.
  placebo_delta         delta = AUC_injected - AUC_placebo with 95% CI per detector, for
                        fixed_time_60 and fixed_count_50, one row per dataset.
                        Reads results/placebo/placebo_sweep_{dataset}.csv.

Writes figures/<name>.png (or --out-dir).

Usage:
    python src/make_figures.py --figure all
    python src/make_figures.py --figure windowing_mechanism
    python src/make_figures.py --figure invariance_vs_window --placebo-csv results/placebo/placebo_sweep_bgl.csv results/placebo/placebo_sweep_thunderbird.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
DETECTOR_COLORS = {
    "count_vector_pca": OKABE_ITO["blue"],
    "isolation_forest_counts": OKABE_ITO["vermillion"],
    "loganomaly": OKABE_ITO["green"],
    "z_score_threshold": OKABE_ITO["orange"],
    "log_ratio_threshold": OKABE_ITO["purple"],
    "timeaware_isolation_forest": OKABE_ITO["black"],
}
DETECTOR_ORDER = ["count_vector_pca", "isolation_forest_counts", "loganomaly", "z_score_threshold", "log_ratio_threshold"]
DPI = 300
FIGURES_DIR = Path("figures")


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_event_row(ax, y, x_events, color, label):
    ax.scatter(x_events, np.full_like(x_events, y, dtype=float), s=40, color=color, zorder=3, label=label)


def make_windowing_mechanism(out_path):
    rng = np.random.default_rng(7)
    n_events = 24
    x_clean = np.sort(rng.uniform(0.3, 29.7, n_events))
    stall_idx = 12
    x_injected = x_clean.copy()
    x_injected[stall_idx:] += 6.0

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2))
    fig.subplots_adjust(top=0.86, hspace=0.55)

    ax = axes[0]
    draw_event_row(ax, 1, x_clean, OKABE_ITO["blue"], "clean")
    draw_event_row(ax, 0, x_injected, OKABE_ITO["vermillion"], "injected (stalled)")
    boundaries = np.arange(0, 31, 6)
    for b in boundaries:
        ax.axvline(b, linestyle="--", color=OKABE_ITO["black"], linewidth=0.8, alpha=0.6)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        n_clean_w = int(((x_clean >= lo) & (x_clean < hi)).sum())
        n_inj_w = int(((x_injected >= lo) & (x_injected < hi)).sum())
        ax.annotate(f"{n_clean_w}", xy=((lo + hi) / 2, 1.35), ha="center", fontsize=9, color=OKABE_ITO["blue"])
        ax.annotate(f"{n_inj_w}", xy=((lo + hi) / 2, -0.35), ha="center", fontsize=9, color=OKABE_ITO["vermillion"])
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([])
    ax.set_xlim(-0.5, 30.5)
    ax.set_xlabel("wall-clock time")
    ax.set_title("(a) fixed-time windowing -- a stall shifts events into different windows, changing per-window counts", fontsize=10, pad=10)
    style_axes(ax)

    ax = axes[1]
    draw_event_row(ax, 1, np.arange(n_events), OKABE_ITO["blue"], "clean")
    draw_event_row(ax, 0, np.arange(n_events), OKABE_ITO["vermillion"], "injected (stalled)")
    boundaries_pos = np.arange(0, n_events + 1, 6)
    for b in boundaries_pos:
        ax.axvline(b - 0.5, linestyle="--", color=OKABE_ITO["black"], linewidth=0.8, alpha=0.6)
    for lo, hi in zip(boundaries_pos[:-1], boundaries_pos[1:]):
        n_w = hi - lo
        ax.annotate(f"{n_w}", xy=((lo + hi - 1) / 2, 1.35), ha="center", fontsize=9, color=OKABE_ITO["blue"])
        ax.annotate(f"{n_w}", xy=((lo + hi - 1) / 2, -0.35), ha="center", fontsize=9, color=OKABE_ITO["vermillion"])
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([])
    ax.set_xlim(-0.5, n_events - 0.5)
    ax.set_xlabel("event position within node (timestamps not shown -- irrelevant to window membership)")
    ax.set_title("(b) fixed-count windowing -- window membership depends only on position, counts never change", fontsize=10, pad=10)
    style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, fontsize=9, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_placebo_delta(placebo_csvs, out_path):
    frames = []
    for path in placebo_csvs:
        if not Path(path).exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        print("ERROR: no placebo CSVs found", file=sys.stderr)
        return
    df = pd.concat(frames, ignore_index=True)
    df = df[df["scheme"].isin(["fixed_time_60", "fixed_count_50"])]

    datasets = sorted(df["dataset"].unique())
    schemes = ["fixed_time_60", "fixed_count_50"]
    fig, axes = plt.subplots(len(datasets), len(schemes), figsize=(5 * len(schemes), 3.2 * len(datasets)), squeeze=False)

    for i, dataset in enumerate(datasets):
        for j, scheme in enumerate(schemes):
            ax = axes[i][j]
            sub = df[(df["dataset"] == dataset) & (df["scheme"] == scheme)]
            sub = sub.groupby("detector", as_index=False).agg(
                delta=("delta", "mean"), lo=("delta_ci_lower", "mean"), hi=("delta_ci_upper", "mean")
            )
            sub["order"] = sub["detector"].map({d: k for k, d in enumerate(DETECTOR_ORDER)})
            sub = sub.sort_values("order")
            y = np.arange(len(sub))
            colors = [DETECTOR_COLORS.get(d, OKABE_ITO["black"]) for d in sub["detector"]]
            err_lo = (sub["delta"] - sub["lo"]).clip(lower=0).to_numpy()
            err_hi = (sub["hi"] - sub["delta"]).clip(lower=0).to_numpy()
            ax.barh(y, sub["delta"], xerr=[err_lo, err_hi], color=colors, capsize=3, height=0.6)
            ax.axvline(0, color=OKABE_ITO["black"], linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(sub["detector"], fontsize=8)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useMathText=True)
            ax.set_xlabel("delta = AUC_injected - AUC_placebo")
            ax.set_title(f"{dataset} / {scheme}", fontsize=10)
            style_axes(ax)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_invariance_vs_window(placebo_csvs, out_path):
    frames = []
    for path in placebo_csvs:
        if not Path(path).exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        print("ERROR: no placebo CSVs found", file=sys.stderr)
        return
    df = pd.concat(frames, ignore_index=True)

    datasets = sorted(df["dataset"].unique())
    scheme_kinds = ["fixed_time", "fixed_count"]
    fig, axes = plt.subplots(len(datasets), len(scheme_kinds), figsize=(5.5 * len(scheme_kinds), 3.6 * len(datasets)), squeeze=False)

    for i, dataset in enumerate(datasets):
        for j, scheme_kind in enumerate(scheme_kinds):
            ax = axes[i][j]
            sub = df[(df["dataset"] == dataset) & (df["scheme_kind"] == scheme_kind)]
            for detector in DETECTOR_ORDER:
                dsub = sub[sub["detector"] == detector].groupby("scheme_size", as_index=False)["max_abs_score_diff"].mean()
                dsub = dsub.sort_values("scheme_size")
                if dsub.empty:
                    continue
                ax.plot(
                    dsub["scheme_size"], dsub["max_abs_score_diff"],
                    marker="o", color=DETECTOR_COLORS.get(detector, OKABE_ITO["black"]), label=detector, linewidth=1.5,
                )
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_xlabel(f"{scheme_kind} window size")
            ax.set_ylabel("max_abs_score_diff (symlog)")
            ax.set_title(f"{dataset} / {scheme_kind}", fontsize=10)
            style_axes(ax)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(DETECTOR_ORDER), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=["windowing_mechanism", "placebo_delta", "invariance_vs_window", "all"], default="all")
    ap.add_argument("--placebo-csv", nargs="+", default=["results/placebo/placebo_sweep_bgl.csv", "results/placebo/placebo_sweep_thunderbird.csv"])
    ap.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    return ap.parse_args()


def main():
    args = parse_args()
    figures = ["windowing_mechanism", "placebo_delta", "invariance_vs_window"] if args.figure == "all" else [args.figure]

    if "windowing_mechanism" in figures:
        make_windowing_mechanism(args.out_dir / "windowing_mechanism.png")
    if "placebo_delta" in figures:
        make_placebo_delta(args.placebo_csv, args.out_dir / "placebo_delta.png")
    if "invariance_vs_window" in figures:
        make_invariance_vs_window(args.placebo_csv, args.out_dir / "invariance_vs_window.png")


if __name__ == "__main__":
    main()
