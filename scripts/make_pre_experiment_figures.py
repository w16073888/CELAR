from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "pre_experiment_figures"

ACCENT = "#B4433C"
SECONDARY = "#477A8E"
TERTIARY = "#D18A39"
NEUTRAL = "#87939D"
INK = "#28323A"
MUTED = "#66727A"
GRID = "#D9DEE2"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.9,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _clean_axis(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _source(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.012,
        0.012,
        text,
        color=MUTED,
        fontsize=8.5,
        ha="left",
        va="bottom",
    )


def make_c1() -> Path:
    frame = pd.read_csv(RESULTS / "celar_pilot_v1" / "c1_summary.csv")
    order = [
        "celar",
        "celar_no_cycle",
        "celar_no_outcome",
        "celar_no_align",
        "shared_ae",
        "celar_shuffled",
        "independent_ae",
    ]
    frame = frame.set_index("method").loc[order].reset_index()
    labels = {
        "celar": "CELAR",
        "celar_no_cycle": "No cycle",
        "celar_no_outcome": "No outcome",
        "celar_no_align": "No alignment",
        "shared_ae": "Shared AE",
        "celar_shuffled": "Shuffled pairs",
        "independent_ae": "Independent AE",
    }
    y = np.arange(len(frame))
    height = 0.34
    fig, axis = plt.subplots(figsize=(10.6, 6.3))
    for offset, metric, color, label, marker in (
        (-height / 2, "retrieval_r1", SECONDARY, "Semantic R@1", "o"),
        (height / 2, "retrieval_r5", ACCENT, "Semantic R@5", "D"),
    ):
        values = frame[metric].to_numpy()
        lower = frame[f"{metric}_ci95_lower"].to_numpy()
        upper = frame[f"{metric}_ci95_upper"].to_numpy()
        axis.barh(
            y + offset,
            values,
            height=height,
            color=color,
            alpha=0.88 if metric == "retrieval_r1" else 0.72,
            label=label,
            edgecolor=color,
            linewidth=0.8,
        )
        axis.errorbar(
            values,
            y + offset,
            xerr=np.vstack([values - lower, upper - values]),
            fmt=marker,
            color="white",
            markeredgecolor=color,
            ecolor=INK,
            elinewidth=1.0,
            capsize=2.5,
            markersize=4.5,
            zorder=4,
        )
        for ypos, value in zip(y + offset, values):
            axis.text(
                min(value + 0.022, 0.985),
                ypos,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color=INK,
            )
    axis.set_yticks(y)
    axis.set_yticklabels([labels[value] for value in frame["method"]])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.04)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlabel("Cross-embodiment action retrieval success")
    axis.set_title(
        "C1 · Shared latent actions align across embodiments",
        loc="left",
        pad=18,
    )
    axis.text(
        0,
        1.012,
        "3 seeds · 60 held-out paired action chunks · bars show 95% bootstrap CI",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10,
        va="bottom",
    )
    axis.legend(loc="lower right", frameon=False, ncol=2)
    _clean_axis(axis)
    _source(fig, "Source: results/celar_pilot_v1/c1_summary.csv")
    fig.tight_layout(rect=(0.015, 0.03, 1, 0.98))
    path = OUTPUT / "c1_cross_embodiment_retrieval.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_c2() -> Path:
    frame = pd.read_csv(RESULTS / "celar_c2_v4" / "c2_summary.csv")
    frame = frame.set_index("method").loc[["native_bc", "celar_bc"]].reset_index()
    labels = ["Native action BC", "CELAR latent BC"]
    colors = [NEUTRAL, ACCENT]
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2))

    for axis, metric, title, formatter in (
        (axes[0], "semantic_mae", "Semantic action error ↓", None),
        (axes[1], "executable_rate", "Executable action rate ↑", PercentFormatter(1.0)),
    ):
        values = frame[metric].to_numpy()
        lower = frame[f"{metric}_ci95_lower"].to_numpy()
        upper = frame[f"{metric}_ci95_upper"].to_numpy()
        axis.bar(
            x,
            values,
            color=colors,
            width=0.58,
            alpha=0.9,
            edgecolor=colors,
        )
        axis.errorbar(
            x,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="none",
            ecolor=INK,
            elinewidth=1.1,
            capsize=4,
        )
        for xpos, value in zip(x, values):
            label = f"{value:.3f}" if formatter is None else f"{value:.1%}"
            axis.text(
                xpos,
                value * 1.03 if formatter is None else min(value + 0.035, 0.93),
                label,
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=INK,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=0)
        axis.set_title(title, loc="left", fontsize=12.5, pad=10)
        axis.set_ylim(0, max(0.38, values.max() * 1.25) if formatter is None else 1.0)
        if formatter is not None:
            axis.yaxis.set_major_formatter(formatter)
        _clean_axis(axis, grid_axis="y")

    semantic_ratio = frame.loc[1, "semantic_mae"] / frame.loc[0, "semantic_mae"]
    execution_ratio = frame.loc[1, "executable_rate"] / frame.loc[0, "executable_rate"]
    jerk_reduction = 1.0 - frame.loc[1, "jerk"] / frame.loc[0, "jerk"]
    fig.suptitle("C2 · Latent BC stays within the usable control envelope", x=0.02, ha="left", fontsize=16)
    fig.text(
        0.02,
        0.92,
        (
            f"Semantic error ×{semantic_ratio:.2f} · "
            f"executable rate ×{execution_ratio:.2f} · "
            f"jerk −{jerk_reduction:.0%} · bars show 95% bootstrap CI"
        ),
        color=MUTED,
        fontsize=10,
    )
    _source(fig, "Source: results/celar_c2_v4/c2_summary.csv · 3 seeds × 2 embodiments")
    fig.tight_layout(rect=(0.015, 0.04, 1, 0.89))
    path = OUTPUT / "c2_latent_vs_native_bc.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_c3() -> Path:
    curves = pd.read_csv(RESULTS / "celar_value_v3" / "c3_curves.csv")
    summary = pd.read_csv(RESULTS / "celar_value_v3" / "c3_summary.csv").set_index("method")
    seed_step = (
        curves.groupby(["seed", "method", "environment_steps"], as_index=False)[
            "eval_success"
        ]
        .mean()
    )
    grouped = (
        seed_step.groupby(["method", "environment_steps"])["eval_success"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    styles = {
        "celar_ppo": ("CELAR PPO", ACCENT, 2.8),
        "shared_ppo": ("Shared decoder PPO", SECONDARY, 2.1),
        "native_ppo": ("Native action PPO", NEUTRAL, 2.1),
    }
    fig, axis = plt.subplots(figsize=(10.8, 6.0))
    for method in ("native_ppo", "shared_ppo", "celar_ppo"):
        group = grouped[grouped["method"] == method]
        label, color, width = styles[method]
        axis.fill_between(
            group["environment_steps"],
            group["min"],
            group["max"],
            color=color,
            alpha=0.09,
            linewidth=0,
        )
        axis.plot(
            group["environment_steps"],
            group["mean"],
            color=color,
            linewidth=width,
            marker="o",
            markersize=3.5,
            label=label,
        )
        final = summary.loc[method]
        axis.scatter(
            [49152],
            [final["final_success"]],
            color=color,
            edgecolor="white",
            linewidth=1.2,
            s=65,
            zorder=5,
        )
        axis.text(
            49152 + 900,
            final["final_success"],
            f"{final['final_success']:.1%}",
            color=color,
            va="center",
            fontsize=10,
            fontweight="medium",
        )
    axis.axhline(0.5, color=INK, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.55)
    axis.text(900, 0.515, "50% target", color=MUTED, fontsize=9)
    axis.set_xlim(0, 55500)
    axis.set_ylim(0, 0.88)
    axis.set_xticks([0, 12288, 24576, 36864, 49152])
    axis.set_xlabel("Online environment steps")
    axis.set_ylabel("Held-out success rate")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_title("C3 · CELAR improves equal-budget PPO learning", loc="left", pad=18)
    axis.text(
        0,
        1.012,
        "Mean across 3 seeds; band = seed min–max; final policy marked at 49,152 steps",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10,
        va="bottom",
    )
    axis.legend(frameon=False, loc="upper left")
    _clean_axis(axis, grid_axis="y")
    _source(fig, "Source: results/celar_value_v3/c3_curves.csv and c3_summary.csv")
    fig.tight_layout(rect=(0.015, 0.035, 1, 0.98))
    path = OUTPUT / "c3_equal_budget_ppo.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_c4() -> Path:
    frame = pd.read_csv(RESULTS / "celar_pilot_v1" / "c4_adaptation_summary.csv")
    methods = {
        "celar_teacher_decoder": ("CELAR: frozen encoder + new decoder", ACCENT, "o", "-"),
        "independent_action_ae": ("Independent action AE", NEUTRAL, "s", "-"),
        "celar_decoder_adapter": ("CELAR: frozen policy + adapter", TERTIARY, "^", "--"),
    }
    fig, axis = plt.subplots(figsize=(9.8, 5.7))
    for method, (label, color, marker, linestyle) in methods.items():
        group = frame[frame["method"] == method].sort_values("shots")
        values = group["semantic_mae"].to_numpy()
        lower = group["semantic_mae_ci95_lower"].to_numpy()
        upper = group["semantic_mae_ci95_upper"].to_numpy()
        axis.errorbar(
            group["shots"],
            values,
            yerr=np.vstack([values - lower, upper - values]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.4,
            markersize=7,
            capsize=4,
            elinewidth=1.1,
            label=label,
        )
        for xpos, value, upper_value in zip(group["shots"], values, upper):
            axis.text(
                xpos,
                upper_value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color=color,
            )
    teacher_5 = frame[
        (frame["method"] == "celar_teacher_decoder") & (frame["shots"] == 5)
    ].iloc[0]
    independent_5 = frame[
        (frame["method"] == "independent_action_ae") & (frame["shots"] == 5)
    ].iloc[0]
    reduction = 1.0 - teacher_5["semantic_mae"] / independent_5["semantic_mae"]
    axis.set_xscale("log")
    axis.set_xticks([5, 20, 50])
    axis.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.set_xlim(4.1, 58)
    axis.set_ylim(0, 0.92)
    axis.set_xlabel("Demonstration trajectories on unseen Sawyer")
    axis.set_ylabel("Held-out semantic action MAE ↓")
    axis.set_title("C4 · Few-shot decoder adaptation to a new embodiment", loc="left", pad=18)
    axis.text(
        0,
        1.012,
        (
            f"5-shot teacher decoder: {teacher_5['semantic_mae']:.3f} MAE "
            f"({reduction:.0%} below independent AE); executable rate 100%"
        ),
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10,
        va="bottom",
    )
    axis.legend(frameon=False, loc="upper right")
    _clean_axis(axis, grid_axis="y")
    _source(fig, "Source: results/celar_pilot_v1/c4_adaptation_summary.csv · 3 seeds")
    fig.tight_layout(rect=(0.015, 0.035, 1, 0.98))
    path = OUTPUT / "c4_unseen_embodiment_adaptation.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    _style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [make_c1(), make_c2(), make_c3(), make_c4()]
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
