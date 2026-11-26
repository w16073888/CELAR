from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(r"D:\大学编程学习\rl+transformer+具身智能")
RESULTS = ROOT / "attm5" / "results"
OUT = ROOT / "attm5" / "docx_work" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#163A59"
BLUE = "#2F6F9F"
TEAL = "#2B8A83"
CORAL = "#C95755"
GRAY = "#8997A5"
LIGHT = "#DCE7EF"


def c0_figure() -> None:
    previews = RESULTS / "c0_mujoco_v1" / "previews"
    robots = [
        ("panda", "Panda · 7 DoF"),
        ("ur5e", "UR5e · 6 DoF"),
        ("sawyer", "Sawyer · 7 DoF"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.3), constrained_layout=True)
    for column, (robot, title) in enumerate(robots):
        rgb = np.asarray(Image.open(previews / f"{robot}_rgb.png").convert("RGB"))
        with np.load(previews / f"{robot}_rgbd.npz") as data:
            depth = data["depth"]
        axes[0, column].imshow(rgb)
        axes[0, column].set_title(title, fontsize=11, color=NAVY, fontweight="bold")
        axes[0, column].axis("off")
        image = axes[1, column].imshow(depth, cmap="viridis_r", vmin=0.5, vmax=1.6)
        axes[1, column].set_title("RGB-D depth (m)", fontsize=10, color=NAVY)
        axes[1, column].axis("off")
    colorbar = fig.colorbar(image, ax=axes[1, :], shrink=0.86, pad=0.01)
    colorbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Unified MuJoCo C0 data: 3 embodiments × 4 tasks × 100 paired scenes",
        fontsize=13,
        color=NAVY,
        fontweight="bold",
    )
    fig.savefig(OUT / "c0_rgbd_examples.png", dpi=300, facecolor="white")
    plt.close(fig)


def retrieval_rl_figure() -> None:
    c1 = pd.read_csv(RESULTS / "celar_pilot_v1" / "c1_summary.csv").set_index("method")
    c3 = pd.read_csv(RESULTS / "celar_value_v3" / "c3_summary.csv").set_index("method")
    curves = pd.read_csv(RESULTS / "celar_value_v3" / "c3_curves.csv")
    methods = [
        ("independent_ae", "Independent\nAE"),
        ("shared_ae", "Shared\nAE"),
        ("celar_no_align", "CELAR\nno align"),
        ("celar", "CELAR"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    x = np.arange(len(methods))
    width = 0.36
    for offset, metric, label in (
        (-width / 2, "retrieval_r1", "R@1"),
        (width / 2, "retrieval_r5", "R@5"),
    ):
        values = np.asarray([c1.loc[key, metric] for key, _ in methods])
        lower = values - np.asarray(
            [c1.loc[key, f"{metric}_ci95_lower"] for key, _ in methods]
        )
        upper = np.asarray(
            [c1.loc[key, f"{metric}_ci95_upper"] for key, _ in methods]
        ) - values
        axes[0, 0].bar(
            x + offset,
            values,
            width,
            yerr=[lower, upper],
            capsize=3,
            label=label,
            color=NAVY if metric == "retrieval_r1" else BLUE,
        )
    axes[0, 0].set_xticks(x, [label for _, label in methods], fontsize=8)
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_ylabel("Cross-embodiment retrieval")
    axes[0, 0].set_title("A  Action-semantic alignment", loc="left", color=NAVY, fontweight="bold")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.2)

    order = ["native_ppo", "shared_ppo", "celar_ppo"]
    labels = ["Native PPO", "Shared decoder", "CELAR PPO"]
    colors = [GRAY, BLUE, CORAL]
    values = np.asarray([c3.loc[key, "final_success"] for key in order])
    lower = values - np.asarray([c3.loc[key, "final_success_ci95_lower"] for key in order])
    upper = np.asarray([c3.loc[key, "final_success_ci95_upper"] for key in order]) - values
    axes[0, 1].bar(
        np.arange(3),
        values,
        yerr=[lower, upper],
        capsize=4,
        color=colors,
    )
    axes[0, 1].set_xticks(np.arange(3), labels, fontsize=8)
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylabel("Final success rate")
    axes[0, 1].set_title("B  Equal-budget PPO", loc="left", color=NAVY, fontweight="bold")
    axes[0, 1].grid(axis="y", alpha=0.2)

    for key, label, color in (
        ("native_ppo", "Native PPO", GRAY),
        ("shared_ppo", "Shared decoder", BLUE),
        ("celar_ppo", "CELAR PPO", CORAL),
    ):
        subset = curves[curves["method"] == key]
        grouped = subset.groupby("environment_steps")["eval_success"]
        mean = grouped.mean()
        low = grouped.min()
        high = grouped.max()
        axes[1, 0].plot(mean.index, mean.values, marker="o", markersize=3, label=label, color=color)
        axes[1, 0].fill_between(mean.index, low.values, high.values, color=color, alpha=0.12)
    axes[1, 0].set_xlabel("Online environment steps")
    axes[1, 0].set_ylabel("Held-out success")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("C  Learning curves across 3 seeds", loc="left", color=NAVY, fontweight="bold")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(frameon=False, fontsize=8)

    steps = np.arange(3)
    native_steps = np.asarray([c3.loc["native_ppo", "steps_to_50"] / 49152.0,
                               c3.loc["shared_ppo", "steps_to_50"] / 49152.0,
                               c3.loc["celar_ppo", "steps_to_50"] / 49152.0])
    axes[1, 1].bar(steps, native_steps, color=[GRAY, BLUE, CORAL])
    axes[1, 1].set_xticks(steps, labels, fontsize=8)
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_ylabel("Fraction of 49,152-step budget")
    axes[1, 1].set_title("D  Steps to 50% success", loc="left", color=NAVY, fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.2)
    for index, value in enumerate(native_steps):
        axes[1, 1].text(index, value + 0.025, f"{value:.2f}", ha="center", fontsize=9, color=NAVY)
    fig.savefig(OUT / "alignment_rl_results.png", dpi=300, facecolor="white")
    plt.close(fig)


def adaptation_group_figure() -> None:
    groups = pd.read_csv(RESULTS / "celar_v3_group_analysis" / "c3_group_summary.csv")
    c4 = pd.read_csv(RESULTS / "celar_pilot_v1" / "c4_adaptation_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), constrained_layout=True)

    pivot = groups.pivot_table(
        index=["body_name", "task_name"], columns="method", values="success_rate"
    ).reset_index()
    labels = [
        f"{row.body_name.replace('franka_', '').replace('_', ' ').title()} / {row.task_name}"
        for row in pivot.itertuples()
    ]
    positions = np.arange(len(labels))
    width = 0.26
    for offset, method, color, label in (
        (-width, "native_ppo", GRAY, "Native"),
        (0.0, "shared_ppo", BLUE, "Shared"),
        (width, "celar_ppo", CORAL, "CELAR"),
    ):
        axes[0].bar(
            positions + offset,
            pivot[method].to_numpy(),
            width,
            color=color,
            label=label,
        )
    axes[0].set_xticks(positions, labels, rotation=22, ha="right", fontsize=8)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("A  Body-task breakdown (3 seeds)", loc="left", color=NAVY, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    selected = c4[c4["method"].isin(["celar_teacher_decoder", "independent_action_ae"])]
    shots = sorted(selected["shots"].unique())
    for method, color, label in (
        ("celar_teacher_decoder", CORAL, "CELAR decoder"),
        ("independent_action_ae", GRAY, "Independent action AE"),
    ):
        row = selected[selected["method"] == method].set_index("shots")
        values = np.asarray([row.loc[shot, "semantic_mae"] for shot in shots])
        lower = values - np.asarray(
            [row.loc[shot, "semantic_mae_ci95_lower"] for shot in shots]
        )
        upper = np.asarray(
            [row.loc[shot, "semantic_mae_ci95_upper"] for shot in shots]
        ) - values
        axes[1].errorbar(
            shots,
            values,
            yerr=[lower, upper],
            marker="o",
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )
    axes[1].set_xticks(shots, [f"{shot}-shot" for shot in shots])
    axes[1].set_xlabel("Demonstration trajectories")
    axes[1].set_ylabel("Semantic action MAE")
    axes[1].set_title("B  Unseen Sawyer adaptation", loc="left", color=NAVY, fontweight="bold")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "group_adaptation_results.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    c0_figure()
    retrieval_rl_figure()
    adaptation_group_figure()
    for path in sorted(OUT.glob("*.png")):
        print(path, path.stat().st_size)
