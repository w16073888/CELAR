from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import (
    MORPHOLOGIES,
    generate_dataset,
    make_pairs,
    safety_clip,
    select_shot_demos,
    semantic_from_native,
    source_digest,
)
from .metrics import (
    bootstrap_mean_ci,
    evaluate_frozen_adapter,
    evaluate_action_interface,
    evaluate_latent_model,
    evaluate_policy,
    holm_adjust,
    paired_difference,
)
from .training import (
    LATENT_METHODS,
    train_adaptation_policy,
    train_independent_action_ae,
    train_latent_models,
    train_policy,
    train_teacher_latent_decoder,
)


SEEN_BODIES = (0, 1)
HELDOUT_BODY = 2


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_seed_rows(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metric_columns: list[str],
    bootstrap_samples: int,
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record["runs"] = int(len(group))
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)
            lower, upper = bootstrap_mean_ci(
                values, samples=bootstrap_samples, seed=8801
            )
            record[metric] = float(values.mean())
            record[f"{metric}_ci95_lower"] = lower
            record[f"{metric}_ci95_upper"] = upper
        output.append(record)
    return pd.DataFrame(output)


def _expert_action_audit(test_samples) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for body_index, morphology in enumerate(MORPHOLOGIES):
        selected = [sample for sample in test_samples if sample.body_index == body_index]
        native = np.stack([sample.native_action for sample in selected])
        mask = np.stack([sample.action_mask for sample in selected])
        semantic_target = np.stack([sample.semantic_action for sample in selected])
        clipped, valid = safety_clip(native, body_index)
        semantic = semantic_from_native(clipped, body_index)
        semantic_error = np.abs(semantic - semantic_target).mean(axis=(1, 2))
        maximum_step = 0.0
        if clipped.shape[1] > 1:
            maximum_step = float(
                np.max(np.abs(np.diff(clipped, axis=1))[mask[:, 1:] > 0.5])
            )
        rows.append(
            {
                "body_index": body_index,
                "morphology": morphology.name,
                "action_dim": morphology.action_dim,
                "samples": len(selected),
                "expert_executable_rate": float(
                    (valid.all(axis=1) & (semantic_error <= 0.34)).mean()
                ),
                "expert_semantic_mae": float(semantic_error.mean()),
                "maximum_step": maximum_step,
                "velocity_limit": morphology.velocity_limit,
            }
        )
    return {
        "passed": bool(
            all(row["expert_executable_rate"] >= 0.95 for row in rows)
            and all(row["maximum_step"] <= row["velocity_limit"] + 1e-5 for row in rows)
        ),
        "bodies": rows,
    }


def _make_c1_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("retrieval_semantic_r1", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for axis, metric, title in (
        (axes[0], "retrieval_semantic_r1", "Cross-embodiment semantic R@1"),
        (axes[1], "semantic_mae", "Executable semantic action MAE"),
    ):
        values = ordered[metric].to_numpy(dtype=float)
        lower = ordered[f"{metric}_ci95_lower"].to_numpy(dtype=float)
        upper = ordered[f"{metric}_ci95_upper"].to_numpy(dtype=float)
        axis.bar(
            ordered["method"],
            values,
            yerr=[values - lower, upper - values],
            capsize=3,
            color=[
                "#c84a52" if method == "celar" else "#88a2b5"
                for method in ordered["method"]
            ],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=28, labelsize=8)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "c1_latent_alignment.png", dpi=180)
    plt.close(fig)


def _make_c4_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for method, group in summary.groupby("method"):
        group = group.sort_values("shots")
        axes[0].plot(
            group["shots"],
            group["semantic_mae"],
            marker="o",
            linewidth=2.2 if "celar" in method else 1.5,
            label=method,
        )
        axes[1].plot(
            group["shots"],
            group["executable_rate"],
            marker="o",
            linewidth=2.2 if "celar" in method else 1.5,
            label=method,
        )
    axes[0].set_title("Held-out morphology semantic MAE")
    axes[0].set_xlabel("Demonstration trajectories")
    axes[0].set_ylabel("Lower is better")
    axes[0].set_xscale("log")
    axes[0].set_xticks(summary["shots"].unique())
    axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].set_title("Held-out morphology executable rate")
    axes[1].set_xlabel("Demonstration trajectories")
    axes[1].set_ylabel("Higher is better")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xscale("log")
    axes[1].set_xticks(summary["shots"].unique())
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "c4_heldout_adaptation.png", dpi=180)
    plt.close(fig)


def _report(
    *,
    c0: dict[str, object],
    c1_summary: pd.DataFrame,
    c2_summary: pd.DataFrame,
    c4_summary: pd.DataFrame,
    c1_comparisons: list[dict[str, object]],
    c4_comparisons: list[dict[str, object]],
    gates: dict[str, bool],
    decision: str,
    reference_shots: int,
    manifest: dict[str, object],
) -> str:
    c1 = c1_summary.set_index("method")
    c2 = c2_summary.set_index("method")
    c4 = c4_summary.set_index(["shots", "method"])
    full = c1.loc["celar"]
    shared = c1.loc["shared_ae"]
    no_align = c1.loc["celar_no_align"]
    comparison_map = {str(item["name"]): item for item in c1_comparisons}
    c4_map = {str(item["name"]): item for item in c4_comparisons}
    c4_20 = c4.loc[(reference_shots, "celar_teacher_decoder")]
    independent_20 = c4.loc[(reference_shots, "independent_action_ae")]
    reduction_20 = 1.0 - float(c4_20["semantic_mae"]) / float(
        independent_20["semantic_mae"]
    )
    lines = [
        "# CELAR 跨本体潜在动作立项实验",
        "",
        "> 本报告由实验脚本自动生成。证据来自受控运动学代理，不是 MuJoCo、"
        "LIBERO 或真机论文级结果；它只用于判断是否值得进入下一阶段。",
        "",
        "## 立项判断",
        "",
        f"- **决策：`{decision}`**",
        f"- C0 配对数据与安全动作：{'通过' if gates['c0'] else '未通过'}",
        f"- C1 潜在动作对齐：{'通过' if gates['c1'] else '未通过'}",
        f"- C2 同本体 BC：{'通过' if gates['c2'] else '未通过'}",
        f"- C4 新本体适配：{'通过' if gates['c4'] else '未通过'}",
        "",
        "预注册判断规则：C0 要求无 pair/task 泄漏且三个本体专家动作安全可执行率"
        "均 ≥95%；C1 要求完整 CELAR 的跨本体 R@1 ≥0.60、比普通共享 AE 至少高"
        "0.10、动作可执行率 ≥0.85、latent 有效秩 ≥3；C2 要求 latent BC 的语义误差"
        "不高于原生动作 BC 的 115%；C4 要求冻结动作编码器的 5-shot 适配相对从零 AE "
        "至少降低 15%，20-shot 均值不退化和方向一致，且可执行率 ≥85%。"
        "C0+C1 加上 C2/C4 之一才进入 MuJoCo 阶段。",
        "",
        "## C0：配对数据与动作安全",
        "",
        f"- pair 泄漏数：{c0['pair_overlap_count']}",
        f"- task 泄漏数：{c0['task_overlap_count']}",
        f"- 数据切分：`{json.dumps(c0['split_counts'], ensure_ascii=False)}`",
        "",
        "| 本体 | 动作维度 | 专家可执行率 | 语义 MAE | 最大单步 | 速度上限 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for body in c0["bodies"]:
        lines.append(
            f"| {body['morphology']} | {body['action_dim']} | "
            f"{body['expert_executable_rate']:.3f} | {body['expert_semantic_mae']:.4f} | "
            f"{body['maximum_step']:.4f} | {body['velocity_limit']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## C1：动作重建与跨本体对齐",
            "",
            "| 方法 | Semantic R@1 | Semantic R@5 | Semantic MAE | 可执行率 | 循环误差 | 有效秩 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in c1_summary.sort_values("retrieval_semantic_r1", ascending=False)["method"]:
        row = c1.loc[method]
        lines.append(
            f"| {method} | {row['retrieval_semantic_r1']:.3f} | {row['retrieval_semantic_r5']:.3f} | "
            f"{row['semantic_mae']:.4f} | {row['executable_rate']:.3f} | "
            f"{row['cycle_error']:.4f} | {row['latent_effective_rank']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"完整 CELAR 相对普通共享 AE 的 R@1 配对提升为 "
            f"**{comparison_map['c1_retrieval_semantic_r1_celar_vs_shared_ae']['mean_difference']:+.3f}** "
            f"(95% CI "
            f"[{comparison_map['c1_retrieval_semantic_r1_celar_vs_shared_ae']['ci95_lower']:.3f}, "
            f"{comparison_map['c1_retrieval_semantic_r1_celar_vs_shared_ae']['ci95_upper']:.3f}])。"
            f"相对无跨本体对齐消融的提升为 "
            f"**{comparison_map['c1_retrieval_semantic_r1_celar_vs_celar_no_align']['mean_difference']:+.3f}**。",
            "",
            "## C2：同本体行为克隆",
            "",
            "| 方法 | 语义 MAE | 可执行率 | Jerk |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in c2_summary["method"]:
        row = c2.loc[method]
        lines.append(
            f"| {method} | {row['semantic_mae']:.4f} | "
            f"{row['executable_rate']:.3f} | {row['jerk']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## C4：未见第三机械臂适配",
            "",
            "| 演示数 | 方法 | 语义 MAE | 可执行率 |",
            "|---:|---|---:|---:|",
        ]
    )
    for (shots, method), row in c4.sort_index().iterrows():
        lines.append(
            f"| {shots} | {method} | {row['semantic_mae']:.4f} | "
            f"{row['executable_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"{reference_shots} 条演示下，CELAR 冻结动作编码器 + 新本体 decoder 的语义误差相对从零独立动作 AE "
            f"降低 **{100.0 * reduction_20:.1f}%**；配对均值为 "
            f"{c4_map[f'c4_{reference_shots}_semantic_mae_celar_teacher_vs_independent_ae']['mean_difference']:+.4f}，"
            f"95% CI "
            f"[{c4_map[f'c4_{reference_shots}_semantic_mae_celar_teacher_vs_independent_ae']['ci95_lower']:+.4f}, "
            f"{c4_map[f'c4_{reference_shots}_semantic_mae_celar_teacher_vs_independent_ae']['ci95_upper']:+.4f}]。"
            "表中另保留了冻结策略端到端 adapter，用于分别审计“动作接口迁移”与“高层策略迁移”。",
            "",
            "## 证据边界与下一阶段",
            "",
            "- 当前环境是高可控运动学/动作空间代理，没有接触动力学、真实视觉主干或真机延迟。",
            "- 本实验验证了动作语义可对齐、decoder 可执行和新本体少样本接口，但不直接证明在线 RL 样本效率。",
            "- 若本报告为 CONDITIONAL_GO 或 GO，下一步应在 MuJoCo/Robosuite 三个机械臂上复现 C0/C1，并加入 C3 PPO 与固定动作块预算。",
            "- 若 C1 通过而 C2/C4 失败，应缩小主张为“跨本体动作表示有价值”，不能声称端到端方法成立。",
            "",
            "## 审计文件",
            "",
            "`c0_protocol.json`、`c1_raw.csv`、`c1_summary.csv`、`c1_seed_summary.csv`、"
            "`c2_raw.csv`、`c2_summary.csv`、`c4_adaptation_raw.csv`、"
            "`c4_adaptation_summary.csv`、`dataset_index.csv`、`results.json`、"
            "`run_manifest.json`、`checkpoints/`、`checksums.sha256`。",
            "",
            f"源码 SHA-256：`{manifest['source_sha256']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(config: ExperimentConfig) -> Path:
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    protocols: dict[str, object] = {}
    c1_raw_rows: list[dict[str, object]] = []
    c1_seed_rows: list[dict[str, object]] = []
    c2_raw_rows: list[dict[str, object]] = []
    c4_raw_rows: list[dict[str, object]] = []
    training_histories: dict[str, object] = {}
    checkpoint_hashes: dict[str, str] = {}
    expert_audits: dict[str, object] = {}
    dataset_index_rows: list[dict[str, object]] = []

    for seed in config.evaluation.seeds:
        print(f"[seed {seed}] generating paired cross-embodiment data")
        dataset = generate_dataset(config.data, seed)
        protocols[str(seed)] = dataset.protocol
        if not dataset.protocol["passed"]:
            raise RuntimeError(f"C0 split gate failed: {dataset.protocol}")
        expert_audits[str(seed)] = _expert_action_audit(dataset.test)
        if not expert_audits[str(seed)]["passed"]:
            raise RuntimeError(f"Expert action audit failed: {expert_audits[str(seed)]}")
        for split, samples in (
            ("train", dataset.train),
            ("val", dataset.val),
            ("test", dataset.test),
        ):
            indexed = {
                (
                    sample.pair_id,
                    sample.body_index,
                    sample.episode_id,
                    sample.task_id,
                    sample.trajectory_id,
                )
                for sample in samples
            }
            for pair_id, body_index, episode_id, task_id, trajectory_id in indexed:
                dataset_index_rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "pair_id": pair_id,
                        "body_index": body_index,
                        "body_name": MORPHOLOGIES[body_index].name,
                        "episode_id": episode_id,
                        "task_id": task_id,
                        "trajectory_id": trajectory_id,
                    }
                )
        test_pairs = make_pairs(
            dataset.test,
            SEEN_BODIES,
            limit=config.evaluation.retrieval_candidates,
            seed=seed + 911,
        )
        models: dict[str, object] = {}
        print(f"[seed {seed}] training latent-action baselines and CELAR")
        for method in LATENT_METHODS:
            model = train_latent_models(
                dataset.train,
                bodies=SEEN_BODIES,
                method=method,
                cfg=config.model,
                seed=seed,
            )
            models[method] = model
            training_histories[f"c1/{method}/seed_{seed}"] = model.history
            summary, raw = evaluate_latent_model(
                model,
                test_pairs,
                tolerance=config.evaluation.executable_semantic_tolerance,
            )
            summary["seed"] = seed
            c1_seed_rows.append(summary)
            c1_raw_rows.extend({"seed": seed, **row} for row in raw)
            checkpoint_path = checkpoint_dir / f"c1_seed_{seed}_{method}.pt"
            source_model = next(iter(model.models.values()))
            torch.save(
                {
                    "method": method,
                    "seed": seed,
                    "model": {
                        str(body): candidate.state_dict()
                        for body, candidate in model.models.items()
                    },
                    "context_normalizer": model.context_normalizer.state_dict(),
                    "source_decoder_count": len(source_model.decoders),
                },
                checkpoint_path,
            )
            checkpoint_hashes[f"c1/{method}/seed_{seed}"] = _sha256(checkpoint_path)

        print(f"[seed {seed}] training native and latent behavior cloning")
        native_policy = train_policy(
            dataset.train,
            bodies=SEEN_BODIES,
            method="native_bc",
            cfg=config.model,
            seed=seed,
            latent_model=None,
        )
        latent_policy = train_policy(
            dataset.train,
            bodies=SEEN_BODIES,
            method="celar_bc",
            cfg=config.model,
            seed=seed,
            latent_model=models["celar"],
        )
        for body_index in SEEN_BODIES:
            for method, policy in (
                ("native_bc", native_policy[body_index]),
                ("celar_bc", latent_policy[body_index]),
            ):
                record = evaluate_policy(
                    policy,
                    dataset.test,
                    body_index=body_index,
                    tolerance=config.evaluation.executable_semantic_tolerance,
                )
                c2_raw_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "body_index": body_index,
                        "body_name": MORPHOLOGIES[body_index].name,
                        **record,
                    }
                )
        policy_checkpoint = checkpoint_dir / f"c2_seed_{seed}_policies.pt"
        torch.save(
            {
                "seed": seed,
                "native_policy": native_policy[0].policy.state_dict(),
                "celar_policy": latent_policy[0].policy.state_dict(),
                "native_normalizer": native_policy[0].normalizer.state_dict(),
                "celar_normalizer": latent_policy[0].normalizer.state_dict(),
            },
            policy_checkpoint,
        )
        checkpoint_hashes[f"c2/seed_{seed}"] = _sha256(policy_checkpoint)

        print(f"[seed {seed}] adapting unseen {MORPHOLOGIES[HELDOUT_BODY].name}")
        for shots in config.evaluation.shots:
            demonstrations = select_shot_demos(
                dataset.train, HELDOUT_BODY, shots, seed=seed + 12_001
            )
            independent = train_adaptation_policy(
                demonstrations,
                body_index=HELDOUT_BODY,
                cfg=config.model,
                seed=seed + shots * 101,
            )
            independent_metrics = evaluate_policy(
                independent,
                dataset.test,
                body_index=HELDOUT_BODY,
                tolerance=config.evaluation.executable_semantic_tolerance,
            )
            c4_raw_rows.append(
                {
                    "seed": seed,
                    "shots": shots,
                    "method": "independent_bc",
                    "body_index": HELDOUT_BODY,
                    **independent_metrics,
                }
            )
            frozen_metrics, _ = evaluate_frozen_adapter(
                models["celar"],
                latent_policy[0],
                demonstrations,
                dataset.test,
                body_index=HELDOUT_BODY,
                tolerance=config.evaluation.executable_semantic_tolerance,
                cfg=config.model,
                seed=seed + shots * 103,
            )
            c4_raw_rows.append(
                {
                    "seed": seed,
                    "shots": shots,
                    "method": "celar_decoder_adapter",
                    "body_index": HELDOUT_BODY,
                    **frozen_metrics,
                }
            )
            independent_ae = train_independent_action_ae(
                demonstrations,
                body_index=HELDOUT_BODY,
                cfg=config.model,
                seed=seed + shots * 107,
            )
            c4_raw_rows.append(
                {
                    "seed": seed,
                    "shots": shots,
                    "method": "independent_action_ae",
                    "body_index": HELDOUT_BODY,
                    **evaluate_action_interface(
                        independent_ae,
                        independent_ae,
                        dataset.test,
                        body_index=HELDOUT_BODY,
                        tolerance=config.evaluation.executable_semantic_tolerance,
                    ),
                }
            )
            teacher_decoder = train_teacher_latent_decoder(
                models["celar"],
                demonstrations,
                body_index=HELDOUT_BODY,
                cfg=config.model,
                seed=seed + shots * 109,
            )
            c4_raw_rows.append(
                {
                    "seed": seed,
                    "shots": shots,
                    "method": "celar_teacher_decoder",
                    "body_index": HELDOUT_BODY,
                    **evaluate_action_interface(
                        models["celar"],
                        teacher_decoder,
                        dataset.test,
                        body_index=HELDOUT_BODY,
                        tolerance=config.evaluation.executable_semantic_tolerance,
                    ),
                }
            )
        c4_checkpoint = checkpoint_dir / f"c4_seed_{seed}_adaptation_complete.pt"
        torch.save({"seed": seed, "shots": config.evaluation.shots}, c4_checkpoint)
        checkpoint_hashes[f"c4/seed_{seed}"] = _sha256(c4_checkpoint)

    c1_raw = pd.DataFrame(c1_raw_rows)
    c1_seed = pd.DataFrame(c1_seed_rows)
    c1_summary = _aggregate_seed_rows(
        c1_seed,
        group_columns=["method"],
        metric_columns=[
            "retrieval_semantic_r1",
            "retrieval_semantic_r5",
            "retrieval_exact_pair_r1",
            "retrieval_exact_pair_r5",
            "reconstruction_mae",
            "semantic_mae",
            "executable_rate",
            "cycle_error",
            "progress_pair_gap",
            "latent_effective_rank",
            "body_balance_gap",
        ],
        bootstrap_samples=config.evaluation.bootstrap_samples,
    )
    c1_comparisons: list[dict[str, object]] = []
    for metric, control in (
        ("retrieval_semantic_r1", "shared_ae"),
        ("retrieval_semantic_r1", "celar_no_align"),
        ("semantic_mae", "shared_ae"),
        ("semantic_mae", "celar_no_cycle"),
        ("semantic_mae", "celar_no_outcome"),
    ):
        treatment_rows = c1_seed[c1_seed["method"] == "celar"].sort_values("seed")
        control_rows = c1_seed[c1_seed["method"] == control].sort_values("seed")
        difference = paired_difference(
            treatment_rows[metric].to_numpy(),
            control_rows[metric].to_numpy(),
            samples=config.evaluation.bootstrap_samples,
            seed=730 + len(c1_comparisons),
        )
        c1_comparisons.append(
            {
                "name": f"c1_{metric}_celar_vs_{control}",
                "metric": metric,
                "treatment": "celar",
                "control": control,
                **difference,
            }
        )

    c2_raw = pd.DataFrame(c2_raw_rows)
    c2_summary = _aggregate_seed_rows(
        c2_raw,
        group_columns=["method"],
        metric_columns=["semantic_mae", "native_mae", "executable_rate", "jerk"],
        bootstrap_samples=config.evaluation.bootstrap_samples,
    )
    c4_raw = pd.DataFrame(c4_raw_rows)
    c4_summary = _aggregate_seed_rows(
        c4_raw,
        group_columns=["shots", "method"],
        metric_columns=["semantic_mae", "native_mae", "executable_rate", "jerk"],
        bootstrap_samples=config.evaluation.bootstrap_samples,
    )
    c4_comparisons: list[dict[str, object]] = []
    for shots in [value for value in config.evaluation.shots if value in {5, 20}]:
        for treatment_method, control_method, label in (
            (
                "celar_teacher_decoder",
                "independent_action_ae",
                "celar_teacher_vs_independent_ae",
            ),
            (
                "celar_decoder_adapter",
                "independent_bc",
                "celar_policy_adapter_vs_independent_bc",
            ),
        ):
            for metric in ("semantic_mae", "executable_rate"):
                treatment = c4_raw[
                    (c4_raw["shots"] == shots)
                    & (c4_raw["method"] == treatment_method)
                ].sort_values("seed")
                control = c4_raw[
                    (c4_raw["shots"] == shots)
                    & (c4_raw["method"] == control_method)
                ].sort_values("seed")
                difference = paired_difference(
                    treatment[metric].to_numpy(),
                    control[metric].to_numpy(),
                    samples=config.evaluation.bootstrap_samples,
                    seed=930 + len(c4_comparisons),
                )
                c4_comparisons.append(
                    {
                        "name": f"c4_{shots}_{metric}_{label}",
                        "shots": shots,
                        "metric": metric,
                        "treatment": treatment_method,
                        "control": control_method,
                        **difference,
                    }
                )
    adjusted = holm_adjust(
        [float(item["paired_t_p"]) for item in c4_comparisons]
    )
    for item, value in zip(c4_comparisons, adjusted):
        item["holm_p"] = float(value)

    all_protocol = {
        "passed": all(bool(value["passed"]) for value in protocols.values()),
        "pair_overlap_count": sum(
            int(value["pair_overlap_count"]) for value in protocols.values()
        ),
        "task_overlap_count": sum(
            int(value["task_overlap_count"]) for value in protocols.values()
        ),
        "split_counts": protocols[str(config.evaluation.seeds[0])]["split_counts"],
        "body_counts": protocols[str(config.evaluation.seeds[0])]["body_counts"],
        "by_seed": protocols,
        "expert_action_audit": expert_audits,
        "bodies": expert_audits[str(config.evaluation.seeds[0])]["bodies"],
    }
    c1_table = c1_summary.set_index("method")
    c2_table = c2_summary.set_index("method")
    c4_table = c4_summary.set_index(["shots", "method"])
    c1_shared_comparison = next(
        item for item in c1_comparisons if item["name"] == "c1_retrieval_semantic_r1_celar_vs_shared_ae"
    )
    reference_shots = (
        5 if 5 in config.evaluation.shots else min(config.evaluation.shots)
    )
    c4_20_comparison = next(
        item
        for item in c4_comparisons
        if item["name"]
        == f"c4_{reference_shots}_semantic_mae_celar_teacher_vs_independent_ae"
    )
    c4_20_direction = (
        c4_raw[
            (c4_raw["shots"] == reference_shots)
            & (c4_raw["method"] == "celar_teacher_decoder")
        ]
        .sort_values("seed")["semantic_mae"]
        .to_numpy()
        < c4_raw[
            (c4_raw["shots"] == reference_shots)
            & (c4_raw["method"] == "independent_action_ae")
        ]
        .sort_values("seed")["semantic_mae"]
        .to_numpy()
    )
    c4_shot_checks: list[bool] = []
    for shots in [value for value in config.evaluation.shots if value in {5, 20}]:
        treatment = c4_raw[
            (c4_raw["shots"] == shots)
            & (c4_raw["method"] == "celar_teacher_decoder")
        ].sort_values("seed")
        control = c4_raw[
            (c4_raw["shots"] == shots)
            & (c4_raw["method"] == "independent_action_ae")
        ].sort_values("seed")
        if len(treatment) != len(control) or not len(treatment):
            c4_shot_checks.append(False)
            continue
        direction = (
            treatment["semantic_mae"].to_numpy()
            < control["semantic_mae"].to_numpy()
        )
        aggregate_ok = (
            float(treatment["semantic_mae"].mean())
            <= (0.85 if shots == 5 else 1.0)
            * float(control["semantic_mae"].mean())
        )
        c4_shot_checks.append(
            bool(
                aggregate_ok
                and float(direction.mean()) >= 2.0 / 3.0
                and float(treatment["executable_rate"].mean()) >= 0.85
            )
        )
    gates = {
        "c0": bool(all_protocol["passed"] and all(value["passed"] for value in expert_audits.values())),
        "c1": bool(
            float(c1_table.loc["celar", "retrieval_semantic_r1"]) >= 0.60
            and float(c1_shared_comparison["mean_difference"]) >= 0.10
            and float(c1_table.loc["celar", "executable_rate"]) >= 0.85
            and float(c1_table.loc["celar", "latent_effective_rank"]) >= 3.0
        ),
        "c2": bool(
            float(c2_table.loc["celar_bc", "semantic_mae"])
            <= 1.15 * float(c2_table.loc["native_bc", "semantic_mae"])
        ),
        "c4": bool(all(c4_shot_checks)),
    }
    if gates["c0"] and gates["c1"] and gates["c2"] and gates["c4"]:
        decision = "GO_TO_MUJOCO_C0_C3"
    elif gates["c0"] and gates["c1"] and (gates["c2"] or gates["c4"]):
        decision = "CONDITIONAL_GO_TO_MUJOCO"
    else:
        decision = "NO_GO_ON_CURRENT_PROXY"

    _write_json(output_dir / "c0_protocol.json", all_protocol)
    c1_raw.to_csv(output_dir / "c1_raw.csv", index=False, encoding="utf-8")
    c1_seed.to_csv(output_dir / "c1_seed_summary.csv", index=False, encoding="utf-8")
    c1_summary.to_csv(output_dir / "c1_summary.csv", index=False, encoding="utf-8")
    c2_raw.to_csv(output_dir / "c2_raw.csv", index=False, encoding="utf-8")
    c2_summary.to_csv(output_dir / "c2_summary.csv", index=False, encoding="utf-8")
    c4_raw.to_csv(output_dir / "c4_adaptation_raw.csv", index=False, encoding="utf-8")
    c4_summary.to_csv(
        output_dir / "c4_adaptation_summary.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(dataset_index_rows).to_csv(
        output_dir / "dataset_index.csv", index=False, encoding="utf-8"
    )
    _make_c1_plot(c1_summary, output_dir)
    _make_c4_plot(c4_summary, output_dir)
    manifest = {
        "config": config.to_dict(),
        "source_sha256": source_digest(root),
        "checkpoint_sha256": checkpoint_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": os.cpu_count(),
        },
        "total_runtime_seconds": time.perf_counter() - started,
        "scope": "controlled_cross_embodiment_kinematic_proxy",
        "paper_grade": False,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    results = {
        "scope": "controlled_cross_embodiment_kinematic_proxy",
        "paper_grade": False,
        "decision": decision,
        "gates": gates,
        "c0": all_protocol,
        "c1_summary": c1_summary.to_dict(orient="records"),
        "c1_comparisons": c1_comparisons,
        "c2_summary": c2_summary.to_dict(orient="records"),
        "c4_summary": c4_summary.to_dict(orient="records"),
        "c4_comparisons": c4_comparisons,
        "training_histories": training_histories,
        "runtime_seconds": manifest["total_runtime_seconds"],
    }
    _write_json(output_dir / "results.json", results)
    report = _report(
        c0=all_protocol,
        c1_summary=c1_summary,
        c2_summary=c2_summary,
        c4_summary=c4_summary,
        c1_comparisons=c1_comparisons,
        c4_comparisons=c4_comparisons,
        gates=gates,
        decision=decision,
        reference_shots=reference_shots,
        manifest=manifest,
    )
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    checksum_paths = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    ]
    checksum_lines = [
        f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in sorted(checksum_paths)
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"CELAR pilot written to {output_dir.resolve()}")
    return output_dir
