from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import MORPHOLOGIES, generate_dataset, source_digest
from .metrics import bootstrap_mean_ci, evaluate_policy, paired_difference
from .rl import train_ppo
from .training import train_latent_models, train_policy


SEEN_BODIES = (0, 1)


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


def _aggregate(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
    samples: int,
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record["runs"] = len(group)
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)
            lower, upper = bootstrap_mean_ci(values, samples=samples, seed=9101)
            record[metric] = float(values.mean())
            record[f"{metric}_ci95_lower"] = lower
            record[f"{metric}_ci95_upper"] = upper
        output.append(record)
    return pd.DataFrame(output)


def _make_curves_plot(curves: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    for method, group in curves.groupby("method"):
        aggregated = group.groupby("environment_steps")["eval_success"].mean()
        axis.plot(
            aggregated.index,
            aggregated.values,
            marker="o",
            linewidth=2.2 if method == "celar_ppo" else 1.5,
            label=method,
        )
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Online environment steps")
    axis.set_ylabel("Held-out success rate")
    axis.set_title("Equal-budget PPO on the cross-embodiment control proxy")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "c3_learning_curves.png", dpi=180)
    plt.close(fig)


def run_c2_study(config: ExperimentConfig) -> Path:
    """Train and evaluate only the longer behavior-cloning comparison."""

    root = Path(__file__).resolve().parents[2]
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    checkpoint_hashes: dict[str, str] = {}
    for seed in config.evaluation.seeds:
        print(f"[c2 seed {seed}] paired data and CELAR latents")
        dataset = generate_dataset(config.data, seed)
        if not dataset.protocol["passed"]:
            raise RuntimeError(dataset.protocol)
        model = train_latent_models(
            dataset.train,
            bodies=SEEN_BODIES,
            method="celar",
            cfg=config.model,
            seed=seed,
        )
        native = train_policy(
            dataset.train,
            bodies=SEEN_BODIES,
            method="native_bc",
            cfg=config.model,
            seed=seed,
            latent_model=None,
        )
        celar = train_policy(
            dataset.train,
            bodies=SEEN_BODIES,
            method="celar_bc",
            cfg=config.model,
            seed=seed,
            latent_model=model,
        )
        for body in SEEN_BODIES:
            for method, policy in (("native_bc", native[body]), ("celar_bc", celar[body])):
                metrics = evaluate_policy(
                    policy,
                    dataset.test,
                    body_index=body,
                    tolerance=config.evaluation.executable_semantic_tolerance,
                )
                rows.append(
                    {
                        "seed": seed,
                        "body_index": body,
                        "body_name": MORPHOLOGIES[body].name,
                        "method": method,
                        **metrics,
                    }
                )
        checkpoint = checkpoint_dir / f"seed_{seed}_policies.pt"
        torch.save(
            {
                "seed": seed,
                "native": native[0].policy.state_dict(),
                "celar": celar[0].policy.state_dict(),
                "latent_mean": celar[0].latent_mean,
                "latent_scale": celar[0].latent_scale,
            },
            checkpoint,
        )
        checkpoint_hashes[str(seed)] = _sha256(checkpoint)
    raw = pd.DataFrame(rows)
    summary = _aggregate(
        raw,
        ["method"],
        ["semantic_mae", "native_mae", "executable_rate", "jerk"],
        config.evaluation.bootstrap_samples,
    )
    table = summary.set_index("method")
    c2_ratio = float(table.loc["celar_bc", "semantic_mae"]) / float(
        table.loc["native_bc", "semantic_mae"]
    )
    execution_ratio = float(table.loc["celar_bc", "executable_rate"]) / float(
        table.loc["native_bc", "executable_rate"]
    )
    passed = bool(c2_ratio <= 1.15 and execution_ratio >= 0.80)
    raw.to_csv(output_dir / "c2_raw.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "c2_summary.csv", index=False, encoding="utf-8")
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
        "scope": "controlled_celar_c2_long_bc",
        "paper_grade": False,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    results = {
        "decision": "C2_PASS" if passed else "C2_FAIL",
        "c2_pass": passed,
        "semantic_error_ratio": c2_ratio,
        "execution_ratio": execution_ratio,
        "summary": summary.to_dict(orient="records"),
        "paper_grade": False,
        "runtime_seconds": manifest["total_runtime_seconds"],
    }
    _write_json(output_dir / "results.json", results)
    report = "\n".join(
        [
            "# CELAR 长训练 C2 验证",
            "",
            f"- 结论：**{'通过' if passed else '未通过'}**",
            f"- 语义误差比：**{c2_ratio:.3f}**（门槛 ≤1.15）",
            f"- 可执行率比：**{execution_ratio:.3f}**（门槛 ≥0.80）",
            "",
            "| 方法 | 语义 MAE | 可执行率 | Jerk |",
            "|---|---:|---:|---:|",
            f"| native_bc | {table.loc['native_bc', 'semantic_mae']:.4f} | "
            f"{table.loc['native_bc', 'executable_rate']:.3f} | "
            f"{table.loc['native_bc', 'jerk']:.4f} |",
            f"| celar_bc | {table.loc['celar_bc', 'semantic_mae']:.4f} | "
            f"{table.loc['celar_bc', 'executable_rate']:.3f} | "
            f"{table.loc['celar_bc', 'jerk']:.4f} |",
            "",
            "该实验仅延长 BC 到 800 epoch，数据和切分与 v3 一致；用于判断 v3 的可执行率差距"
            "是否来自优化不足。它仍是二维运动学代理，不是论文级机器人结果。",
            "",
            f"源码 SHA-256：`{manifest['source_sha256']}`。",
            "",
        ]
    )
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    paths = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
            for path in sorted(paths)
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"CELAR C2 study written to {output_dir.resolve()}")
    return output_dir


def _markdown_report(
    *,
    c2: pd.DataFrame,
    c3: pd.DataFrame,
    comparisons: dict[str, object],
    gates: dict[str, bool],
    decision: str,
    manifest: dict[str, object],
) -> str:
    training_steps = int(manifest["config"]["rl"]["training_steps"])
    c2i = c2.set_index("method")
    c3i = c3.set_index("method")
    c2_ratio = float(c2i.loc["celar_bc", "semantic_mae"]) / float(
        c2i.loc["native_bc", "semantic_mae"]
    )
    native_steps = float(c3i.loc["native_ppo", "steps_to_50"])
    celar_steps = float(c3i.loc["celar_ppo", "steps_to_50"])
    if celar_steps < training_steps:
        sample_gain = 1.0 - celar_steps / max(1.0, native_steps)
    else:
        sample_gain = 0.0
    return "\n".join(
        [
            "# CELAR 价值延伸实验 C2/C3",
            "",
            "> 本报告是冻结旧版 `celar_pilot_v1` 后新增的代理实验。它检验 latent BC "
            "和等在线预算 PPO，结果仍不是 MuJoCo 或真机论文级证据。",
            "",
            "## 结论",
            "",
            f"- **决策：`{decision}`**",
            f"- 改进后的 C2 latent BC：{'通过' if gates['c2'] else '未通过'}",
            f"- C3 等预算在线 PPO：{'通过' if gates['c3'] else '未通过'}",
            "",
            "## C2：标准化 latent BC",
            "",
            "| 方法 | 语义 MAE | 可执行率 | Jerk |",
            "|---|---:|---:|---:|",
            f"| native_bc | {c2i.loc['native_bc', 'semantic_mae']:.4f} | "
            f"{c2i.loc['native_bc', 'executable_rate']:.3f} | "
            f"{c2i.loc['native_bc', 'jerk']:.4f} |",
            f"| celar_bc | {c2i.loc['celar_bc', 'semantic_mae']:.4f} | "
            f"{c2i.loc['celar_bc', 'executable_rate']:.3f} | "
            f"{c2i.loc['celar_bc', 'jerk']:.4f} |",
            "",
            f"CELAR BC / 原生 BC 的语义误差比为 **{c2_ratio:.3f}**；"
            "门槛为语义误差不高于 1.15，且可执行率不低于原生 BC 的 80%。",
            "",
            "## C3：同预算 PPO",
            "",
            "| 方法 | 最终成功率 | 50% 步数 | 80% 步数 |",
            "|---|---:|---:|---:|",
            *[
                f"| {method} | {c3i.loc[method, 'final_success']:.3f} | "
                f"{c3i.loc[method, 'steps_to_50']:.0f} | "
                f"{c3i.loc[method, 'steps_to_80']:.0f} |"
                for method in ("native_ppo", "shared_ppo", "celar_ppo")
            ],
            "",
            f"所有方法使用相同 {training_steps:,} 个在线环境步、相同 PPO 超参数、同一环境随机种子和"
            "同一组 Panda/UR5e 本体。CELAR 只冻结预训练动作 decoder；策略网络仍从头学习。",
            f"相对于原生 PPO，CELAR 达到 50% 成功率的步数变化为 **{sample_gain:+.1%}**。",
            f"最终成功率配对差值：`{comparisons['final_success']}`。",
            "",
            "## 证据边界",
            "",
            "- 环境是二维抓放控制代理，奖励和状态显式可观测。",
            "- 策略没有视觉主干、接触动力学、IK 或真实控制延迟。",
            "- C3 只能判断动作 decoder 先验是否在该代理中产生在线学习优势，不能代替 LIBERO/Robosuite。",
            "- 若 C3 未通过，下一阶段应保留 C1/C4 的动作表示价值，但先修复策略条件化，不能扩大论文主张。",
            "",
            "## 审计文件",
            "",
            "`c2_raw.csv`、`c2_summary.csv`、`c3_seed_summary.csv`、"
            "`c3_curves.csv`、`results.json`、`run_manifest.json`、`checkpoints/`、"
            "`checksums.sha256`。",
            "",
            f"源码 SHA-256：`{manifest['source_sha256']}`。",
            "",
        ]
    )


def run_value_study(config: ExperimentConfig) -> Path:
    root = Path(__file__).resolve().parents[2]
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    c2_rows: list[dict[str, object]] = []
    c3_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    checkpoint_hashes: dict[str, str] = {}
    for seed in config.evaluation.seeds:
        print(f"[value seed {seed}] paired data and latent models")
        dataset = generate_dataset(config.data, seed)
        if not dataset.protocol["passed"]:
            raise RuntimeError(dataset.protocol)
        latent_models = {}
        for method in ("shared_ae", "celar"):
            latent_models[method] = train_latent_models(
                dataset.train,
                bodies=SEEN_BODIES,
                method=method,
                cfg=config.model,
                seed=seed,
            )
            path = checkpoint_dir / f"seed_{seed}_{method}.pt"
            torch.save(
                {
                    "method": method,
                    "seed": seed,
                    "models": {
                        str(key): value.state_dict()
                        for key, value in latent_models[method].models.items()
                    },
                    "context_normalizer": latent_models[method].context_normalizer.state_dict(),
                },
                path,
            )
            checkpoint_hashes[f"latent/{method}/{seed}"] = _sha256(path)

        print(f"[value seed {seed}] standardized latent and native BC")
        native = train_policy(
            dataset.train,
            bodies=SEEN_BODIES,
            method="native_bc",
            cfg=config.model,
            seed=seed,
            latent_model=None,
        )
        celar = train_policy(
            dataset.train,
            bodies=SEEN_BODIES,
            method="celar_bc",
            cfg=config.model,
            seed=seed,
            latent_model=latent_models["celar"],
        )
        for body in SEEN_BODIES:
            for method, policy in (("native_bc", native[body]), ("celar_bc", celar[body])):
                record = evaluate_policy(
                    policy,
                    dataset.test,
                    body_index=body,
                    tolerance=config.evaluation.executable_semantic_tolerance,
                )
                c2_rows.append(
                    {
                        "seed": seed,
                        "body_index": body,
                        "body_name": MORPHOLOGIES[body].name,
                        "method": method,
                        **record,
                    }
                )
        bc_path = checkpoint_dir / f"seed_{seed}_bc_policies.pt"
        torch.save(
            {
                "native": native[0].policy.state_dict(),
                "celar": celar[0].policy.state_dict(),
                "celar_latent_mean": celar[0].latent_mean,
                "celar_latent_scale": celar[0].latent_scale,
            },
            bc_path,
        )
        checkpoint_hashes[f"bc/{seed}"] = _sha256(bc_path)

        print(f"[value seed {seed}] equal-budget PPO")
        for method, decoder in (
            ("native_ppo", None),
            ("shared_ppo", latent_models["shared_ae"]),
            ("celar_ppo", latent_models["celar"]),
        ):
            result = train_ppo(
                latent_model=decoder,
                bodies=SEEN_BODIES,
                seed=seed,
                **asdict(config.rl),
            )
            c3_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "final_success": result.final_success,
                    "steps_to_50": result.steps_to_50,
                    "steps_to_80": result.steps_to_80,
                    "runtime_seconds": result.runtime_seconds,
                }
            )
            for record in result.history:
                curve_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        **record,
                    }
                )
            path = checkpoint_dir / f"seed_{seed}_{method}.pt"
            torch.save(
                {"seed": seed, "method": method, "policy": result.policy.state_dict()},
                path,
            )
            checkpoint_hashes[f"ppo/{method}/{seed}"] = _sha256(path)

    c2_raw = pd.DataFrame(c2_rows)
    c2_summary = _aggregate(
        c2_raw,
        ["method"],
        ["semantic_mae", "native_mae", "executable_rate", "jerk"],
        config.evaluation.bootstrap_samples,
    )
    c3_frame = pd.DataFrame(c3_rows)
    c3_summary = _aggregate(
        c3_frame,
        ["method"],
        ["final_success", "steps_to_50", "steps_to_80"],
        config.evaluation.bootstrap_samples,
    )
    curves = pd.DataFrame(curve_rows)
    treatment = c3_frame[c3_frame["method"] == "celar_ppo"].sort_values("seed")
    native_rows = c3_frame[c3_frame["method"] == "native_ppo"].sort_values("seed")
    success_difference = paired_difference(
        treatment["final_success"].to_numpy(),
        native_rows["final_success"].to_numpy(),
        samples=config.evaluation.bootstrap_samples,
        seed=1101,
    )
    step_difference = paired_difference(
        treatment["steps_to_50"].to_numpy(),
        native_rows["steps_to_50"].to_numpy(),
        samples=config.evaluation.bootstrap_samples,
        seed=1102,
    )
    c2_table = c2_summary.set_index("method")
    c3_table = c3_summary.set_index("method")
    gates = {
        "c2": bool(
            float(c2_table.loc["celar_bc", "semantic_mae"])
            <= 1.15 * float(c2_table.loc["native_bc", "semantic_mae"])
            and float(c2_table.loc["celar_bc", "executable_rate"])
            >= 0.80 * float(c2_table.loc["native_bc", "executable_rate"])
        ),
        "c3": bool(
            (
                float(c3_table.loc["celar_ppo", "final_success"])
                >= float(c3_table.loc["native_ppo", "final_success"]) + 0.05
                or float(c3_table.loc["celar_ppo", "steps_to_50"])
                <= 0.70 * float(c3_table.loc["native_ppo", "steps_to_50"])
            )
            and float(c3_table.loc["celar_ppo", "final_success"]) >= 0.45
        ),
    }
    if gates["c2"] and gates["c3"]:
        decision = "GO_TO_MUJOCO_FULL_PILOT"
    elif gates["c2"] or gates["c3"]:
        decision = "CONDITIONAL_GO_KEEP_C1_C4_SCOPE"
    else:
        decision = "NO_GO_ON_POLICY_EXTENSION"
    _make_curves_plot(curves, output_dir)
    c2_raw.to_csv(output_dir / "c2_raw.csv", index=False, encoding="utf-8")
    c2_summary.to_csv(output_dir / "c2_summary.csv", index=False, encoding="utf-8")
    c3_frame.to_csv(output_dir / "c3_seed_summary.csv", index=False, encoding="utf-8")
    c3_summary.to_csv(output_dir / "c3_summary.csv", index=False, encoding="utf-8")
    curves.to_csv(output_dir / "c3_curves.csv", index=False, encoding="utf-8")
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
        "scope": "controlled_celar_c2_c3_proxy",
        "paper_grade": False,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    comparisons = {
        "final_success": success_difference,
        "steps_to_50": step_difference,
    }
    results = {
        "decision": decision,
        "gates": gates,
        "c2_summary": c2_summary.to_dict(orient="records"),
        "c3_summary": c3_summary.to_dict(orient="records"),
        "comparisons": comparisons,
        "paper_grade": False,
        "scope": manifest["scope"],
        "runtime_seconds": manifest["total_runtime_seconds"],
    }
    _write_json(output_dir / "results.json", results)
    report = _markdown_report(
        c2=c2_summary,
        c3=c3_summary,
        comparisons=comparisons,
        gates=gates,
        decision=decision,
        manifest=manifest,
    )
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    paths = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
            for path in sorted(paths)
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"CELAR value study written to {output_dir.resolve()}")
    return output_dir
