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
    load_c0_dataset,
    make_pairs,
    source_digest,
)
from .metrics import bootstrap_mean_ci, evaluate_latent_model, paired_difference
from .models import TrainedLatentModel
from .training import LATENT_METHODS, train_latent_models


ALL_BODIES = (0, 1, 2)


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
    rows: pd.DataFrame,
    metrics: list[str],
    bootstrap_samples: int,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for method, group in rows.groupby("method"):
        record: dict[str, Any] = {"method": method, "runs": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            lower, upper = bootstrap_mean_ci(
                values, samples=bootstrap_samples, seed=1701
            )
            record[metric] = float(values.mean())
            record[f"{metric}_ci95_lower"] = lower
            record[f"{metric}_ci95_upper"] = upper
        output.append(record)
    return pd.DataFrame(output)


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values(
        "retrieval_semantic_r1", ascending=False
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, metric, title in (
        (
            axes[0],
            "retrieval_semantic_r1",
            "Semantic retrieval R@1",
        ),
        (
            axes[1],
            "retrieval_exact_pair_r1",
            "Exact paired-window R@1",
        ),
        (
            axes[2],
            "body_balance_gap",
            "Body semantic-MAE imbalance",
        ),
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
                "#c84a52" if method == "celar" else "#89a2b4"
                for method in ordered["method"]
            ],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=32, labelsize=8)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.02)
    figure.tight_layout()
    figure.savefig(output_dir / "c1_mujoco_retrieval.png", dpi=180)
    plt.close(figure)


def _report(
    *,
    protocol: dict[str, Any],
    summary: pd.DataFrame,
    comparisons: list[dict[str, Any]],
    gates: dict[str, bool],
    decision: str,
    manifest: dict[str, Any],
) -> str:
    table = summary.set_index("method")
    comparison = {
        item["name"]: item for item in comparisons
    }
    full = table.loc["celar"]
    shared = table.loc["shared_ae"]
    no_align = table.loc["celar_no_align"]
    lines = [
        "# CELAR C1 正式 MuJoCo 实验",
        "",
        f"**{decision}**",
        "",
        "> 数据来自正式 `C0_PASS` MuJoCo 数据。C1 同时训练和评估 Panda、"
        "UR5e、Sawyer；RGB-D 下采样、本体状态、任务和原生动作块均来自 C0 NPZ。",
        "",
        "## 门控",
        "",
        "| 项目 | 结果 |",
        "|---|---|",
        *[
            f"| {name} | {'通过' if passed else '未通过'} |"
            for name, passed in gates.items()
        ],
        "",
        f"- 训练/验证/测试窗口：{protocol['split_counts']['train_samples']} / "
        f"{protocol['split_counts']['val_samples']} / "
        f"{protocol['split_counts']['test_samples']}",
        f"- 配对数据：{protocol['split_counts']['train_pairs']} train、"
        f"{protocol['split_counts']['val_pairs']} val、"
        f"{protocol['split_counts']['test_pairs']} test",
        f"- pair/task/scene 泄漏：{protocol['pair_overlap_count']} / "
        f"{protocol['task_overlap_count']} / {protocol['scene_overlap_count']}",
        f"- 三本体 canonical semantic 不一致窗口："
        f"{protocol['canonical_semantic_mismatch_count']}",
        "",
        "## 三本体评估",
        "",
        "| 方法 | Semantic R@1 | Exact pair R@1 | Semantic MAE | "
        "可执行率 | 有效秩 | Min std | 本体失衡 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in summary.sort_values(
        "retrieval_semantic_r1", ascending=False
    )["method"]:
        row = table.loc[method]
        lines.append(
            f"| {method} | {row['retrieval_semantic_r1']:.3f} | "
            f"{row['retrieval_exact_pair_r1']:.3f} | "
            f"{row['semantic_mae']:.4f} | "
            f"{row['executable_rate']:.3f} | "
            f"{row['latent_effective_rank']:.2f} | "
            f"{row['latent_usage_min_std']:.4f} | "
            f"{row['body_balance_gap']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"完整 CELAR 相对普通共享 AE 的 semantic R@1 配对提升为 "
            f"**{comparison['c1_retrieval_semantic_r1_celar_vs_shared_ae']['mean_difference']:+.3f}**，"
            f"95% CI "
            f"[{comparison['c1_retrieval_semantic_r1_celar_vs_shared_ae']['ci95_lower']:+.3f}, "
            f"{comparison['c1_retrieval_semantic_r1_celar_vs_shared_ae']['ci95_upper']:+.3f}]。",
            f"相对无跨本体对齐消融的提升为 "
            f"**{comparison['c1_retrieval_semantic_r1_celar_vs_celar_no_align']['mean_difference']:+.3f}**。",
            f"完整 CELAR 相对普通共享 AE 的 exact-pair R@1 配对提升为 "
            f"**{comparison['c1_retrieval_exact_pair_r1_celar_vs_shared_ae']['mean_difference']:+.3f}**，"
            f"95% CI "
            f"[{comparison['c1_retrieval_exact_pair_r1_celar_vs_shared_ae']['ci95_lower']:+.3f}, "
            f"{comparison['c1_retrieval_exact_pair_r1_celar_vs_shared_ae']['ci95_upper']:+.3f}]。",
            "",
            "`Semantic R@1` 与 `Exact pair R@1` 是不同指标：前者接受同一语义容差内的"
            "动作邻域，后者要求找回完全相同的 C0 配对窗口。raw CSV 中使用"
            "`retrieval_semantic_r1_hit` 和 `retrieval_exact_pair_hit`，不再复用同一个"
            "模糊字段名。",
            "",
            "## 损失实现",
            "",
            "- outcome：逐样本 progress MSE + 同 pair 三本体预测方差，直接约束 "
            "`pred_progress(z_A) ≈ pred_progress(z_B) ≈ pred_progress(z_C)`。",
            "- smoothing：只构造同一 `(episode, body, window_t, window_t+1)` 的真实相邻窗口，"
            "随机 batch 的相邻行不再参与平滑损失。",
            "- cycle：三本体所有有序跨本体 decoder/re-encode 组合。",
            "",
            "## 证据边界",
            "",
            "C1 使用正式 C0 MuJoCo 数据和三本体评估，但视觉输入为 16×16 下采样 RGB-D，"
            "机械臂仍是最小统一 MJCF。不能把该结果代替厂商机械臂动力学、LIBERO 或真机结论。",
            "",
            f"源码 SHA-256：`{manifest['source_sha256']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_sim_c1(config: ExperimentConfig) -> Path:
    root = Path(__file__).resolve().parents[2]
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    c0_dir = Path(config.c0_data_dir)
    if not c0_dir.is_absolute():
        c0_dir = root / c0_dir
    started = time.perf_counter()
    print(f"[C1] loading formal C0 data from {c0_dir}")
    dataset, c0_manifest = load_c0_dataset(c0_dir)
    test_pairs = make_pairs(
        dataset.test,
        ALL_BODIES,
        limit=config.evaluation.retrieval_candidates,
        seed=20261126,
    )
    raw_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}
    checkpoints: dict[str, str] = {}
    for seed in config.evaluation.seeds:
        print(f"[C1 seed {seed}] training 7 action representations on 3 robots")
        for method in LATENT_METHODS:
            model = train_latent_models(
                dataset.train,
                bodies=ALL_BODIES,
                method=method,
                cfg=config.model,
                seed=seed,
            )
            summary, raw = evaluate_latent_model(
                model,
                test_pairs,
                tolerance=config.evaluation.executable_semantic_tolerance,
            )
            summary["seed"] = seed
            seed_rows.append(summary)
            raw_rows.extend({"seed": seed, **row} for row in raw)
            histories[f"{method}/seed_{seed}"] = model.history
            path = checkpoint_dir / f"seed_{seed}_{method}.pt"
            torch.save(
                {
                    "method": method,
                    "seed": seed,
                    "models": {
                        str(body): candidate.state_dict()
                        for body, candidate in model.models.items()
                    },
                    "context_normalizer": model.context_normalizer.state_dict(),
                },
                path,
            )
            checkpoints[f"{method}/seed_{seed}"] = _sha256(path)
    raw = pd.DataFrame(raw_rows)
    seed_frame = pd.DataFrame(seed_rows)
    metrics = [
        "retrieval_semantic_r1",
        "retrieval_semantic_r5",
        "retrieval_exact_pair_r1",
        "retrieval_exact_pair_r5",
        "reconstruction_mae",
        "semantic_mae",
        "executable_rate",
        "cycle_error",
        "progress_pair_gap",
        "progress_mae",
        "latent_effective_rank",
        "latent_usage_min_std",
        "body_balance_gap",
    ]
    summary = _aggregate(
        seed_frame,
        metrics,
        config.evaluation.bootstrap_samples,
    )
    table = summary.set_index("method")
    comparisons: list[dict[str, Any]] = []
    for metric, control in (
        ("retrieval_semantic_r1", "shared_ae"),
        ("retrieval_semantic_r1", "celar_no_align"),
        ("retrieval_exact_pair_r1", "shared_ae"),
        ("progress_pair_gap", "celar_no_outcome"),
    ):
        treatment_rows = seed_frame[seed_frame["method"] == "celar"].sort_values(
            "seed"
        )
        control_rows = seed_frame[seed_frame["method"] == control].sort_values(
            "seed"
        )
        comparison = paired_difference(
            treatment_rows[metric].to_numpy(),
            control_rows[metric].to_numpy(),
            samples=config.evaluation.bootstrap_samples,
            seed=2201 + len(comparisons),
        )
        comparisons.append(
            {
                "name": f"c1_{metric}_celar_vs_{control}",
                "metric": metric,
                "treatment": "celar",
                "control": control,
                **comparison,
            }
        )
    comparison_map = {item["name"]: item for item in comparisons}
    gates = {
        "c0_pass": True,
        "three_body_eval": bool(
            set(range(3))
            == {int(value) for value in test_pairs.body_index[0]}
        ),
        "pairing": bool(
            dataset.protocol["pair_overlap_count"] == 0
            and dataset.protocol["task_overlap_count"] == 0
            and dataset.protocol["scene_overlap_count"] == 0
            and dataset.protocol["canonical_semantic_mismatch_count"] == 0
        ),
        "semantic_retrieval": bool(
            float(table.loc["celar", "retrieval_semantic_r1"]) >= 0.90
        ),
        "exact_pair_retrieval": bool(
            float(table.loc["celar", "retrieval_exact_pair_r1"]) >= 0.60
            and comparison_map[
                "c1_retrieval_exact_pair_r1_celar_vs_shared_ae"
            ]["mean_difference"]
            >= 0.10
        ),
        "execution": bool(
            float(table.loc["celar", "executable_rate"]) >= 0.85
        ),
        "latent_usage": bool(
            float(table.loc["celar", "latent_effective_rank"]) >= 3.0
            and float(table.loc["celar", "latent_usage_min_std"]) >= 0.03
        ),
        "body_balance": bool(
            float(table.loc["celar", "body_balance_gap"]) <= 0.10
        ),
    }
    decision = "C1_PASS" if all(gates.values()) else "C1_BLOCKED"
    _plot(summary, output_dir)
    raw.to_csv(output_dir / "c1_raw.csv", index=False, encoding="utf-8")
    seed_frame.to_csv(
        output_dir / "c1_seed_summary.csv", index=False, encoding="utf-8"
    )
    summary.to_csv(
        output_dir / "c1_summary.csv", index=False, encoding="utf-8"
    )
    _write_json(output_dir / "pairing_audit.json", dataset.protocol)
    manifest = {
        "config": config.to_dict(),
        "c0_manifest_rows": len(c0_manifest),
        "source_sha256": source_digest(root),
        "checkpoint_sha256": checkpoints,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": os.cpu_count(),
        },
        "runtime_seconds": time.perf_counter() - started,
        "scope": "formal_c0_mujoco_three_embodiment_c1",
        "paper_grade": False,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    results = {
        "decision": decision,
        "gates": gates,
        "protocol": dataset.protocol,
        "summary": summary.to_dict(orient="records"),
        "comparisons": comparisons,
        "training_histories": histories,
        "paper_grade": False,
        "runtime_seconds": manifest["runtime_seconds"],
    }
    _write_json(output_dir / "results.json", results)
    report = _report(
        protocol=dataset.protocol,
        summary=summary,
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
    print(f"CELAR formal C1 written to {output_dir.resolve()}")
    return output_dir
