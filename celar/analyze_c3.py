from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import CONTEXT_DIM, MORPH_DIM, MORPHOLOGIES
from .metrics import bootstrap_mean_ci
from .models import LatentActionModel, Normalizer, TrainedLatentModel
from .rl import (
    OBS_DIM,
    TASK_FAMILIES,
    GaussianActorCritic,
    _decode_actions,
    CrossEmbodimentControlEnv,
)


def _load_latent_model(path: Path, config) -> TrainedLatentModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = LatentActionModel(
        context_dim=CONTEXT_DIM,
        morphology_dim=MORPH_DIM,
        chunk_length=config.data.chunk_length,
        n_decoders=2,
        cfg=config.model,
    )
    model.load_state_dict(checkpoint["models"]["0"])
    normalizer = Normalizer.fit(np.zeros((2, CONTEXT_DIM), dtype=np.float32))
    normalizer.load_state_dict(checkpoint["context_normalizer"])
    return TrainedLatentModel(
        method=checkpoint.get("method", path.stem),
        body_indices=(0, 1),
        context_normalizer=normalizer,
        models={0: model},
        history=[],
    )


def _load_policy(path: Path) -> GaussianActorCritic:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy = GaussianActorCritic(OBS_DIM, 8)
    policy.load_state_dict(checkpoint["policy"])
    return policy


def _evaluate_group(
    policy: GaussianActorCritic,
    decoder: TrainedLatentModel | None,
    *,
    body: int,
    task: int,
    seed: int,
    episodes: int = 24,
) -> float:
    env = CrossEmbodimentControlEnv(
        (body,),
        envs=episodes,
        seed=seed,
        auto_reset=False,
        fixed_body=body,
        fixed_task=task,
    )
    observation = env.reset()
    success = np.zeros(episodes, dtype=bool)
    with torch.no_grad():
        for _ in range(env.max_episode_steps):
            mean, _ = policy(torch.as_tensor(observation, dtype=torch.float32))
            semantic = _decode_actions(
                mean.cpu().numpy(),
                env.body,
                latent_model=decoder,
            )
            result = env.step(semantic)
            success |= result.success
            observation = result.observation
            if success.all():
                break
    return float(success.mean())


def run_analysis(config_path: Path, source_dir: Path, output_dir: Path) -> Path:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder_by_seed: dict[int, dict[str, TrainedLatentModel | None]] = {}
    for seed in config.evaluation.seeds:
        decoder_by_seed[seed] = {
            "native_ppo": None,
            "shared_ppo": _load_latent_model(
                source_dir / "checkpoints" / f"seed_{seed}_shared_ae.pt",
                config,
            ),
            "celar_ppo": _load_latent_model(
                source_dir / "checkpoints" / f"seed_{seed}_celar.pt",
                config,
            ),
        }
    rows: list[dict[str, object]] = []
    for seed in config.evaluation.seeds:
        for method in ("native_ppo", "shared_ppo", "celar_ppo"):
            policy = _load_policy(
                source_dir / "checkpoints" / f"seed_{seed}_{method}.pt"
            )
            for body in (0, 1):
                for task_index, task in enumerate(TASK_FAMILIES):
                    success = _evaluate_group(
                        policy,
                        decoder_by_seed[seed][method],
                        body=body,
                        task=task_index,
                        seed=seed * 10_000 + body * 100 + task_index + 77,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "body_index": body,
                            "body_name": MORPHOLOGIES[body].name,
                            "task_index": task_index,
                            "task_name": task,
                            "success_rate": success,
                        }
                    )
    raw = pd.DataFrame(rows)
    group = (
        raw.groupby(
            ["method", "body_index", "body_name", "task_index", "task_name"],
            as_index=False,
        )["success_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "success_rate", "std": "seed_std", "count": "seeds"})
    )
    treatment = raw[raw["method"] == "celar_ppo"]
    control = raw[raw["method"] == "native_ppo"]
    merged = treatment.merge(
        control,
        on=["seed", "body_index", "task_index"],
        suffixes=("_celar", "_native"),
    )
    differences = (
        merged["success_rate_celar"] - merged["success_rate_native"]
    ).to_numpy(dtype=np.float64)
    lower, upper = bootstrap_mean_ci(differences, samples=3000, seed=1201)
    positive = float((differences > 0).mean())
    nonnegative = float((differences >= 0).mean())
    passed = bool(float(differences.mean()) > 0.0 and positive >= 2.0 / 3.0)
    raw.to_csv(output_dir / "c3_group_raw.csv", index=False, encoding="utf-8")
    group.to_csv(output_dir / "c3_group_summary.csv", index=False, encoding="utf-8")
    summary = {
        "comparison": "celar_ppo_vs_native_ppo",
        "seed_body_task_pairs": int(len(differences)),
        "mean_success_difference": float(differences.mean()),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "positive_fraction": positive,
        "nonnegative_fraction": nonnegative,
        "passed_majority_direction": passed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pivoted = group.pivot_table(
        index=["body_name", "task_name"],
        columns="method",
        values="success_rate",
    ).reset_index()
    lines = [
        "# CELAR C3 分组审计",
        "",
        "对已保存的 v3 PPO 策略重新做固定本体、固定任务评估，不重新训练。",
        "",
        f"- 种子 × 本体 × 任务配对数：{summary['seed_body_task_pairs']}",
        f"- CELAR - 原生 PPO 平均成功率差：{summary['mean_success_difference']:+.3f}",
        f"- bootstrap 95% CI：[{summary['ci95_lower']:+.3f}, {summary['ci95_upper']:+.3f}]",
        f"- 方向为正比例：{summary['positive_fraction']:.1%}",
        f"- 多数方向门槛：{'通过' if passed else '未通过'}",
        "",
        "| 本体 | 任务 | 原生 PPO | 共享 decoder PPO | CELAR PPO |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in pivoted.sort_values(["body_name", "task_name"]).iterrows():
        lines.append(
            f"| {row['body_name']} | {row['task_name']} | "
            f"{row['native_ppo']:.3f} | {row['shared_ppo']:.3f} | "
            f"{row['celar_ppo']:.3f} |"
        )
    lines.extend(
        [
            "",
            "该分析仍使用二维抓放控制代理和 24 个固定种子的评估回合。",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved C3 policies by task/body")
    parser.add_argument("--config", default="configs/value_v3.yaml")
    parser.add_argument("--source", default="results/celar_value_v3")
    parser.add_argument("--output", default="results/celar_v3_group_analysis")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_analysis(
        root / args.config,
        root / args.source,
        root / args.output,
    )


if __name__ == "__main__":
    main()
