from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_tasks: int = 6
    val_tasks: int = 2
    test_tasks: int = 2
    train_episodes_per_task: int = 18
    val_episodes_per_task: int = 4
    test_episodes_per_task: int = 5
    trajectory_length: int = 24
    chunk_length: int = 4
    stride: int = 4
    observation_noise: float = 0.018
    action_noise: float = 0.025
    scene_angle_train: float = 25.0
    scene_angle_test: float = 110.0
    failure_rate: float = 0.10


@dataclass
class ModelConfig:
    hidden_dim: int = 96
    latent_dim: int = 8
    epochs: int = 45
    batch_size: int = 128
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    align_weight: float = 0.65
    cycle_weight: float = 0.20
    outcome_weight: float = 0.20
    smooth_weight: float = 0.06
    usage_weight: float = 0.05
    reconstruction_weight: float = 1.0
    pair_margin: float = 0.10
    bc_epochs: int = 80
    bc_batch_size: int = 128
    bc_action_weight: float = 1.5
    adaptation_steps: int = 280
    adaptation_learning_rate: float = 0.004


@dataclass
class EvalConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    bootstrap_samples: int = 3000
    retrieval_candidates: int = 64
    shots: list[int] = field(default_factory=lambda: [5, 20, 50])
    executable_semantic_tolerance: float = 0.34


@dataclass
class RLConfig:
    training_steps: int = 24_576
    rollout_envs: int = 8
    rollout_steps: int = 64
    update_epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 0.0003
    gamma: float = 0.98
    gae_lambda: float = 0.94
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    max_episode_steps: int = 36
    eval_episodes: int = 24
    eval_interval: int = 4096


@dataclass
class ExperimentConfig:
    output_dir: str = "results/celar_pilot_v1"
    c0_data_dir: str = "results/c0_mujoco_v1"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_known(dataclass_type: type, values: dict[str, Any] | None):
    if not values:
        return dataclass_type()
    known = set(dataclass_type.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {dataclass_type.__name__} fields: {sorted(unknown)}")
    return dataclass_type(**values)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    known = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown experiment fields: {sorted(unknown)}")
    return ExperimentConfig(
        output_dir=str(raw.get("output_dir", "results/celar_pilot_v1")),
        c0_data_dir=str(raw.get("c0_data_dir", "results/c0_mujoco_v1")),
        data=_merge_known(DataConfig, raw.get("data")),
        model=_merge_known(ModelConfig, raw.get("model")),
        evaluation=_merge_known(EvalConfig, raw.get("evaluation")),
        rl=_merge_known(RLConfig, raw.get("rl")),
    )
