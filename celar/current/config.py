from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    chunk_length: int = 8


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
    usage_weight: float = 1.0
    reconstruction_weight: float = 1.0
    pair_margin: float = 0.10


@dataclass
class EvaluationConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    bootstrap_samples: int = 3000
    retrieval_candidates: int = 60
    executable_semantic_tolerance: float = 0.34


@dataclass
class ExperimentConfig:
    output_dir: str = "results/c1_mujoco_v7"
    c0_data_dir: str = "results/c0_mujoco_v1"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

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
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    known = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown current CELAR fields: {sorted(unknown)}")
    return ExperimentConfig(
        output_dir=str(raw.get("output_dir", "results/c1_mujoco_v7")),
        c0_data_dir=str(raw.get("c0_data_dir", "results/c0_mujoco_v1")),
        data=_merge_known(DataConfig, raw.get("data")),
        model=_merge_known(ModelConfig, raw.get("model")),
        evaluation=_merge_known(EvaluationConfig, raw.get("evaluation")),
    )
