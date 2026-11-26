from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import ModelConfig
from .data import MAX_ACTION_DIM


@dataclass
class Normalizer:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, values: np.ndarray) -> "Normalizer":
        mean = np.asarray(values, dtype=np.float64).mean(axis=0)
        scale = np.asarray(values, dtype=np.float64).std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(
            mean=torch.from_numpy(mean.astype(np.float32)),
            scale=torch.from_numpy(scale.astype(np.float32)),
        )

    def transform(self, values: np.ndarray | torch.Tensor) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(np.asarray(values), dtype=torch.float32)
        return (values - self.mean) / self.scale

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "scale": self.scale}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.mean = torch.as_tensor(state["mean"]).clone()
        self.scale = torch.as_tensor(state["scale"]).clone()


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class NativeDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        morphology_dim: int,
        chunk_length: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.chunk_length = chunk_length
        self.network = _mlp(
            latent_dim + morphology_dim,
            cfg.hidden_dim,
            chunk_length * MAX_ACTION_DIM,
        )

    def forward(self, latent: torch.Tensor, morphology: torch.Tensor) -> torch.Tensor:
        output = self.network(torch.cat([latent, morphology], dim=-1))
        return torch.tanh(output).reshape(
            len(output), self.chunk_length, MAX_ACTION_DIM
        )


class LatentActionModel(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        morphology_dim: int,
        chunk_length: int,
        n_decoders: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.chunk_length = chunk_length
        self.context_encoder = _mlp(context_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.action_encoder = _mlp(
            chunk_length * MAX_ACTION_DIM + morphology_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
        )
        self.project = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        self.decoders = nn.ModuleList(
            [
                NativeDecoder(
                    cfg.latent_dim,
                    morphology_dim,
                    chunk_length,
                    cfg,
                )
                for _ in range(n_decoders)
            ]
        )
        self.outcome_head = _mlp(
            cfg.latent_dim + morphology_dim, cfg.hidden_dim, 1
        )

    def encode(
        self,
        context: torch.Tensor,
        native_action: torch.Tensor,
        morphology: torch.Tensor,
    ) -> torch.Tensor:
        action = torch.cat(
            [native_action.reshape(len(native_action), -1), morphology],
            dim=-1,
        )
        return self.project(
            torch.cat(
                [
                    self.context_encoder(context),
                    self.action_encoder(action),
                ],
                dim=-1,
            )
        )

    def decode(
        self,
        latent: torch.Tensor,
        morphology: torch.Tensor,
        body_index: torch.Tensor,
    ) -> torch.Tensor:
        body_index = torch.as_tensor(
            body_index, dtype=torch.long, device=latent.device
        )
        output = torch.zeros(
            (len(latent), self.chunk_length, MAX_ACTION_DIM),
            dtype=latent.dtype,
            device=latent.device,
        )
        for index, decoder in enumerate(self.decoders):
            mask = body_index == index
            if mask.any():
                output[mask] = decoder(latent[mask], morphology[mask])
        return output

    def predict_progress(
        self, latent: torch.Tensor, morphology: torch.Tensor
    ) -> torch.Tensor:
        return self.outcome_head(torch.cat([latent, morphology], dim=-1)).squeeze(-1)


@dataclass
class TrainedLatentModel:
    method: str
    body_indices: tuple[int, ...]
    context_normalizer: Normalizer
    models: dict[int, LatentActionModel]
    history: list[dict[str, float]]

    def model_for(self, body_index: int) -> LatentActionModel:
        if body_index in self.models:
            return self.models[body_index]
        if len(self.models) == 1:
            return next(iter(self.models.values()))
        raise KeyError(f"No model trained for body {body_index}")

    @torch.no_grad()
    def encode(
        self,
        context: np.ndarray,
        native_action: np.ndarray,
        morphology: np.ndarray,
        body_index: int,
    ) -> np.ndarray:
        model = self.model_for(body_index)
        model.eval()
        return model.encode(
            self.context_normalizer.transform(context),
            torch.as_tensor(native_action, dtype=torch.float32),
            torch.as_tensor(morphology, dtype=torch.float32),
        ).cpu().numpy()

    @torch.no_grad()
    def decode(
        self,
        latent: np.ndarray,
        morphology: np.ndarray,
        body_index: int,
    ) -> np.ndarray:
        model = self.model_for(body_index)
        model.eval()
        local_index = 0 if len(model.decoders) == 1 else int(body_index)
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32)
        body_tensor = torch.full(
            (len(latent_tensor),), local_index, dtype=torch.long
        )
        return model.decode(
            latent_tensor,
            torch.as_tensor(morphology, dtype=torch.float32),
            body_tensor,
        ).cpu().numpy()

    @torch.no_grad()
    def predict_progress(
        self,
        latent: np.ndarray,
        morphology: np.ndarray,
        body_index: int,
    ) -> np.ndarray:
        model = self.model_for(body_index)
        model.eval()
        return model.predict_progress(
            torch.as_tensor(latent, dtype=torch.float32),
            torch.as_tensor(morphology, dtype=torch.float32),
        ).cpu().numpy()
