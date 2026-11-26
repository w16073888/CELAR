from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np
import torch
from torch.nn import functional as F

from .config import ModelConfig
from .data import PairBatch, Sample, make_pairs
from .models import LatentActionModel, Normalizer, TrainedLatentModel


LATENT_METHODS = (
    "independent_ae",
    "shared_ae",
    "celar_no_align",
    "celar_no_cycle",
    "celar_no_outcome",
    "celar_shuffled",
    "celar",
)


def _fit_context_normalizer(samples: list[Sample]) -> Normalizer:
    return Normalizer.fit(np.stack([sample.context for sample in samples]))


def _reconstruction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return (
        F.smooth_l1_loss(predicted, target, reduction="none") * mask
    ).sum() / mask.sum().clamp_min(1.0)


def _usage_loss(latent: torch.Tensor) -> torch.Tensor:
    centered = latent - latent.mean(dim=0, keepdim=True)
    std = centered.pow(2).mean(dim=0).sqrt()
    variance_target = (std - 0.12).pow(2).mean()
    normalized = centered / (std + 1e-4)
    correlation = normalized.transpose(0, 1) @ normalized / max(
        1, len(centered) - 1
    )
    off_diagonal = correlation - torch.diag_embed(
        torch.diagonal(correlation)
    )
    return variance_target + 0.10 * off_diagonal.pow(2).mean()


def _adjacent_sample_pairs(samples: list[Sample]) -> list[tuple[Sample, Sample]]:
    lookup = {
        (sample.episode_id, sample.body_index, sample.window_index): sample
        for sample in samples
    }
    result: list[tuple[Sample, Sample]] = []
    for sample in samples:
        following = lookup.get(
            (sample.episode_id, sample.body_index, sample.window_index + 1)
        )
        if following is not None:
            result.append((sample, following))
    return result


def _local_body_index(
    model: LatentActionModel,
    body_index: torch.Tensor,
) -> torch.Tensor:
    return body_index if len(model.decoders) > 1 else torch.zeros_like(body_index)


def _train_shared(
    pairs: PairBatch,
    train_samples: list[Sample],
    *,
    method: str,
    bodies: tuple[int, ...],
    cfg: ModelConfig,
    seed: int,
) -> tuple[LatentActionModel, Normalizer, list[dict[str, float]]]:
    torch.manual_seed(seed + 1301)
    rng = np.random.default_rng(seed + 541)
    model = LatentActionModel(
        context_dim=pairs.context.shape[-1],
        morphology_dim=pairs.morphology.shape[-1],
        chunk_length=pairs.native.shape[2],
        n_decoders=max(bodies) + 1,
        cfg=cfg,
    )
    normalizer = _fit_context_normalizer(
        [sample for sample in train_samples if sample.body_index in bodies]
    )
    adjacent = _adjacent_sample_pairs(train_samples)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    native = torch.as_tensor(pairs.native, dtype=torch.float32)
    mask = torch.as_tensor(pairs.action_mask, dtype=torch.float32)
    context = torch.as_tensor(pairs.context, dtype=torch.float32)
    morphology = torch.as_tensor(pairs.morphology, dtype=torch.float32)
    body_index = torch.as_tensor(pairs.body_index, dtype=torch.long)
    progress = torch.as_tensor(pairs.progress, dtype=torch.float32)
    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(pairs))
        totals = {
            key: 0.0
            for key in (
                "loss",
                "reconstruction",
                "alignment",
                "cycle",
                "outcome",
                "smooth",
                "usage",
            )
        }
        batches = 0
        for start in range(0, len(order), cfg.batch_size):
            indices_np = order[start : start + cfg.batch_size]
            indices = torch.from_numpy(indices_np.copy())
            batch_native = native[indices]
            batch_mask = mask[indices]
            batch_context = normalizer.transform(context[indices])
            batch_morph = morphology[indices]
            batch_body = body_index[indices]
            flat_native = batch_native.reshape(-1, *batch_native.shape[-2:])
            flat_mask = batch_mask.reshape(-1, *batch_mask.shape[-2:])
            flat_context = batch_context.reshape(-1, batch_context.shape[-1])
            flat_morph = batch_morph.reshape(-1, batch_morph.shape[-1])
            flat_body = batch_body.reshape(-1)
            latent = model.encode(flat_context, flat_native, flat_morph)
            decoded = model.decode(
                latent,
                flat_morph,
                _local_body_index(model, flat_body),
            )
            reconstruction = _reconstruction_loss(
                decoded, flat_native, flat_mask
            )
            latent_pairs = latent.reshape(len(indices), len(bodies), -1)
            alignment = torch.zeros((), dtype=torch.float32)
            cycle = torch.zeros((), dtype=torch.float32)
            outcome = torch.zeros((), dtype=torch.float32)
            smooth = torch.zeros((), dtype=torch.float32)
            usage = _usage_loss(latent)

            if method not in {"shared_ae", "celar_no_align"}:
                terms: list[torch.Tensor] = []
                for left, right in combinations(range(len(bodies)), 2):
                    right_latent = latent_pairs[:, right]
                    if method == "celar_shuffled":
                        right_latent = right_latent[
                            torch.randperm(
                                len(right_latent),
                                generator=torch.Generator().manual_seed(
                                    seed + epoch * 17 + left * 3 + right
                                ),
                            )
                        ]
                    terms.append(
                        F.mse_loss(latent_pairs[:, left], right_latent)
                    )
                alignment = torch.stack(terms).mean()

            if method in {"celar", "celar_no_outcome"}:
                terms = []
                for source, target in permutations(range(len(bodies)), 2):
                    source_slice = slice(source, None, len(bodies))
                    target_slice = slice(target, None, len(bodies))
                    cross_decoded = model.decode(
                        latent_pairs[:, source],
                        flat_morph[target_slice],
                        _local_body_index(model, flat_body[target_slice]),
                    )
                    cycled = model.encode(
                        flat_context[target_slice],
                        cross_decoded,
                        flat_morph[target_slice],
                    )
                    terms.append(
                        F.mse_loss(cycled, latent_pairs[:, source])
                    )
                cycle = torch.stack(terms).mean()

            if method in {"celar", "celar_no_cycle"}:
                predictions = model.predict_progress(
                    latent, flat_morph
                ).reshape(len(indices), len(bodies))
                prediction_error = F.mse_loss(
                    predictions, progress[indices]
                )
                pair_consistency = (
                    predictions - predictions.mean(dim=1, keepdim=True)
                ).pow(2).mean()
                outcome = prediction_error + pair_consistency

            if (
                cfg.smooth_weight > 0.0
                and adjacent
                and method not in {"shared_ae", "celar_no_align"}
            ):
                smooth_count = min(len(adjacent), cfg.batch_size)
                chosen = rng.choice(
                    len(adjacent), size=smooth_count, replace=False
                )
                current = [adjacent[int(index)][0] for index in chosen]
                following = [adjacent[int(index)][1] for index in chosen]
                current_latent = model.encode(
                    normalizer.transform(
                        np.stack([sample.context for sample in current])
                    ),
                    torch.as_tensor(
                        np.stack([sample.native_action for sample in current]),
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(
                        np.stack([sample.morphology for sample in current]),
                        dtype=torch.float32,
                    ),
                )
                following_latent = model.encode(
                    normalizer.transform(
                        np.stack([sample.context for sample in following])
                    ),
                    torch.as_tensor(
                        np.stack([sample.native_action for sample in following]),
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(
                        np.stack([sample.morphology for sample in following]),
                        dtype=torch.float32,
                    ),
                )
                smooth = F.mse_loss(current_latent, following_latent)

            loss = (
                cfg.reconstruction_weight * reconstruction
                + cfg.align_weight * alignment
                + cfg.cycle_weight * cycle
                + cfg.outcome_weight * outcome
                + cfg.smooth_weight * smooth
                + cfg.usage_weight * usage
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values = {
                "loss": loss,
                "reconstruction": reconstruction,
                "alignment": alignment,
                "cycle": cycle,
                "outcome": outcome,
                "smooth": smooth,
                "usage": usage,
            }
            for key, value in values.items():
                totals[key] += float(value.detach())
            batches += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                **{
                    key: value / max(1, batches)
                    for key, value in totals.items()
                },
            }
        )
    return model, normalizer, history


def _train_independent(
    samples: list[Sample],
    *,
    bodies: tuple[int, ...],
    cfg: ModelConfig,
    seed: int,
) -> tuple[dict[int, LatentActionModel], Normalizer, list[dict[str, float]]]:
    normalizer = _fit_context_normalizer(
        [sample for sample in samples if sample.body_index in bodies]
    )
    models: dict[int, LatentActionModel] = {}
    histories: list[dict[str, float]] = []
    for body in bodies:
        selected = [sample for sample in samples if sample.body_index == body]
        native = torch.as_tensor(
            np.stack([sample.native_action for sample in selected]),
            dtype=torch.float32,
        )
        mask = torch.as_tensor(
            np.stack([sample.action_mask for sample in selected]),
            dtype=torch.float32,
        )
        context = normalizer.transform(
            np.stack([sample.context for sample in selected])
        )
        morphology = torch.as_tensor(
            np.stack([sample.morphology for sample in selected]),
            dtype=torch.float32,
        )
        progress = torch.as_tensor(
            np.asarray([sample.progress for sample in selected]),
            dtype=torch.float32,
        )
        torch.manual_seed(seed + 9101 + body)
        rng = np.random.default_rng(seed + 771 + body)
        model = LatentActionModel(
            context_dim=selected[0].context.shape[0],
            morphology_dim=selected[0].morphology.shape[0],
            chunk_length=native.shape[1],
            n_decoders=1,
            cfg=cfg,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        history: list[dict[str, float]] = []
        for epoch in range(cfg.epochs):
            order = rng.permutation(len(selected))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(order), cfg.batch_size):
                indices = torch.from_numpy(
                    order[start : start + cfg.batch_size].copy()
                )
                latent = model.encode(
                    context[indices], native[indices], morphology[indices]
                )
                decoded = model.decode(
                    latent,
                    morphology[indices],
                    torch.zeros(len(indices), dtype=torch.long),
                )
                loss = _reconstruction_loss(
                    decoded, native[indices], mask[indices]
                )
                loss = loss + cfg.outcome_weight * F.mse_loss(
                    model.predict_progress(latent, morphology[indices]),
                    progress[indices],
                )
                loss = loss + cfg.usage_weight * _usage_loss(latent)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss.detach())
                batches += 1
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": epoch_loss / max(1, batches),
                }
            )
        models[body] = model
        histories.extend(
            {"body": float(body), **record} for record in history
        )
    return models, normalizer, histories


def train_latent_models(
    train_samples: list[Sample],
    *,
    bodies: tuple[int, ...],
    method: str,
    cfg: ModelConfig,
    seed: int,
) -> TrainedLatentModel:
    if method not in LATENT_METHODS:
        raise ValueError(f"Unknown latent method: {method}")
    if method == "independent_ae":
        models, normalizer, history = _train_independent(
            train_samples, bodies=bodies, cfg=cfg, seed=seed
        )
    else:
        pairs = make_pairs(train_samples, bodies, seed=seed)
        model, normalizer, history = _train_shared(
            pairs,
            train_samples,
            method=method,
            bodies=bodies,
            cfg=cfg,
            seed=seed,
        )
        models = {bodies[0]: model}
    return TrainedLatentModel(
        method=method,
        body_indices=bodies,
        context_normalizer=normalizer,
        models=models,
        history=history,
    )
