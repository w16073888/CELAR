from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np
import torch
from torch.nn import functional as F

from .config import ModelConfig
from .data import (
    CONTEXT_DIM,
    MORPH_DIM,
    MORPHOLOGIES,
    PairBatch,
    Sample,
    make_pairs,
    samples_for_body,
)
from .models import (
    ActionPolicy,
    LatentAdapter,
    LatentActionModel,
    NativeDecoder,
    Normalizer,
    TrainedLatentModel,
    TrainedPolicy,
)


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


def _loss_reconstruction(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(predicted, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _latent_usage_loss(latent: torch.Tensor) -> torch.Tensor:
    centered = latent - latent.mean(dim=0, keepdim=True)
    variance = centered.pow(2).mean(dim=0)
    std = variance.sqrt()
    variance_target = (std - 0.12).pow(2).mean()
    normalized = centered / (std + 1e-4)
    correlation = normalized.transpose(0, 1) @ normalized / max(
        1, len(centered) - 1
    )
    off_diagonal = correlation - torch.diag_embed(
        torch.diagonal(correlation)
    )
    return variance_target + 0.10 * off_diagonal.pow(2).mean()


def _local_body_index(model: LatentActionModel, global_index: torch.Tensor) -> torch.Tensor:
    return global_index if len(model.decoders) > 1 else torch.zeros_like(global_index)


def _adjacent_sample_pairs(samples: list[Sample]) -> list[tuple[Sample, Sample]]:
    lookup = {
        (sample.episode_id, sample.body_index, sample.window_index): sample
        for sample in samples
    }
    pairs: list[tuple[Sample, Sample]] = []
    for sample in samples:
        following = lookup.get(
            (sample.episode_id, sample.body_index, sample.window_index + 1)
        )
        if following is not None:
            pairs.append((sample, following))
    return pairs


def _train_shared_model(
    pairs: PairBatch,
    train_samples: list[Sample],
    *,
    method: str,
    cfg: ModelConfig,
    bodies: tuple[int, ...],
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
    adjacent_pairs = _adjacent_sample_pairs(train_samples)
    normalizer = _fit_context_normalizer(
        [
            _sample_from_pair_row(pairs, index, body_slot)
            for index in range(min(len(pairs), 512))
            for body_slot in range(len(bodies))
        ]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    history: list[dict[str, float]] = []
    native = torch.as_tensor(pairs.native, dtype=torch.float32)
    mask = torch.as_tensor(pairs.action_mask, dtype=torch.float32)
    context = torch.as_tensor(pairs.context, dtype=torch.float32)
    morphology = torch.as_tensor(pairs.morphology, dtype=torch.float32)
    body_index = torch.as_tensor(pairs.body_index, dtype=torch.long)
    progress = torch.as_tensor(pairs.progress, dtype=torch.float32)
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(pairs))
        totals = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "alignment": 0.0,
            "cycle": 0.0,
            "outcome": 0.0,
            "smooth": 0.0,
            "usage": 0.0,
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
            local_body = _local_body_index(model, flat_body)
            reconstructed = model.decode(latent, flat_morph, local_body)
            reconstruction = _loss_reconstruction(
                reconstructed, flat_native, flat_mask
            )
            alignment = torch.zeros((), dtype=torch.float32)
            cycle = torch.zeros((), dtype=torch.float32)
            outcome = torch.zeros((), dtype=torch.float32)
            smooth = torch.zeros((), dtype=torch.float32)
            usage = _latent_usage_loss(latent)
            latent_pairs = latent.reshape(len(indices), len(bodies), -1)
            if method not in {"shared_ae", "celar_no_align"}:
                alignment_terms: list[torch.Tensor] = []
                for left, right in combinations(range(len(bodies)), 2):
                    target_right = latent_pairs[:, right]
                    if method == "celar_shuffled":
                        target_right = target_right[
                            torch.randperm(
                                len(target_right),
                                generator=torch.Generator().manual_seed(
                                    seed + epoch * 17 + left * 3 + right
                                ),
                            )
                        ]
                    alignment_terms.append(
                        F.mse_loss(latent_pairs[:, left], target_right)
                    )
                alignment = torch.stack(alignment_terms).mean()
            if method in {"celar", "celar_no_outcome"}:
                cycle_terms: list[torch.Tensor] = []
                for source, target in permutations(range(len(bodies)), 2):
                    source_slice = slice(source, None, len(bodies))
                    target_slice = slice(target, None, len(bodies))
                    decoded = model.decode(
                        latent_pairs[:, source],
                        flat_morph[target_slice],
                        _local_body_index(model, flat_body[target_slice]),
                    )
                    cycled = model.encode(
                        flat_context[target_slice],
                        decoded,
                        flat_morph[target_slice],
                    )
                    cycle_terms.append(
                        F.mse_loss(cycled, latent_pairs[:, source])
                    )
                cycle = torch.stack(cycle_terms).mean()
            if method in {"celar", "celar_no_cycle"}:
                predicted_progress = model.predict_progress(
                    latent, flat_morph
                ).reshape(len(indices), len(bodies))
                target_progress = progress[indices]
                prediction_loss = F.mse_loss(
                    predicted_progress, target_progress
                )
                pair_consistency = (
                    predicted_progress
                    - predicted_progress.mean(dim=1, keepdim=True)
                ).pow(2).mean()
                outcome = prediction_loss + pair_consistency
            if (
                cfg.smooth_weight > 0.0
                and adjacent_pairs
                and method not in {"shared_ae", "celar_no_align"}
            ):
                smooth_count = min(len(adjacent_pairs), cfg.batch_size)
                smooth_indices = rng.choice(
                    len(adjacent_pairs), size=smooth_count, replace=False
                )
                smooth_current = [
                    adjacent_pairs[int(index)][0] for index in smooth_indices
                ]
                smooth_following = [
                    adjacent_pairs[int(index)][1] for index in smooth_indices
                ]
                current_context = normalizer.transform(
                    np.stack([sample.context for sample in smooth_current])
                )
                following_context = normalizer.transform(
                    np.stack([sample.context for sample in smooth_following])
                )
                current_native = torch.as_tensor(
                    np.stack([sample.native_action for sample in smooth_current]),
                    dtype=torch.float32,
                )
                following_native = torch.as_tensor(
                    np.stack([sample.native_action for sample in smooth_following]),
                    dtype=torch.float32,
                )
                current_morph = torch.as_tensor(
                    np.stack([sample.morphology for sample in smooth_current]),
                    dtype=torch.float32,
                )
                following_morph = torch.as_tensor(
                    np.stack([sample.morphology for sample in smooth_following]),
                    dtype=torch.float32,
                )
                current_latent = model.encode(
                    current_context, current_native, current_morph
                )
                following_latent = model.encode(
                    following_context, following_native, following_morph
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
            totals["loss"] += float(loss.detach())
            totals["reconstruction"] += float(reconstruction.detach())
            totals["alignment"] += float(alignment.detach())
            totals["cycle"] += float(cycle.detach())
            totals["outcome"] += float(outcome.detach())
            totals["smooth"] += float(smooth.detach())
            totals["usage"] += float(usage.detach())
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


def _sample_from_pair_row(pairs: PairBatch, index: int, body_slot: int) -> Sample:
    return Sample(
        body_index=int(pairs.body_index[index, body_slot]),
        pair_id=pairs.pair_ids[index],
        episode_id=pairs.episode_ids[index],
        task_id=pairs.task_ids[index],
        trajectory_id="",
        window_index=int(pairs.window_indices[index, body_slot]),
        native_action=pairs.native[index, body_slot],
        action_mask=pairs.action_mask[index, body_slot],
        context=pairs.context[index, body_slot],
        morphology=pairs.morphology[index, body_slot],
        semantic_action=pairs.semantic[index, body_slot],
        progress=float(pairs.progress[index, body_slot]),
        success=float(pairs.success[index, body_slot]),
    )


def _train_independent(
    train_samples: list[Sample],
    *,
    bodies: tuple[int, ...],
    cfg: ModelConfig,
    seed: int,
) -> tuple[dict[int, LatentActionModel], Normalizer, list[dict[str, float]]]:
    models: dict[int, LatentActionModel] = {}
    normalizer = _fit_context_normalizer(
        [sample for sample in train_samples if sample.body_index in bodies]
    )
    histories: list[dict[str, float]] = []
    for body_index in bodies:
        body_samples = samples_for_body(train_samples, body_index)
        native = torch.as_tensor(
            np.stack([sample.native_action for sample in body_samples]), dtype=torch.float32
        )
        mask = torch.as_tensor(
            np.stack([sample.action_mask for sample in body_samples]), dtype=torch.float32
        )
        context = torch.as_tensor(
            normalizer.transform(np.stack([sample.context for sample in body_samples])),
            dtype=torch.float32,
        )
        morphology = torch.as_tensor(
            np.stack([sample.morphology for sample in body_samples]), dtype=torch.float32
        )
        progress_target = torch.as_tensor(
            np.asarray([sample.progress for sample in body_samples]), dtype=torch.float32
        )
        torch.manual_seed(seed + 9101 + body_index)
        rng = np.random.default_rng(seed + 771 + body_index)
        model = LatentActionModel(
            context_dim=body_samples[0].context.shape[0],
            morphology_dim=body_samples[0].morphology.shape[0],
            chunk_length=native.shape[1],
            n_decoders=1,
            cfg=cfg,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        history: list[dict[str, float]] = []
        for epoch in range(cfg.epochs):
            order = rng.permutation(len(body_samples))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(order), cfg.batch_size):
                indices = torch.from_numpy(order[start : start + cfg.batch_size].copy())
                latent = model.encode(
                    context[indices], native[indices], morphology[indices]
                )
                reconstructed = model.decode(
                    latent,
                    morphology[indices],
                    torch.zeros(len(indices), dtype=torch.long),
                )
                reconstruction = _loss_reconstruction(
                    reconstructed, native[indices], mask[indices]
                )
                predicted_progress = model.predict_progress(
                    latent, morphology[indices]
                )
                loss = reconstruction + cfg.outcome_weight * F.mse_loss(
                    predicted_progress, progress_target[indices]
                )
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
        models[body_index] = model
        for record in history:
            histories.append({"body": float(body_index), **record})
    return models, normalizer, histories


def train_latent_models(
    train_samples: list[Sample],
    *,
    bodies: tuple[int, ...],
    method: str,
    cfg: ModelConfig,
    seed: int,
    pair_limit: int | None = None,
) -> TrainedLatentModel:
    pairs = make_pairs(train_samples, bodies, limit=pair_limit, seed=seed)
    if method not in LATENT_METHODS:
        raise ValueError(f"Unknown latent method: {method}")
    if method == "independent_ae":
        models, normalizer, history = _train_independent(
            train_samples,
            bodies=bodies,
            cfg=cfg,
            seed=seed,
        )
    else:
        model, normalizer, history = _train_shared_model(
            pairs,
            train_samples,
            method=method,
            cfg=cfg,
            bodies=bodies,
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


def train_policy(
    train_samples: list[Sample],
    *,
    bodies: tuple[int, ...],
    method: str,
    cfg: ModelConfig,
    seed: int,
    latent_model: TrainedLatentModel | None = None,
) -> dict[int, TrainedPolicy]:
    selected = [sample for sample in train_samples if sample.body_index in bodies]
    if not selected:
        raise ValueError("No samples for behavior cloning")
    rng = np.random.default_rng(seed + 30_001)
    torch.manual_seed(seed + 40_001)
    normalizer = (
        copy.deepcopy(latent_model.context_normalizer)
        if latent_model is not None
        else _fit_context_normalizer(selected)
    )
    contexts = normalizer.transform(np.stack([sample.context for sample in selected]))
    native = torch.as_tensor(
        np.stack([sample.native_action for sample in selected]), dtype=torch.float32
    )
    body = np.asarray([sample.body_index for sample in selected], dtype=np.int64)
    body_tensor = torch.as_tensor(body, dtype=torch.long)
    morphology = torch.as_tensor(
        np.stack([sample.morphology for sample in selected]), dtype=torch.float32
    )
    latent_targets: np.ndarray | None = None
    latent_mean: np.ndarray | None = None
    latent_scale: np.ndarray | None = None
    output_dim = native.shape[-2] * native.shape[-1]
    if latent_model is not None:
        latent_targets = np.zeros((len(selected), latent_model.model_for(bodies[0]).latent_dim), dtype=np.float32)
        for body_index in bodies:
            mask = body == body_index
            body_positions = np.flatnonzero(mask)
            latent_targets[body_positions] = latent_model.encode(
                np.stack([selected[index].context for index in body_positions]),
                native.numpy()[mask],
                morphology.numpy()[mask],
                body_index,
            )
        latent_mean = latent_targets.mean(axis=0)
        latent_scale = latent_targets.std(axis=0)
        latent_scale = np.where(latent_scale < 1e-6, 1.0, latent_scale)
        latent_targets = (latent_targets - latent_mean) / latent_scale
        output_dim = latent_targets.shape[-1]
    target = (
        torch.as_tensor(latent_targets, dtype=torch.float32)
        if latent_targets is not None
        else native.reshape(len(native), -1)
    )
    policy = ActionPolicy(selected[0].context.shape[0], output_dim, cfg)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)
    start_time = time.perf_counter()
    losses: list[float] = []
    for _ in range(cfg.bc_epochs):
        indices = torch.from_numpy(
            rng.integers(0, len(selected), size=min(cfg.bc_batch_size, len(selected)))
        )
        predicted = policy(contexts[indices])
        loss = F.smooth_l1_loss(predicted, target[indices])
        if latent_model is not None:
            source_model = latent_model.model_for(latent_model.body_indices[0])
            local_body = (
                body_tensor[indices]
                if len(source_model.decoders) > 1
                else torch.zeros(len(indices), dtype=torch.long)
            )
            decoded = source_model.decode(
                predicted * torch.as_tensor(latent_scale, dtype=torch.float32)
                + torch.as_tensor(latent_mean, dtype=torch.float32),
                morphology[indices],
                local_body,
            )
            action_loss = _loss_reconstruction(
                decoded,
                native[indices],
                torch.as_tensor(
                    np.stack([sample.action_mask for sample in selected]),
                    dtype=torch.float32,
                )[indices],
            )
            loss = loss + cfg.bc_action_weight * action_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    common = {
        "policy": policy,
        "normalizer": normalizer,
        "method": method,
        "output_kind": "latent" if latent_model is not None else "native",
        "latent_model": latent_model,
        "latent_mean": latent_mean,
        "latent_scale": latent_scale,
        "chunk_length": int(native.shape[-2]),
        "runtime_seconds": time.perf_counter() - start_time,
        "train_loss": float(np.mean(losses[-20:])),
    }
    return {
        body_index: TrainedPolicy(body_index=body_index, **common)
        for body_index in bodies
    }


def attach_adapted_decoder(
    latent_model: TrainedLatentModel,
    body_index: int,
    demonstrations: list[Sample],
    *,
    cfg: ModelConfig,
    seed: int,
    latent_policy: TrainedPolicy,
) -> tuple[TrainedLatentModel, LatentAdapter]:
    source_model = latent_model.model_for(latent_model.body_indices[0])
    adapted_model = copy.deepcopy(source_model)
    new_decoder: NativeDecoder = copy.deepcopy(source_model.decoders[0])
    while len(adapted_model.decoders) <= body_index:
        adapted_model.decoders.append(copy.deepcopy(source_model.decoders[0]))
    adapted_model.decoders[body_index] = new_decoder
    policy = latent_policy.policy
    if policy is None:
        raise RuntimeError("Adaptation requires a latent policy")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in adapted_model.parameters():
        parameter.requires_grad_(False)
    for parameter in new_decoder.parameters():
        parameter.requires_grad_(True)
    contexts = latent_policy.normalizer.transform(
        np.stack([sample.context for sample in demonstrations])
    )
    with torch.no_grad():
        latent = policy(contexts)
    morphology = torch.as_tensor(
        np.stack([sample.morphology for sample in demonstrations]), dtype=torch.float32
    )
    adapter = LatentAdapter(
        latent_dim=latent.shape[-1],
        context_dim=contexts.shape[-1],
        morphology_dim=morphology.shape[-1],
    )
    native = torch.as_tensor(
        np.stack([sample.native_action for sample in demonstrations]), dtype=torch.float32
    )
    mask = torch.as_tensor(
        np.stack([sample.action_mask for sample in demonstrations]), dtype=torch.float32
    )
    optimizer = torch.optim.AdamW(
        [*new_decoder.parameters(), *adapter.parameters()],
        lr=cfg.adaptation_learning_rate,
    )
    torch.manual_seed(seed + 70_001 + body_index)
    rng = np.random.default_rng(seed + 80_003 + body_index)
    for _ in range(cfg.adaptation_steps):
        indices = torch.from_numpy(
            rng.integers(0, len(demonstrations), size=min(cfg.bc_batch_size, len(demonstrations)))
        )
        adapted_latent = adapter(
            latent[indices], contexts[indices], morphology[indices]
        )
        reconstructed = new_decoder(adapted_latent, morphology[indices])
        loss = _loss_reconstruction(reconstructed, native[indices], mask[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*new_decoder.parameters(), *adapter.parameters()], 1.0
        )
        optimizer.step()
    adapted_model.eval()
    return (
        TrainedLatentModel(
            method=f"{latent_model.method}+decoder_adapter",
            body_indices=latent_model.body_indices,
            context_normalizer=latent_model.context_normalizer,
            models={latent_model.body_indices[0]: adapted_model},
            history=[
                *latent_model.history,
                {
                    "epoch": float(cfg.adaptation_steps),
                    "adaptation_loss": float(loss.detach()),
                },
            ],
        ),
        adapter,
    )


def train_adaptation_policy(
    demonstrations: list[Sample],
    *,
    body_index: int,
    cfg: ModelConfig,
    seed: int,
) -> TrainedPolicy:
    return train_policy(
        demonstrations,
        bodies=(body_index,),
        method="independent_bc",
        cfg=cfg,
        seed=seed,
        latent_model=None,
    )[body_index]


def train_independent_action_ae(
    demonstrations: list[Sample],
    *,
    body_index: int,
    cfg: ModelConfig,
    seed: int,
) -> TrainedLatentModel:
    """Train a new-body action autoencoder from the same few demonstrations."""

    native = torch.as_tensor(
        np.stack([sample.native_action for sample in demonstrations]), dtype=torch.float32
    )
    mask = torch.as_tensor(
        np.stack([sample.action_mask for sample in demonstrations]), dtype=torch.float32
    )
    normalizer = _fit_context_normalizer(demonstrations)
    context = normalizer.transform(
        np.stack([sample.context for sample in demonstrations])
    )
    morphology_np = np.stack([sample.morphology for sample in demonstrations])
    morphology = torch.as_tensor(morphology_np, dtype=torch.float32)
    model = LatentActionModel(
        context_dim=demonstrations[0].context.shape[0],
        morphology_dim=demonstrations[0].morphology.shape[0],
        chunk_length=native.shape[1],
        n_decoders=1,
        cfg=cfg,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.adaptation_learning_rate
    )
    torch.manual_seed(seed + 105_001)
    rng = np.random.default_rng(seed + 106_003)
    start = time.perf_counter()
    losses: list[float] = []
    for _ in range(cfg.adaptation_steps):
        indices = torch.from_numpy(
            rng.integers(0, len(demonstrations), size=min(cfg.bc_batch_size, len(demonstrations)))
        )
        latent = model.encode(context[indices], native[indices], morphology[indices])
        reconstructed = model.decode(
            latent,
            morphology[indices],
            torch.zeros(len(indices), dtype=torch.long),
        )
        loss = _loss_reconstruction(reconstructed, native[indices], mask[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return TrainedLatentModel(
        method="independent_action_ae",
        body_indices=(body_index,),
        context_normalizer=normalizer,
        models={body_index: model},
        history=[
            {
                "epoch": float(cfg.adaptation_steps),
                "adaptation_loss": float(np.mean(losses[-20:])),
                "runtime_seconds": time.perf_counter() - start,
            }
        ],
    )


def train_teacher_latent_decoder(
    base_model: TrainedLatentModel,
    demonstrations: list[Sample],
    *,
    body_index: int,
    cfg: ModelConfig,
    seed: int,
) -> TrainedLatentModel:
    """Freeze CELAR's encoder and learn only a new-body decoder."""

    source_model = base_model.model_for(base_model.body_indices[0])
    adapted_model = copy.deepcopy(source_model)
    new_decoder: NativeDecoder = copy.deepcopy(source_model.decoders[0])
    while len(adapted_model.decoders) <= body_index:
        adapted_model.decoders.append(copy.deepcopy(source_model.decoders[0]))
    adapted_model.decoders[body_index] = new_decoder
    for parameter in adapted_model.parameters():
        parameter.requires_grad_(False)
    for parameter in new_decoder.parameters():
        parameter.requires_grad_(True)
    native = torch.as_tensor(
        np.stack([sample.native_action for sample in demonstrations]), dtype=torch.float32
    )
    mask = torch.as_tensor(
        np.stack([sample.action_mask for sample in demonstrations]), dtype=torch.float32
    )
    contexts = np.stack([sample.context for sample in demonstrations])
    morphology_np = np.stack([sample.morphology for sample in demonstrations])
    morphology = torch.as_tensor(morphology_np, dtype=torch.float32)
    with torch.no_grad():
        latent = torch.as_tensor(
            base_model.encode(
                contexts,
                native.numpy(),
                morphology_np,
                body_index,
            ),
            dtype=torch.float32,
        )
    optimizer = torch.optim.AdamW(
        new_decoder.parameters(), lr=cfg.adaptation_learning_rate
    )
    torch.manual_seed(seed + 107_001)
    rng = np.random.default_rng(seed + 108_007)
    losses: list[float] = []
    for _ in range(cfg.adaptation_steps):
        indices = torch.from_numpy(
            rng.integers(0, len(demonstrations), size=min(cfg.bc_batch_size, len(demonstrations)))
        )
        reconstructed = new_decoder(latent[indices], morphology[indices])
        loss = _loss_reconstruction(reconstructed, native[indices], mask[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new_decoder.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    adapted_model.eval()
    return TrainedLatentModel(
        method="celar_teacher_decoder",
        body_indices=base_model.body_indices,
        context_normalizer=base_model.context_normalizer,
        models={base_model.body_indices[0]: adapted_model},
        history=[
            *base_model.history,
            {
                "epoch": float(cfg.adaptation_steps),
                "adaptation_loss": float(np.mean(losses[-20:])),
            },
        ],
    )
