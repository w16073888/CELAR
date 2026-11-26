from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from .data import (
    MORPHOLOGIES,
    PairBatch,
    Sample,
    safety_clip,
    semantic_from_native,
)
from .models import TrainedLatentModel, TrainedPolicy
from .training import attach_adapted_decoder


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    *,
    samples: int = 3000,
    seed: int = 0,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = array[rng.integers(0, len(array), len(array))].mean()
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_difference(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    samples: int = 3000,
    seed: int = 0,
) -> dict[str, float | int]:
    treatment = np.asarray(treatment, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if treatment.shape != control.shape:
        raise ValueError("Paired arrays must have identical shape")
    difference = treatment - control
    lower, upper = bootstrap_mean_ci(difference, samples=samples, seed=seed)
    if len(difference) > 1 and np.std(difference) > 1e-12:
        p_value = float(stats.ttest_1samp(difference, 0.0).pvalue)
    elif len(difference):
        p_value = 0.0 if abs(float(difference.mean())) > 1e-12 else 1.0
    else:
        p_value = float("nan")
    return {
        "n_pairs": int(len(difference)),
        "mean_difference": float(difference.mean()),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "paired_t_p": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    output = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        output[index] = running
    return output


def retrieval_at_k(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    k: int,
) -> float:
    query = np.asarray(query, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    distances = ((query[:, None, :] - candidates[None, :, :]) ** 2).sum(axis=-1)
    nearest = np.argsort(distances, axis=1)[:, :k]
    target = np.arange(len(query))[:, None]
    return float((nearest == target).any(axis=1).mean())


def semantic_retrieval_at_k(
    query: np.ndarray,
    candidates: np.ndarray,
    query_semantics: np.ndarray,
    candidate_semantics: np.ndarray,
    *,
    k: int,
    tolerance: float,
) -> float:
    """R@k where positives are actions within a semantic tolerance."""

    query = np.asarray(query, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    query_semantics = np.asarray(query_semantics, dtype=np.float64)
    candidate_semantics = np.asarray(candidate_semantics, dtype=np.float64)
    distances = ((query[:, None, :] - candidates[None, :, :]) ** 2).sum(axis=-1)
    nearest = np.argsort(distances, axis=1)[:, :k]
    semantic_distance = np.abs(
        query_semantics[:, None, :, :] - candidate_semantics[None, :, :, :]
    ).mean(axis=(2, 3))
    hits = np.take_along_axis(
        semantic_distance,
        nearest,
        axis=1,
    ) <= tolerance
    return float(hits.any(axis=1).mean())


def effective_rank(latent: np.ndarray) -> float:
    values = np.asarray(latent, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    covariance = values.T @ values / max(1, len(values) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 1e-12:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum()
    return float(np.exp(entropy))


def _semantic_mae(
    predicted: np.ndarray,
    target: np.ndarray,
    action_mask: np.ndarray | None = None,
) -> np.ndarray:
    error = np.abs(predicted - target)
    if action_mask is None:
        return error.mean(axis=tuple(range(1, predicted.ndim)))
    step_mask = action_mask[..., :3].mean(axis=-1) > 0.5
    denominator = np.maximum(step_mask.sum(axis=1) * predicted.shape[-1], 1)
    return (error * step_mask[..., None]).sum(axis=(1, 2)) / denominator


def _jerk(native: np.ndarray, mask: np.ndarray) -> float:
    if native.shape[1] < 3:
        return 0.0
    acceleration = np.diff(native, n=2, axis=1)
    valid = mask[:, 2:] > 0.5
    if not valid.any():
        return 0.0
    return float(np.abs(acceleration[valid]).mean())


def evaluate_latent_model(
    model: TrainedLatentModel,
    pairs: PairBatch,
    *,
    tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    bodies = tuple(int(value) for value in pairs.body_index[0].tolist())
    if len(bodies) < 2:
        raise ValueError("Paired evaluation requires at least two embodiments")
    latent_rows: list[np.ndarray] = []
    decoded_rows: list[np.ndarray] = []
    executable_rows: list[np.ndarray] = []
    semantic_errors_by_body: dict[int, np.ndarray] = {}
    native_mae_by_body: dict[int, np.ndarray] = {}
    progress_errors_by_body: dict[int, np.ndarray] = {}
    for body_slot, body_index in enumerate(bodies):
        context = pairs.context[:, body_slot]
        native = pairs.native[:, body_slot]
        mask = pairs.action_mask[:, body_slot]
        morphology = pairs.morphology[:, body_slot]
        semantic_target = pairs.semantic[:, body_slot]
        latent = model.encode(context, native, morphology, body_index)
        decoded = model.decode(latent, morphology, body_index)
        clipped, valid = safety_clip(decoded, body_index)
        semantic = semantic_from_native(clipped, body_index)
        semantic_error = _semantic_mae(semantic, semantic_target, mask)
        native_error = (np.abs(decoded - native) * mask).sum(
            axis=(1, 2)
        ) / np.maximum(mask.sum(axis=(1, 2)), 1.0)
        predicted_progress = model.predict_progress(latent, morphology, body_index)
        progress_error = np.abs(predicted_progress - pairs.progress[:, body_slot])
        latent_rows.append(latent)
        decoded_rows.append(decoded)
        executable_rows.append(valid.all(axis=1) & (semantic_error <= tolerance))
        semantic_errors_by_body[body_index] = semantic_error
        native_mae_by_body[body_index] = native_error
        progress_errors_by_body[body_index] = progress_error

    cycle_terms: list[np.ndarray] = []
    for source, target in (
        (source, target)
        for source in range(len(bodies))
        for target in range(len(bodies))
        if source != target
    ):
        decoded = model.decode(
            latent_rows[source],
            pairs.morphology[:, target],
            bodies[target],
        )
        cycled = model.encode(
            pairs.context[:, target],
            decoded,
            pairs.morphology[:, target],
            bodies[target],
        )
        cycle_terms.append(
            np.mean((cycled - latent_rows[source]) ** 2, axis=1)
        )
    cycle_error = np.mean(np.stack(cycle_terms), axis=0)
    predictions = np.stack(
        [
            model.predict_progress(
                latent_rows[index],
                pairs.morphology[:, index],
                body,
            )
            for index, body in enumerate(bodies)
        ],
        axis=1,
    )
    progress_pair_gap = (
        predictions - predictions.mean(axis=1, keepdims=True)
    ) ** 2
    exact_hits: list[np.ndarray] = []
    semantic_hits_r1: list[np.ndarray] = []
    semantic_hits_r5: list[np.ndarray] = []
    exact_r1_values: list[float] = []
    exact_r5_values: list[float] = []
    semantic_r1_values: list[float] = []
    semantic_r5_values: list[float] = []
    for source in range(len(bodies)):
        source_exact_hits = []
        source_semantic_hits_r1 = []
        source_semantic_hits_r5 = []
        for target in range(len(bodies)):
            if source == target:
                continue
            query = latent_rows[source]
            candidate = latent_rows[target]
            distances = ((query[:, None, :] - candidate[None, :, :]) ** 2).sum(
                axis=-1
            )
            nearest = np.argsort(distances, axis=1)
            source_exact_hits.append(nearest[:, 0] == np.arange(len(query)))
            semantic_distance = np.abs(
                pairs.semantic[:, source][None, :, :, :]
                - pairs.semantic[:, target][:, None, :, :]
            ).mean(axis=(2, 3))
            source_semantic_hits_r1.append(
                np.take_along_axis(
                    semantic_distance,
                    nearest[:, :1],
                    axis=1,
                )[:, 0]
                <= 0.18
            )
            source_semantic_hits_r5.append(
                np.take_along_axis(
                    semantic_distance,
                    nearest[:, :5],
                    axis=1,
                ).any(axis=1)
            )
            exact_r1_values.append(
                float(np.mean(nearest[:, 0] == np.arange(len(query))))
            )
            exact_r5_values.append(
                float(np.mean((nearest[:, :5] == np.arange(len(query))[:, None]).any(axis=1)))
            )
            source_union = np.concatenate(
                [
                    latent_rows[other]
                    for other in range(len(bodies))
                    if other != source
                ]
            )
            source_semantic_targets = np.concatenate(
                [
                    pairs.semantic[:, other]
                    for other in range(len(bodies))
                    if other != source
                ]
            )
            semantic_r1_values.append(
                semantic_retrieval_at_k(
                    query,
                    source_union,
                    pairs.semantic[:, source],
                    source_semantic_targets,
                    k=1,
                    tolerance=0.18,
                )
            )
            semantic_r5_values.append(
                semantic_retrieval_at_k(
                    query,
                    source_union,
                    pairs.semantic[:, source],
                    source_semantic_targets,
                    k=5,
                    tolerance=0.18,
                )
            )
        exact_hits.append(np.mean(np.stack(source_exact_hits), axis=0))
        semantic_hits_r1.append(
            np.mean(np.stack(source_semantic_hits_r1), axis=0)
        )
        semantic_hits_r5.append(
            np.mean(np.stack(source_semantic_hits_r5), axis=0)
        )
    exact_any = np.mean(np.stack(exact_hits), axis=0)
    semantic_r1_any = np.mean(np.stack(semantic_hits_r1), axis=0)
    semantic_r5_any = np.mean(np.stack(semantic_hits_r5), axis=0)
    all_latent = np.concatenate(latent_rows)
    hidden = all_latent.std(axis=0)
    row_records: list[dict[str, object]] = []
    for index, pair_id in enumerate(pairs.pair_ids):
        row_records.append(
            {
                "method": model.method,
                "pair_id": pair_id,
                "episode_id": pairs.episode_ids[index],
                "task_id": pairs.task_ids[index],
                "window_index": int(pairs.window_indices[index, 0]),
                "retrieval_exact_pair_hit": float(exact_any[index]),
                "retrieval_semantic_r1_hit": float(semantic_r1_any[index]),
                "retrieval_semantic_r5_hit": float(semantic_r5_any[index]),
                **{
                    f"semantic_error_body_{body}": float(
                        semantic_errors_by_body[body][index]
                    )
                    for body in bodies
                },
                **{
                    f"executable_body_{body}": float(
                        executable_rows[body_slot][index]
                    )
                    for body_slot, body in enumerate(bodies)
                },
                "cycle_error": float(cycle_error[index]),
                "progress_pair_gap": float(progress_pair_gap[index].mean()),
            }
        )
    summary = {
        "method": model.method,
        "pairs": len(pairs),
        "retrieval_semantic_r1": float(np.mean(semantic_r1_values)),
        "retrieval_semantic_r5": float(np.mean(semantic_r5_values)),
        "retrieval_exact_pair_r1": float(np.mean(exact_r1_values)),
        "retrieval_exact_pair_r5": float(np.mean(exact_r5_values)),
        "reconstruction_mae": float(
            np.mean(
                [
                    value.mean()
                    for body in bodies
                    for value in [native_mae_by_body[body]]
                ]
            )
        ),
        "semantic_mae": float(
            np.mean([semantic_errors_by_body[body].mean() for body in bodies])
        ),
        "executable_rate": float(np.concatenate(executable_rows).mean()),
        "cycle_error": float(cycle_error.mean()),
        "progress_pair_gap": float(progress_pair_gap.mean()),
        "progress_mae": float(
            np.mean([progress_errors_by_body[body].mean() for body in bodies])
        ),
        "latent_usage_min_std": float(hidden.min()),
        "latent_effective_rank": effective_rank(all_latent),
        "jerk": float(
            np.mean(
                [
                    _jerk(decoded_rows[index], pairs.action_mask[:, index])
                    for index in range(len(bodies))
                ]
            )
        ),
        "body_balance_gap": float(
            max(
                semantic_errors_by_body[body].mean()
                for body in bodies
            )
            - min(
                semantic_errors_by_body[body].mean()
                for body in bodies
            )
        ),
    }
    for body, values in semantic_errors_by_body.items():
        summary[f"semantic_mae_body_{body}"] = float(values.mean())
    return summary, row_records


def evaluate_policy(
    policy: TrainedPolicy,
    samples: list[Sample],
    *,
    body_index: int,
    tolerance: float,
) -> dict[str, float]:
    selected = [sample for sample in samples if sample.body_index == body_index]
    context = np.stack([sample.context for sample in selected])
    native = np.stack([sample.native_action for sample in selected])
    mask = np.stack([sample.action_mask for sample in selected])
    semantic_target = np.stack([sample.semantic_action for sample in selected])
    predicted = policy.predict(context)
    clipped, valid = safety_clip(predicted, body_index)
    semantic = semantic_from_native(clipped, body_index)
    semantic_error = _semantic_mae(semantic, semantic_target, mask)
    native_error = (np.abs(predicted - native) * mask).sum(
        axis=(1, 2)
    ) / np.maximum(mask.sum(axis=(1, 2)), 1.0)
    return {
        "samples": float(len(selected)),
        "semantic_mae": float(semantic_error.mean()),
        "native_mae": float(native_error.mean()),
        "executable_rate": float(
            (valid.all(axis=1) & (semantic_error <= tolerance)).mean()
        ),
        "jerk": _jerk(predicted, mask),
        "train_loss": policy.train_loss,
        "runtime_seconds": policy.runtime_seconds,
    }


def evaluate_frozen_adapter(
    base_latent_model: TrainedLatentModel,
    latent_policy: TrainedPolicy,
    demonstrations: list[Sample],
    test_samples: list[Sample],
    *,
    body_index: int,
    tolerance: float,
    cfg,
    seed: int,
) -> tuple[dict[str, float], TrainedLatentModel]:
    adapted, adapter = attach_adapted_decoder(
        base_latent_model,
        body_index,
        demonstrations,
        cfg=cfg,
        seed=seed,
        latent_policy=latent_policy,
    )
    selected = [sample for sample in test_samples if sample.body_index == body_index]
    context = np.stack([sample.context for sample in selected])
    native = np.stack([sample.native_action for sample in selected])
    mask = np.stack([sample.action_mask for sample in selected])
    semantic_target = np.stack([sample.semantic_action for sample in selected])
    latent = latent_policy.predict_raw(context)
    morphology = np.stack([sample.morphology for sample in selected])
    normalized_context = latent_policy.normalizer.transform(context)
    predicted_latent = adapter(
        torch.as_tensor(latent, dtype=torch.float32),
        normalized_context,
        torch.as_tensor(morphology, dtype=torch.float32),
    ).detach().numpy()
    predicted = adapted.decode(predicted_latent, morphology, body_index)
    clipped, valid = safety_clip(predicted, body_index)
    semantic = semantic_from_native(clipped, body_index)
    semantic_error = _semantic_mae(semantic, semantic_target, mask)
    native_error = (np.abs(predicted - native) * mask).sum(
        axis=(1, 2)
    ) / np.maximum(mask.sum(axis=(1, 2)), 1.0)
    return (
        {
            "samples": float(len(selected)),
            "semantic_mae": float(semantic_error.mean()),
            "native_mae": float(native_error.mean()),
            "executable_rate": float(
                (valid.all(axis=1) & (semantic_error <= tolerance)).mean()
            ),
            "jerk": _jerk(predicted, mask),
        },
        adapted,
    )


def evaluate_action_interface(
    encoder_model: TrainedLatentModel,
    decoder_model: TrainedLatentModel,
    test_samples: list[Sample],
    *,
    body_index: int,
    tolerance: float,
) -> dict[str, float]:
    selected = [sample for sample in test_samples if sample.body_index == body_index]
    context = np.stack([sample.context for sample in selected])
    native = np.stack([sample.native_action for sample in selected])
    mask = np.stack([sample.action_mask for sample in selected])
    morphology = np.stack([sample.morphology for sample in selected])
    semantic_target = np.stack([sample.semantic_action for sample in selected])
    latent = encoder_model.encode(context, native, morphology, body_index)
    predicted = decoder_model.decode(latent, morphology, body_index)
    clipped, valid = safety_clip(predicted, body_index)
    semantic = semantic_from_native(clipped, body_index)
    semantic_error = _semantic_mae(semantic, semantic_target, mask)
    native_error = (np.abs(predicted - native) * mask).sum(
        axis=(1, 2)
    ) / np.maximum(mask.sum(axis=(1, 2)), 1.0)
    return {
        "samples": float(len(selected)),
        "semantic_mae": float(semantic_error.mean()),
        "native_mae": float(native_error.mean()),
        "executable_rate": float(
            (valid.all(axis=1) & (semantic_error <= tolerance)).mean()
        ),
        "jerk": _jerk(predicted, mask),
    }
