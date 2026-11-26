from __future__ import annotations

import numpy as np
from scipy import stats

from .data import PairBatch
from .models import TrainedLatentModel


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
    means = np.asarray(
        [
            array[rng.integers(0, len(array), len(array))].mean()
            for _ in range(samples)
        ]
    )
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
    lower, upper = bootstrap_mean_ci(
        difference, samples=samples, seed=seed
    )
    p_value = (
        float(stats.ttest_1samp(difference, 0.0).pvalue)
        if len(difference) > 1 and np.std(difference) > 1e-12
        else float("nan")
    )
    return {
        "n_pairs": int(len(difference)),
        "mean_difference": float(difference.mean()),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "paired_t_p": p_value,
    }


def effective_rank(latent: np.ndarray) -> float:
    values = np.asarray(latent, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    covariance = values.T @ values / max(1, len(values) - 1)
    eigenvalues = np.clip(
        np.linalg.eigvalsh(covariance), 0.0, None
    )
    total = eigenvalues.sum()
    if total <= 1e-12:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(
        probabilities * np.log(probabilities + 1e-12)
    ).sum()
    return float(np.exp(entropy))


def _semantic_mae(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    step_mask = mask[..., :3].mean(axis=-1) > 0.5
    denominator = np.maximum(
        step_mask.sum(axis=1) * predicted.shape[-1], 1
    )
    return (
        np.abs(predicted - target) * step_mask[..., None]
    ).sum(axis=(1, 2)) / denominator


def _predicted_semantic_action(predicted_native: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [predicted_native[..., :3], predicted_native[..., 7:8]],
        axis=-1,
    )


def semantic_retrieval_at_k(
    query: np.ndarray,
    candidates: np.ndarray,
    query_semantics: np.ndarray,
    candidate_semantics: np.ndarray,
    *,
    k: int,
    tolerance: float,
) -> float:
    query = np.asarray(query, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    distances = (
        (query[:, None, :] - candidates[None, :, :]) ** 2
    ).sum(axis=-1)
    nearest = np.argsort(distances, axis=1)[:, :k]
    semantic_distance = np.abs(
        np.asarray(query_semantics)[:, None, :, :]
        - np.asarray(candidate_semantics)[None, :, :, :]
    ).mean(axis=(2, 3))
    hits = np.take_along_axis(
        semantic_distance, nearest, axis=1
    ) <= tolerance
    return float(hits.any(axis=1).mean())


def evaluate_latent_model(
    model: TrainedLatentModel,
    pairs: PairBatch,
    *,
    tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    bodies = tuple(int(value) for value in pairs.body_index[0])
    latents: list[np.ndarray] = []
    decoded: list[np.ndarray] = []
    executable: list[np.ndarray] = []
    semantic_errors: dict[int, np.ndarray] = {}
    native_errors: dict[int, np.ndarray] = {}
    progress_errors: dict[int, np.ndarray] = {}
    for slot, body in enumerate(bodies):
        native = pairs.native[:, slot]
        mask = pairs.action_mask[:, slot]
        morphology = pairs.morphology[:, slot]
        latent = model.encode(
            pairs.context[:, slot],
            native,
            morphology,
            body,
        )
        prediction = model.decode(latent, morphology, body)
        semantic_error = _semantic_mae(
            _predicted_semantic_action(prediction) * mask[..., :4],
            pairs.semantic[:, slot],
            mask,
        )
        native_error = (
            np.abs(prediction - native) * mask
        ).sum(axis=(1, 2)) / np.maximum(
            mask.sum(axis=(1, 2)), 1.0
        )
        progress_prediction = model.predict_progress(
            latent, morphology, body
        )
        latents.append(latent)
        decoded.append(prediction)
        executable.append(
            (semantic_error <= tolerance)
            & np.isfinite(prediction).all(axis=(1, 2))
        )
        semantic_errors[body] = semantic_error
        native_errors[body] = native_error
        progress_errors[body] = np.abs(
            progress_prediction - pairs.progress[:, slot]
        )

    cycle_terms: list[np.ndarray] = []
    for source in range(len(bodies)):
        for target in range(len(bodies)):
            if source == target:
                continue
            cross = model.decode(
                latents[source],
                pairs.morphology[:, target],
                bodies[target],
            )
            cycled = model.encode(
                pairs.context[:, target],
                cross,
                pairs.morphology[:, target],
                bodies[target],
            )
            cycle_terms.append(
                ((cycled - latents[source]) ** 2).mean(axis=1)
            )
    cycle_error = np.mean(np.stack(cycle_terms), axis=0)
    progress_prediction = np.stack(
        [
            model.predict_progress(
                latents[slot],
                pairs.morphology[:, slot],
                body,
            )
            for slot, body in enumerate(bodies)
        ],
        axis=1,
    )
    progress_pair_gap = (
        progress_prediction
        - progress_prediction.mean(axis=1, keepdims=True)
    ) ** 2

    exact_r1: list[float] = []
    exact_r5: list[float] = []
    semantic_r1: list[float] = []
    semantic_r5: list[float] = []
    exact_hit_rows: list[np.ndarray] = []
    semantic_hit_rows: list[np.ndarray] = []
    semantic5_hit_rows: list[np.ndarray] = []
    for source in range(len(bodies)):
        source_exact: list[np.ndarray] = []
        source_semantic: list[np.ndarray] = []
        source_semantic5: list[np.ndarray] = []
        candidate_bodies = [
            value for value in range(len(bodies)) if value != source
        ]
        union = np.concatenate(
            [latents[value] for value in candidate_bodies]
        )
        union_semantics = np.concatenate(
            [pairs.semantic[:, value] for value in candidate_bodies]
        )
        for target in candidate_bodies:
            distances = (
                (
                    latents[source][:, None, :]
                    - latents[target][None, :, :]
                )
                ** 2
            ).sum(axis=-1)
            nearest = np.argsort(distances, axis=1)
            target_indices = np.arange(len(distances))
            source_exact.append(nearest[:, 0] == target_indices)
            semantic_distance = np.abs(
                pairs.semantic[:, source][:, None, :, :]
                - pairs.semantic[:, target][None, :, :, :]
            ).mean(axis=(2, 3))
            source_semantic.append(
                np.take_along_axis(
                    semantic_distance, nearest[:, :1], axis=1
                )[:, 0]
                <= 0.18
            )
            source_semantic5.append(
                np.take_along_axis(
                    semantic_distance, nearest[:, :5], axis=1
                ).any(axis=1)
            )
            exact_r1.append(
                float(np.mean(nearest[:, 0] == target_indices))
            )
            exact_r5.append(
                float(
                    np.mean(
                        (
                            nearest[:, :5]
                            == target_indices[:, None]
                        ).any(axis=1)
                    )
                )
            )
        semantic_r1.append(
            semantic_retrieval_at_k(
                latents[source],
                union,
                pairs.semantic[:, source],
                union_semantics,
                k=1,
                tolerance=0.18,
            )
        )
        semantic_r5.append(
            semantic_retrieval_at_k(
                latents[source],
                union,
                pairs.semantic[:, source],
                union_semantics,
                k=5,
                tolerance=0.18,
            )
        )
        exact_hit_rows.append(np.mean(np.stack(source_exact), axis=0))
        semantic_hit_rows.append(
            np.mean(np.stack(source_semantic), axis=0)
        )
        semantic5_hit_rows.append(
            np.mean(np.stack(source_semantic5), axis=0)
        )

    all_latent = np.concatenate(latents)
    rows = [
        {
            "method": model.method,
            "pair_id": pair_id,
            "episode_id": pairs.episode_ids[index],
            "task_id": pairs.task_ids[index],
            "window_index": int(pairs.window_indices[index, 0]),
            "retrieval_exact_pair_hit": float(
                np.mean(np.stack(exact_hit_rows), axis=0)[index]
            ),
            "retrieval_semantic_r1_hit": float(
                np.mean(np.stack(semantic_hit_rows), axis=0)[index]
            ),
            "retrieval_semantic_r5_hit": float(
                np.mean(np.stack(semantic5_hit_rows), axis=0)[index]
            ),
            **{
                f"semantic_error_body_{body}": float(
                    semantic_errors[body][index]
                )
                for body in bodies
            },
            **{
                f"executable_body_{body}": float(
                    executable[slot][index]
                )
                for slot, body in enumerate(bodies)
            },
            "cycle_error": float(cycle_error[index]),
            "progress_pair_gap": float(
                progress_pair_gap[index].mean()
            ),
        }
        for index, pair_id in enumerate(pairs.pair_ids)
    ]
    summary: dict[str, object] = {
        "method": model.method,
        "pairs": len(pairs),
        "retrieval_semantic_r1": float(np.mean(semantic_r1)),
        "retrieval_semantic_r5": float(np.mean(semantic_r5)),
        "retrieval_exact_pair_r1": float(np.mean(exact_r1)),
        "retrieval_exact_pair_r5": float(np.mean(exact_r5)),
        "reconstruction_mae": float(
            np.mean([value.mean() for value in native_errors.values()])
        ),
        "semantic_mae": float(
            np.mean([value.mean() for value in semantic_errors.values()])
        ),
        "executable_rate": float(np.concatenate(executable).mean()),
        "cycle_error": float(cycle_error.mean()),
        "progress_pair_gap": float(progress_pair_gap.mean()),
        "progress_mae": float(
            np.mean([value.mean() for value in progress_errors.values()])
        ),
        "latent_usage_min_std": float(all_latent.std(axis=0).min()),
        "latent_effective_rank": effective_rank(all_latent),
        "body_balance_gap": float(
            max(value.mean() for value in semantic_errors.values())
            - min(value.mean() for value in semantic_errors.values())
        ),
    }
    for body, value in semantic_errors.items():
        summary[f"semantic_mae_body_{body}"] = float(value.mean())
    return summary, rows
