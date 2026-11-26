from __future__ import annotations

from pathlib import Path

import numpy as np

from celar.current.config import ExperimentConfig, ModelConfig
from celar.current.data import (
    GRIPPER_ACTION_INDEX,
    TASK_FAMILIES,
    Dataset,
    Sample,
    _canonical_semantic_window,
    make_pairs,
)
from celar.current.metrics import evaluate_latent_model
from celar.current.training import (
    method_uses_alignment,
    method_uses_smoothing,
    train_latent_models,
)


def _current_sample(pair: int, body: int, window: int) -> Sample:
    rng = np.random.default_rng(pair * 1000 + body * 100 + window)
    native = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros_like(native)
    native[:4, : 7 if body != 1 else 6] = rng.normal(
        0.0, 0.05, size=(4, 7 if body != 1 else 6)
    )
    mask[:4, : 7 if body != 1 else 6] = 1.0
    native[:4, GRIPPER_ACTION_INDEX] = 1.0
    mask[:4, GRIPPER_ACTION_INDEX] = 1.0
    semantic = np.zeros((8, 4), dtype=np.float32)
    semantic[:4, :3] = 0.05
    semantic[:4, 3] = 1.0
    context = rng.normal(size=32).astype(np.float32)
    context[:3] += body
    morphology = np.eye(5, dtype=np.float32)[body]
    return Sample(
        body_index=body,
        pair_id=f"pair-{pair}",
        episode_id=f"scene-{pair}",
        task_id=f"{TASK_FAMILIES[pair % 4]}-task",
        trajectory_id=f"scene-{pair}-body-{body}",
        window_index=window,
        native_action=native,
        action_mask=mask,
        context=context,
        morphology=morphology,
        semantic_action=semantic,
        progress=0.1 * (window + 1),
        success=0.0,
    )


def _current_dataset() -> Dataset:
    train = [
        _current_sample(pair, body, window)
        for pair in range(4)
        for body in range(3)
        for window in range(4)
    ]
    test = [
        _current_sample(pair + 10, body, window)
        for pair in range(2)
        for body in range(3)
        for window in range(4)
    ]
    return Dataset(
        catalog={},
        train=train,
        val=[],
        test=test,
        protocol={"passed": True},
    )


def test_current_package_has_only_four_task_schema() -> None:
    assert TASK_FAMILIES == ("pick", "place", "open_drawer", "push")
    source_root = Path(__file__).resolve().parents[1] / "celar" / "current"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_root.glob("*.py")
    )
    assert "pre_experiment" not in source
    assert 'TASK_FAMILIES = ("pick", "place", "open")' not in source
    assert ExperimentConfig().output_dir.endswith("c1_mujoco_v7")


def test_configured_chunk_length_controls_semantic_window() -> None:
    semantic = _canonical_semantic_window(
        task_family="pick",
        chunk_index=1,
        object_position=np.asarray([0.1, 0.0, 0.45]),
        goal_position=np.asarray([0.3, 0.0, 0.45]),
        initial_eef=np.asarray([0.0, 0.0, 0.6]),
        chunk_length=6,
    )
    assert semantic.shape == (6, 4)


def test_no_align_keeps_smoothing() -> None:
    assert not method_uses_alignment("celar_no_align")
    assert method_uses_smoothing("celar_no_align")
    dataset = _current_dataset()
    model = train_latent_models(
        dataset.train,
        bodies=(0, 1, 2),
        method="celar_no_align",
        cfg=ModelConfig(
            hidden_dim=32,
            latent_dim=8,
            epochs=1,
            batch_size=16,
        ),
        seed=7,
    )
    assert model.history[-1]["alignment"] == 0.0
    assert model.history[-1]["smooth"] > 0.0


def test_current_pairs_and_celar_training() -> None:
    dataset = _current_dataset()
    pairs = make_pairs(dataset.test, (0, 1, 2))
    assert pairs.native.shape == (2, 3, 8, 8)
    np.testing.assert_allclose(pairs.semantic[:, 0], pairs.semantic[:, 1])
    np.testing.assert_allclose(pairs.semantic[:, 0], pairs.semantic[:, 2])
    model = train_latent_models(
        dataset.train,
        bodies=(0, 1, 2),
        method="celar",
        cfg=ModelConfig(
            hidden_dim=32,
            latent_dim=8,
            epochs=2,
            batch_size=16,
        ),
        seed=3,
    )
    summary, raw = evaluate_latent_model(model, pairs, tolerance=0.34)
    assert 0.0 <= summary["retrieval_exact_pair_r1"] <= 1.0
    assert "retrieval_exact_pair_hit" in raw[0]
    assert "retrieval_semantic_r1_hit" in raw[0]
    assert "retrieval_r1" not in raw[0]
    assert model.history[-1]["outcome"] > 0.0
    assert model.history[-1]["smooth"] > 0.0
