from __future__ import annotations

import numpy as np

from celar.config import DataConfig
from celar.data import (
    CONTEXT_DIM,
    MORPHOLOGIES,
    generate_dataset,
    make_pairs,
    safety_clip,
    semantic_from_native,
)


def test_grouped_split_has_no_pair_or_task_leakage() -> None:
    cfg = DataConfig(
        train_tasks=3,
        val_tasks=1,
        test_tasks=1,
        train_episodes_per_task=2,
        val_episodes_per_task=1,
        test_episodes_per_task=1,
        trajectory_length=12,
    )
    dataset = generate_dataset(cfg, seed=7)
    assert dataset.protocol["passed"]
    assert dataset.protocol["pair_overlap_count"] == 0
    assert dataset.protocol["task_overlap_count"] == 0
    assert dataset.protocol["dimensions"]["context"] == [CONTEXT_DIM]


def test_paired_chunks_share_semantics_but_not_native_actions() -> None:
    cfg = DataConfig(
        train_tasks=2,
        val_tasks=1,
        test_tasks=1,
        train_episodes_per_task=1,
        val_episodes_per_task=1,
        test_episodes_per_task=1,
        trajectory_length=12,
    )
    dataset = generate_dataset(cfg, seed=11)
    pairs = make_pairs(dataset.test, (0, 1, 2))
    np.testing.assert_allclose(pairs.semantic[:, 0], pairs.semantic[:, 1])
    np.testing.assert_allclose(pairs.semantic[:, 0], pairs.semantic[:, 2])
    np.testing.assert_allclose(pairs.progress[:, 0], pairs.progress[:, 1])
    np.testing.assert_allclose(pairs.progress[:, 0], pairs.progress[:, 2])
    assert pairs.native.shape[-1] == 8
    assert pairs.native.shape[1] == 3
    assert not np.allclose(pairs.native[:, 0, :, :6], pairs.native[:, 1, :, :6])


def test_expert_actions_are_safe_and_recover_semantics() -> None:
    cfg = DataConfig(
        train_tasks=2,
        val_tasks=1,
        test_tasks=1,
        train_episodes_per_task=1,
        val_episodes_per_task=1,
        test_episodes_per_task=2,
        trajectory_length=12,
    )
    dataset = generate_dataset(cfg, seed=13)
    for body_index, _ in enumerate(MORPHOLOGIES):
        samples = [sample for sample in dataset.test if sample.body_index == body_index]
        native = np.stack([sample.native_action for sample in samples])
        semantic = np.stack([sample.semantic_action for sample in samples])
        clipped, valid = safety_clip(native, body_index)
        recovered = semantic_from_native(clipped, body_index)
        assert valid.all()
        assert np.abs(recovered - semantic).mean() < 0.34
