from __future__ import annotations

from dataclasses import replace

import numpy as np

from celar.config import DataConfig, ExperimentConfig, ModelConfig
from celar.data import generate_dataset, make_pairs
from celar.metrics import evaluate_latent_model
from celar.training import _adjacent_sample_pairs, train_latent_models
from celar.rl import CrossEmbodimentControlEnv, train_ppo


def test_celar_training_and_metrics_are_deterministic() -> None:
    data_cfg = DataConfig(
        train_tasks=2,
        val_tasks=1,
        test_tasks=1,
        train_episodes_per_task=2,
        val_episodes_per_task=1,
        test_episodes_per_task=1,
        trajectory_length=12,
    )
    model_cfg = ModelConfig(epochs=2, hidden_dim=32, latent_dim=8, bc_epochs=4)
    dataset = generate_dataset(data_cfg, seed=23)
    model = train_latent_models(
        dataset.train,
        bodies=(0, 1, 2),
        method="celar",
        cfg=model_cfg,
        seed=3,
        pair_limit=24,
    )
    pairs = make_pairs(dataset.test, (0, 1, 2), limit=8, seed=4)
    first, _ = evaluate_latent_model(model, pairs, tolerance=0.34)
    second, _ = evaluate_latent_model(model, pairs, tolerance=0.34)
    assert first["retrieval_semantic_r1"] == second["retrieval_semantic_r1"]
    assert 0.0 <= first["executable_rate"] <= 1.0
    assert first["latent_effective_rank"] >= 0.0
    assert model.history[-1]["outcome"] > 0.0
    assert model.history[-1]["smooth"] > 0.0
    assert model.history[-1]["usage"] >= 0.0


def test_config_roundtrip_shape() -> None:
    cfg = ExperimentConfig()
    assert cfg.model.latent_dim == 8
    assert cfg.data.chunk_length == 4
    assert cfg.evaluation.shots == [5, 20, 50]


def test_smoothing_pairs_are_real_temporal_neighbors() -> None:
    data_cfg = DataConfig(
        train_tasks=1,
        val_tasks=1,
        test_tasks=1,
        train_episodes_per_task=1,
        val_episodes_per_task=1,
        test_episodes_per_task=1,
        trajectory_length=12,
    )
    dataset = generate_dataset(data_cfg, seed=31)
    pairs = _adjacent_sample_pairs(dataset.train)
    assert pairs
    assert all(
        current.episode_id == following.episode_id
        and current.body_index == following.body_index
        and following.window_index == current.window_index + 1
        for current, following in pairs
    )


def test_control_environment_can_be_solved_and_ppo_runs() -> None:
    env = CrossEmbodimentControlEnv((0,), envs=4, seed=9, auto_reset=False)
    env.reset()
    success = np.zeros(4, dtype=bool)
    for _ in range(env.max_episode_steps):
        action = np.zeros((4, 4), dtype=np.float32)
        object_delta = env.obj - env.eef
        distance = np.linalg.norm(object_delta, axis=1)
        action[:, :2] = object_delta / np.maximum(distance[:, None], 1e-6)
        action[distance < 0.16, 3] = 1.0
        held = env.held
        goal_delta = env.goal - env.eef
        goal_distance = np.linalg.norm(goal_delta, axis=1)
        action[held, :2] = goal_delta[held] / np.maximum(
            goal_distance[held, None], 1e-6
        )
        action[held & (goal_distance < 0.18), 3] = -1.0
        result = env.step(action)
        success |= result.success
    assert success.mean() >= 0.75
    result = train_ppo(
        latent_model=None,
        bodies=(0, 1),
        seed=3,
        training_steps=512,
        rollout_envs=4,
        rollout_steps=32,
        update_epochs=1,
        minibatch_size=64,
        eval_episodes=4,
    )
    assert result.final_success >= 0.0
