from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import (
    MAX_ACTION_DIM,
    MORPHOLOGIES,
    safety_clip,
    semantic_from_native,
)
from .models import TrainedLatentModel


TASK_FAMILIES = ("pick", "place", "open")
OBS_DIM = 19


def _one_hot(index: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros((len(index), size), dtype=np.float32)
    output[np.arange(len(index)), index] = 1.0
    return output


@dataclass
class StepResult:
    observation: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    success: np.ndarray


class CrossEmbodimentControlEnv:
    """Small vectorized pick/place controller for equal-budget PPO tests."""

    def __init__(
        self,
        bodies: tuple[int, ...],
        *,
        envs: int = 8,
        seed: int = 0,
        max_episode_steps: int = 36,
        auto_reset: bool = True,
        fixed_body: int | None = None,
        fixed_task: int | None = None,
    ) -> None:
        self.bodies = tuple(int(value) for value in bodies)
        self.envs = int(envs)
        self.rng = np.random.default_rng(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.auto_reset = bool(auto_reset)
        self.fixed_body = fixed_body
        self.fixed_task = fixed_task
        self.body = np.zeros(self.envs, dtype=np.int64)
        self.task = np.zeros(self.envs, dtype=np.int64)
        self.eef = np.zeros((self.envs, 2), dtype=np.float32)
        self.obj = np.zeros((self.envs, 2), dtype=np.float32)
        self.goal = np.zeros((self.envs, 2), dtype=np.float32)
        self.held = np.zeros(self.envs, dtype=bool)
        self.gripper_open = np.ones(self.envs, dtype=np.float32)
        self.steps = np.zeros(self.envs, dtype=np.int64)
        self.success = np.zeros(self.envs, dtype=bool)
        self.previous_potential = np.zeros(self.envs, dtype=np.float32)
        self.reset()

    def reset(self, indices: np.ndarray | None = None) -> np.ndarray:
        if indices is None:
            indices = np.arange(self.envs)
        indices = np.asarray(indices, dtype=np.int64)
        count = len(indices)
        self.body[indices] = (
            int(self.fixed_body)
            if self.fixed_body is not None
            else self.rng.choice(self.bodies, size=count)
        )
        self.task[indices] = (
            int(self.fixed_task)
            if self.fixed_task is not None
            else self.rng.integers(0, len(TASK_FAMILIES), size=count)
        )
        self.eef[indices] = self.rng.uniform(-0.9, 0.9, size=(count, 2))
        self.obj[indices] = self.rng.uniform(-0.65, 0.65, size=(count, 2))
        angle = self.rng.uniform(-np.pi, np.pi, size=count)
        radius = self.rng.uniform(0.65, 1.15, size=count)
        offset = np.stack([np.cos(angle), np.sin(angle)], axis=1) * radius[:, None]
        self.goal[indices] = np.clip(self.obj[indices] + offset, -1.35, 1.35)
        self.held[indices] = False
        self.gripper_open[indices] = 1.0
        self.steps[indices] = 0
        self.success[indices] = False
        self.previous_potential[indices] = self._potential(indices)
        return self.observation()

    def _potential(self, indices: np.ndarray | None = None) -> np.ndarray:
        if indices is None:
            indices = np.arange(self.envs)
        object_distance = np.linalg.norm(self.obj[indices] - self.eef[indices], axis=1)
        goal_distance = np.linalg.norm(self.goal[indices] - self.eef[indices], axis=1)
        approach = 0.45 * np.exp(-object_distance / 0.48)
        transport = 0.55 * np.exp(-goal_distance / 0.52)
        return np.where(self.held[indices], 0.45 + transport, approach).astype(np.float32)

    def observation(self) -> np.ndarray:
        stage = np.where(
            self.success,
            1.0,
            np.where(self.held, 0.66, np.clip(self._potential() / 0.45, 0.0, 1.0)),
        )
        return np.concatenate(
            [
                self.eef,
                self.obj,
                self.goal,
                self.obj - self.eef,
                self.goal - self.eef,
                self.held[:, None].astype(np.float32),
                self.gripper_open[:, None],
                stage[:, None],
                _one_hot(self.task, len(TASK_FAMILIES)),
                _one_hot(self.body, len(MORPHOLOGIES)),
            ],
            axis=1,
        ).astype(np.float32)

    def step(self, semantic_action: np.ndarray) -> StepResult:
        semantic_action = np.asarray(semantic_action, dtype=np.float32)
        if semantic_action.shape != (self.envs, 4):
            raise ValueError(
                f"Expected semantic action {(self.envs, 4)}, got {semantic_action.shape}"
            )
        semantic_action = np.clip(semantic_action, -1.0, 1.0)
        self.eef = np.clip(
            self.eef + 0.15 * semantic_action[:, :2],
            -1.2,
            1.2,
        ).astype(np.float32)
        grasp_command = semantic_action[:, 3]
        just_grasped = np.zeros(self.envs, dtype=bool)
        just_released = np.zeros(self.envs, dtype=bool)
        object_distance = np.linalg.norm(self.obj - self.eef, axis=1)
        can_grasp = (~self.held) & (grasp_command > 0.05) & (object_distance < 0.18)
        self.held |= can_grasp
        just_grasped |= can_grasp
        self.gripper_open = np.where(grasp_command > 0.05, 0.0, 1.0)
        self.obj = np.where(self.held[:, None], self.eef, self.obj)
        goal_distance = np.linalg.norm(self.goal - self.eef, axis=1)
        release_success = self.held & (grasp_command < -0.05) & (goal_distance < 0.20)
        self.success |= release_success
        just_released |= release_success
        self.held &= ~release_success
        potential = self._potential()
        reward = (
            3.0 * (potential - self.previous_potential)
            + 0.25 * just_grasped.astype(np.float32)
            + 5.0 * release_success.astype(np.float32)
            - 0.01
        ).astype(np.float32)
        self.previous_potential = potential
        self.steps += 1
        terminated = self.success.copy()
        truncated = self.steps >= self.max_episode_steps
        done = terminated | truncated
        if self.auto_reset:
            for index in np.flatnonzero(done):
                self.reset(np.asarray([index], dtype=np.int64))
        return StepResult(
            observation=self.observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=just_released,
        )


class GaussianActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.value = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.65))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(observation)
        return self.mean(hidden), self.value(hidden).squeeze(-1)

    def distribution(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, value = self(observation)
        std = self.log_std.clamp(-3.0, 0.5).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), value


@dataclass
class PPOResult:
    policy: GaussianActorCritic
    history: list[dict[str, float]]
    runtime_seconds: float
    final_success: float
    steps_to_50: float
    steps_to_80: float


def _decode_actions(
    raw_actions: np.ndarray,
    body: np.ndarray,
    *,
    latent_model: TrainedLatentModel | None,
) -> np.ndarray:
    bounded = np.tanh(raw_actions).astype(np.float32)
    native = np.zeros((len(body), 1, MAX_ACTION_DIM), dtype=np.float32)
    if latent_model is None:
        for body_index in np.unique(body):
            mask = body == body_index
            native[mask, 0, : MORPHOLOGIES[int(body_index)].action_dim] = bounded[mask, : MORPHOLOGIES[int(body_index)].action_dim]
    else:
        morphology = np.stack(
            [MORPHOLOGIES[int(body_index)].descriptor for body_index in body]
        )
        native = (
            latent_model.decode(bounded, morphology, int(body[0]))[:, :1]
            if np.all(body == body[0])
            else native
        )
        if not np.all(body == body[0]):
            for body_index in np.unique(body):
                mask = body == body_index
                native[mask] = latent_model.decode(
                    bounded[mask], morphology[mask], int(body_index)
                )[:, :1]
    clipped = native.copy()
    for body_index in np.unique(body):
        mask = body == body_index
        clipped[mask], _ = safety_clip(native[mask], int(body_index))
    semantic = np.zeros((len(body), 4), dtype=np.float32)
    for body_index in np.unique(body):
        mask = body == body_index
        semantic[mask] = semantic_from_native(clipped[mask], int(body_index))[:, 0]
    return semantic.astype(np.float32)


def _evaluate(
    policy: GaussianActorCritic,
    *,
    latent_model: TrainedLatentModel | None,
    bodies: tuple[int, ...],
    seed: int,
    episodes: int = 32,
) -> float:
    env = CrossEmbodimentControlEnv(
        bodies,
        envs=episodes,
        seed=seed,
        auto_reset=False,
    )
    observation = env.reset()
    success = np.zeros(episodes, dtype=bool)
    with torch.no_grad():
        for _ in range(env.max_episode_steps):
            mean, _ = policy(torch.as_tensor(observation, dtype=torch.float32))
            semantic = _decode_actions(
                mean.cpu().numpy(),
                env.body,
                latent_model=latent_model,
            )
            result = env.step(semantic)
            success |= result.success
            observation = result.observation
            if success.all():
                break
    return float(success.mean())


def train_ppo(
    *,
    latent_model: TrainedLatentModel | None,
    bodies: tuple[int, ...],
    seed: int,
    training_steps: int = 24_576,
    rollout_envs: int = 8,
    rollout_steps: int = 64,
    update_epochs: int = 4,
    minibatch_size: int = 128,
    learning_rate: float = 3e-4,
    gamma: float = 0.98,
    gae_lambda: float = 0.94,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.003,
    value_coef: float = 0.5,
    max_episode_steps: int = 36,
    eval_episodes: int = 24,
    eval_interval: int = 4096,
) -> PPOResult:
    torch.manual_seed(seed + 10_001)
    np.random.seed(seed + 20_003)
    rng = np.random.default_rng(seed + 30_007)
    env = CrossEmbodimentControlEnv(
        bodies,
        envs=rollout_envs,
        seed=seed + 40_009,
        max_episode_steps=max_episode_steps,
        auto_reset=True,
    )
    observation = env.reset()
    policy = GaussianActorCritic(OBS_DIM, MAX_ACTION_DIM)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_success = _evaluate(
        policy,
        latent_model=latent_model,
        bodies=bodies,
        seed=seed + 50_021,
        episodes=eval_episodes,
    )
    history.append(
        {
            "environment_steps": 0.0,
            "eval_success": best_success,
            "train_reward": 0.0,
        }
    )
    total_steps = 0
    update = 0
    start_time = time.perf_counter()
    while total_steps < training_steps:
        observations: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []
        log_probabilities: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        values: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        for _ in range(rollout_steps):
            distribution, value = policy.distribution(
                torch.as_tensor(observation, dtype=torch.float32)
            )
            raw_action = distribution.sample()
            semantic = _decode_actions(
                raw_action.cpu().numpy(),
                env.body,
                latent_model=latent_model,
            )
            result = env.step(semantic)
            observations.append(observation)
            raw_actions.append(raw_action.cpu().numpy())
            log_probabilities.append(distribution.log_prob(raw_action).sum(-1).detach().numpy())
            rewards.append(result.reward)
            values.append(value.detach().numpy())
            dones.append((result.terminated | result.truncated).astype(np.float32))
            observation = result.observation
        with torch.no_grad():
            _, next_value = policy(torch.as_tensor(observation, dtype=torch.float32))
        reward_array = np.stack(rewards)
        value_array = np.stack(values)
        done_array = np.stack(dones)
        advantages = np.zeros_like(reward_array)
        running = np.zeros(rollout_envs, dtype=np.float32)
        for step in reversed(range(rollout_steps)):
            next_values = (
                next_value.numpy()
                if step == rollout_steps - 1
                else value_array[step + 1]
            )
            not_done = 1.0 - done_array[step]
            delta = (
                reward_array[step]
                + gamma * next_values * not_done
                - value_array[step]
            )
            running = delta + gamma * gae_lambda * not_done * running
            advantages[step] = running
        returns = advantages + value_array
        flat_observations = np.concatenate(observations)
        flat_actions = np.concatenate(raw_actions)
        flat_old_log_probabilities = np.concatenate(log_probabilities)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / max(
            1e-8, flat_advantages.std()
        )
        observation_tensor = torch.as_tensor(flat_observations, dtype=torch.float32)
        action_tensor = torch.as_tensor(flat_actions, dtype=torch.float32)
        old_log_probability = torch.as_tensor(
            flat_old_log_probabilities, dtype=torch.float32
        )
        advantage_tensor = torch.as_tensor(flat_advantages, dtype=torch.float32)
        return_tensor = torch.as_tensor(flat_returns, dtype=torch.float32)
        for _ in range(update_epochs):
            order = rng.permutation(len(observation_tensor))
            for batch_start in range(0, len(order), minibatch_size):
                indices = torch.from_numpy(
                    order[batch_start : batch_start + minibatch_size].copy()
                )
                distribution, values_pred = policy.distribution(
                    observation_tensor[indices]
                )
                log_probability = distribution.log_prob(action_tensor[indices]).sum(-1)
                ratio = torch.exp(log_probability - old_log_probability[indices])
                unclipped = ratio * advantage_tensor[indices]
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - clip_epsilon,
                        1.0 + clip_epsilon,
                    )
                    * advantage_tensor[indices]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(values_pred, return_tensor[indices])
                entropy = distribution.entropy().sum(-1).mean()
                loss = (
                    policy_loss
                    + value_coef * value_loss
                    - entropy_coef * entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
        total_steps += len(flat_observations)
        update += 1
        if update % max(1, eval_interval // (rollout_envs * rollout_steps)) == 0:
            eval_success = _evaluate(
                policy,
                latent_model=latent_model,
                bodies=bodies,
                seed=seed + 60_013,
                episodes=eval_episodes,
            )
            best_success = max(best_success, eval_success)
            history.append(
                {
                    "environment_steps": float(total_steps),
                    "eval_success": eval_success,
                    "train_reward": float(reward_array.mean()),
                }
            )
    final_success = _evaluate(
        policy,
        latent_model=latent_model,
        bodies=bodies,
        seed=seed + 70_017,
        episodes=eval_episodes * 2,
    )
    history.append(
        {
            "environment_steps": float(total_steps),
            "eval_success": final_success,
            "train_reward": float(reward_array.mean()),
        }
    )

    def steps_to(threshold: float) -> float:
        for record in history:
            if record["eval_success"] >= threshold:
                return float(record["environment_steps"])
        return float(training_steps)

    return PPOResult(
        policy=policy,
        history=history,
        runtime_seconds=time.perf_counter() - start_time,
        final_success=final_success,
        steps_to_50=steps_to(0.50),
        steps_to_80=steps_to(0.80),
    )
