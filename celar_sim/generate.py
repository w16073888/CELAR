from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .environment import UnifiedManipulationEnv
from .schema import ROBOT_SPECS, ROBOTS, TASK_FAMILIES, SceneSpec, make_scene_spec


@dataclass
class GenerationConfig:
    pairs_per_task: int = 100
    render_size: int = 128
    output_dir: str = "results/c0_mujoco_v1"
    seed: int = 20261106

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GenerationConfig":
        return cls(**{key: raw.get(key, getattr(cls(), key)) for key in cls.__dataclass_fields__})


def _normalized_goal_direction(scene: SceneSpec) -> np.ndarray:
    direction = np.asarray(scene.goal_position) - np.asarray(scene.object_position)
    norm = float(np.linalg.norm(direction[:2]))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0])
    return np.asarray([direction[0] / norm, direction[1] / norm, 0.0])


def _waypoint_and_grip(
    env: UnifiedManipulationEnv,
    scene: SceneSpec,
    substep: int,
) -> tuple[np.ndarray, float]:
    object_position = np.asarray(scene.object_position, dtype=np.float64)
    goal_position = np.asarray(scene.goal_position, dtype=np.float64)
    eef = env.eef_position()
    if scene.task_family == "pick":
        if substep < 10:
            target = object_position + np.asarray([0.0, 0.0, 0.11])
            grip = -0.8
        elif substep < 18:
            target = object_position + np.asarray([0.0, 0.0, 0.035])
            grip = -0.8
        elif substep < 24:
            target = object_position + np.asarray([0.0, 0.0, 0.035])
            grip = 0.9
        else:
            target = object_position + np.asarray([0.0, 0.0, 0.15])
            grip = 0.9
    elif scene.task_family == "place":
        if substep < 13:
            target = goal_position + np.asarray([0.0, 0.0, 0.14])
            grip = 0.9
        elif substep < 21:
            target = goal_position + np.asarray([0.0, 0.0, 0.045])
            grip = 0.9
        else:
            target = goal_position + np.asarray([0.0, 0.0, 0.07])
            grip = -0.9
    elif scene.task_family == "open_drawer":
        handle = np.asarray(env.data.site_xpos[env.handle_site_id], dtype=np.float64)
        if substep < 14:
            target = handle + np.asarray([-0.10, 0.0, 0.03])
            grip = -0.7
        elif substep < 20:
            target = handle
            grip = 0.9
        else:
            target = handle + np.asarray([0.20, 0.0, 0.0])
            grip = 0.9
    else:
        direction = _normalized_goal_direction(scene)
        contact = object_position - 0.085 * direction
        if substep < 12:
            target = contact + np.asarray([0.0, 0.0, 0.05])
            grip = -0.8
        else:
            target = goal_position
            grip = -0.8
    if scene.failure_mode == "failure":
        if scene.task_family == "pick":
            grip = -0.9
        elif scene.task_family == "place":
            grip = -0.9 if substep > 13 else 0.9
        elif scene.task_family == "open_drawer":
            grip = -0.9
        else:
            direction = _normalized_goal_direction(scene)
            target = object_position + np.asarray([-direction[1], direction[0], 0.0]) * 0.18
    elif scene.failure_mode == "perturb" and 8 <= substep < 14:
        target = target + np.asarray([0.0, 0.07, 0.035])
    del eef
    return target, grip


def _expert_chunk(
    env: UnifiedManipulationEnv,
    scene: SceneSpec,
    *,
    chunk_index: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    spec = ROBOT_SPECS[env.robot_name]
    chunk_length = spec.action_chunk_length
    actions = np.zeros((chunk_length, spec.arm_dof), dtype=np.float32)
    grippers = np.zeros(chunk_length, dtype=np.float32)
    max_delta = np.asarray(spec.max_joint_delta, dtype=np.float64)
    virtual_eef = env.eef_position()
    for local_step in range(chunk_length):
        substep = chunk_index * chunk_length + local_step
        target, grip = _waypoint_and_grip(env, scene, substep)
        delta_world = target - virtual_eef
        delta_world *= 0.65
        executed_delta = np.clip(
            delta_world[:3],
            -max_delta[:3],
            max_delta[:3],
        )
        actions[local_step, :3] = executed_delta
        virtual_eef = virtual_eef + executed_delta
        if scene.failure_mode == "perturb":
            actions[local_step, 1] += rng.normal(0.0, 0.004)
        else:
            actions[local_step] += rng.normal(
                0.0, 0.0015, size=spec.arm_dof
            ).astype(np.float32)
        grippers[local_step] = grip
    return actions, grippers


def _sha256_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_trajectory(
    env: UnifiedManipulationEnv,
    scene: SceneSpec,
    *,
    output_path: Path,
) -> dict[str, Any]:
    env.reset(scene)
    initial_qpos = np.asarray(env.data.qpos).copy()
    rng = np.random.default_rng(scene.seed + ROBOTS.index(env.robot_name) * 7)
    spec = ROBOT_SPECS[env.robot_name]
    started = time.perf_counter()
    records: dict[str, list[np.ndarray]] = {
        "rgb": [],
        "depth": [],
        "native_action": [],
        "gripper_action": [],
        "eef_pose": [],
        "arm_qpos": [],
        "arm_qvel": [],
        "object_pose": [],
        "drawer_position": [],
        "progress": [],
        "success": [],
        "safety_pre_limit_violation": [],
        "safety_pre_velocity_violation": [],
        "safety_post_limit_violation": [],
        "safety_post_velocity_violation": [],
        "contact_count": [],
        "extrinsics": [],
    }
    for chunk_index in range(8):
        action, gripper = _expert_chunk(
            env, scene, chunk_index=chunk_index, rng=rng
        )
        result = env.step_chunk(action, gripper)
        records["rgb"].append(result.rgb)
        records["depth"].append(result.depth.astype(np.float16))
        records["native_action"].append(result.native_action)
        records["gripper_action"].append(result.gripper_action)
        records["eef_pose"].append(result.eef_pose)
        records["arm_qpos"].append(result.observation["arm_qpos"])
        records["arm_qvel"].append(result.observation["arm_qvel"])
        records["object_pose"].append(result.object_pose)
        records["drawer_position"].append(np.asarray([result.drawer_position]))
        records["progress"].append(np.asarray([result.progress]))
        records["success"].append(np.asarray([float(result.success)]))
        records["safety_pre_limit_violation"].append(
            result.safety_pre_limit_violation
        )
        records["safety_pre_velocity_violation"].append(
            result.safety_pre_velocity_violation
        )
        records["safety_post_limit_violation"].append(
            result.safety_post_limit_violation
        )
        records["safety_post_velocity_violation"].append(
            result.safety_post_velocity_violation
        )
        records["contact_count"].append(np.asarray([result.contact_count]))
        records["extrinsics"].append(result.extrinsics)
    intrinsics, _ = env.camera_parameters()
    arrays = {
        key: np.stack(values)
        for key, values in records.items()
    }
    arrays["intrinsics"] = intrinsics.astype(np.float64)
    arrays["initial_qpos"] = initial_qpos
    np.savez_compressed(output_path, **arrays)
    duration = time.perf_counter() - started
    return {
        "trajectory_file": output_path.name,
        "pair_id": scene.pair_id,
        "task_id": scene.task_id,
        "task_family": scene.task_family,
        "task_instance": scene.task_instance,
        "scene_id": scene.scene_id,
        "split": scene.split,
        "robot": env.robot_name,
        "arm_dof": spec.arm_dof,
        "chunk_length": spec.action_chunk_length,
        "chunks": 8,
        "success": float(arrays["success"][-1, 0] > 0.5),
        "failure_mode": scene.failure_mode,
        "max_progress": float(arrays["progress"].max()),
        "post_limit_violation_count": int(
            arrays["safety_post_limit_violation"].sum()
        ),
        "post_velocity_violation_count": int(
            arrays["safety_post_velocity_violation"].sum()
        ),
        "mean_chunk_latency_ms": 1000.0 * duration / 8.0,
        "initial_state_sha256": _sha256_json(scene.to_dict()),
        "rgb_shape": list(arrays["rgb"].shape),
        "depth_shape": list(arrays["depth"].shape),
    }


def generate_dataset(
    config: GenerationConfig,
    *,
    output_dir: Path,
) -> pd.DataFrame:
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    environments = {
        robot: UnifiedManipulationEnv(
            robot,
            render_size=config.render_size,
            enable_rendering=True,
        )
        for robot in ROBOTS
    }
    try:
        for family_index, family in enumerate(TASK_FAMILIES):
            for pair_index in range(config.pairs_per_task):
                scene = make_scene_spec(
                    family=family,
                    pair_index=pair_index,
                    total_pairs=config.pairs_per_task,
                    seed=config.seed + family_index * 77,
                )
                for robot in ROBOTS:
                    env = environments[robot]
                    filename = (
                        f"{scene.split}_{family}_{robot}_{pair_index:04d}.npz"
                    )
                    row = generate_trajectory(
                        env,
                        scene,
                        output_path=trajectory_dir / filename,
                    )
                    row["scene_seed"] = scene.seed
                    row["object_position"] = json.dumps(scene.object_position)
                    row["goal_position"] = json.dumps(scene.goal_position)
                    row["initial_eef_position"] = json.dumps(
                        scene.initial_eef_position
                    )
                    rows.append(row)
                if (pair_index + 1) % 10 == 0:
                    print(f"[{family}] generated {pair_index + 1}/{config.pairs_per_task} pairs")
    finally:
        for env in environments.values():
            env.close()
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "trajectory_manifest.csv", index=False)
    return manifest
