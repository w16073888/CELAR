from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


TASK_FAMILIES = ("pick", "place", "open_drawer", "push")
ROBOTS = ("panda", "ur5e", "sawyer")
SPLITS = ("train", "val", "test")
COLORS = np.asarray(
    [
        [0.85, 0.18, 0.15, 1.0],
        [0.15, 0.55, 0.85, 1.0],
        [0.20, 0.75, 0.35, 1.0],
        [0.90, 0.70, 0.15, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class SceneSpec:
    split: str
    pair_id: str
    scene_id: str
    task_id: str
    task_family: str
    task_instance: int
    color_index: int
    seed: int
    object_position: tuple[float, float, float]
    goal_position: tuple[float, float, float]
    initial_eef_position: tuple[float, float, float]
    drawer_initial: float
    held_initial: bool
    failure_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobotSpec:
    name: str
    arm_dof: int
    extra_hinges: int
    action_chunk_length: int
    max_joint_delta: tuple[float, ...]
    velocity_limit: tuple[float, ...]
    workspace_low: tuple[float, float, float]
    workspace_high: tuple[float, float, float]
    base_position: tuple[float, float, float] = (-1.15, 0.0, 0.60)
    base_reach: float = 0.02


ROBOT_SPECS: dict[str, RobotSpec] = {
    "panda": RobotSpec(
        name="panda",
        arm_dof=7,
        extra_hinges=4,
        action_chunk_length=4,
        max_joint_delta=(0.045, 0.045, 0.038, 0.035, 0.035, 0.030, 0.030),
        velocity_limit=(0.42, 0.42, 0.36, 1.40, 1.40, 1.40, 1.40),
        workspace_low=(-0.12, -0.46, 0.34),
        workspace_high=(0.48, 0.46, 0.82),
    ),
    "ur5e": RobotSpec(
        name="ur5e",
        arm_dof=6,
        extra_hinges=3,
        action_chunk_length=6,
        max_joint_delta=(0.050, 0.050, 0.042, 0.038, 0.038, 0.034),
        velocity_limit=(0.46, 0.46, 0.40, 1.55, 1.55, 1.55),
        workspace_low=(-0.10, -0.48, 0.34),
        workspace_high=(0.50, 0.48, 0.82),
    ),
    "sawyer": RobotSpec(
        name="sawyer",
        arm_dof=7,
        extra_hinges=4,
        action_chunk_length=8,
        max_joint_delta=(0.040, 0.040, 0.034, 0.030, 0.030, 0.028, 0.028),
        velocity_limit=(0.38, 0.38, 0.32, 1.25, 1.25, 1.25, 1.25),
        workspace_low=(-0.12, -0.44, 0.34),
        workspace_high=(0.46, 0.44, 0.80),
    ),
}


def split_counts(total: int) -> dict[str, int]:
    train = int(round(total * 0.70))
    val = int(round(total * 0.15))
    return {"train": train, "val": val, "test": total - train - val}


def split_for_index(index: int, total: int, seed: int) -> str:
    rng = np.random.default_rng(seed + 9109)
    order = rng.permutation(total)
    counts = split_counts(total)
    position = int(np.flatnonzero(order == index)[0])
    if position < counts["train"]:
        return "train"
    if position < counts["train"] + counts["val"]:
        return "val"
    return "test"


def make_scene_spec(
    *,
    family: str,
    pair_index: int,
    total_pairs: int,
    seed: int,
) -> SceneSpec:
    """Create identical scene geometry for every embodiment in a pair."""

    split = split_for_index(pair_index, total_pairs, seed)
    mode_roll = pair_index % 10
    failure_mode = "failure" if mode_roll == 8 else "perturb" if mode_roll == 9 else "nominal"
    family_index = TASK_FAMILIES.index(family)
    split_task_count = {"train": 4, "val": 2, "test": 2}[split]
    task_instance = pair_index % split_task_count
    color_index = (family_index * 3 + task_instance) % len(COLORS)
    rng = np.random.default_rng(
        seed + family_index * 1_000_003 + pair_index * 7919
    )
    object_xy = np.asarray(
        [
            0.06 + rng.uniform(-0.05, 0.07),
            rng.uniform(-0.28, 0.28),
        ],
        dtype=np.float64,
    )
    goal_xy = np.asarray(
        [
            0.31 + rng.uniform(-0.05, 0.06),
            rng.uniform(-0.28, 0.28),
        ],
        dtype=np.float64,
    )
    object_z = 0.475
    goal_z = 0.455
    if family == "open_drawer":
        object_xy = np.asarray([0.18, -0.28], dtype=np.float64)
        goal_xy = np.asarray([0.34, -0.28], dtype=np.float64)
        initial_eef = np.asarray([0.10, -0.28, 0.60], dtype=np.float64)
    elif family == "place":
        initial_eef = np.asarray([0.02, object_xy[1], 0.58], dtype=np.float64)
    else:
        initial_eef = np.asarray(
            [object_xy[0] - 0.10, object_xy[1], 0.69],
            dtype=np.float64,
        )
    return SceneSpec(
        split=split,
        pair_id=f"{family}-{split}-p{pair_index:04d}",
        scene_id=f"{family}-{split}-s{pair_index:04d}",
        task_id=f"{family}-{split}-t{task_instance}",
        task_family=family,
        task_instance=task_instance,
        color_index=color_index,
        seed=int(rng.integers(0, 2**31 - 1)),
        object_position=(float(object_xy[0]), float(object_xy[1]), float(object_z)),
        goal_position=(float(goal_xy[0]), float(goal_xy[1]), float(goal_z)),
        initial_eef_position=tuple(float(value) for value in initial_eef),
        drawer_initial=float(rng.uniform(-0.08, -0.02)),
        held_initial=family == "place",
        failure_mode=failure_mode,
    )
