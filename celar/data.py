from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import DataConfig


MAX_ACTION_DIM = 8
SEMANTIC_DIM = 4
VISUAL_DIM = 18
LANGUAGE_DIM = 11
PROPRIO_DIM = 8
MORPH_DIM = 5
CONTEXT_DIM = VISUAL_DIM + LANGUAGE_DIM + PROPRIO_DIM + MORPH_DIM

TASK_FAMILIES = ("pick", "place", "open")
COLORS = 4
ZONES = 4


@dataclass(frozen=True)
class Morphology:
    name: str
    action_dim: int
    joint_count: int
    joint_limit: float
    velocity_limit: float
    reach: float
    payload: float
    control_hz: float
    mapping: np.ndarray
    bias: np.ndarray
    camera_angle: float

    @property
    def descriptor(self) -> np.ndarray:
        return np.asarray(
            [
                self.joint_count / 10.0,
                self.reach,
                self.velocity_limit,
                self.payload,
                self.control_hz / 100.0,
            ],
            dtype=np.float32,
        )


def _morphologies() -> tuple[Morphology, ...]:
    matrices = [
        np.asarray(
            [
                [0.62, -0.10, 0.12, 0.00],
                [0.18, 0.55, -0.08, 0.02],
                [-0.20, 0.24, 0.58, -0.05],
                [0.12, -0.30, 0.16, 0.42],
                [0.45, 0.20, -0.12, 0.00],
                [-0.10, 0.36, 0.20, 0.01],
                [0.08, -0.12, 0.28, 0.58],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [0.48, 0.20, -0.10, 0.00],
                [-0.18, 0.56, 0.15, 0.04],
                [0.24, -0.22, 0.52, 0.00],
                [0.10, 0.30, -0.25, 0.35],
                [-0.35, 0.16, 0.22, 0.03],
                [0.08, -0.18, 0.34, 0.55],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [0.54, -0.18, 0.08, 0.01],
                [0.22, 0.49, 0.18, -0.03],
                [-0.12, 0.27, 0.61, 0.02],
                [0.16, -0.31, 0.21, 0.46],
                [0.39, 0.11, -0.20, 0.00],
                [-0.06, 0.38, 0.16, 0.54],
                [0.11, -0.08, 0.30, 0.10],
            ],
            dtype=np.float32,
        ),
    ]
    return (
        Morphology(
            "franka_panda",
            7,
            7,
            1.0,
            0.78,
            0.86,
            0.72,
            20.0,
            matrices[0],
            np.asarray([0.06, -0.03, 0.04, -0.02, 0.02, -0.01, 0.01], dtype=np.float32),
            25.0,
        ),
        Morphology(
            "ur5e",
            6,
            6,
            1.0,
            0.84,
            0.92,
            0.58,
            18.0,
            matrices[1],
            np.asarray([-0.02, 0.04, -0.01, 0.03, 0.01, -0.03], dtype=np.float32),
            -32.0,
        ),
        Morphology(
            "sawyer",
            7,
            7,
            1.0,
            0.72,
            0.80,
            0.63,
            16.0,
            matrices[2],
            np.asarray([0.03, 0.02, -0.04, 0.01, -0.02, 0.03, -0.01], dtype=np.float32),
            48.0,
        ),
    )


MORPHOLOGIES = _morphologies()


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    color: int
    goal_zone: int
    split: str


@dataclass
class Sample:
    body_index: int
    pair_id: str
    episode_id: str
    task_id: str
    trajectory_id: str
    window_index: int
    native_action: np.ndarray
    action_mask: np.ndarray
    context: np.ndarray
    morphology: np.ndarray
    semantic_action: np.ndarray
    progress: float
    success: float


@dataclass
class Dataset:
    catalog: dict[str, list[TaskSpec]]
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]
    protocol: dict[str, object]


def one_hot(index: int, size: int) -> np.ndarray:
    output = np.zeros(size, dtype=np.float32)
    output[int(index)] = 1.0
    return output


def _catalog(cfg: DataConfig, seed: int) -> dict[str, list[TaskSpec]]:
    combinations = [
        (family, color, zone)
        for family in TASK_FAMILIES
        for color in range(COLORS)
        for zone in range(ZONES)
    ]
    rng = np.random.default_rng(seed + 811)
    rng.shuffle(combinations)
    counts = {
        "train": cfg.train_tasks,
        "val": cfg.val_tasks,
        "test": cfg.test_tasks,
    }
    if sum(counts.values()) > len(combinations):
        raise ValueError("Task catalog is too small")
    output: dict[str, list[TaskSpec]] = {}
    cursor = 0
    for split, count in counts.items():
        output[split] = []
        for family, color, zone in combinations[cursor : cursor + count]:
            output[split].append(
                TaskSpec(
                    task_id=f"{family}-c{color}-z{zone}",
                    family=family,
                    color=color,
                    goal_zone=zone,
                    split=split,
                )
            )
        cursor += count
    return output


def _language(task: TaskSpec) -> np.ndarray:
    return np.concatenate(
        [
            one_hot(TASK_FAMILIES.index(task.family), len(TASK_FAMILIES)),
            one_hot(task.color, COLORS),
            one_hot(task.goal_zone, ZONES),
        ]
    ).astype(np.float32)


def _rotate(task_space: np.ndarray, angle_degrees: float) -> np.ndarray:
    output = np.asarray(task_space, dtype=np.float32).copy()
    angle = np.deg2rad(angle_degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    if output.shape[-1] == 3:
        output[..., :2] = output[..., :2] @ rotation.T
    else:
        for start in range(0, min(9, output.shape[-1]), 3):
            if start + 2 <= output.shape[-1]:
                output[..., start : start + 2] = (
                    output[..., start : start + 2] @ rotation.T
                )
    return output


def _semantic_trajectory(
    task: TaskSpec,
    rng: np.random.Generator,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.arange(length, dtype=np.float32)
    phase = steps / max(1, length - 1)
    controls = np.zeros((length, SEMANTIC_DIM), dtype=np.float32)
    if task.family == "pick":
        controls[:, 0] = 0.55 * np.sin(2.0 * np.pi * phase + 0.2)
        controls[:, 1] = 0.47 * np.cos(1.4 * np.pi * phase - 0.5)
        controls[:, 2] = np.where(phase < 0.50, -0.25, 0.58)
        controls[:, 3] = np.where(phase < 0.42, -0.10, 0.82)
    elif task.family == "place":
        controls[:, 0] = 0.48 * np.cos(1.6 * np.pi * phase + 0.4)
        controls[:, 1] = -0.52 * np.sin(1.3 * np.pi * phase - 0.1)
        controls[:, 2] = np.where(phase < 0.34, -0.60, 0.34)
        controls[:, 3] = np.where(phase < 0.56, 0.70, -0.72)
    else:
        controls[:, 0] = 0.35 * np.sin(1.2 * np.pi * phase)
        controls[:, 1] = np.where(phase < 0.38, -0.76, 0.64)
        controls[:, 2] = 0.26 * np.cos(1.8 * np.pi * phase + 0.2)
        controls[:, 3] = np.where(phase < 0.28, -0.42, 0.48)
    controls += rng.normal(0.0, 0.05, size=controls.shape).astype(np.float32)
    controls = np.clip(controls, -0.92, 0.92)
    progress = np.clip(0.06 + 0.88 * phase, 0.0, 1.0).astype(np.float32)
    # A short recovery branch ends incompletely and remains in the data.
    success = np.ones(length, dtype=np.float32)
    return controls, progress, success


def _native_action(
    semantic: np.ndarray,
    morphology: Morphology,
    body_index: int,
    rng: np.random.Generator,
    noise: float,
) -> np.ndarray:
    raw = semantic @ morphology.mapping.T + morphology.bias
    phase = np.linspace(0.0, 1.0, len(semantic), dtype=np.float32)[:, None]
    body_style = np.linspace(-1.0, 1.0, morphology.action_dim, dtype=np.float32)[None, :]
    raw += (0.035 + 0.02 * body_index) * np.sin(np.pi * phase) * body_style
    coupling = np.asarray(
        [
            semantic[:, 0] * semantic[:, 3],
            semantic[:, 1] * semantic[:, 2],
        ],
        dtype=np.float32,
    ).T
    raw[:, 1:3] += 0.05 * coupling
    raw += rng.normal(0.0, noise, size=raw.shape).astype(np.float32)
    return np.clip(raw, -morphology.joint_limit, morphology.joint_limit).astype(np.float32)


def _context(
    task: TaskSpec,
    morphology: Morphology,
    eef: np.ndarray,
    victim: np.ndarray,
    goal: np.ndarray,
    gripper: float,
    progress: float,
) -> np.ndarray:
    velocity = np.zeros(3, dtype=np.float32)
    if len(eef) > 1:
        velocity = np.asarray(eef[-1] - eef[-2], dtype=np.float32)
    visual = np.concatenate(
        [
            victim - eef[-1],
            victim,
            goal - eef[-1],
            np.asarray([gripper, progress], dtype=np.float32),
            one_hot(TASK_FAMILIES.index(task.family), len(TASK_FAMILIES)),
            one_hot(task.color, COLORS),
        ]
    ).astype(np.float32)
    if visual.shape[0] != VISUAL_DIM:
        raise AssertionError((visual.shape, VISUAL_DIM))
    visual = _rotate(visual[None, :], morphology.camera_angle)[0]
    proprio = np.concatenate(
        [eef[-1], np.asarray([gripper, progress], dtype=np.float32), velocity]
    ).astype(np.float32)
    return np.concatenate(
        [visual, _language(task), proprio, morphology.descriptor]
    ).astype(np.float32)


def _episode_samples(
    task: TaskSpec,
    body_index: int,
    episode_index: int,
    cfg: DataConfig,
    seed: int,
    scene_angle: float,
) -> list[Sample]:
    # Scene geometry, task phase, progress, success/failure labels and the
    # intended semantic controls are generated before any embodiment-specific
    # transform. Every body paired on `episode_id` therefore receives exactly
    # the same episode seed and shared episode content.
    shared_rng = np.random.default_rng(seed)
    body_rng = np.random.default_rng(seed + body_index * 37)
    morphology = MORPHOLOGIES[body_index]
    semantic, progress, success = _semantic_trajectory(
        task, shared_rng, cfg.trajectory_length
    )
    failed = bool(shared_rng.random() < cfg.failure_rate)
    if failed:
        cutoff = int(
            shared_rng.integers(
                cfg.trajectory_length // 2, cfg.trajectory_length
            )
        )
        semantic[cutoff:] *= 0.25
        progress[cutoff:] = progress[cutoff] * np.linspace(1.0, 0.65, len(progress[cutoff:]))
        success[:] = 0.0
    native = _native_action(
        semantic, morphology, body_index, body_rng, cfg.action_noise
    )
    start = shared_rng.normal(0.0, 0.22, size=3).astype(np.float32)
    victim = shared_rng.normal(0.0, 0.30, size=3).astype(np.float32)
    goal = victim + shared_rng.normal(1.15, 0.20, size=3).astype(np.float32)
    eef_rows = [start]
    for step in range(1, cfg.trajectory_length):
        eef_rows.append(
            np.clip(eef_rows[-1] + 0.13 * semantic[step - 1, :3], -1.2, 1.2)
        )
    eef_trace = np.stack(eef_rows).astype(np.float32)
    gripper_trace = np.sign(semantic[:, 3]).astype(np.float32)
    if scene_angle > 0.0:
        angle = scene_angle if body_index % 2 == 0 else -0.55 * scene_angle
        victim_view = _rotate(victim[None, :], angle)[0]
        goal_view = _rotate(goal[None, :], angle)[0]
        eef_view = _rotate(eef_trace, angle)
    else:
        victim_view, goal_view, eef_view = victim, goal, eef_trace
    episode_id = f"{task.task_id}-e{episode_index:04d}"
    output: list[Sample] = []
    starts = range(0, cfg.trajectory_length - cfg.chunk_length + 1, cfg.stride)
    for window_index, chunk_start in enumerate(starts):
        stop = chunk_start + cfg.chunk_length
        chunk_semantic = semantic[chunk_start:stop].copy()
        chunk_native = native[chunk_start:stop].copy()
        padded = np.zeros((cfg.chunk_length, MAX_ACTION_DIM), dtype=np.float32)
        mask = np.zeros((cfg.chunk_length, MAX_ACTION_DIM), dtype=np.float32)
        padded[:, : morphology.action_dim] = chunk_native
        mask[:, : morphology.action_dim] = 1.0
        context = _context(
            task,
            morphology,
            eef_view[: stop],
            victim_view,
            goal_view,
            float(gripper_trace[stop - 1]),
            float(progress[stop - 1]),
        )
        context += body_rng.normal(
            0.0, cfg.observation_noise, size=context.shape
        ).astype(np.float32)
        output.append(
            Sample(
                body_index=body_index,
                pair_id=f"{episode_id}-w{window_index:03d}",
                episode_id=episode_id,
                task_id=task.task_id,
                trajectory_id=f"{episode_id}-b{body_index}",
                window_index=window_index,
                native_action=padded,
                action_mask=mask,
                context=context,
                morphology=morphology.descriptor.copy(),
                semantic_action=chunk_semantic,
                progress=float(progress[stop - 1]),
                success=float(success[stop - 1]),
            )
        )
    return output


def _generate_split(
    tasks: Iterable[TaskSpec],
    *,
    episodes_per_task: int,
    cfg: DataConfig,
    seed: int,
    split: str,
) -> list[Sample]:
    output: list[Sample] = []
    rng = np.random.default_rng(seed)
    for task_index, task in enumerate(tasks):
        for episode_index in range(episodes_per_task):
            angle_draw = rng.uniform(0.0, cfg.scene_angle_train)
            scene_angle = (
                angle_draw
                if split == "train"
                else cfg.scene_angle_test * (0.82 + 0.18 * rng.random())
            )
            for body_index in range(len(MORPHOLOGIES)):
                output.extend(
                    _episode_samples(
                        task,
                        body_index,
                        episode_index,
                        cfg,
                        seed + task_index * 100_003 + episode_index * 997,
                        scene_angle,
                    )
                )
    return output


def _protocol(
    train: list[Sample],
    val: list[Sample],
    test: list[Sample],
    cfg: DataConfig,
) -> dict[str, object]:
    train_pairs = {sample.pair_id for sample in train}
    val_pairs = {sample.pair_id for sample in val}
    test_pairs = {sample.pair_id for sample in test}
    train_tasks = {sample.task_id for sample in train}
    val_tasks = {sample.task_id for sample in val}
    test_tasks = {sample.task_id for sample in test}
    pair_overlap = len(train_pairs & val_pairs) + len(train_pairs & test_pairs) + len(
        val_pairs & test_pairs
    )
    task_overlap = len(train_tasks & val_tasks) + len(train_tasks & test_tasks) + len(
        val_tasks & test_tasks
    )
    body_counts = {
        split: {
            morphology.name: sum(sample.body_index == index for sample in samples)
            for index, morphology in enumerate(MORPHOLOGIES)
        }
        for split, samples in (("train", train), ("val", val), ("test", test))
    }
    dimensions = {
        "context": sorted({sample.context.shape[0] for sample in [*train, *val, *test]}),
        "native_action": sorted(
            {sample.native_action.shape for sample in [*train, *val, *test]}
        ),
        "semantic_action": sorted(
            {sample.semantic_action.shape for sample in [*train, *val, *test]}
        ),
    }
    passed = (
        pair_overlap == 0
        and task_overlap == 0
        and dimensions["context"] == [CONTEXT_DIM]
        and dimensions["native_action"] == [(cfg.chunk_length, MAX_ACTION_DIM)]
        and dimensions["semantic_action"] == [(cfg.chunk_length, SEMANTIC_DIM)]
        and all(count > 0 for counts in body_counts.values() for count in counts.values())
    )
    return {
        "passed": bool(passed),
        "pair_overlap_count": int(pair_overlap),
        "task_overlap_count": int(task_overlap),
        "split_counts": {
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
            "train_tasks": len(train_tasks),
            "val_tasks": len(val_tasks),
            "test_tasks": len(test_tasks),
        },
        "body_counts": body_counts,
        "dimensions": dimensions,
        "chunk_length": cfg.chunk_length,
        "stride": cfg.stride,
    }


def generate_dataset(cfg: DataConfig, seed: int) -> Dataset:
    catalog = _catalog(cfg, seed)
    train = _generate_split(
        catalog["train"],
        episodes_per_task=cfg.train_episodes_per_task,
        cfg=cfg,
        seed=seed * 10_000 + 101,
        split="train",
    )
    val = _generate_split(
        catalog["val"],
        episodes_per_task=cfg.val_episodes_per_task,
        cfg=cfg,
        seed=seed * 10_000 + 211,
        split="val",
    )
    test = _generate_split(
        catalog["test"],
        episodes_per_task=cfg.test_episodes_per_task,
        cfg=cfg,
        seed=seed * 10_000 + 307,
        split="test",
    )
    return Dataset(
        catalog=catalog,
        train=train,
        val=val,
        test=test,
        protocol=_protocol(train, val, test, cfg),
    )


@dataclass
class PairBatch:
    native: np.ndarray
    action_mask: np.ndarray
    context: np.ndarray
    morphology: np.ndarray
    body_index: np.ndarray
    semantic: np.ndarray
    progress: np.ndarray
    success: np.ndarray
    pair_ids: list[str]
    episode_ids: list[str]
    task_ids: list[str]
    window_indices: np.ndarray

    def __len__(self) -> int:
        return int(self.native.shape[0])


def _common_pair_ids(samples: list[Sample], bodies: tuple[int, ...]) -> list[str]:
    groups: dict[str, set[int]] = {}
    for sample in samples:
        if sample.body_index in bodies:
            groups.setdefault(sample.pair_id, set()).add(sample.body_index)
    required = set(bodies)
    return sorted(pair_id for pair_id, present in groups.items() if present == required)


def make_pairs(
    samples: list[Sample],
    bodies: tuple[int, ...],
    *,
    limit: int | None = None,
    seed: int = 0,
) -> PairBatch:
    body_list = tuple(int(body) for body in bodies)
    if len(body_list) < 2:
        raise ValueError("At least two bodies are required for paired evaluation")
    pair_ids = _common_pair_ids(samples, body_list)
    if limit is not None and len(pair_ids) > limit:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(pair_ids), size=limit, replace=False))
        pair_ids = [pair_ids[index] for index in selected]
    lookup = {
        (sample.pair_id, sample.body_index): sample
        for sample in samples
        if sample.body_index in body_list
    }
    native: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    morphologies: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    semantics: list[np.ndarray] = []
    progress: list[np.ndarray] = []
    success: list[np.ndarray] = []
    episode_ids: list[str] = []
    task_ids: list[str] = []
    window_indices: list[np.ndarray] = []
    for pair_id in pair_ids:
        first = lookup[(pair_id, body_list[0])]
        native.append(np.stack([lookup[(pair_id, body)].native_action for body in body_list]))
        masks.append(np.stack([lookup[(pair_id, body)].action_mask for body in body_list]))
        contexts.append(np.stack([lookup[(pair_id, body)].context for body in body_list]))
        morphologies.append(
            np.stack([lookup[(pair_id, body)].morphology for body in body_list])
        )
        indices.append(np.asarray(body_list, dtype=np.int64))
        semantics.append(
            np.stack(
                [
                    lookup[(pair_id, body)].semantic_action
                    for body in body_list
                ]
            )
        )
        progress.append(
            np.asarray([lookup[(pair_id, body)].progress for body in body_list], dtype=np.float32)
        )
        success.append(
            np.asarray([lookup[(pair_id, body)].success for body in body_list], dtype=np.float32)
        )
        episode_ids.append(first.episode_id)
        task_ids.append(first.task_id)
        window_indices.append(
            np.asarray(
                [lookup[(pair_id, body)].window_index for body in body_list],
                dtype=np.int64,
            )
        )
    return PairBatch(
        native=np.stack(native).astype(np.float32),
        action_mask=np.stack(masks).astype(np.float32),
        context=np.stack(contexts).astype(np.float32),
        morphology=np.stack(morphologies).astype(np.float32),
        body_index=np.stack(indices).astype(np.int64),
        semantic=np.stack(semantics).astype(np.float32),
        progress=np.stack(progress).astype(np.float32),
        success=np.stack(success).astype(np.float32),
        pair_ids=pair_ids,
        episode_ids=episode_ids,
        task_ids=task_ids,
        window_indices=np.stack(window_indices).astype(np.int64),
    )


def samples_for_body(samples: list[Sample], body_index: int) -> list[Sample]:
    return [sample for sample in samples if sample.body_index == body_index]


def select_shot_demos(
    samples: list[Sample],
    body_index: int,
    shots: int,
    seed: int,
) -> list[Sample]:
    body_samples = samples_for_body(samples, body_index)
    trajectory_ids = sorted({sample.trajectory_id for sample in body_samples})
    rng = np.random.default_rng(seed + body_index * 101 + shots * 17)
    selected = set(
        rng.choice(trajectory_ids, size=min(shots, len(trajectory_ids)), replace=False).tolist()
    )
    return [sample for sample in body_samples if sample.trajectory_id in selected]


def semantic_from_native(native: np.ndarray, body_index: int) -> np.ndarray:
    morphology = MORPHOLOGIES[body_index]
    chunk = np.asarray(native, dtype=np.float32)[..., : morphology.action_dim]
    centered = chunk - morphology.bias
    inverse = np.linalg.pinv(morphology.mapping)
    return np.einsum("...a,sa->...s", centered, inverse)


def safety_clip(native: np.ndarray, body_index: int) -> tuple[np.ndarray, np.ndarray]:
    morphology = MORPHOLOGIES[body_index]
    array = np.asarray(native, dtype=np.float32).copy()
    valid = np.isfinite(array).all(axis=-1)
    clipped = np.clip(array, -morphology.joint_limit, morphology.joint_limit)
    if clipped.shape[-2] > 1:
        step = np.diff(clipped, axis=-2)
        velocity_scale = np.maximum(
            1.0, np.max(np.abs(step), axis=-1, keepdims=True) / morphology.velocity_limit
        )
        step = step / velocity_scale
        clipped[..., 1:, :] = clipped[..., :-1, :] + step
        velocity_valid = np.ones_like(valid, dtype=bool)
        velocity_valid[..., 1:] = (
            np.max(np.abs(np.diff(clipped, axis=-2)), axis=-1)
            <= morphology.velocity_limit + 1e-5
        )
        valid &= velocity_valid
    return clipped.astype(np.float32), valid.astype(bool)


def source_digest(root: str | Path) -> str:
    digest = hashlib.sha256()
    package_root = Path(root) / "celar"
    for path in sorted(package_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
