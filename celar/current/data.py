from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROBOT_TO_BODY = {"panda": 0, "ur5e": 1, "sawyer": 2}
TASK_FAMILIES = ("pick", "place", "open_drawer", "push")
SIM_CONTEXT_DIM = 1061
MAX_ACTION_DIM = 8
GRIPPER_ACTION_INDEX = 7


@dataclass(frozen=True)
class Morphology:
    name: str
    action_dim: int
    joint_count: int
    velocity_limit: float
    reach: float
    payload: float
    control_hz: float

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


MORPHOLOGIES = (
    Morphology("franka_panda", 7, 7, 0.78, 0.86, 0.72, 20.0),
    Morphology("ur5e", 6, 6, 0.84, 0.92, 0.58, 18.0),
    Morphology("sawyer", 7, 7, 0.72, 0.80, 0.63, 16.0),
)


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
    catalog: dict[str, list[object]]
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]
    protocol: dict[str, object]


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


def _downsample(value: np.ndarray, output_size: int = 16) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    height, width = array.shape[:2]
    block_h = height // output_size
    block_w = width // output_size
    if array.ndim == 3:
        return array.reshape(
            output_size,
            block_h,
            output_size,
            block_w,
            array.shape[2],
        ).mean(axis=(1, 3))
    return array.reshape(
        output_size,
        block_h,
        output_size,
        block_w,
    ).mean(axis=(1, 3))


def _one_hot(index: int, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.float32)
    result[int(index)] = 1.0
    return result


def _pad7(value: np.ndarray) -> np.ndarray:
    output = np.zeros(7, dtype=np.float32)
    source = np.asarray(value, dtype=np.float32).reshape(-1)
    output[: min(7, len(source))] = source[:7]
    return output


def _sim_context(
    *,
    frame_index: int,
    data: np.lib.npyio.NpzFile,
    task_family: str,
) -> np.ndarray:
    rgb = _downsample(data["rgb"][frame_index]) / 255.0
    depth = _downsample(data["depth"][frame_index].astype(np.float32))
    depth = np.clip(depth / 2.0, 0.0, 1.0)
    eef = _pad7(data["eef_pose"][frame_index])
    obj = _pad7(data["object_pose"][frame_index])
    qpos = _pad7(data["arm_qpos"][frame_index])
    qvel = _pad7(data["arm_qvel"][frame_index])
    gripper = np.asarray(data["gripper_action"][frame_index], dtype=np.float32).mean(
        keepdims=True
    )
    finger = np.asarray(
        [float(gripper[0] > 0.0), float(gripper[0] <= 0.0)], dtype=np.float32
    )
    drawer = np.asarray(data["drawer_position"][frame_index], dtype=np.float32)
    progress = np.asarray(data["progress"][frame_index], dtype=np.float32)
    success = np.asarray(data["success"][frame_index], dtype=np.float32)
    return np.concatenate(
        [
            rgb.reshape(-1),
            depth.reshape(-1),
            eef,
            obj,
            qpos,
            qvel,
            finger,
            drawer,
            _one_hot(TASK_FAMILIES.index(task_family), len(TASK_FAMILIES)),
            progress,
            success,
        ]
    ).astype(np.float32)


def _canonical_semantic_window(
    *,
    task_family: str,
    chunk_index: int,
    object_position: np.ndarray,
    goal_position: np.ndarray,
    initial_eef: np.ndarray,
    chunk_length: int,
) -> np.ndarray:
    """Shared task-stage action distilled only from the common scene semantics."""

    def direction(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = np.asarray(target, dtype=np.float32) - np.asarray(
            source, dtype=np.float32
        )
        norm = float(np.linalg.norm(delta))
        return delta / max(norm, 1e-6)

    object_direction = direction(initial_eef, object_position) * 0.05
    goal_direction = direction(initial_eef, goal_position) * 0.05
    window = np.zeros((int(chunk_length), 4), dtype=np.float32)
    if task_family == "pick":
        movement = object_direction if chunk_index < 4 else np.asarray(
            [0.0, 0.0, 0.05], dtype=np.float32
        )
        grip = 1.0 if chunk_index >= 4 else -1.0
    elif task_family == "place":
        movement = goal_direction if chunk_index < 6 else np.asarray(
            [0.0, 0.0, -0.04], dtype=np.float32
        )
        grip = -1.0 if chunk_index >= 6 else 1.0
    elif task_family == "open_drawer":
        movement = (
            goal_direction
            if chunk_index < 4
            else np.asarray([0.05, 0.0, 0.0], dtype=np.float32)
        )
        grip = 1.0 if chunk_index >= 4 else -1.0
    else:
        movement = object_direction if chunk_index < 4 else goal_direction
        grip = -1.0
    window[:, :3] = movement
    window[:, 3] = grip
    return window


def load_c0_dataset(
    root: str | Path,
    *,
    chunk_length: int = 8,
) -> tuple[Dataset, pd.DataFrame]:
    """Load formal C0 trajectories as genuine body-paired action windows."""

    root = Path(root)
    results_path = root / "c0_results.json"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    c0_results = json.loads(results_path.read_text(encoding="utf-8"))
    if c0_results.get("decision") != "C0_PASS":
        raise RuntimeError(f"C0 has not passed: {c0_results.get('decision')}")
    manifest = pd.read_csv(root / "trajectory_manifest.csv")
    samples_by_split: dict[str, list[Sample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for row in manifest.itertuples(index=False):
        body_index = ROBOT_TO_BODY[str(row.robot)]
        data_path = root / "trajectories" / str(row.trajectory_file)
        object_position = np.asarray(
            json.loads(str(row.object_position)), dtype=np.float32
        )
        goal_position = np.asarray(
            json.loads(str(row.goal_position)), dtype=np.float32
        )
        initial_eef = np.asarray(
            json.loads(str(row.initial_eef_position)), dtype=np.float32
        )
        with np.load(data_path) as data:
            chunks = int(data["native_action"].shape[0])
            for chunk_index in range(chunks):
                native = np.asarray(
                    data["native_action"][chunk_index], dtype=np.float32
                )
                gripper = np.asarray(
                    data["gripper_action"][chunk_index], dtype=np.float32
                )
                if native.shape[0] > chunk_length:
                    raise ValueError(
                        f"Configured chunk_length={chunk_length} is shorter than "
                        f"{row.robot} recorded chunk length {native.shape[0]}"
                    )
                # C0 uses robot-specific chunks (4/6/8). C1 aligns the coarse
                # chunk index to a padded length-8 tensor and masks missing
                # substeps; no native action is fabricated.
                padded = np.zeros(
                    (int(chunk_length), MAX_ACTION_DIM), dtype=np.float32
                )
                padded[: native.shape[0], : native.shape[1]] = native
                mask = np.zeros_like(padded)
                mask[: native.shape[0], : native.shape[1]] = 1.0
                padded[: native.shape[0], GRIPPER_ACTION_INDEX] = gripper
                mask[: native.shape[0], GRIPPER_ACTION_INDEX] = 1.0
                # Each embodiment derives semantics from its own executed EEF
                # displacement and gripper command. Nothing is copied from body 0.
                semantic = _canonical_semantic_window(
                    task_family=str(row.task_family),
                    chunk_index=chunk_index,
                    object_position=object_position,
                    goal_position=goal_position,
                    initial_eef=initial_eef,
                    chunk_length=chunk_length,
                )
                context = _sim_context(
                    frame_index=chunk_index,
                    data=data,
                    task_family=str(row.task_family),
                )
                if context.shape[0] != SIM_CONTEXT_DIM:
                    raise AssertionError((context.shape, SIM_CONTEXT_DIM))
                samples_by_split[str(row.split)].append(
                    Sample(
                        body_index=body_index,
                        pair_id=str(row.pair_id),
                        episode_id=str(row.scene_id),
                        task_id=str(row.task_id),
                        trajectory_id=str(row.trajectory_file),
                        window_index=chunk_index,
                        native_action=padded,
                        action_mask=mask,
                        context=context,
                        morphology=MORPHOLOGIES[body_index].descriptor.copy(),
                        semantic_action=semantic,
                        progress=float(data["progress"][chunk_index, 0]),
                        success=float(data["success"][chunk_index, 0]),
                    )
                )
    train = samples_by_split["train"]
    val = samples_by_split["val"]
    test = samples_by_split["test"]
    pair_sets = {
        split: {sample.pair_id for sample in samples}
        for split, samples in (
            ("train", train),
            ("val", val),
            ("test", test),
        )
    }
    task_sets = {
        split: {sample.task_id for sample in samples}
        for split, samples in (
            ("train", train),
            ("val", val),
            ("test", test),
        )
    }
    scene_sets = {
        split: {sample.episode_id for sample in samples}
        for split, samples in (
            ("train", train),
            ("val", val),
            ("test", test),
        )
    }
    overlaps = {}
    for name, split_sets in (
        ("pair", pair_sets),
        ("task", task_sets),
        ("scene", scene_sets),
    ):
        overlaps[name] = int(
            len(split_sets["train"] & split_sets["val"])
            + len(split_sets["train"] & split_sets["test"])
            + len(split_sets["val"] & split_sets["test"])
        )
    # Every pair must be represented by all three embodiments in every split.
    actual_pair_bodies: dict[str, set[int]] = {}
    for sample in [*train, *val, *test]:
        actual_pair_bodies.setdefault(sample.pair_id, set()).add(sample.body_index)
    incomplete_pairs = [
        pair_id
        for pair_id, bodies in actual_pair_bodies.items()
        if bodies != {0, 1, 2}
    ]
    canonical_mismatches: list[str] = []
    semantic_lookup: dict[tuple[str, int], list[np.ndarray]] = {}
    for sample in [*train, *val, *test]:
        semantic_lookup.setdefault(
            (sample.pair_id, sample.window_index), []
        ).append(sample.semantic_action)
    for (pair_id, window_index), values in semantic_lookup.items():
        if any(not np.array_equal(values[0], value) for value in values[1:]):
            canonical_mismatches.append(f"{pair_id}:w{window_index}")
    protocol = {
        "passed": bool(
            not any(overlaps.values())
            and not incomplete_pairs
            and not canonical_mismatches
            and len(train) > 0
            and len(test) > 0
        ),
        "pair_overlap_count": overlaps["pair"],
        "task_overlap_count": overlaps["task"],
        "scene_overlap_count": overlaps["scene"],
        "incomplete_pairs": incomplete_pairs[:20],
        "canonical_semantic_mismatch_count": len(canonical_mismatches),
        "split_counts": {
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "train_pairs": len(pair_sets["train"]),
            "val_pairs": len(pair_sets["val"]),
            "test_pairs": len(pair_sets["test"]),
        },
        "body_counts": {
            split: {
                MORPHOLOGIES[body].name: sum(
                    sample.body_index == body for sample in samples
                )
                for body in (0, 1, 2)
            }
            for split, samples in (
                ("train", train),
                ("val", val),
                ("test", test),
            )
        },
        "dimensions": {
            "context": sorted({sample.context.shape[0] for sample in samples_by_split["train"]}),
            "native_action": sorted(
                {sample.native_action.shape for sample in samples_by_split["train"]}
            ),
            "semantic_action": sorted(
                {sample.semantic_action.shape for sample in samples_by_split["train"]}
            ),
        },
        "chunk_length": int(chunk_length),
    }
    if not protocol["passed"]:
        raise RuntimeError(f"Formal C0 pairing protocol failed: {protocol}")
    return (
        Dataset(
            catalog={},
            train=train,
            val=val,
            test=test,
            protocol=protocol,
        ),
        manifest,
    )


def _common_pair_ids(
    samples: list[Sample],
    bodies: tuple[int, ...],
) -> list[str]:
    required = set(bodies)
    present: dict[str, set[int]] = {}
    for sample in samples:
        if sample.body_index in required:
            present.setdefault(sample.pair_id, set()).add(sample.body_index)
    return sorted(
        pair_id
        for pair_id, body_set in present.items()
        if body_set == required
    )


def make_pairs(
    samples: list[Sample],
    bodies: tuple[int, ...] = (0, 1, 2),
    *,
    limit: int | None = None,
    seed: int = 0,
) -> PairBatch:
    body_list = tuple(int(body) for body in bodies)
    if len(body_list) < 2:
        raise ValueError("At least two bodies are required")
    pair_ids = _common_pair_ids(samples, body_list)
    if limit is not None and len(pair_ids) > limit:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(pair_ids), size=limit, replace=False))
        pair_ids = [pair_ids[int(index)] for index in selected]
    lookup = {
        (sample.pair_id, sample.body_index): sample
        for sample in samples
        if sample.body_index in body_list
    }
    return PairBatch(
        native=np.stack(
            [
                np.stack(
                    [lookup[(pair_id, body)].native_action for body in body_list]
                )
                for pair_id in pair_ids
            ]
        ).astype(np.float32),
        action_mask=np.stack(
            [
                np.stack(
                    [lookup[(pair_id, body)].action_mask for body in body_list]
                )
                for pair_id in pair_ids
            ]
        ).astype(np.float32),
        context=np.stack(
            [
                np.stack(
                    [lookup[(pair_id, body)].context for body in body_list]
                )
                for pair_id in pair_ids
            ]
        ).astype(np.float32),
        morphology=np.stack(
            [
                np.stack(
                    [lookup[(pair_id, body)].morphology for body in body_list]
                )
                for pair_id in pair_ids
            ]
        ).astype(np.float32),
        body_index=np.stack(
            [
                np.asarray(body_list, dtype=np.int64)
                for _ in pair_ids
            ]
        ),
        semantic=np.stack(
            [
                np.stack(
                    [lookup[(pair_id, body)].semantic_action for body in body_list]
                )
                for pair_id in pair_ids
            ]
        ).astype(np.float32),
        progress=np.stack(
            [
                np.asarray(
                    [lookup[(pair_id, body)].progress for body in body_list],
                    dtype=np.float32,
                )
                for pair_id in pair_ids
            ]
        ),
        success=np.stack(
            [
                np.asarray(
                    [lookup[(pair_id, body)].success for body in body_list],
                    dtype=np.float32,
                )
                for pair_id in pair_ids
            ]
        ),
        pair_ids=pair_ids,
        episode_ids=[
            lookup[(pair_id, body_list[0])].episode_id for pair_id in pair_ids
        ],
        task_ids=[
            lookup[(pair_id, body_list[0])].task_id for pair_id in pair_ids
        ],
        window_indices=np.stack(
            [
                np.asarray(
                    [
                        lookup[(pair_id, body)].window_index
                        for body in body_list
                    ],
                    dtype=np.int64,
                )
                for pair_id in pair_ids
            ]
        ),
    )


def source_digest(root: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    package_root = Path(root) / "celar" / "current"
    for path in sorted(package_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
