from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import robot_xml
from .schema import COLORS, ROBOT_SPECS, SceneSpec


def _require_mujoco():
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo is required for C0. Install requirements-sim.txt in the "
            "remote simulation environment and set MUJOCO_GL=egl."
        ) from error
    return mujoco


@dataclass
class ChunkResult:
    observation: dict[str, Any]
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    native_action: np.ndarray
    gripper_action: np.ndarray
    eef_pose: np.ndarray
    object_pose: np.ndarray
    drawer_position: float
    progress: float
    success: bool
    safety_pre_limit_violation: np.ndarray
    safety_pre_velocity_violation: np.ndarray
    safety_post_limit_violation: np.ndarray
    safety_post_velocity_violation: np.ndarray
    contact_count: int


class UnifiedManipulationEnv:
    """Minimal shared MuJoCo scene with robot-specific arm and native actions."""

    def __init__(
        self,
        robot_name: str,
        *,
        render_size: int = 128,
        enable_rendering: bool = True,
    ) -> None:
        self.mujoco = _require_mujoco()
        if robot_name not in ROBOT_SPECS:
            raise ValueError(f"Unknown robot: {robot_name}")
        self.robot_name = robot_name
        self.spec = ROBOT_SPECS[robot_name]
        self.render_size = int(render_size)
        self.xml = robot_xml(robot_name)
        self.model = self.mujoco.MjModel.from_xml_string(self.xml)
        self.data = self.mujoco.MjData(self.model)
        self.arm_joint_names = tuple(
            f"arm_joint_{index}" for index in range(self.spec.arm_dof)
        )
        self.arm_qpos_adr = np.asarray(
            [
                int(self.model.jnt_qposadr[self._joint_id(name)])
                for name in self.arm_joint_names
            ],
            dtype=np.int64,
        )
        self.arm_qvel_adr = np.asarray(
            [
                int(self.model.jnt_dofadr[self._joint_id(name)])
                for name in self.arm_joint_names
            ],
            dtype=np.int64,
        )
        self.arm_ranges = np.asarray(
            [self.model.jnt_range[self._joint_id(name)] for name in self.arm_joint_names],
            dtype=np.float64,
        )
        self.left_finger_adr = int(
            self.model.jnt_qposadr[self._joint_id("left_finger_joint")]
        )
        self.right_finger_adr = int(
            self.model.jnt_qposadr[self._joint_id("right_finger_joint")]
        )
        self.object_qpos_adr = int(
            self.model.jnt_qposadr[self._joint_id("object_free")]
        )
        self.drawer_qpos_adr = int(
            self.model.jnt_qposadr[self._joint_id("drawer_slide")]
        )
        self.eef_site_id = self._site_id("eef_site")
        self.handle_site_id = self._site_id("drawer_handle")
        self.goal_body_id = self._body_id("goal_marker")
        self.goal_mocap_id = int(self.model.body_mocapid[self.goal_body_id])
        self.finger_geom_ids = {
            self._geom_id("left_finger_geom"),
            self._geom_id("right_finger_geom"),
        }
        self.object_geom_id = self._geom_id("object_geom")
        self.renderer = None
        if enable_rendering:
            self.renderer = self.mujoco.Renderer(
                self.model,
                height=self.render_size,
                width=self.render_size,
            )
        self.scene: SceneSpec | None = None
        self.held = False
        self.held_by_drawer = False
        self.held_offset = np.zeros(3, dtype=np.float64)
        self.drawer_grasp_reference = 0.0
        self.chunk_index = 0
        self.success = False
        self.last_progress = 0.0
        self.initial_qpos: np.ndarray | None = None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _joint_id(self, name: str) -> int:
        return int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name))

    def _site_id(self, name: str) -> int:
        return int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, name))

    def _body_id(self, name: str) -> int:
        return int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name))

    def _geom_id(self, name: str) -> int:
        return int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, name))

    def _world_to_arm_qpos(self, position: np.ndarray) -> np.ndarray:
        base_x, base_y, base_z = self.spec.base_position
        extension = 0.30 + 0.10 * self.spec.extra_hinges
        qpos = np.zeros(self.spec.arm_dof, dtype=np.float64)
        qpos[0] = position[0] - extension - base_x
        qpos[1] = position[1] - base_y
        qpos[2] = position[2] - base_z
        return np.clip(qpos, self.arm_ranges[:, 0], self.arm_ranges[:, 1])

    def reset(self, scene: SceneSpec) -> dict[str, Any]:
        self.scene = scene
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        initial_arm = self._world_to_arm_qpos(
            np.asarray(scene.initial_eef_position, dtype=np.float64)
        )
        self.data.qpos[self.arm_qpos_adr] = initial_arm
        self.data.qpos[self.left_finger_adr] = 0.0
        self.data.qpos[self.right_finger_adr] = 0.0
        object_position = np.asarray(scene.object_position, dtype=np.float64)
        if scene.held_initial:
            object_position = np.asarray(scene.initial_eef_position) + np.asarray(
                [0.0, 0.0, -0.065]
            )
        self._set_free_body_position(object_position)
        self.data.qpos[self.drawer_qpos_adr] = scene.drawer_initial
        self.data.mocap_pos[self.goal_mocap_id] = scene.goal_position
        self.data.mocap_quat[self.goal_mocap_id] = np.asarray(
            [1.0, 0.0, 0.0, 0.0]
        )
        self.mujoco.mj_forward(self.model, self.data)
        self.model.geom_rgba[self.object_geom_id] = COLORS[scene.color_index]
        self.held = bool(scene.held_initial)
        self.held_by_drawer = False
        self.held_offset = (
            object_position - self.eef_position() if self.held else np.zeros(3)
        )
        self.drawer_grasp_reference = float(
            self.data.qpos[self.drawer_qpos_adr]
        )
        self.chunk_index = 0
        self.success = False
        self.last_progress = self._progress()
        self.mujoco.mj_forward(self.model, self.data)
        self.initial_qpos = np.asarray(self.data.qpos).copy()
        return self.observation()

    def _set_free_body_position(self, position: np.ndarray) -> None:
        self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 3] = position
        self.data.qpos[self.object_qpos_adr + 3 : self.object_qpos_adr + 7] = np.asarray(
            [1.0, 0.0, 0.0, 0.0]
        )

    def object_position(self) -> np.ndarray:
        return np.asarray(
            self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 3],
            dtype=np.float64,
        ).copy()

    def eef_position(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self.eef_site_id], dtype=np.float64).copy()

    def eef_pose(self) -> np.ndarray:
        position = self.eef_position()
        quaternion = np.zeros(4, dtype=np.float64)
        self.mujoco.mju_mat2Quat(
            quaternion,
            np.asarray(self.data.site_xmat[self.eef_site_id], dtype=np.float64),
        )
        return np.concatenate([position, quaternion])

    def gripper_position(self) -> np.ndarray:
        return np.asarray(
            [
                self.data.qpos[self.left_finger_adr],
                self.data.qpos[self.right_finger_adr],
            ],
            dtype=np.float64,
        )

    def drawer_position(self) -> float:
        return float(self.data.qpos[self.drawer_qpos_adr])

    def observation(self) -> dict[str, Any]:
        return {
            "eef_pose": self.eef_pose(),
            "arm_qpos": np.asarray(self.data.qpos[self.arm_qpos_adr]).copy(),
            "arm_qvel": np.asarray(self.data.qvel[self.arm_qvel_adr]).copy(),
            "gripper": self.gripper_position(),
            "object_pose": np.asarray(
                self.data.qpos[
                    self.object_qpos_adr : self.object_qpos_adr + 7
                ]
            ).copy(),
            "drawer_position": self.drawer_position(),
            "held": self.held or self.held_by_drawer,
            "progress": self._progress(),
            "success": self.success,
        }

    def _goal_distance(self) -> float:
        if self.scene is None:
            return float("inf")
        return float(
            np.linalg.norm(
                np.asarray(self.scene.goal_position) - self.object_position()
            )
        )

    def _progress(self) -> float:
        if self.scene is None:
            return 0.0
        if self.scene.task_family == "pick":
            object_height = self.object_position()[2]
            return float(
                np.clip(
                    0.65 * float(self.held) + 0.35 * (object_height - 0.43) / 0.20,
                    0.0,
                    1.0,
                )
            )
        if self.scene.task_family == "place":
            object_distance = self._goal_distance()
            return float(
                np.clip(
                    0.75 * (1.0 - object_distance / 0.55)
                    + 0.25 * float(not self.held and object_distance < 0.09),
                    0.0,
                    1.0,
                )
            )
        if self.scene.task_family == "open_drawer":
            return float(
                np.clip(
                    (self.drawer_position() - self.scene.drawer_initial) / 0.20,
                    0.0,
                    1.0,
                )
            )
        object_distance = self._goal_distance()
        return float(np.clip(1.0 - object_distance / 0.55, 0.0, 1.0))

    def _update_success(self) -> bool:
        if self.scene is None:
            return False
        if self.scene.task_family == "pick":
            return bool(self.held and self.object_position()[2] > 0.535)
        if self.scene.task_family == "place":
            return bool(not self.held and self._goal_distance() < 0.085)
        if self.scene.task_family == "open_drawer":
            return bool(
                self.drawer_position()
                >= self.scene.drawer_initial + 0.155
            )
        return bool(self._goal_distance() < 0.085)

    def _apply_attachment(self, grip_command: float) -> None:
        if self.scene is None:
            return
        eef = self.eef_position()
        if grip_command > 0.15:
            if not self.held:
                if self.scene.task_family == "open_drawer":
                    handle = np.asarray(
                        self.data.site_xpos[self.handle_site_id], dtype=np.float64
                    )
                    if np.linalg.norm(handle - eef) < 0.075:
                        self.held = True
                        self.held_by_drawer = True
                        self.drawer_grasp_reference = self.drawer_position()
                        self.held_offset = np.zeros(3, dtype=np.float64)
                else:
                    object_position = self.object_position()
                    if np.linalg.norm(object_position - eef) < 0.072:
                        self.held = True
                        self.held_by_drawer = False
                        self.held_offset = object_position - eef
        elif grip_command < -0.15 and self.held:
            self.held = False
            self.held_by_drawer = False
            self.held_offset[:] = 0.0

    def _synchronize_attachment(self) -> None:
        if not self.held or self.scene is None:
            return
        eef = self.eef_position()
        if self.held_by_drawer:
            delta = eef[0] - float(self.scene.initial_eef_position[0])
            value = self.drawer_grasp_reference + np.clip(delta, -0.10, 0.22)
            self.data.qpos[self.drawer_qpos_adr] = np.clip(value, -0.08, 0.23)
        else:
            self._set_free_body_position(eef + self.held_offset)

    def _apply_push_contact(
        self,
        previous_eef: np.ndarray,
        current_eef: np.ndarray,
    ) -> None:
        if self.scene is None or self.scene.task_family != "push" or self.held:
            return
        movement = current_eef - previous_eef
        object_position = self.object_position()
        if (
            np.linalg.norm(object_position - current_eef) < 0.085
            and np.linalg.norm(movement) > 1e-5
        ):
            self._set_free_body_position(object_position + 0.75 * movement)

    def _safe_arm_action(
        self,
        delta: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool, bool, bool]:
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != (self.spec.arm_dof,):
            raise ValueError(
                f"Expected action {(self.spec.arm_dof,)}, got {delta.shape}"
            )
        current = np.asarray(self.data.qpos[self.arm_qpos_adr], dtype=np.float64)
        maximum_delta = np.asarray(self.spec.max_joint_delta, dtype=np.float64)
        pre_velocity = np.abs(delta) > maximum_delta + 1e-9
        clipped_delta = np.clip(delta, -maximum_delta, maximum_delta)
        requested = current + clipped_delta
        pre_limit = (requested < self.arm_ranges[:, 0] - 1e-9) | (
            requested > self.arm_ranges[:, 1] + 1e-9
        )
        safe = np.clip(requested, self.arm_ranges[:, 0], self.arm_ranges[:, 1])
        post_limit = (safe < self.arm_ranges[:, 0] - 1e-9) | (
            safe > self.arm_ranges[:, 1] + 1e-9
        )
        # MuJoCo stores qpos as double, but range clipping and bindings may
        # introduce tiny representation noise; 1e-6 is far below physical
        # relevance and is the C0 numerical safety tolerance.
        post_velocity = np.abs(safe - current) > maximum_delta + 1e-6
        return (
            safe,
            bool(pre_limit.any()),
            bool(pre_velocity.any()),
            bool(post_limit.any()),
            bool(post_velocity.any()),
        )

    def step_chunk(
        self,
        native_action: np.ndarray,
        gripper_action: np.ndarray,
    ) -> ChunkResult:
        native_action = np.asarray(native_action, dtype=np.float64)
        gripper_action = np.asarray(gripper_action, dtype=np.float64).reshape(-1)
        expected_length = self.spec.action_chunk_length
        if native_action.shape != (expected_length, self.spec.arm_dof):
            raise ValueError(
                f"Expected native action {(expected_length, self.spec.arm_dof)}, "
                f"got {native_action.shape}"
            )
        if gripper_action.shape != (expected_length,):
            raise ValueError(
                f"Expected gripper action {(expected_length,)}, got {gripper_action.shape}"
            )
        pre_limit_flags = np.zeros(expected_length, dtype=bool)
        pre_velocity_flags = np.zeros(expected_length, dtype=bool)
        post_limit_flags = np.zeros(expected_length, dtype=bool)
        post_velocity_flags = np.zeros(expected_length, dtype=bool)
        for substep in range(expected_length):
            previous_eef = self.eef_position()
            safe_qpos, pre_limit, pre_velocity, post_limit, post_velocity = (
                self._safe_arm_action(native_action[substep])
            )
            pre_limit_flags[substep] = pre_limit
            pre_velocity_flags[substep] = pre_velocity
            post_limit_flags[substep] = post_limit
            post_velocity_flags[substep] = post_velocity
            self.data.qpos[self.arm_qpos_adr] = safe_qpos
            grip = float(np.clip(gripper_action[substep], -1.0, 1.0))
            finger_offset = 0.044 * (0.5 + 0.5 * grip)
            self.data.qpos[self.left_finger_adr] = finger_offset
            self.data.qpos[self.right_finger_adr] = -finger_offset
            # The arm is a minimal kinematic manipulator. Contact/equality
            # forces must not integrate arm coordinates away from the command.
            self.data.qvel[self.arm_qvel_adr] = 0.0
            self.mujoco.mj_step(self.model, self.data)
            self.data.qpos[self.arm_qpos_adr] = np.clip(
                self.data.qpos[self.arm_qpos_adr],
                self.arm_ranges[:, 0],
                self.arm_ranges[:, 1],
            )
            self.mujoco.mj_forward(self.model, self.data)
            self._apply_attachment(grip)
            self._synchronize_attachment()
            self._apply_push_contact(previous_eef, self.eef_position())
            self.mujoco.mj_forward(self.model, self.data)
            self.success = self.success or self._update_success()
        self.chunk_index += 1
        observation = self.observation()
        if self.renderer is None:
            rgb = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)
            depth = np.zeros((self.render_size, self.render_size), dtype=np.float32)
            intrinsics, extrinsics = self.camera_parameters()
        else:
            rgb, depth, intrinsics, extrinsics, _ = self.render_rgbd(
                include_segmentation=False
            )
        self.last_progress = float(observation["progress"])
        return ChunkResult(
            observation=observation,
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            native_action=native_action.copy(),
            gripper_action=gripper_action.copy(),
            eef_pose=np.asarray(observation["eef_pose"]).copy(),
            object_pose=np.asarray(observation["object_pose"]).copy(),
            drawer_position=float(observation["drawer_position"]),
            progress=float(observation["progress"]),
            success=bool(self.success),
            safety_pre_limit_violation=pre_limit_flags,
            safety_pre_velocity_violation=pre_velocity_flags,
            safety_post_limit_violation=post_limit_flags,
            safety_post_velocity_violation=post_velocity_flags,
            contact_count=int(self.data.ncon),
        )

    def camera_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        camera_id = int(
            self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, "front"
            )
        )
        position = np.asarray(self.data.cam_xpos[camera_id], dtype=np.float64)
        rotation = np.asarray(
            self.data.cam_xmat[camera_id], dtype=np.float64
        ).reshape(3, 3)
        extrinsics = np.eye(4, dtype=np.float64)
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = position
        fovy = np.deg2rad(float(self.model.cam_fovy[camera_id]))
        focal = 0.5 * self.render_size / np.tan(0.5 * fovy)
        intrinsics = np.asarray(
            [
                [focal, 0.0, self.render_size / 2.0],
                [0.0, focal, self.render_size / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return intrinsics, extrinsics

    def project_world(
        self,
        point: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> tuple[float, float, float]:
        rotation = extrinsics[:3, :3]
        translation = extrinsics[:3, 3]
        camera_point = rotation.T @ (np.asarray(point, dtype=np.float64) - translation)
        depth = -float(camera_point[2])
        if depth <= 1e-8:
            return float("nan"), float("nan"), depth
        u = intrinsics[0, 0] * camera_point[0] / depth + intrinsics[0, 2]
        v = intrinsics[1, 1] * -camera_point[1] / depth + intrinsics[1, 2]
        return float(u), float(v), depth

    def render_rgbd(
        self,
        *,
        include_segmentation: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        if self.renderer is None:
            raise RuntimeError("Rendering is disabled")
        camera_id = int(
            self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, "front"
            )
        )
        self.renderer.update_scene(self.data, camera=camera_id)
        self.renderer.enable_depth_rendering()
        depth = np.asarray(self.renderer.render(), dtype=np.float32).copy()
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera_id)
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8).copy()
        segmentation = None
        if include_segmentation:
            self.renderer.enable_segmentation_rendering()
            segmentation = np.asarray(self.renderer.render()).copy()
            self.renderer.disable_segmentation_rendering()
            self.renderer.update_scene(self.data, camera=camera_id)
        intrinsics, extrinsics = self.camera_parameters()
        return rgb, depth, intrinsics, extrinsics, segmentation

    def rgbd_alignment_check(self) -> dict[str, float | bool]:
        _, depth, intrinsics, extrinsics, segmentation = self.render_rgbd(
            include_segmentation=True
        )
        if segmentation is None:
            raise RuntimeError("Segmentation rendering failed")
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        mask = np.isin(segmentation, np.fromiter(self.finger_geom_ids, dtype=np.int32))
        if not mask.any():
            return {
                "passed": False,
                "pixel_error": float("inf"),
                "depth_error": float("inf"),
            }
        rows, columns = np.nonzero(mask)
        observed_u = float(columns.mean())
        observed_v = float(rows.mean())
        expected_u, expected_v, expected_depth = self.project_world(
            self.eef_position(), intrinsics, extrinsics
        )
        pixel_error = float(
            np.hypot(observed_u - expected_u, observed_v - expected_v)
        )
        observed_depth = float(np.median(depth[mask]))
        depth_error = float(abs(observed_depth - expected_depth))
        return {
            "passed": bool(
                pixel_error <= 4.0
                and depth_error <= 0.09
                and np.isfinite(expected_depth)
            ),
            "pixel_error": pixel_error,
            "observed_depth": observed_depth,
            "expected_depth": float(expected_depth),
            "depth_error": depth_error,
        }
