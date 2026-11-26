from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from celar_sim.models import robot_xml
from celar_sim.schema import ROBOT_SPECS, SceneSpec, make_scene_spec


def test_robot_xml_contains_morphology_and_camera() -> None:
    xml = robot_xml("panda")
    assert '<camera name="front"' in xml
    assert 'name="eef_site"' in xml
    assert 'name="left_finger_joint"' in xml
    assert all(f'name="arm_joint_{index}"' in xml for index in range(7))


def test_scene_pairs_and_split_are_deterministic() -> None:
    first = make_scene_spec(family="pick", pair_index=12, total_pairs=100, seed=9)
    second = make_scene_spec(family="pick", pair_index=12, total_pairs=100, seed=9)
    assert first == second
    assert first.pair_id == second.pair_id
    assert first.scene_id == second.scene_id
    assert first.task_id == second.task_id


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="MuJoCo is only installed in the remote simulation environment",
)
def test_environment_reset_and_full_chunk() -> None:
    from celar_sim.environment import UnifiedManipulationEnv

    scene = make_scene_spec(family="pick", pair_index=3, total_pairs=100, seed=7)
    env = UnifiedManipulationEnv("ur5e", enable_rendering=False)
    try:
        env.reset(scene)
        first = np.asarray(env.data.qpos).copy()
        env.reset(scene)
        np.testing.assert_allclose(first, env.data.qpos, atol=1e-10)
        action = np.zeros(
            (
                ROBOT_SPECS["ur5e"].action_chunk_length,
                ROBOT_SPECS["ur5e"].arm_dof,
            )
        )
        env.step_chunk(action, np.zeros(len(action)))
        assert env.chunk_index == 1
    finally:
        env.close()
