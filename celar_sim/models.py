from __future__ import annotations

from .schema import COLORS, ROBOT_SPECS


def robot_xml(robot_name: str, object_color_index: int = 0) -> str:
    """Return a compact MJCF model with robot-specific arm DOF and morphology."""

    spec = ROBOT_SPECS[robot_name]
    base_x, base_y, base_z = spec.base_position
    finger_gap = 0.026
    color = COLORS[object_color_index]
    rgba = " ".join(f"{float(value):.4f}" for value in color)
    extension = 0.08
    arm_parts = [
        """
        <body name="x_stage" pos="0 0 0">
          <joint name="arm_joint_0" type="slide" axis="1 0 0" range="-0.10 0.95" damping="0.8"/>
          <geom name="x_link" type="capsule" fromto="0 0 0 0.12 0 0" size="0.026" rgba="0.20 0.24 0.28 1"/>
          <body name="y_stage" pos="0.12 0 0">
            <joint name="arm_joint_1" type="slide" axis="0 1 0" range="-0.38 0.38" damping="0.8"/>
            <geom name="y_link" type="capsule" fromto="0 0 0 0.10 0 0" size="0.024" rgba="0.25 0.30 0.34 1"/>
            <body name="z_stage" pos="0.10 0 0">
              <joint name="arm_joint_2" type="slide" axis="0 0 1" range="-0.25 0.22" damping="0.8"/>
              <geom name="z_link" type="capsule" fromto="0 0 0 0.10 0 0" size="0.022" rgba="0.30 0.36 0.40 1"/>
        """
    ]
    for index in range(spec.extra_hinges):
        arm_parts.append(
            f"""
              <body name="wrist_{index}" pos="0.10 0 0">
                <joint name="arm_joint_{index + 3}" type="hinge" axis="0 1 0" range="-0.75 0.75" damping="1.2"/>
                <geom name="wrist_link_{index}" type="capsule" fromto="0 0 0 {extension:.4f} 0 0" size="0.018" rgba="0.40 0.43 0.47 1"/>
            """
        )
    deepest = max(0, spec.extra_hinges - 1)
    tail_pos = extension if spec.extra_hinges else 0.10
    close_count = spec.extra_hinges
    arm_parts.append(
        f"""
                <site name="eef_site" pos="{tail_pos:.4f} 0 0" size="0.012" rgba="1 0.1 0.1 1"/>
                <body name="left_finger" pos="{tail_pos:.4f} {finger_gap:.4f} 0">
                  <joint name="left_finger_joint" type="slide" axis="0 1 0" range="0 0.045" damping="0.4"/>
                  <geom name="left_finger_geom" type="box" size="0.018 0.009 0.045" rgba="0.75 0.78 0.80 1"/>
                </body>
                <body name="right_finger" pos="{tail_pos:.4f} {-finger_gap:.4f} 0">
                  <joint name="right_finger_joint" type="slide" axis="0 1 0" range="-0.045 0" damping="0.4"/>
                  <geom name="right_finger_geom" type="box" size="0.018 0.009 0.045" rgba="0.75 0.78 0.80 1"/>
                </body>
        """
    )
    for _ in range(close_count):
        arm_parts.append("</body>")
    arm_parts.extend(
        [
            "</body></body></body>",
        ]
    )
    arm_xml = "".join(arm_parts)
    return f"""
<mujoco model="celar_{robot_name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
  <size njmax="512" nconmax="128"/>
  <visual>
    <global offwidth="128" offheight="128"/>
    <quality shadowsize="1024"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.65 0.65 0.65"/>
  </visual>
  <worldbody>
    <light name="key" pos="0.8 -0.7 1.6" dir="-0.4 0.4 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="front" pos="1.25 0.0 0.72" xyaxes="0 1 0 0 0 1" fovy="45"/>
    <geom name="table" type="box" pos="0.08 0 0.40" size="0.58 0.55 0.035" rgba="0.55 0.48 0.40 1"/>
    <geom name="table_leg_0" type="box" pos="-0.34 -0.38 0.19" size="0.035 0.035 0.19" rgba="0.35 0.30 0.25 1"/>
    <geom name="table_leg_1" type="box" pos="0.50 -0.38 0.19" size="0.035 0.035 0.19" rgba="0.35 0.30 0.25 1"/>
    <geom name="table_leg_2" type="box" pos="-0.34 0.38 0.19" size="0.035 0.035 0.19" rgba="0.35 0.30 0.25 1"/>
    <geom name="table_leg_3" type="box" pos="0.50 0.38 0.19" size="0.035 0.035 0.19" rgba="0.35 0.30 0.25 1"/>
    <body name="goal_marker" mocap="true" pos="0.30 0 0.455">
      <geom name="goal_disc" type="cylinder" size="0.065 0.004" rgba="0.10 0.75 0.90 0.55" contype="0" conaffinity="0"/>
    </body>
    <body name="object" pos="0.05 0 0.475">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="0.032 0.032 0.032" rgba="{rgba}" mass="0.08" friction="1.2 0.02 0.02"/>
    </body>
    <body name="drawer_base" pos="0.18 -0.28 0.43">
      <geom name="drawer_shell" type="box" size="0.18 0.15 0.11" pos="0.16 0 0.11" rgba="0.34 0.35 0.38 1"/>
      <body name="drawer" pos="0 0 0.10">
        <joint name="drawer_slide" type="slide" axis="1 0 0" range="-0.08 0.23" damping="2.0"/>
        <geom name="drawer_front" type="box" size="0.025 0.15 0.11" pos="0.0 0 0" rgba="0.52 0.54 0.58 1"/>
        <geom name="drawer_bin" type="box" size="0.13 0.13 0.03" pos="0.12 0 0.05" rgba="0.42 0.44 0.48 1"/>
        <site name="drawer_handle" pos="-0.026 0 0.02" size="0.012" rgba="1 0.75 0.1 1"/>
      </body>
    </body>
    <body name="robot_base" pos="{base_x:.4f} {base_y:.4f} {base_z:.4f}">
      <geom name="pedestal" type="cylinder" size="0.13 0.09" pos="0 0 -0.09" rgba="0.19 0.21 0.23 1"/>
      {arm_xml}
    </body>
  </worldbody>
</mujoco>
"""
