# C0 Minimal Unified MuJoCo Environment

本模块实现 CELAR 进入视觉/动作模型训练前的真实 MuJoCo 工程层。它替代此前的二维
运动学代理，但仍是**最小统一 MJCF**，不是 Panda、UR5e、Sawyer 的厂商原始动力学模型。

## 覆盖范围

- 三个机械臂：Panda 7 DoF、UR5e 6 DoF、Sawyer 7 DoF；
- 四个任务族：`pick`、`place`、`open_drawer`、`push`；
- 每个任务族 100 个配对场景，三本体共享 `pair_id`、`scene_id`、初始状态和目标；
- 70/15/15 train/val/test，按 pair、task instance、scene 分组隔离；
- 每本体独立原生动作维度、关节范围、速度上限和安全裁剪；
- Panda/UR5e/Sawyer 动作块长度分别为 4/6/8，完整执行整块；
- 每块记录动作、夹爪、EEP pose、本体关节、物体姿态、抽屉位置、进度、成功、接触和安全 flags；
- 每块末尾输出 128×128 RGB、metric depth、相机内参和外参；
- 固定种子，保留 nominal、failure、perturb 三类轨迹。

## 数据格式

`results/c0_mujoco_v1/trajectory_manifest.csv` 是索引。每条轨迹为一个压缩 NPZ：

| 字段 | 形状 |
|---|---|
| `rgb` | `[8, 128, 128, 3]` |
| `depth` | `[8, 128, 128]`，meter，存储为 FP16 |
| `native_action` | `[8, chunk_length, arm_dof]` |
| `gripper_action` | `[8, chunk_length]` |
| `eef_pose` | `[8, 7]` |
| `arm_qpos`, `arm_qvel` | `[8, 6或7]` |
| `object_pose` | `[8, 7]` |
| `drawer_position` | `[8, 1]` |
| `progress`, `success` | `[8, 1]` |
| `safety_*` | `[8, chunk_length]` |
| `intrinsics` | `[3, 3]` |
| `extrinsics` | `[8, 4, 4]` |

## 远端运行

```bash
cd /home/scu101/celar/attm5
bash scripts/run_c0_remote.sh
```

脚本先跑 `configs/c0_smoke.yaml`，再跑正式 `configs/c0_mujoco.yaml`。建议环境：

```bash
export MUJOCO_GL=egl
python -m pip install -r requirements-sim.txt
```

正式运行包含 `3 robots × 4 tasks × 100 scenes = 1,200` 条轨迹。全部通过后才应启动
视觉动作模型训练。

## C0 门控

- 三本体 reset 与完整动作块 step 可重复；
- segmentation 像素中心与投影 EEP 差异 ≤4 px，深度差 ≤0.09 m；
- 专家动作经安全层后的关节限位/速度越界数均为 0；
- pair/task/scene 跨 split 重叠数均为 0；
- batch 8 的 RGB-D 前向/反向峰值 reserved memory <10.5 GB；
- 单动作块执行中位延迟 ≤180 ms、P95 ≤300 ms。

显存探针使用 128×128 RGB-D、batch 8 和 ResNet-18 级视觉 backbone。正式报告位于
`results/c0_mujoco_v1/README.md`。
