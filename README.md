# CELAR 跨本体潜在动作强化学习

本目录依据第五份规划文档
[`CELAR跨本体潜在动作嵌入强化学习实施计划.md`](../规划/CELAR跨本体潜在动作嵌入强化学习实施计划.md)
实现当前四任务、三机械臂流程。

## 当前正式流程

- `celar_sim/`：Panda、UR5e、Sawyer 的最小统一 MuJoCo 环境；
- `celar/current/`：只读取正式 C0 数据，定义当前四任务、三本体 C1；
- `configs/c0_mujoco.yaml`：1,200 条配对轨迹的 C0 正式配置；
- `configs/c1_mujoco_v7.yaml`：C1 最终配置；
- `results/c0_mujoco_v1/`：正式 C0 数据和 `C0_PASS` 报告；
- `results/c1_mujoco_v7/`：正式 C1 数据和 `C1_PASS` 报告。

当前任务只有：

```text
pick / place / open_drawer / push
```

当前机械臂只有：

```text
Panda / UR5e / Sawyer
```

## 运行当前流程

```powershell
cd attm5
python -m pytest
python -m celar_sim.cli --config configs/c0_mujoco.yaml
python -m celar.cli --config configs/c1_mujoco_v7.yaml
```

当前 C1 完全从 C0 NPZ 加载：

- 每个 `pair_id` 有三本体；
- 场景、初始状态、目标、split 和 canonical semantic stage 共享；
- 每个本体的 native action 独立保留；
- C1 使用该本体 native action 索引第 8 维追加的 gripper 命令；
- decoder 仍按本体独立；
- 所有检索、循环、进度一致性和 latent usage 都覆盖三个本体。

## 历史预实验

旧的三任务二维代理、C2/C3 PPO 和 C4 代理已经隔离到：

```text
celar/pre_experiment/
configs/pre_experiment/
```

历史复现命令：

```powershell
python -m celar.pre_experiment.cli --study pilot --config configs/pre_experiment/pilot.yaml
python -m celar.pre_experiment.cli --study c2 --config configs/pre_experiment/c2_v4.yaml
python -m celar.pre_experiment.cli --study value --config configs/pre_experiment/value_v3.yaml
```

历史结论见
[`celar/pre_experiment/PROJECT_RESULTS.md`](celar/pre_experiment/PROJECT_RESULTS.md)。
根目录当前流程不会 import 历史包。

## 结论

C0 已通过：1,200 条轨迹、四任务、三本体、RGB-D/位姿对齐、安全层、切分审计、
batch 8 显存和动作块延迟全部满足门槛。

C1 v5 已通过：三本体 Semantic R@1 1.000，Exact-pair R@1 0.705，
相对共享 AE 提升 25.1 个百分点，动作可执行率 98.3%，latent 有效秩 7.16/8。

完整判断见 [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md)。
