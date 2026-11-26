# CELAR 跨本体潜在动作立项实验

本目录依据第五份规划文档
[`CELAR跨本体潜在动作嵌入强化学习实施计划.md`](../规划/CELAR跨本体潜在动作嵌入强化学习实施计划.md)
实现一个可审计的跨本体动作表示代理实验。

正式结论见 [`PROJECT_RESULTS.md`](PROJECT_RESULTS.md)：C0–C4 代理阶段均已通过，
联合决策为 `GO_TO_MUJOCO_FULL_PILOT`。这支持进入 MuJoCo 全量试点，但仍不支持
把当前二维代理结果直接写成真实机器人或论文级结论。

实验回答四个决定是否立项的问题：

1. 三个动作维度和运动学各异的机械臂，是否能在相同任务阶段映射到相近的潜在动作邻域？
2. 共享 latent 经每个本体独立 decoder 和安全层后，是否仍能恢复可执行动作？
3. 对训练中完全未见过的第三机械臂，冻结动作编码器并只训练 decoder，是否比从零训练更省演示？
4. 在相同 BC 与 PPO 在线预算下，共享 latent 动作接口是否带来成功率或达标效率优势？

## 运行

```powershell
cd attm5
python -m pytest
python -m celar.cli --config configs/smoke.yaml
python -m celar.cli --config configs/pilot.yaml
python -m celar.cli --study value --config configs/value_v3.yaml
python -m celar.cli --study c1-mujoco --config configs/c1_mujoco_v4.yaml
python -m celar.cli --study c2 --config configs/c2_v4.yaml
python -m celar.analyze_c3
python -m celar.combine_results
```

`smoke.yaml` 只验证链路；`pilot.yaml` 是正式代理实验，包含三个随机种子。

## 主要产物

- `c0_protocol.json`：切分泄漏、配对维度、三本体专家动作安全审计；
- `c1_*`：七种动作表示方法的跨本体检索、重建、执行、循环和消融；
- `c2_*`：原生动作 BC 与 CELAR latent BC；
- `c3_*`：原生动作、普通共享 decoder 与 CELAR 的同预算 PPO 曲线；
- `c4_*`：5/20/50 条演示的新本体 decoder 适配；
- `results.json`：预注册门槛、配对差异和立项决策；
- `run_manifest.json`：配置、环境、源码与 checkpoint SHA-256；
- `checksums.sha256`：所有结果文件完整性校验。

## 证据边界

当前实现是显式构造的跨本体运动学与二维抓放代理，适合在低成本下检验动作维数差异、
配对对齐、decoder 可执行性、小样本适配和同预算 PPO。它不包含接触动力学、真实
RGB-D、LIBERO/MuJoCo、IK、控制延迟或真机结果，因此报告中的决策只能是
“是否值得进入下一阶段”。

C0 的真实 MuJoCo 工程层位于 [`celar_sim/`](celar_sim/)，运行说明见
[`celar_sim/README.md`](celar_sim/README.md)。C0 已在远端 RTX 3060 12 GB 服务器
执行，1,200 条轨迹和完整门控结果保存在
[`results/c0_mujoco_v1/`](results/c0_mujoco_v1/)；`C0_PASS`。
