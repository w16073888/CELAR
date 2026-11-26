# CELAR 跨本体潜在动作立项实验

> 本报告由实验脚本自动生成。证据来自受控运动学代理，不是 MuJoCo、LIBERO 或真机论文级结果；它只用于判断是否值得进入下一阶段。

## 立项判断

- **决策：`CONDITIONAL_GO_TO_MUJOCO`**
- C0 配对数据与安全动作：通过
- C1 潜在动作对齐：通过
- C2 同本体 BC：未通过
- C4 新本体适配：通过

预注册判断规则：C0 要求无 pair/task 泄漏且三个本体专家动作安全可执行率均 ≥95%；C1 要求完整 CELAR 的跨本体 R@1 ≥0.60、比普通共享 AE 至少高0.10、动作可执行率 ≥0.85、latent 有效秩 ≥3；C2 要求 latent BC 的语义误差不高于原生动作 BC 的 115%；C4 要求冻结动作编码器的 5-shot 适配相对从零 AE 至少降低 15%，20-shot 均值不退化和方向一致，且可执行率 ≥85%。C0+C1 加上 C2/C4 之一才进入 MuJoCo 阶段。

## C0：配对数据与动作安全

- pair 泄漏数：0
- task 泄漏数：0
- 数据切分：`{"train_samples": 1944, "val_samples": 144, "test_samples": 180, "train_pairs": 648, "val_pairs": 48, "test_pairs": 60, "train_tasks": 6, "val_tasks": 2, "test_tasks": 2}`

| 本体 | 动作维度 | 专家可执行率 | 语义 MAE | 最大单步 | 速度上限 |
|---|---:|---:|---:|---:|---:|
| franka_panda | 7 | 1.000 | 0.0400 | 0.7800 | 0.7800 |
| ur5e | 6 | 1.000 | 0.0549 | 0.8400 | 0.8400 |
| sawyer | 7 | 1.000 | 0.0449 | 0.6752 | 0.7200 |

## C1：动作重建与跨本体对齐

| 方法 | R@1 | R@5 | 语义 MAE | 可执行率 | 循环误差 | 有效秩 |
|---|---:|---:|---:|---:|---:|---:|
| celar_no_cycle | 0.922 | 0.972 | 0.1166 | 0.986 | 0.0002 | 4.02 |
| celar_no_outcome | 0.922 | 0.967 | 0.1074 | 0.986 | 0.0001 | 3.89 |
| celar | 0.911 | 0.967 | 0.1155 | 0.981 | 0.0002 | 4.01 |
| celar_no_align | 0.550 | 0.706 | 0.0953 | 0.989 | 0.0006 | 5.51 |
| shared_ae | 0.550 | 0.706 | 0.0953 | 0.989 | 0.0006 | 5.51 |
| celar_shuffled | 0.378 | 0.550 | 0.1195 | 0.978 | 0.0002 | 4.70 |
| independent_ae | 0.217 | 0.267 | 0.0938 | 0.981 | 0.2493 | 4.47 |

完整 CELAR 相对普通共享 AE 的 R@1 配对提升为 **+0.361** (95% CI [0.300, 0.400])。相对无跨本体对齐消融的提升为 **+0.361**。

## C2：同本体行为克隆

| 方法 | 语义 MAE | 可执行率 | Jerk |
|---|---:|---:|---:|
| celar_bc | 0.4115 | 0.389 | 0.0724 |
| native_bc | 0.2928 | 0.711 | 0.1342 |

## C4：未见第三机械臂适配

| 演示数 | 方法 | 语义 MAE | 可执行率 |
|---:|---|---:|---:|
| 5 | celar_decoder_adapter | 0.6284 | 0.206 |
| 5 | celar_teacher_decoder | 0.1110 | 1.000 |
| 5 | independent_action_ae | 0.1500 | 1.000 |
| 5 | independent_bc | 0.3980 | 0.350 |
| 20 | celar_decoder_adapter | 0.4798 | 0.344 |
| 20 | celar_teacher_decoder | 0.1085 | 1.000 |
| 20 | independent_action_ae | 0.1153 | 1.000 |
| 20 | independent_bc | 0.3209 | 0.606 |
| 50 | celar_decoder_adapter | 0.5042 | 0.289 |
| 50 | celar_teacher_decoder | 0.1119 | 1.000 |
| 50 | independent_action_ae | 0.0944 | 1.000 |
| 50 | independent_bc | 0.3198 | 0.622 |

5 条演示下，CELAR 冻结动作编码器 + 新本体 decoder 的语义误差相对从零独立动作 AE 降低 **26.0%**；配对均值为 -0.0390，95% CI [-0.0749, +0.0009]。表中另保留了冻结策略端到端 adapter，用于分别审计“动作接口迁移”与“高层策略迁移”。

## 证据边界与下一阶段

- 当前环境是高可控运动学/动作空间代理，没有接触动力学、真实视觉主干或真机延迟。
- 本实验验证了动作语义可对齐、decoder 可执行和新本体少样本接口，但不直接证明在线 RL 样本效率。
- 若本报告为 CONDITIONAL_GO 或 GO，下一步应在 MuJoCo/Robosuite 三个机械臂上复现 C0/C1，并加入 C3 PPO 与固定动作块预算。
- 若 C1 通过而 C2/C4 失败，应缩小主张为“跨本体动作表示有价值”，不能声称端到端方法成立。

## 审计文件

`c0_protocol.json`、`c1_raw.csv`、`c1_summary.csv`、`c1_seed_summary.csv`、`c2_raw.csv`、`c2_summary.csv`、`c4_adaptation_raw.csv`、`c4_adaptation_summary.csv`、`dataset_index.csv`、`results.json`、`run_manifest.json`、`checkpoints/`、`checksums.sha256`。

源码 SHA-256：`cd82b2d69086ba312c80d25eca0fb8d9e673d77b27c30663f7faea8aa2280781`。
