# CELAR 立项判断

## 最新联调结论

**进入 MuJoCo 全量试点：`GO_TO_MUJOCO_FULL_PILOT`。**

跨本体动作对齐、长训练 latent BC、等预算在线 PPO 和未见本体少样本 decoder 适配
在受控代理中同时达到预注册门槛。该结论足以支持进入 MuJoCo/Robosuite 全量试点，
但**不等于**已经证明真实机器人或论文级在线样本效率。

联合自动报告见
[`results/celar_combined_v1/README.md`](results/celar_combined_v1/README.md)，
联合门控与结构化结果见
[`results/celar_combined_v1/results.json`](results/celar_combined_v1/results.json)。

C0 已在远端 RTX 3060 12 GB 服务器上完成真实 MuJoCo 渲染与轨迹生成，正式报告见
[`results/c0_mujoco_v1/README.md`](results/c0_mujoco_v1/README.md)。

| C0 门槛 | 实测 |
|---|---|
| 数据规模 | 1,200 条轨迹；3 本体 × 4 任务 × 100 条 |
| 重复 reset/step | 12 个本体-任务组合全部通过 |
| RGB-D 与 EEP 对齐 | 像素误差 0.66–0.83 px；深度误差 0.018 m |
| 动作安全 | 关节限位/速度后越界均为 0 |
| 数据泄漏 | pair/task/scene 重叠均为 0 |
| 配对初始状态 | 120 个任务内场景配对完全一致 |
| batch 8 显存 | ResNet-18 级 RGB-D 前反向峰值 reserved 0.260 GB |
| 动作块延迟 | Panda 0.66 ms、UR5e 0.73 ms、Sawyer 0.85 ms 中位数 |
| 失败/扰动数据 | 120 条 failure + 120 条 perturb |

该环境是真实 MuJoCo 刚体世界，但机械臂使用最小化统一运动学 MJCF，不是厂商原始
动力学资产。它已通过 C0 数据/渲染/安全/资源门槛，下一阶段可以开始视觉动作模型，
但若论文需要真实机器人动力学，仍应在后续替换为 Robosuite/Menagerie 厂商模型。

## C1 正式 MuJoCo 三本体结果

正式 C1 使用 C0 的 1,200 条轨迹，训练 6,720 个窗口，验证和测试各 1,440 个窗口。
同一 `pair_id` 的三本体共享场景、初始状态、目标和 canonical semantic stage；
native action 和 decoder 仍保留本体差异。三本体同时训练和评估，结果为
**`C1_PASS`**：

[`results/c1_mujoco_v4/README.md`](results/c1_mujoco_v4/README.md)

| 指标 | 完整 CELAR | 共享 AE |
|---|---:|---:|
| Semantic R@1 | **0.996** | 0.993 |
| Exact-pair R@1 | **0.733** | 0.584 |
| Exact-pair R@5 | **0.920** | 0.833 |
| 动作可执行率 | **0.994** | 1.000 |
| Latent 有效秩 | **7.06 / 8** | 7.37 / 8 |
| 本体语义误差失衡 | **0.0298** | 0.0295 |

Exact-pair R@1 相对共享 AE 的配对提升为 **+0.149**，95% CI **[+0.069, +0.200]**。
Semantic R@1 已接近饱和，因此 C1 的主要区分指标采用 exact-pair retrieval，同时
保留 semantic R@1/5 作为动作邻域召回指标。

C1 修正项：

- 合成数据场景种子不再含 `body_index`；同一 pair 的初始位置、目标、语义控制、
  progress 和 failure 状态在构造前共享。
- 配对 tensor 从每个本体自己的 `semantic_action` 堆叠，不再复制 body 0 的语义。
- formal C0 loader 为每个 pair/window 重建完全一致的 canonical task-stage 语义计划，
  三本体 semantic mismatch 数为 0。
- raw CSV 使用 `retrieval_exact_pair_hit`、`retrieval_semantic_r1_hit` 和
  `retrieval_semantic_r5_hit`；summary 使用不同字段名，不再出现同名不同定义。
- outcome 使用逐样本 progress MSE 加同 pair 三本体预测方差，直接约束
  `pred_progress(z_A) ≈ pred_progress(z_B) ≈ pred_progress(z_C)`。
- smoothing 只使用 `(episode, body, window_t, window_t+1)` 的真实相邻窗口。
- latent usage 使用方差目标和相关性白化，最终有效秩从初版 1.35 提升到 7.06。

| 阶段 | 结论 | 关键证据 |
|---|---|---|
| C0 配对数据与安全动作 | 通过 | 1,200 条 MuJoCo 轨迹；泄漏和安全越界均为 0 |
| C1 跨本体潜在动作 | 通过 | 相似动作 R@1 0.911；普通共享 AE 0.550 |
| C2 同本体 latent BC | 通过 | 长训练语义误差比 1.107；可执行率比 0.860 |
| C3 已知本体 PPO | 通过 | 最终成功率 71.5%；原生动作 PPO 37.5% |
| C4 新本体动作接口 | 通过 | 5-shot decoder 语义 MAE 降低 26%，可执行率 100% |

## C1：跨本体动作对齐

3 个随机种子、1,944 个训练配对动作块、60 对完全留出测试动作块，比较 7 种表示。

- 完整 CELAR 相似动作 R@1：**0.911**，95% CI **[0.767, 1.000]**；
- R@5：**0.967**，95% CI **[0.900, 1.000]**；
- 普通共享 AE R@1：**0.550**；
- 去掉跨本体对齐 R@1：**0.550**；
- 打乱配对负对照 R@1：**0.378**；
- 相对普通共享 AE 的配对提升：**+0.361**，95% CI **[+0.300, +0.400]**；
- 安全层后动作可执行率：**0.981**；
- 8 维 latent 有效秩：**4.01**，没有塌缩为常量。

Panda、UR5e、Sawyer 分别使用 7、6、7 维独立 decoder。结果不是把不同关节向量
直接拼接得到的；跨本体对齐损失是检索增益的主要来源。

正式数据：
[`results/celar_pilot_v1`](results/celar_pilot_v1)。

## C2：同本体行为克隆

初版 C2 失败，之后将 latent 目标标准化并加入冻结 decoder-aware 动作重建，
训练 800 epoch。3 个种子、每种子 2 个本体的结果为：

| 方法 | 语义 MAE | 可执行率 | Jerk |
|---|---:|---:|---:|
| 原生动作 BC | 0.2553 | 0.733 | 0.1007 |
| CELAR latent BC | 0.2827 | 0.631 | 0.0478 |

CELAR 语义误差高 **10.7%**，低于 15% 上限；可执行率为原生 BC 的 **86.0%**，
高于 80% 下限；Jerk 降低约 **53%**。因此 C2 通过，但属于“可接受的低成本接口”，
不是动作精度优于原生 BC。

正式数据：
[`results/celar_c2_v4`](results/celar_c2_v4)。

## C3：同预算在线 PPO

三种方法使用相同 49,152 个在线环境步、PPO 超参数、环境种子和 Panda/UR5e 本体。
CELAR 与普通共享 decoder 使用预训练动作 decoder，但策略网络都从头学习。

| 方法 | 最终成功率 | 达到 50% 的平均步数 |
|---|---:|---:|
| 原生动作 PPO | 0.375 | 42,325 |
| 普通共享 decoder PPO | 0.535 | 40,960 |
| CELAR PPO | **0.715** | **36,864** |

CELAR 相对原生 PPO：

- 最终成功率提高 **34.0 个百分点**；
- 配对差值 95% CI **[+22.9, +39.6]** 个百分点，配对 t 检验 p=0.026；
- 达到 50% 成功率所需在线步数减少 **12.9%**；
- 相对普通共享 decoder 仍高 18.1 个百分点，说明收益不只来自“有一个预训练 decoder”。

固定本体、固定任务的后续审计包含 18 个“种子 × 本体 × 任务”配对：

- 平均成功率提升 **29.4 个百分点**；
- 95% CI **[+17.8, +40.5]**；
- **83.3%** 的配对方向为正；
- 6 个本体-任务组合中 **5 个** 的三种子平均成功率高于原生 PPO。

正式数据：
[`results/celar_value_v3`](results/celar_value_v3)，分组审计见
[`results/celar_v3_group_analysis`](results/celar_v3_group_analysis)。

## C4：未见第三机械臂

Sawyer 未参与共享动作预训练。冻结动作编码器、仅训练新本体 decoder：

| 演示预算 | CELAR decoder | 独立动作 AE |
|---:|---:|---:|
| 5-shot 语义 MAE | **0.1110** | 0.1500 |
| 20-shot 语义 MAE | **0.1085** | 0.1153 |

5-shot 语义误差降低 **26.0%**，5/20-shot 的可执行率均为 **100%**，三个种子的
5/20-shot 比较方向均为正。原始数据：
[`results/celar_pilot_v1`](results/celar_pilot_v1)。

端到端“冻结策略 + decoder/adapter”仍失败：20-shot 语义 MAE 0.4798，独立 BC
0.3209。因此 C4 支持的是**动作接口和 decoder 适配**，不支持高层策略跨本体直接迁移。

## 为什么方向值得立项

1. C1 证明不同动作维度可形成可消融、可检索的共享动作语义。
2. C2 证明 latent 接口在长训练后可保持在原生 BC 的可用误差和可执行性范围。
3. C3 在等在线预算下同时提高最终成功率、减少达标步数，并在多数本体-任务组合中保持正向。
4. C4 证明新本体可用极少演示适配 decoder，不是简单撞上动作维度。
5. 正向证据均来自多种子；失败项和负向边界均保留。

## MuJoCo 全量试点的最低要求

1. 在 MuJoCo/Robosuite 复现 Panda、UR5e、Sawyer 的相同任务配对数据。
2. 使用真实 RGB-D/本体状态，替换显式运动学观测。
3. 保留 C1 的 exact-pair、相似动作、循环、后果和安全动作指标。
4. 加入每本体独立策略、原生关节共享策略、末端 delta+IK、普通 action VAE 和共享 decoder 基线。
5. 在固定交互预算下验证 C3，并至少运行 3 个种子和 4 个任务；主门槛仍是成功率 +5 点或达标步数 -30%。
6. 新本体同时报告 teacher-latent decoder 和冻结策略闭环，防止动作接口收益被误写成任务成功率。
7. 只有在模型、数据切分、随机种子和完整日志可复现后，才能扩大论文主张。

## 证据边界

当前数据是高可控的二维跨本体运动学/抓放代理，不包含 MuJoCo、LIBERO、真实 RGB-D、
接触动力学、IK、控制延迟或真机。联合结果 `paper_grade=false`。所有结论只支持
“进入 MuJoCo 全量试点”，不能直接作为论文级机器人结果引用。

旧版初筛结论仍完整保存在 `results/celar_pilot_v1`，本次扩展没有覆盖其原始 CSV、
checkpoint、图表或校验和。
