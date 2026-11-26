# CELAR 价值延伸实验 C2/C3

> 本报告是冻结旧版 `celar_pilot_v1` 后新增的代理实验。它检验 latent BC 和等在线预算 PPO，结果仍不是 MuJoCo 或真机论文级证据。

## 结论

- **决策：`NO_GO_ON_POLICY_EXTENSION`**
- 改进后的 C2 latent BC：未通过
- C3 等预算在线 PPO：未通过

## C2：标准化 latent BC

| 方法 | 语义 MAE | 可执行率 | Jerk |
|---|---:|---:|---:|
| native_bc | 0.2732 | 0.733 | 0.1214 |
| celar_bc | 0.2862 | 0.575 | 0.0503 |

CELAR BC / 原生 BC 的语义误差比为 **1.048**；门槛为不高于 1.15。

## C3：同预算 PPO

| 方法 | 最终成功率 | 50% 步数 | 80% 步数 |
|---|---:|---:|---:|
| native_ppo | 0.174 | 24576 | 24576 |
| shared_ppo | 0.194 | 24576 | 24576 |
| celar_ppo | 0.194 | 24576 | 24576 |

所有方法使用相同 24,576 个在线环境步、相同 PPO 超参数、同一环境随机种子和同一组 Panda/UR5e 本体。CELAR 只冻结预训练动作 decoder；策略网络仍从头学习。
相对于原生 PPO，CELAR 达到 50% 成功率的步数变化为 **+0.0%**。
最终成功率配对差值：`{'n_pairs': 3, 'mean_difference': 0.020833333333333343, 'ci95_lower': -0.20833333333333334, 'ci95_upper': 0.16666666666666666, 'paired_t_p': 0.8740118423302575}`。

## 证据边界

- 环境是二维抓放控制代理，奖励和状态显式可观测。
- 策略没有视觉主干、接触动力学、IK 或真实控制延迟。
- C3 只能判断动作 decoder 先验是否在该代理中产生在线学习优势，不能代替 LIBERO/Robosuite。
- 若 C3 未通过，下一阶段应保留 C1/C4 的动作表示价值，但先修复策略条件化，不能扩大论文主张。

## 审计文件

`c2_raw.csv`、`c2_summary.csv`、`c3_seed_summary.csv`、`c3_curves.csv`、`results.json`、`run_manifest.json`、`checkpoints/`、`checksums.sha256`。

源码 SHA-256：`27b325b7e87369a7c47355cf90fc9aa31e3f9ce54de52b9e33ab8f8e8413e658`。
