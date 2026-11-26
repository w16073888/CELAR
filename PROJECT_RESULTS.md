# CELAR 当前正式结论

## 决策

**C0 与正式 C1 通过，可以进入正式 C2；C3/C4 尚未用当前 MuJoCo 数据验证。**

旧二维代理上的 C2/C3/C4 结论仅作历史参考，不再和当前四任务正式流程混在一起。

## C0

正式报告：
[`results/c0_mujoco_v1/README.md`](results/c0_mujoco_v1/README.md)

| 门槛 | 实测 |
|---|---|
| 数据规模 | 3 本体 × 4 任务 × 100 条 = 1,200 轨迹 |
| 重复 reset/step | 12 个本体-任务组合通过 |
| RGB-D 与 EEP 对齐 | 0.66–0.83 px；深度误差 0.018 m |
| 安全越界 | 关节限位和速度越界均为 0 |
| pair/task/scene 泄漏 | 全部为 0 |
| batch 8 显存 | 0.260 GB reserved |
| 动作块延迟 | 0.64–0.85 ms 中位数 |
| 困难样本 | 120 failure + 120 perturb |

## C1

正式报告：
[`results/c1_mujoco_v7/README.md`](results/c1_mujoco_v7/README.md)

| 指标 | CELAR | 共享 AE |
|---|---:|---:|
| Semantic R@1 | **1.000** | 0.974 |
| Exact-pair R@1 | **0.705** | 0.454 |
| Exact-pair R@5 | **0.927** | 0.781 |
| Semantic MAE | 0.0529 | 0.0514 |
| 动作可执行率 | **0.983** | 0.983 |
| Latent 有效秩 | **7.16 / 8** | 7.71 / 8 |
| 最小维标准差 | **0.0946** | 0.1080 |
| 本体语义误差失衡 | **0.0023** | 0.0052 |

相对共享 AE：

- Semantic R@1 提升 **2.6 个百分点**；
- Exact-pair R@1 提升 **25.1 个百分点**，95% CI **[+23.3, +26.1]**；
- Exact-pair R@5 提升 **14.5 个百分点**；
- 本体语义误差失衡更小。

Semantic R@1 已接近饱和，当前主要区分指标为 exact-pair retrieval。

## C1 数据与损失审计

- 训练 6,720 个窗口，验证和测试各 1,440 个窗口；
- train/val/test 分别为 280/60/60 个三本体 pair；
- pair/task/scene 泄漏均为 0；
- 每个 pair/window 的 canonical semantic 在三个本体间完全一致；
- 每个本体的 native action 独立加载，并在第 8 维追加独立 gripper 命令；
- raw CSV 使用 `retrieval_exact_pair_hit` 和 `retrieval_semantic_r1_hit` 等明确字段；
- outcome 损失使用逐样本 progress MSE + 同 pair 三本体预测方差；
- smoothing 只使用同一 episode/body 的 `window_t -> window_t+1`；
- usage 损失使用方差目标和相关性白化，避免 latent collapse。
- `celar_no_align` 只关闭 alignment；其训练历史中 alignment=0、smooth 仍为正，
  不再误删平滑损失；
- C1 loader 的动作 padding/semantic window 长度来自 `data.chunk_length`；
- current 默认输出目录为 `results/c1_mujoco_v7`。

## 代码边界

当前代码：

- [`celar/current/data.py`](celar/current/data.py)
- [`celar/current/training.py`](celar/current/training.py)
- [`celar/current/metrics.py`](celar/current/metrics.py)
- [`celar/current/experiment.py`](celar/current/experiment.py)

当前包不 import 旧二维预实验代码，任务表只有抓取、放置、开抽屉和推物体。

历史三任务代理和旧 C2/C3/C4 位于
[`celar/pre_experiment/`](celar/pre_experiment/)，
其结论见
[`celar/pre_experiment/PROJECT_RESULTS.md`](celar/pre_experiment/PROJECT_RESULTS.md)。

## 下一步

1. 使用当前四任务 MuJoCo 数据训练同本体 BC；
2. 评价隐动作接口的可执行率和任务成功率；
3. 在正式环境运行等预算 PPO；
4. 单独验证 Sawyer 极少样本适配；
5. C2/C3/C4 未在当前环境通过前，不沿用旧代理的阶段通过结论。
