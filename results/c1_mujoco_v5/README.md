# CELAR C1 正式 MuJoCo 实验

**C1_PASS**

> 数据来自正式 `C0_PASS` MuJoCo 数据。C1 同时训练和评估 Panda、UR5e、Sawyer；RGB-D 下采样、本体状态、任务和原生动作块均来自 C0 NPZ。

## 门控

| 项目 | 结果 |
|---|---|
| c0_pass | 通过 |
| three_body_eval | 通过 |
| pairing | 通过 |
| semantic_retrieval | 通过 |
| exact_pair_retrieval | 通过 |
| execution | 通过 |
| latent_usage | 通过 |
| body_balance | 通过 |

- 训练/验证/测试窗口：6720 / 1440 / 1440
- 配对数据：280 train、60 val、60 test
- pair/task/scene 泄漏：0 / 0 / 0
- 三本体 canonical semantic 不一致窗口：0

## 三本体评估

| 方法 | Semantic R@1 | Exact pair R@1 | Semantic MAE | 可执行率 | 有效秩 | Min std | 本体失衡 |
|---|---:|---:|---:|---:|---:|---:|---:|
| celar | 1.000 | 0.705 | 0.0529 | 0.983 | 7.16 | 0.0946 | 0.0023 |
| celar_no_cycle | 1.000 | 0.696 | 0.0519 | 0.983 | 7.18 | 0.0950 | 0.0018 |
| celar_no_outcome | 1.000 | 0.699 | 0.0510 | 0.983 | 7.45 | 0.1039 | 0.0033 |
| celar_no_align | 0.974 | 0.454 | 0.0514 | 0.983 | 7.71 | 0.1080 | 0.0052 |
| shared_ae | 0.974 | 0.454 | 0.0514 | 0.983 | 7.71 | 0.1080 | 0.0052 |
| celar_shuffled | 0.939 | 0.464 | 0.0527 | 0.983 | 7.62 | 0.0470 | 0.0044 |
| independent_ae | 0.481 | 0.006 | 0.0528 | 0.983 | 6.87 | 0.0895 | 0.0058 |

完整 CELAR 相对普通共享 AE 的 semantic R@1 配对提升为 **+0.026**，95% CI [+0.011, +0.056]。
相对无跨本体对齐消融的提升为 **+0.026**。
完整 CELAR 相对普通共享 AE 的 exact-pair R@1 配对提升为 **+0.251**，95% CI [+0.233, +0.261]。

`Semantic R@1` 与 `Exact pair R@1` 是不同指标：前者接受同一语义容差内的动作邻域，后者要求找回完全相同的 C0 配对窗口。raw CSV 中使用`retrieval_semantic_r1_hit` 和 `retrieval_exact_pair_hit`，不再复用同一个模糊字段名。

## 损失实现

- outcome：逐样本 progress MSE + 同 pair 三本体预测方差，直接约束 `pred_progress(z_A) ≈ pred_progress(z_B) ≈ pred_progress(z_C)`。
- smoothing：只构造同一 `(episode, body, window_t, window_t+1)` 的真实相邻窗口，随机 batch 的相邻行不再参与平滑损失。
- cycle：三本体所有有序跨本体 decoder/re-encode 组合。

## 证据边界

C1 使用正式 C0 MuJoCo 数据和三本体评估，但视觉输入为 16×16 下采样 RGB-D，机械臂仍是最小统一 MJCF。不能把该结果代替厂商机械臂动力学、LIBERO 或真机结论。

源码 SHA-256：`6db3534d90b731fdc32b757b4609db82a28fa6f81558db8e5cfbf9a44b6db657`。
