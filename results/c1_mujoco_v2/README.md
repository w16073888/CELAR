# CELAR C1 正式 MuJoCo 实验

**C1_BLOCKED**

> 数据来自正式 `C0_PASS` MuJoCo 数据。C1 同时训练和评估 Panda、UR5e、Sawyer；RGB-D 下采样、本体状态、任务和原生动作块均来自 C0 NPZ。

## 门控

| 项目 | 结果 |
|---|---|
| c0_pass | 通过 |
| three_body_eval | 通过 |
| pairing | 通过 |
| semantic_retrieval | 未通过 |
| execution | 通过 |
| latent_usage | 未通过 |
| body_balance | 通过 |

- 训练/验证/测试窗口：6720 / 1440 / 1440
- 配对数据：280 train、60 val、60 test
- pair/task/scene 泄漏：0 / 0 / 0
- 三本体 canonical semantic 不一致窗口：0

## 三本体评估

| 方法 | Semantic R@1 | Exact pair R@1 | Semantic MAE | 可执行率 | 有效秩 | 本体失衡 |
|---|---:|---:|---:|---:|---:|---:|
| celar_no_cycle | 0.994 | 0.704 | 0.2876 | 1.000 | 1.50 | 0.0287 |
| celar_no_align | 0.991 | 0.477 | 0.2877 | 1.000 | 1.57 | 0.0293 |
| shared_ae | 0.991 | 0.477 | 0.2877 | 1.000 | 1.57 | 0.0293 |
| celar | 0.983 | 0.716 | 0.2877 | 1.000 | 1.35 | 0.0293 |
| celar_shuffled | 0.980 | 0.605 | 0.2873 | 1.000 | 1.95 | 0.0287 |
| celar_no_outcome | 0.978 | 0.705 | 0.2875 | 1.000 | 1.02 | 0.0289 |
| independent_ae | 0.500 | 0.025 | 0.2879 | 1.000 | 3.63 | 0.0295 |

完整 CELAR 相对普通共享 AE 的 semantic R@1 配对提升为 **-0.007**，95% CI [-0.017, +0.000]。
相对无跨本体对齐消融的提升为 **-0.007**。

`Semantic R@1` 与 `Exact pair R@1` 是不同指标：前者接受同一语义容差内的动作邻域，后者要求找回完全相同的 C0 配对窗口。raw CSV 中使用`retrieval_semantic_r1_hit` 和 `retrieval_exact_pair_hit`，不再复用同一个模糊字段名。

## 损失实现

- outcome：逐样本 progress MSE + 同 pair 三本体预测方差，直接约束 `pred_progress(z_A) ≈ pred_progress(z_B) ≈ pred_progress(z_C)`。
- smoothing：只构造同一 `(episode, body, window_t, window_t+1)` 的真实相邻窗口，随机 batch 的相邻行不再参与平滑损失。
- cycle：三本体所有有序跨本体 decoder/re-encode 组合。

## 证据边界

C1 使用正式 C0 MuJoCo 数据和三本体评估，但视觉输入为 16×16 下采样 RGB-D，机械臂仍是最小统一 MJCF。不能把该结果代替厂商机械臂动力学、LIBERO 或真机结论。

源码 SHA-256：`7a825b69120ce4ce392a8a5e294cf4bca9441cabaa6a77ae668067c02de10f47`。
