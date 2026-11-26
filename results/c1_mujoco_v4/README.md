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
| celar_no_outcome | 1.000 | 0.716 | 0.2878 | 1.000 | 7.34 | 0.0997 | 0.0294 |
| celar | 0.996 | 0.733 | 0.2877 | 0.994 | 7.06 | 0.0944 | 0.0298 |
| celar_no_cycle | 0.996 | 0.728 | 0.2877 | 0.996 | 7.05 | 0.0933 | 0.0297 |
| celar_shuffled | 0.994 | 0.554 | 0.2878 | 1.000 | 7.41 | 0.0399 | 0.0293 |
| celar_no_align | 0.993 | 0.584 | 0.2877 | 1.000 | 7.37 | 0.1033 | 0.0295 |
| shared_ae | 0.993 | 0.584 | 0.2877 | 1.000 | 7.37 | 0.1033 | 0.0295 |
| independent_ae | 0.563 | 0.023 | 0.2867 | 1.000 | 3.59 | 0.2243 | 0.0285 |

完整 CELAR 相对普通共享 AE 的 semantic R@1 配对提升为 **+0.004**，95% CI [+0.000, +0.006]。
相对无跨本体对齐消融的提升为 **+0.004**。
完整 CELAR 相对普通共享 AE 的 exact-pair R@1 配对提升为 **+0.149**，95% CI [+0.069, +0.200]。

`Semantic R@1` 与 `Exact pair R@1` 是不同指标：前者接受同一语义容差内的动作邻域，后者要求找回完全相同的 C0 配对窗口。raw CSV 中使用`retrieval_semantic_r1_hit` 和 `retrieval_exact_pair_hit`，不再复用同一个模糊字段名。

## 损失实现

- outcome：逐样本 progress MSE + 同 pair 三本体预测方差，直接约束 `pred_progress(z_A) ≈ pred_progress(z_B) ≈ pred_progress(z_C)`。
- smoothing：只构造同一 `(episode, body, window_t, window_t+1)` 的真实相邻窗口，随机 batch 的相邻行不再参与平滑损失。
- cycle：三本体所有有序跨本体 decoder/re-encode 组合。

## 证据边界

C1 使用正式 C0 MuJoCo 数据和三本体评估，但视觉输入为 16×16 下采样 RGB-D，机械臂仍是最小统一 MJCF。不能把该结果代替厂商机械臂动力学、LIBERO 或真机结论。

源码 SHA-256：`3d4ff8f4971b6b3450703bb48c0e188a2c71b0ac15c8752d0b7672920f6ac8f4`。
