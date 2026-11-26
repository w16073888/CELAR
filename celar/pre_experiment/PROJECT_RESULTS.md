# CELAR 历史预实验结论

本文件只记录旧二维代理结果，不代表当前四任务 MuJoCo 流程。

## 历史结论

旧代理曾得到：

- C1 合成动作语义检索；
- C2 同本体 latent BC；
- C3 二维抓放 PPO；
- C4 第三本体 decoder 适配；
- 联合代理门控。

这些结果仍保存在 `results/` 的旧目录中，例如：

- `results/celar_pilot_v1/`
- `results/celar_c2_v4/`
- `results/celar_value_v3/`
- `results/celar_v3_group_analysis/`

## 使用限制

- 旧任务表只有 `pick`、`place`、`open`；
- 数据不是正式 C0 MuJoCo NPZ；
- 不包含当前四任务 `open_drawer` 和 `push` 的正式训练；
- 不得把旧 C2/C3/C4 的阶段通过结果写成当前流程结果。

当前正式结论见根目录 [`PROJECT_RESULTS.md`](../../PROJECT_RESULTS.md)。
