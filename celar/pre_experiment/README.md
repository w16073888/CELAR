# CELAR Legacy Pre-Experiment

本目录冻结早期二维跨本体运动学和抓放代理，仅用于复现历史数据。

旧配置特征是：

- 三个任务：`pick`、`place`、`open`；
- 两个训练本体和第三本体适配代理；
- 不使用正式 C0 MuJoCo NPZ；
- C2/C3/C4 结论均来自二维代理。

这些配置不再属于当前四任务正式 C1，也不能被 `celar/current/` 导入。

历史配置位于 [`../../configs/pre_experiment/`](../../configs/pre_experiment/)。

运行示例：

```powershell
python -m celar.pre_experiment.cli --study pilot --config configs/pre_experiment/pilot.yaml
python -m celar.pre_experiment.cli --study c2 --config configs/pre_experiment/c2_v4.yaml
python -m celar.pre_experiment.cli --study value --config configs/pre_experiment/value_v3.yaml
```
