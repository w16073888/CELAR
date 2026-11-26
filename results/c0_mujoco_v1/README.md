# CELAR C0 MuJoCo 工程门槛

**C0_PASS**

| 门槛 | 结果 |
|---|---|
| environment | 通过 |
| alignment | 通过 |
| memory | 通过 |
| latency | 通过 |
| leakage | 通过 |
| pair_consistency | 通过 |
| expert_nominal_success | 通过 |
| schema | 通过 |
| safety | 通过 |
| coverage | 通过 |

- 轨迹数：1200
- 每本体每任务：100 条配对轨迹
- 失败轨迹：120
- 扰动轨迹：120
- 统一初始状态不一致的 pair：0
- 总运行时间：17.9 秒

## RGB-D 对齐

| 本体 | 像素误差 | 深度误差 |
|---|---:|---:|
| panda | 0.74 px | 0.0180 m |
| ur5e | 0.83 px | 0.0180 m |
| sawyer | 0.66 px | 0.0180 m |

## Nominal 专家成功率

| 本体 | 任务 | 成功率 |
|---|---|---:|
| panda | open_drawer | 1.000 |
| sawyer | open_drawer | 1.000 |
| ur5e | open_drawer | 1.000 |
| panda | pick | 1.000 |
| sawyer | pick | 1.000 |
| ur5e | pick | 1.000 |
| panda | place | 1.000 |
| sawyer | place | 1.000 |
| ur5e | place | 1.000 |
| panda | push | 0.838 |
| sawyer | push | 0.825 |
| ur5e | push | 0.812 |

## 显存

- backbone：torchvision_resnet18
- batch：8
- peak reserved：0.260 GB
- 门槛：< 10.5 GB

## 动作块延迟

| 本体 | 中位数 | P95 | 最大 |
|---|---:|---:|---:|
| panda | 0.64 ms | 0.89 ms | 0.94 ms |
| ur5e | 0.74 ms | 0.77 ms | 1.04 ms |
| sawyer | 0.87 ms | 0.88 ms | 0.88 ms |

## 数据边界

该环境是真实 MuJoCo 世界，但机械臂采用最小化统一运动学 MJCF，不声称是厂商原始动力学模型。它适合 C0 数据/渲染/安全/延迟门槛，不替代完整 Robosuite。
