from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .environment import UnifiedManipulationEnv
from .generate import GenerationConfig, generate_dataset
from .schema import ROBOT_SPECS, ROBOTS, TASK_FAMILIES, make_scene_spec


@dataclass
class C0Config:
    pairs_per_task: int = 100   #每个机器人每个任务多少轨迹
    render_size: int = 128      #渲染图像的分辨率
    seed: int = 20261106        #随机种子，生成初始环境用场景
    output_dir: str = "results/c0_mujoco_v1"    
    memory_batch_size: int = 8      #试测会不会爆显存的测试批次大小
    memory_limit_gb: float = 10.5   #试测显存的峰值规定
    latency_trials: int = 20        #试测动作块执行延迟的取样次数
    max_median_latency_ms: float = 180.0    #动作块执行延迟的中位数门槛
    max_p95_latency_ms: float = 300.0       #动作块执行延迟的95%分位数门槛

    def to_generation(self) -> GenerationConfig:    #将C0Config转换为GenerationConfig
        return GenerationConfig(
            pairs_per_task=self.pairs_per_task,
            render_size=self.render_size,
            output_dir=self.output_dir,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, Any]:        #将配置转换为字典形式
        return asdict(self)


def _sha256(path: Path) -> str:     #给文件做哈希校验完整性
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:    #将字典或其他可序列化对象写入JSON文件
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _scene_for(robot_index: int, task_index: int, seed: int):   #根据机器人索引和任务索引生成场景参数信息对象
    return make_scene_spec(
        family=TASK_FAMILIES[task_index],   #schema模块的任务族列表
        pair_index=17 + robot_index * 3,    #每个任务的轨迹索引
        total_pairs=100,                    #每个任务的总轨迹数
        seed=seed + task_index * 101,       #随机种子
    )


def _preflight_environment(config: C0Config) -> dict[str, Any]:     #测试reset和step的稳定性，返回一个字典，包含是否通过和每个机器人每个任务的测试结果
    rows: list[dict[str, Any]] = []
    for robot_index, robot in enumerate(ROBOTS):    #enmerate遍历所有(机器人索引，机器人)
        for task_index, task in enumerate(TASK_FAMILIES):
            scene = _scene_for(robot_index, task_index, config.seed)
            env = UnifiedManipulationEnv(       #这是统一 MuJoCo 操作环境的 Python 类的构造方法，这个类用于把 Panda、UR5e、Sawyer 三种机械臂放进相同任务、场景和初始状态下执行实验
                robot,
                render_size=config.render_size,
            )
            try:
                env.reset(scene)
                first = np.asarray(env.data.qpos).copy()
                env.reset(scene)
                second = np.asarray(env.data.qpos).copy()
                reset_error = float(np.max(np.abs(first - second)))         #测试reset的稳定性,用两次的相减看差看位姿之差是否为0，qpos是机械臂的关节位置向量
                
                action = np.zeros(      #机械臂关节的零动作(不动)构造
                    (   #从ROB全OT_SPECS字典中获取该机器人对应的动作块长度和机械臂自由度
                        ROBOT_SPECS[robot].action_chunk_length,     #动作块长度
                        ROBOT_SPECS[robot].arm_dof,                 #机械臂自由度
                    ),
                    dtype=np.float64,
                )
                gripper = np.zeros(ROBOT_SPECS[robot].action_chunk_length)  #机械臂夹爪零动作构造
                
                result = env.step_chunk(action, gripper)
                state_after_step = np.asarray(env.data.qpos).copy()     #获得step之后的机械臂关节位置向量
                env.reset(scene)
                repeated = env.step_chunk(action, gripper)
                repeated_state = np.asarray(env.data.qpos).copy()       #再reset之后再step一次，获得机械臂关节位置向量
                step_error = float(
                    np.max(np.abs(state_after_step - repeated_state))
                )

                rows.append(            #记录与判定
                    {
                        "robot": robot,
                        "task_family": task,
                        "reset_error": reset_error,
                        "repeat_step_error": step_error,
                        "passed": bool(reset_error <= 1e-10 and step_error <= 1e-6),
                        "rgb_shape": list(result.rgb.shape),
                        "depth_shape": list(result.depth.shape),
                    }
                )
            finally:
                env.close()
    #包含两个键的字典：passed（总体结论）和 trials（逐项结论）
    return {"passed": all(row["passed"] for row in rows),   #生成器，表示rows里面的每一项的passed都是True，all()函数返回True，否则返回False
             "trials": rows}    #返回测评明细


def _alignment_preflight(config: C0Config) -> dict[str, Any]:   #测试RGB-D对齐的稳定性，返回一个字典，包含是否通过和每个机器人每个任务的测试结果
    rows: list[dict[str, Any]] = []
    preview_dir = Path(config.output_dir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)              #创建预览图像的目录
    for robot_index, robot in enumerate(ROBOTS):                #遍历所有机器人
        scene = _scene_for(robot_index, 0, config.seed)
        env = UnifiedManipulationEnv(
            robot,
            render_size=config.render_size,
        )
        try:
            env.reset(scene)
            rgb, depth, intrinsics, extrinsics, _ = env.render_rgbd(    
#intrinsics：内参3×3，把相机坐标系中的三维点投影成图像像素，不负责世界坐标到相机坐标的变换。
#extrinsics：外参4×4，表示相机在世界坐标系中的旋转和平移。当前实现中保存的是相机位姿，世界点转相机点要使用其逆变换。
#第五个参数用_表示不需要
                include_segmentation=True   #设置：除了RGB和深度，再渲染一张逐像素分割图。这就是我们没有赋值的第五个返回值
            )
            check = env.rgbd_alignment_check()  #由环境自己实现的对齐检查
            
            Image.fromarray(rgb).save(preview_dir / f"{robot}_rgb.png")
            depth_vis = np.clip(depth / 1.5, 0.0, 1.0)
            Image.fromarray((depth_vis * 255).astype(np.uint8)).save(
                preview_dir / f"{robot}_depth.png"
            )
            np.savez_compressed(
                preview_dir / f"{robot}_rgbd.npz",
                rgb=rgb,
                depth=depth.astype(np.float32),
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
            rows.append({"robot": robot, **check})
        finally:
            env.close()
    return {"passed": all(row["passed"] for row in rows), "trials": rows}


def _latency_preflight(config: C0Config) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for robot_index, robot in enumerate(ROBOTS):
        scene = _scene_for(robot_index, 1, config.seed)
        env = UnifiedManipulationEnv(
            robot,
            render_size=config.render_size,
        )
        try:
            env.reset(scene)
            spec = ROBOT_SPECS[robot]
            action = np.zeros(
                (spec.action_chunk_length, spec.arm_dof), dtype=np.float64
            )
            gripper = np.zeros(spec.action_chunk_length, dtype=np.float64)
            for _ in range(3):
                env.step_chunk(action, gripper)
            durations: list[float] = []
            for _ in range(config.latency_trials):
                env.reset(scene)
                started = time.perf_counter()
                env.step_chunk(action, gripper)
                durations.append((time.perf_counter() - started) * 1000.0)
            values = np.asarray(durations)
            row = {
                "robot": robot,
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.quantile(values, 0.95)),
                "max_ms": float(values.max()),
            }
            row["passed"] = bool(
                row["median_ms"] <= config.max_median_latency_ms
                and row["p95_ms"] <= config.max_p95_latency_ms
            )
            rows.append(row)
        finally:
            env.close()
    return {"passed": all(row["passed"] for row in rows), "trials": rows}


class _FallbackRgbdModel(torch.nn.Module):
    def __init__(self, action_dim: int = 8, chunk_length: int = 8) -> None:
        super().__init__()
        self.visual = torch.nn.Sequential(
            torch.nn.Conv2d(4, 24, 5, stride=2, padding=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(24, 48, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(48, 96, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(96, 192),
        )
        self.action = torch.nn.Sequential(
            torch.nn.Linear(action_dim * chunk_length, 192),
            torch.nn.SiLU(),
            torch.nn.Linear(192, 192),
        )
        self.project = torch.nn.Sequential(
            torch.nn.Linear(384, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 8),
        )

    def forward(
        self, rgbd: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.project(
            torch.cat(
                [self.visual(rgbd), self.action(actions.reshape(len(actions), -1))],
                dim=-1,
            )
        )


def _memory_probe(config: C0Config) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "passed": False,
            "blocked_reason": "CUDA is unavailable in the execution environment",
        }
    device = torch.device("cuda")
    try:
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        model.conv1 = torch.nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(512, 8)
        backbone = "torchvision_resnet18"
    except Exception:
        model = _FallbackRgbdModel()
        backbone = "builtin_cnn_fallback"
    model = model.to(device)
    rgbd = torch.randn(
        config.memory_batch_size,
        4,
        config.render_size,
        config.render_size,
        device=device,
    )
    actions = torch.randn(
        config.memory_batch_size,
        8,
        8,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        features = model(rgbd)
        # Include the action branch's tensor budget without requiring a second model.
        loss = features.square().mean() + actions.float().square().mean() * 0.01
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    return {
        "passed": bool(peak_reserved < config.memory_limit_gb),
        "backbone": backbone,
        "batch_size": config.memory_batch_size,
        "resolution": [config.render_size, config.render_size],
        "peak_allocated_gb": float(peak_allocated),
        "peak_reserved_gb": float(peak_reserved),
        "limit_gb": config.memory_limit_gb,
        "step_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(device),
    }


def _leakage_audit(
    manifest: pd.DataFrame,
    expected_pairs_per_task: int,
) -> dict[str, Any]:
    groups = {
        name: {
            split: set(manifest[manifest["split"] == split][name].astype(str))
            for split in ("train", "val", "test")
        }
        for name in ("pair_id", "task_id", "scene_id")
    }
    overlaps: dict[str, int] = {}
    for name, split_sets in groups.items():
        overlaps[name] = int(
            len(split_sets["train"] & split_sets["val"])
            + len(split_sets["train"] & split_sets["test"])
            + len(split_sets["val"] & split_sets["test"])
        )
    counts = (
        manifest.groupby(["task_family", "robot"])
        .size()
        .rename("trajectories")
        .reset_index()
    )
    minimum_count = max(1, int(np.floor(0.80 * expected_pairs_per_task)))
    failure_count = int((manifest["failure_mode"] == "failure").sum())
    perturbed_count = int((manifest["failure_mode"] == "perturb").sum())
    requires_difficult_data = expected_pairs_per_task >= 10
    return {
        "passed": bool(
            all(value == 0 for value in overlaps.values())
            and bool((counts["trajectories"] >= minimum_count).all())
            and (not requires_difficult_data or (failure_count > 0 and perturbed_count > 0))
        ),
        "overlaps": overlaps,
        "counts": counts.to_dict(orient="records"),
        "failure_trajectories": failure_count,
        "perturbed_trajectories": perturbed_count,
        "minimum_per_task_body": minimum_count,
    }


def _pair_consistency_audit(manifest: pd.DataFrame) -> dict[str, Any]:
    inconsistent: list[str] = []
    for pair_id, group in manifest.groupby("pair_id"):
        if int(group["initial_state_sha256"].nunique()) != 1:
            inconsistent.append(str(pair_id))
    return {
        "passed": bool(not inconsistent and len(manifest.groupby("pair_id")) > 0),
        "pair_count": int(manifest["pair_id"].nunique()),
        "inconsistent_pairs": inconsistent[:20],
        "inconsistent_pair_count": len(inconsistent),
    }


def _success_audit(manifest: pd.DataFrame) -> dict[str, Any]:
    nominal = manifest[manifest["failure_mode"] == "nominal"]
    table = (
        nominal.groupby(["task_family", "robot"])["success"]
        .mean()
        .rename("success_rate")
        .reset_index()
    )
    return {
        "passed": bool(
            len(table) == len(ROBOTS) * len(TASK_FAMILIES)
            and (table["success_rate"] >= 0.80).all()
        ),
        "nominal_by_task_body": table.to_dict(orient="records"),
        "all_trajectories_by_task_body": (
            manifest.groupby(["task_family", "robot"])["success"]
            .mean()
            .rename("success_rate")
            .reset_index()
            .to_dict(orient="records")
        ),
    }


def _data_schema_audit(
    manifest: pd.DataFrame,
    root: Path,
    config: C0Config,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    required = {
        "rgb",
        "depth",
        "native_action",
        "gripper_action",
        "eef_pose",
        "arm_qpos",
        "arm_qvel",
        "object_pose",
        "drawer_position",
        "progress",
        "success",
        "safety_pre_limit_violation",
        "safety_pre_velocity_violation",
        "safety_post_limit_violation",
        "safety_post_velocity_violation",
        "contact_count",
        "intrinsics",
        "extrinsics",
        "initial_qpos",
    }
    for robot in ROBOTS:
        row = manifest[manifest["robot"] == robot].iloc[0]
        with np.load(root / "trajectories" / row["trajectory_file"]) as data:
            missing = sorted(required - set(data.files))
            rows.append(
                {
                    "robot": robot,
                    "missing_keys": missing,
                    "rgb_shape": list(data["rgb"].shape),
                    "depth_shape": list(data["depth"].shape),
                    "native_action_shape": list(data["native_action"].shape),
                    "chunk_length": int(data["native_action"].shape[1]),
                    "passed": bool(
                        not missing
                        and data["rgb"].shape[-3:] == (
                            config.render_size,
                            config.render_size,
                            3,
                        )
                        and data["depth"].shape[-2:] == (
                            config.render_size,
                            config.render_size,
                        )
                    ),
                }
            )
    return {"passed": all(row["passed"] for row in rows), "robots": rows}


def _report(
    *,
    config: C0Config,
    environment: dict[str, Any],
    alignment: dict[str, Any],
    memory: dict[str, Any],
    latency: dict[str, Any],
    leakage: dict[str, Any],
    pair_consistency: dict[str, Any],
    success: dict[str, Any],
    schema: dict[str, Any],
    manifest: pd.DataFrame,
    runtime: float,
) -> str:
    gates = {
        "environment": environment["passed"],
        "alignment": alignment["passed"],
        "memory": memory["passed"],
        "latency": latency["passed"],
        "leakage": leakage["passed"],
        "pair_consistency": pair_consistency["passed"],
        "expert_nominal_success": success["passed"],
        "schema": schema["passed"],
        "safety": bool(
            manifest["post_limit_violation_count"].sum() == 0
            and manifest["post_velocity_violation_count"].sum() == 0
        ),
        "coverage": bool(
            len(manifest) == len(ROBOTS) * len(TASK_FAMILIES) * config.pairs_per_task
        ),
    }
    decision = "C0_PASS" if all(gates.values()) else "C0_BLOCKED"
    lines = [
        "# CELAR C0 MuJoCo 工程门槛",
        "",
        f"**{decision}**",
        "",
        "| 门槛 | 结果 |",
        "|---|---|",
        *[
            f"| {name} | {'通过' if passed else '未通过'} |"
            for name, passed in gates.items()
        ],
        "",
        f"- 轨迹数：{len(manifest)}",
        f"- 每本体每任务：{config.pairs_per_task} 条配对轨迹",
        f"- 失败轨迹：{leakage['failure_trajectories']}",
        f"- 扰动轨迹：{leakage['perturbed_trajectories']}",
        f"- 统一初始状态不一致的 pair：{pair_consistency['inconsistent_pair_count']}",
        f"- 总运行时间：{runtime:.1f} 秒",
        "",
        "## RGB-D 对齐",
        "",
        "| 本体 | 像素误差 | 深度误差 |",
        "|---|---:|---:|",
        *[
            f"| {row['robot']} | {row['pixel_error']:.2f} px | "
            f"{row.get('depth_error', float('nan')):.4f} m |"
            for row in alignment["trials"]
        ],
        "",
        "## Nominal 专家成功率",
        "",
        "| 本体 | 任务 | 成功率 |",
        "|---|---|---:|",
        *[
            f"| {row['robot']} | {row['task_family']} | "
            f"{row['success_rate']:.3f} |"
            for row in success["nominal_by_task_body"]
        ],
        "",
        "## 显存",
        "",
        f"- backbone：{memory.get('backbone', 'unavailable')}",
        f"- batch：{memory.get('batch_size', 'unavailable')}",
        f"- peak reserved：{memory.get('peak_reserved_gb', float('nan')):.3f} GB",
        f"- 门槛：< {config.memory_limit_gb:.1f} GB",
        "",
        "## 动作块延迟",
        "",
        "| 本体 | 中位数 | P95 | 最大 |",
        "|---|---:|---:|---:|",
        *[
            f"| {row['robot']} | {row['median_ms']:.2f} ms | "
            f"{row['p95_ms']:.2f} ms | {row['max_ms']:.2f} ms |"
            for row in latency["trials"]
        ],
        "",
        "## 数据边界",
        "",
        "该环境是真实 MuJoCo 世界，但机械臂采用最小化统一运动学 MJCF，不声称是厂商"
        "原始动力学模型。它适合 C0 数据/渲染/安全/延迟门槛，不替代完整 Robosuite。",
        "",
    ]
    return "\n".join(lines)


def run_c0(config: C0Config) -> Path:
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    environment = _preflight_environment(config)
    if not environment["passed"]:
        _write_json(output_dir / "preflight_failure.json", environment)
        raise RuntimeError("Reset/step preflight failed")
    alignment = _alignment_preflight(config)
    if not alignment["passed"]:
        _write_json(output_dir / "preflight_failure.json", alignment)
        raise RuntimeError("RGB-D alignment preflight failed")
    memory = _memory_probe(config)
    latency = _latency_preflight(config)
    if not latency["passed"]:
        _write_json(output_dir / "preflight_failure.json", latency)
        raise RuntimeError("Latency preflight failed")
    print("[C0] generating paired rollouts")
    manifest = generate_dataset(config.to_generation(), output_dir=output_dir)
    leakage = _leakage_audit(manifest, config.pairs_per_task)
    pair_consistency = _pair_consistency_audit(manifest)
    success_audit = _success_audit(manifest)
    schema = _data_schema_audit(manifest, output_dir, config)
    safety = {
        "post_limit_violation_count": int(
            manifest["post_limit_violation_count"].sum()
        ),
        "post_velocity_violation_count": int(
            manifest["post_velocity_violation_count"].sum()
        ),
    }
    gates = {
        "environment_reset_step": environment["passed"],
        "rgbd_pose_alignment": alignment["passed"],
        "expert_action_safety": bool(
            safety["post_limit_violation_count"] == 0
            and safety["post_velocity_violation_count"] == 0
        ),
        "split_leakage": leakage["passed"],
        "pair_initial_state_consistency": pair_consistency["passed"],
        "expert_nominal_success": success_audit["passed"],
        "batch8_memory": memory["passed"],
        "chunk_latency": latency["passed"],
        "schema": schema["passed"],
        "coverage": bool(
            len(manifest) == len(ROBOTS) * len(TASK_FAMILIES) * config.pairs_per_task
        ),
    }
    runtime = time.perf_counter() - started
    results = {
        "decision": "C0_PASS" if all(gates.values()) else "C0_BLOCKED",
        "gates": gates,
        "config": config.to_dict(),
        "environment": environment,
        "alignment": alignment,
        "memory": memory,
        "latency": latency,
        "leakage": leakage,
        "pair_consistency": pair_consistency,
        "success_audit": success_audit,
        "safety": safety,
        "schema": schema,
        "manifest_rows": len(manifest),
        "runtime_seconds": runtime,
        "simulator": "minimal_unified_mujoco_kinematic_mjcf",
        "paper_grade": False,
    }
    _write_json(output_dir / "c0_results.json", results)
    report = _report(
        config=config,
        environment=environment,
        alignment=alignment,
        memory=memory,
        latency=latency,
        leakage=leakage,
        pair_consistency=pair_consistency,
        success=success_audit,
        schema=schema,
        manifest=manifest,
        runtime=runtime,
    )
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    manifest_hash = _sha256(output_dir / "trajectory_manifest.csv")
    run_manifest = {
        "config": config.to_dict(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count(),
        },
        "trajectory_manifest_sha256": manifest_hash,
        "runtime_seconds": runtime,
        "paper_grade": False,
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)
    checksums = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(
                f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
            )
    (output_dir / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )
    print(f"CELAR C0 results written to {output_dir.resolve()}")
    return output_dir
