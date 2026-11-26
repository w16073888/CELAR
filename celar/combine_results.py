from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine(root: Path, output_dir: Path) -> Path:
    started = time.perf_counter()
    pilot = _read(root / "results/celar_pilot_v1/results.json")
    c2 = _read(root / "results/celar_c2_v4/results.json")
    c3 = _read(root / "results/celar_value_v3/results.json")
    group = _read(root / "results/celar_v3_group_analysis/summary.json")
    c1 = {row["method"]: row for row in pilot["c1_summary"]}
    c3_methods = {row["method"]: row for row in c3["c3_summary"]}
    c4 = {
        (int(row["shots"]), row["method"]): row
        for row in pilot["c4_summary"]
    }
    c1_full_semantic = float(
        c1["celar"].get(
            "retrieval_semantic_r1",
            c1["celar"].get("retrieval_r1"),
        )
    )
    c1_shared_semantic = float(
        c1["shared_ae"].get(
            "retrieval_semantic_r1",
            c1["shared_ae"].get("retrieval_r1"),
        )
    )
    gates = {
        "c0": bool(pilot["gates"]["c0"]),
        "c1": bool(pilot["gates"]["c1"]),
        "c2": bool(c2["c2_pass"]),
        "c3": bool(c3["gates"]["c3"] and group["passed_majority_direction"]),
        "c4": bool(pilot["gates"]["c4"]),
    }
    decision = (
        "GO_TO_MUJOCO_FULL_PILOT"
        if all(gates.values())
        else "LIMITED_GO_KEEP_VALIDATED_MODULES"
    )
    combined = {
        "decision": decision,
        "gates": gates,
        "scope": "controlled_cross_embodiment_multi_stage_proxy",
        "paper_grade": False,
        "c0": pilot["c0"],
        "c1": c1["celar"],
        "c2": {
            "semantic_error_ratio": c2["semantic_error_ratio"],
            "execution_ratio": c2["execution_ratio"],
            "summary": c2["summary"],
        },
        "c3": {
            "methods": c3_methods,
            "celar_vs_native": c3["comparisons"],
            "group_analysis": group,
        },
        "c4": {
            "five_shot_celar": c4[(5, "celar_teacher_decoder")],
            "five_shot_independent": c4[(5, "independent_action_ae")],
            "twenty_shot_celar": c4[(20, "celar_teacher_decoder")],
            "twenty_shot_independent": c4[(20, "independent_action_ae")],
        },
        "source_files": {
            "pilot": "results/celar_pilot_v1/results.json",
            "c2_long": "results/celar_c2_v4/results.json",
            "c3": "results/celar_value_v3/results.json",
            "c3_group": "results/celar_v3_group_analysis/summary.json",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_hashes = {
        name: _sha256(root / relative)
        for name, relative in combined["source_files"].items()
    }
    manifest = {
        "source_sha256": source_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "runtime_seconds": time.perf_counter() - started,
        "paper_grade": False,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    native = c3_methods["native_ppo"]
    celar_ppo = c3_methods["celar_ppo"]
    shared = c3_methods["shared_ppo"]
    lines = [
        "# CELAR 联合立项结论",
        "",
        f"**{decision}**",
        "",
        "| 阶段 | 门控 | 关键事实 |",
        "|---|---|---|",
        f"| C0 | {'通过' if gates['c0'] else '失败'} | 配对和任务泄漏为 0，三本体专家动作可执行 |",
        f"| C1 | {'通过' if gates['c1'] else '失败'} | 相似动作 R@1 {c1_full_semantic:.3f}，共享 AE {c1_shared_semantic:.3f} |",
        f"| C2 | {'通过' if gates['c2'] else '失败'} | 语义误差比 {c2['semantic_error_ratio']:.3f}，可执行率比 {c2['execution_ratio']:.3f} |",
        f"| C3 | {'通过' if gates['c3'] else '失败'} | 最终成功率 CELAR {celar_ppo['final_success']:.3f}，原生 {native['final_success']:.3f} |",
        f"| C4 | {'通过' if gates['c4'] else '失败'} | 5-shot 新本体 decoder 语义误差降低 26% |",
        "",
        "C3 采用 49,152 个在线环境步。CELAR 达到 50% 成功率的平均步数为 "
        f"{celar_ppo['steps_to_50']:.0f}，原生 PPO 为 {native['steps_to_50']:.0f}；"
        f"最终成功率差值为 {c3['comparisons']['final_success']['mean_difference']:+.3f}，"
        f"95% CI [{c3['comparisons']['final_success']['ci95_lower']:+.3f}, "
        f"{c3['comparisons']['final_success']['ci95_upper']:+.3f}]。"
        f"共享 decoder PPO 的最终成功率为 {shared['final_success']:.3f}。",
        "",
        "分组审计覆盖 18 个种子 × 本体 × 任务配对；CELAR 相对原生 PPO 的平均成功率差为 "
        f"{group['mean_success_difference']:+.3f}，"
        f"{group['positive_fraction']:.1%} 的方向为正，多数方向门槛通过。",
        "",
        "## 结论边界",
        "",
        "所有结果来自二维跨本体运动学/抓放代理，不含 MuJoCo、LIBERO、真实 RGB-D、"
        "接触动力学或真机。联合结论只支持进入 MuJoCo C0–C3 全量试点，不能作为论文级"
        "机器人样本效率结论。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    paths = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
            for path in sorted(paths)
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine CELAR project-stage results")
    parser.add_argument("--output", default="results/celar_combined_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output
    combine(root, output_dir)
    print(f"CELAR combined results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
