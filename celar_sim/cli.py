from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .c0 import C0Config, run_c0


def _load(path: Path) -> C0Config:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = set(C0Config.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown C0 fields: {sorted(unknown)}")
    return C0Config(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CELAR MuJoCo C0")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_c0(_load(Path(args.config)))


if __name__ == "__main__":
    main()
