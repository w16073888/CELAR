from __future__ import annotations

import argparse

from .config import load_config
from .experiment import run_experiment
from .value_study import run_c2_study, run_value_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen CELAR pre-experiment proxies"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--study",
        choices=("pilot", "c2", "value"),
        default="pilot",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.study == "value":
        run_value_study(config)
    elif args.study == "c2":
        run_c2_study(config)
    else:
        run_experiment(config)


if __name__ == "__main__":
    main()
