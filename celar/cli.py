from __future__ import annotations

import argparse

from .config import load_config
from .experiment import run_experiment
from .sim_c1 import run_sim_c1
from .value_study import run_c2_study, run_value_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CELAR cross-embodiment pilot")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--study",
        choices=("pilot", "c1-mujoco", "c2", "value"),
        default="pilot",
        help="pilot=synthetic C0/C1/C2/C4, c1-mujoco=formal C0 C1, c2=long BC, value=extended C2/C3",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.study == "value":
        run_value_study(config)
    elif args.study == "c1-mujoco":
        run_sim_c1(config)
    elif args.study == "c2":
        run_c2_study(config)
    else:
        run_experiment(config)


if __name__ == "__main__":
    main()
