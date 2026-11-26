from __future__ import annotations

import argparse

from .config import load_config
from .experiment import run_sim_c1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal CELAR C1")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_sim_c1(load_config(args.config))


if __name__ == "__main__":
    main()
