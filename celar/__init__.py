"""CELAR formal four-task, three-robot C1 experiments."""

from .current.config import ExperimentConfig, load_config
from .current.experiment import run_sim_c1

__all__ = ["ExperimentConfig", "load_config", "run_sim_c1"]
