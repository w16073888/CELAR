"""Current four-task, three-robot CELAR C1 pipeline."""

from .config import ExperimentConfig, load_config
from .experiment import run_sim_c1

__all__ = ["ExperimentConfig", "load_config", "run_sim_c1"]
