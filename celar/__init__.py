"""CELAR controlled cross-embodiment pilot."""

from .config import ExperimentConfig, load_config
from .experiment import run_experiment

__all__ = ["ExperimentConfig", "load_config", "run_experiment"]
