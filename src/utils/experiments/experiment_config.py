# -*- coding: utf-8 -*-
"""Experiment configuration module."""
# Standard imports
from dataclasses import dataclass

# Third party imports
from lightning.pytorch.trainer.connectors.accelerator_connector import _PRECISION_INPUT

# First party imports
from utils.base_config import BaseConfig
from utils.experiments.model_config import ModelConfig


@dataclass
class ExperimentConfig(BaseConfig):
    """Configuration class for training a model.

    It contains hardware settings, and experiment settings.

    Attributes:
        experiment_name (str): The name of the experiment.
        description (str): A description of the experiment.
        dataset_names (list[str]): A list of dataset names to use for the experiment.
        device (str): The device to use for training ('cpu' or 'cuda').
        precision (str): The precision to use for training ('16', '32', '16-mixed', 'bf16-mixed', ...).
    """

    experiment_name: str
    description: str
    run_version: str
    model_configs: dict[str, ModelConfig]
    dataset_names: list[str]

    # Hardware settings
    device: str
    precision: _PRECISION_INPUT
