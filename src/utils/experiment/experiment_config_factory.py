# -*- coding: utf-8 -*-
"""This module is responsible for creating the experiment configuration for the models."""
# Standard imports
import pathlib

# First party imports
from utils.experiment.config_parser import ConfigParser
from utils.experiment.experiment_config import ExperimentConfig


class ExperimentConfigFactory:
    """Factory class for creating experiment configurations."""

    @staticmethod
    def create_experiment_config(experiment_cfg_path: pathlib.Path | str, model_configs: dict) -> ExperimentConfig:
        """Loads the configuration from a configuration file."""
        exp_config_dict = ConfigParser.parse_config(path=experiment_cfg_path)

        return ExperimentConfig(
            model_configs=model_configs,
            **exp_config_dict,
        )
