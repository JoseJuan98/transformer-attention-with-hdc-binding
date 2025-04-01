# -*- coding: utf-8 -*-
"""This module is responsible for creating the experiment configuration for the models."""
# Standard imports
import json
import pathlib

# Local imports
from .experiments.experiment_config import ExperimentConfig
from .experiments.model_config import ModelConfig


class ExperimentConfigFactory:
    """Factory class for creating experiment configurations."""

    @staticmethod
    def _get_cfg_from_file(path: pathlib.Path | str) -> dict:
        if isinstance(path, str):
            path = pathlib.Path(path)

        if path.suffix not in [".json"]:
            raise ValueError(f"Invalid configuration file type: {path.suffix}")

        with open(path, "r") as file:
            config_dict = json.load(file)

        return config_dict

    def _load_model_configs(self, path: pathlib.Path | str) -> dict[str, ModelConfig]:
        """Loads the configuration from a configuration file."""
        config_dict = self._get_cfg_from_file(path=path)
        cfg = {}
        for k, v in config_dict.items():
            cfg[k] = ModelConfig(**v)

        return cfg

    def create_experiment_config(
        self, model_cfg_path: pathlib.Path | str, experiment_cfg_path: pathlib.Path | str
    ) -> ExperimentConfig:
        """Loads the configuration from a configuration file."""
        model_configs = self._load_model_configs(path=model_cfg_path)
        exp_config_dict = self._get_cfg_from_file(path=experiment_cfg_path)

        return ExperimentConfig(
            model_configs=model_configs,
            **exp_config_dict,
        )
