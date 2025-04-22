# -*- codong: utf-8 -*-
"""Factory class for creating model configurations based on the provided configuration file."""
# Standard imports
import pathlib

# First party imports
from experiment_framework.config.model_config import ModelConfig
from experiment_framework.config.parser import ConfigParser


class ModelConfigFactory:
    """Factory class for creating model configurations based on the provided configuration file."""

    @staticmethod
    def create_model_configs(model_cfg_path: pathlib.Path | str) -> dict[str, ModelConfig]:
        """Loads the configuration from a configuration file."""
        config_dict = ConfigParser.parse_config(path=model_cfg_path)

        # Check that `model_name` is not duplicated
        model_names = [model["model_name"] for model in config_dict.values()]
        if len(model_names) != len(set(model_names)):
            raise ValueError(f"Duplicated model names found in the configuration file: {model_names}")

        cfg = {}
        for k, v in config_dict.items():
            cfg[k] = ModelConfig(**v)

        return cfg
