# -*- coding: utf-8 -*-
"""Experiment configuration module."""
# Standard imports
import json
import pathlib
from dataclasses import dataclass
from typing import Literal

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
        task (str): Machine Learning task. Values are ['classification', 'regression', 'time series forecasting',
            'time series classification', 'machine translation', 'sequence to sequence', 'text generation',
            'question answering', 'sentimend analysis', 'image classification', 'object detection'].
        description (str): A description of the experiment.
        dataset_names (list[str]): A list of dataset names to use for the experiment.
        run_version (str): The version of the experiment.
        runs_per_experiment (int): The number of runs per experiment to reduce variance.
        batch_size (int): The batch size to use for training.
        model_configs (dict[str, ModelConfig]): A dictionary of model configurations. The key is the model name and
            the value is the model configuration.
        accelerator (str): The device to use for training ('cpu', 'gpu', 'tpu', 'hpu', 'mps', 'auto').
        precision (str): The precision to use for training ('16', '32', '16-mixed', 'bf16-mixed', ...).
        profiler (bool): Whether to use the profiler.
        summary (bool): Whether to use the summary writer.
    """

    experiment_name: str
    task: Literal[
        "classification",
        "regression",
        "time series forecasting",
        "time series classification",
        "machine translation",
        "sequence to sequence",
        "text generation",
        "question answering",
        "sentimend analysis",
        "image classification",
        "object detection",
    ]
    model_configs: dict[str, ModelConfig]
    dataset_names: list[str]
    description: str
    run_version: str
    runs_per_experiment: int
    batch_size: int

    # Hardware settings
    accelerator: Literal["cpu", "gpu", "tpu", "hpu", "mps", "auto"]
    precision: _PRECISION_INPUT

    # Training settings
    profiler: bool
    summary: bool

    def pretty_str(self):
        """List the attributes of the class."""
        cfg_str = f"{' Experiment Configuration ':_^100}\n\n"
        cfg_str += "\n".join(
            [f"\t{k}: {v}" for k, v in self.to_dict().items() if k not in ["model_configs", "dataset_names"]]
        )

        cfg_str += "\n\n\tdataset_names:\n\n"
        for k in self.dataset_names:
            cfg_str += f"\t\t{k}\n"

        cfg_str += "\n\tmodel_configs:\n\n"
        for k, v in self.model_configs.items():
            cfg_str += f"\t\t{k}:\n\n{"\n".join([f"\t\t\t{k}: {v}" for k, v in v.to_dict().items()])}\n\n"
        cfg_str += "_" * 100 + "\n\n"
        return cfg_str

    def dump(self, path: str | pathlib.Path):
        """Dumps the configuration to a JSON file."""
        if isinstance(path, str):
            path = pathlib.Path(path)

        # Create the directories if it does not exist
        path.parent.mkdir(parents=True, exist_ok=True)

        obj = self.to_dict().copy()

        obj["model_configs"] = {k: v.to_dict() for k, v in self.model_configs.items()}

        with open(path, "w") as file:
            json.dump(obj=obj, fp=file, indent=4)
