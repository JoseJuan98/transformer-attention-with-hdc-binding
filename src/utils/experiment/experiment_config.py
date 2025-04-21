# -*- coding: utf-8 -*-
"""Model configuration module."""
# Standard imports
import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Literal, Union

# Third party imports
from lightning.pytorch.trainer.connectors.accelerator_connector import _PRECISION_INPUT

# First party imports
from utils.base_config import BaseConfig
from utils.experiment.metrics_handler import METRICS_MODE_STR
from utils.experiment.model_config import ModelConfig


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
        default_batch_size (int): The batch size to use for training.
        model_configs (dict[str, ModelConfig]): A dictionary of model configurations. The key is the model name and
            the value is the model configuration.
        accelerator (str): The device to use for training ('cpu', 'gpu', 'tpu', 'hpu', 'mps', 'auto').
        precision (str): The precision to use for training ('16', '32', '16-mixed', 'bf16-mixed', ...).
        auto_scale_batch_size (bool): Whether to automatically scale the batch size.
        profiler (bool): Whether to use the profiler.
        summary (bool): Whether to use the summary writer.
        plots (bool): Whether to generate the plots.
        development (bool): Whether to run in development mode. This is used to reduce the number of epochs and
            terminates the run after any errors.
        validation_split (float): The percentage of training data to use for validation (0.0-1.0).
        early_stopping_patience (int): The number of epochs to wait for early stopping. Default: 10 epochs. If equals to
            0, early stopping is disabled.
        metrics_mode (str): The mode to aggregate metrics to. Values are ['append', 'write']. If 'append', and the
            metrics file already exist, the new metrics will be appended to the existing file. If 'write', the new
            experiment's metrics will overwrite the existing file.
        accumulate_grad_batches (Union[int, Dict[int, int]]): The number of batches to accumulate gradients over.
            Default: 1 (no accumulation). If a dictionary is provided, the keys are the number of batches to
            accumulate over and the values are the number of batches to accumulate gradients over.
        gradient_clip_val (float): The value to clip gradients to. Default: 0.0 (no clipping). Only use if
            `gradient_clip_algorithm` is set to "value".
        gradient_clip_algorithm (str): The algorithm to use for gradient clipping. Values are ['norm', 'value'].
            Default: "norm" (clip by norm). Only use if `gradient_clip_val` is set to a value greater than 0.
        use_swa (bool): Whether to use Stochastic Weight Averaging (SWA). Default: False.
        swa_learning_rate (float): The learning rate to use for SWA. Default: 1e-2.
        use_lr_finder (bool): Whether to use the learning rate finder. Default: False.
        lr_finder_milestones (list[int]): The milestones to use for the learning rate finder. Default: [].
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
    default_batch_size: int

    # Hardware settings
    accelerator: Literal["cpu", "gpu", "tpu", "hpu", "mps", "auto"]
    precision: _PRECISION_INPUT
    auto_scale_batch_size: bool

    # Other settings
    metrics_mode: METRICS_MODE_STR
    profiler: bool
    summary: bool
    plots: bool
    development: bool

    # Parameters for advanced training techniques
    validation_split: float
    early_stopping_patience: int = 10  # Default: 10 epochs
    accumulate_grad_batches: Union[int, Dict[int, int]] = 1  # Default: no accumulation
    gradient_clip_val: float = 0.0  # Default: no gradient clipping
    gradient_clip_algorithm: Literal["norm", "value"] = "norm"  # Default: clip by norm
    use_swa: bool = False  # Default: don't use SWA
    swa_learning_rate: float = 1e-2  # Default SWA learning rate
    use_lr_finder: bool = False  # Default: don't use LR Finder
    lr_finder_milestones: list[int] = field(default_factory=list)  # Default: LR Finder milestones

    def pretty_str(self) -> str:
        """List the attributes of the class in a pretty format."""
        cfg_str = f"{' Experiment Configuration ':_^100}\n\n"
        cfg_str += "\n".join(
            [f"\t{k}: {v}" for k, v in self.to_dict().items() if k not in ["model_configs", "dataset_names"]]
        )

        cfg_str += "\n\n\tdataset_names:\n\n"
        for dataset_name in self.dataset_names:
            cfg_str += f"\t\t{dataset_name}\n"

        cfg_str += "\n\tmodel_configs:\n\n"
        for model_name, model_config in self.model_configs.items():
            cfg_str += f"\t\t{model_name}:\n\n"
            cfg_str += "\n".join([f"\t\t\t{sub_k}: {sub_v}" for sub_k, sub_v in model_config.to_dict().items()])
            cfg_str += "\n\n"
        cfg_str += "_" * 100 + "\n\n"
        return cfg_str

    def dump(self, path: str | pathlib.Path) -> None:
        """Dumps the configuration to a JSON file."""
        path = pathlib.Path(path) if isinstance(path, str) else path

        # Create the directories if it does not exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert ModelConfig objects to dictionaries for JSON serialization
        model_configs_serializable = {k: v.to_dict() for k, v in self.model_configs.items()}

        with open(path, "w") as file:
            json.dump(obj={**self.to_dict(), "model_configs": model_configs_serializable}, fp=file, indent=4)
