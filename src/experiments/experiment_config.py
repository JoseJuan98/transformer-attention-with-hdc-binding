# -*- coding: utf-8 -*-
"""Experiment configuration module."""
# Standard imports
import json
import pathlib
from dataclasses import dataclass

# Third party imports
from lightning.pytorch.trainer.connectors.accelerator_connector import _PRECISION_INPUT


@dataclass
class ModelConfig:
    """Configuration class for training a model.

    It contains the hyperparameters, hardware settings, and experiment settings.

    Attributes:
        input_size (int): The size of the input features or vocabulary in case of text.
        context_length (int): The maximum sequence length.
        d_model (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the inner layer of the feed-forward network.
        num_layers (int): The number of encoder layers.
        dropout (float): The dropout probability.
        batch_size (int): The batch size.
        learning_rate (float): The learning rate.
        num_epochs (int): The number of training epochs.
        device (str): The device to use for training ('cpu' or 'cuda').
        model_relative_path (str, pathlib.Path): The relative path to ~`Config.model_dir` to save the model to.
        experiment_name (str): The name of the experiment.
        description (str): A description of the experiment.
        dataset (str): The dataset used for training.
        num_classes (str): The number of classes in the dataset
        precision (str): The precision to use for training ('16', '32', '16-mixed').
    """

    # Model hyperparameters
    batch_size: int
    num_epochs: int
    learning_rate: float
    input_size: int
    context_length: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    dropout: float

    # Hardware settings
    device: str
    precision: _PRECISION_INPUT

    # Experiment settings
    model_relative_path: str
    experiment_name: str
    description: str
    dataset: str
    num_classes: int

    def to_dict(self) -> dict:
        """Converts the configuration to a dictionary."""
        return self.__dict__

    def dump(self, path: str | pathlib.Path):
        """Dumps the configuration to a JSON file."""
        if isinstance(path, str):
            path = pathlib.Path(path)

        # Create the directories if it does not exist
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as file:
            json.dump(obj=self.to_dict(), fp=file, indent=4)

    def pretty_str(self):
        """List the attributes of the class."""
        return "\n".join([f"\t{k}: {v}" for k, v in self.to_dict().items()])
