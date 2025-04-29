# -*- coding: utf-8 -*-
"""Dataset configuration module."""

# Standard imports
from dataclasses import dataclass

# First party imports
from utils.base_config import BaseConfig


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration class for training a model for a specific dataset.

    It contains the specific hyperparameters for the dataset.

    Attributes:
        dataset_name (str): The name of the dataset.
        num_channels (int): The number of input channels.
        context_length (int): The maximum sequence length.
        num_classes (int): The number of classes in the dataset.
    """

    dataset_name: str
    num_channels: int
    context_length: int
    num_classes: int
