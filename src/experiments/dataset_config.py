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
        input_size (int): The size of the input features or vocabulary in case of text.
        context_length (int): The maximum sequence length.
        num_classes (int): The number of classes in the dataset.
    """

    dataset_name: str
    input_size: int
    context_length: int
    num_classes: int
