# -*- coding: utf-8 -*-
"""Model configuration module."""
# Standard imports
from dataclasses import dataclass
from typing import Literal

# First party imports
from utils.base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Configuration class for training a model.

    It contains the hyperparameters for the model.

    Attributes:
        model_name (str): The name of the model. Values can be: "transformer_absolute_sinusoidal_pe",
            "transformer_elementwise_pe", "transformer_ciruclarconvolution_pe".
        desc (str): A description of the model.
        num_epochs (int): The number of training epochs.
        batch_size (int): The batch size.
        d_model (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the inner layer of the feed-forward network.
        num_layers (int): The number of encoder layers.
        dropout (float): The dropout probability.

        learning_rate (float): The learning rate.

    """

    model_name: Literal[
        "transformer_absolute_sinusoidal_pe", "transformer_elementwise_pe", "transformer_ciruclarconvolution_pe"
    ]
    desc: str

    # Model hyperparameters
    num_epochs: int
    batch_size: int
    learning_rate: float
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    dropout: float
