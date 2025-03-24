# -*- coding: utf-8 -*-
"""Model configuration module."""
# Standard imports
from dataclasses import dataclass

# First party imports
from utils.base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Configuration class for training a model.

    It contains the hyperparameters for the model.

    Attributes:
        num_epochs (int): The number of training epochs.
        batch_size (int): The batch size.
        input_size (int): The size of the input features or vocabulary in case of text.
        context_length (int): The maximum sequence length.
        d_model (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the inner layer of the feed-forward network.
        num_layers (int): The number of encoder layers.
        dropout (float): The dropout probability.

        learning_rate (float): The learning rate.

    """

    # Model hyperparameters
    num_epochs: int
    batch_size: int
    learning_rate: float
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    dropout: float
