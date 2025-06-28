# -*- coding: utf-8 -*-
"""Model configuration module."""
# Standard imports
from dataclasses import dataclass
from typing import Any, Dict, Union

# First party imports
from models import ModelTypeStr
from models.binding.factory import BindingMethodTypeStr
from models.embedding.factory import EmbeddingTypeStr
from models.positional_encoding.factory import TSPositionalEncodingTypeStr
from models.transformer.attention.factory import AttentionTypeStr
from utils.base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Configuration class for training a model.

    It contains the hyperparameters for the model.

    Attributes:
        model_name (str): The name of the model.
        model (str): The name of the type of model to use. Values can be: "encoder-only-transformer"
        desc (str): A description of the model.
        num_epochs (int): The number of training epochs.
        d_model (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the inner layer of the feed-forward network.
        num_layers (int): The number of encoder layers.
        dropout (float): The dropout probability.
        learning_rate (float): The learning rate.
        positional_encoding (str, Dict[str, Any]): The type of positional encoding. Values can be: "sinusoidal",
            "split_sinusoidal", "random", "null", "fractional_power". If a dictionary is provided, it should contain the
            key "type" with the positional encoding type and any additional parameters for the positional encoding.
        embedding_binding (str): The binding method for the embeddings. Values can be: "additive", "multiplicative",
            "circular_convolution".
        embedding (str): The type of embedding. Values can be: "1d_conv", "2d_conv", "linear_projection".
    """

    model_name: str
    desc: str

    # Model hyperparameters
    model: ModelTypeStr
    num_epochs: int
    learning_rate: float
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    dropout: float
    positional_encoding: Union[TSPositionalEncodingTypeStr, Dict[str, Union[Any, TSPositionalEncodingTypeStr]]]
    embedding_binding: BindingMethodTypeStr
    embedding: Union[EmbeddingTypeStr, Dict[str, Union[Any, EmbeddingTypeStr]]]
    multihead_attention: Union[AttentionTypeStr, Dict[str, Union[Any, AttentionTypeStr]]] = "standard"
