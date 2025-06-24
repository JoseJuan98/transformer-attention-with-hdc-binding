# -*- config: utf-8 -*-
"""Base class for embedding layers."""


# Standard imports
from abc import ABC, abstractmethod

# Third party imports
import torch


class BaseEmbedding(ABC, torch.nn.Module):
    """Base class for embedding layers.

    This class serves as a base for all embedding layers, providing a common interface and basic functionality.
    It is designed to be subclassed by specific embedding implementations.

    Args:
        in_features (int): The dimensionality of the embeddings (d_model).
        out_features (int): The output dimensionality of the embeddings (d_model).
        learnable (bool, optional): If True, the embeddings are learnable parameters. Defaults to False.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, **kwargs):
        super(BaseEmbedding, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    @abstractmethod
    def init_weights(self) -> None:
        """Initializes the weights of the embedding layer."""
        # This method should be implemented in subclasses to initialize weights appropriately.
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the embedding layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, out_features).
        """
        # This method should be implemented in subclasses to define the forward pass.
        raise NotImplementedError("Subclasses must implement this method.")
