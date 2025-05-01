# -*- coding: utf-8 -*-
"""Base class for positional encoding modules."""

# Standard imports
from abc import ABC, abstractmethod

# Third party imports
import torch


class PositionalEncoding(ABC, torch.nn.Module):
    """Base class for positional encoding modules."""

    def __init__(self, d_model: int, num_positions: int = 5000):
        """Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            num_positions (int, optional): The maximum sequence length. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.num_positions = num_positions
        self.encodings = self._init_weight(d_model=d_model, num_positions=num_positions)

    @staticmethod
    @abstractmethod
    def _init_weight(d_model: int, num_positions: int) -> torch.nn.Parameter:
        """Initializes the positional encodings.

        Args:
            d_model (int): The dimensionality of the embeddings.
            num_positions (int): The maximum sequence length.

        Returns:
            torch.nn.Parameter: The initialized positional encodings.
        """
        raise NotImplementedError("Subclasses must implement the `_init_weight` method.")

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """`input_tensor` is the input tensor of shape (batch_size, seq_len, d_model)."""
