# -*- coding: utf-8 -*-
"""HDC positional embedding modules for Time Series.

Implements circular convolution binding methods for hyperdimensional computing (HDC) positional encodings.
"""
# Third party imports
import torch

# First party imports
from models.positional_encoding.base import PositionalEncoding


class NullPositionalEncoding(PositionalEncoding):
    """Creates a null (zero) positional encoding.

    This module initializes positional encodings as a zero tensor, effectively disabling positional information.
    """

    name = "none"

    def __init__(self, d_model: int, num_positions: int = 5000, **kwargs):
        """Initializes the NullPositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            num_positions (int, optional): The maximum sequence length. Defaults to 5000.
        """
        super(NullPositionalEncoding, self).__init__(d_model=d_model, num_positions=num_positions, **kwargs)

    @staticmethod
    def _init_weight(d_model: int, num_positions: int, **kwargs) -> torch.nn.Parameter:
        """Initializes the positional encodings.

        Args:
            d_model (int): The dimensionality of the embeddings.
            num_positions (int): The maximum sequence length.
            seed (int): The random seed for reproducibility.

        Returns:
            torch.nn.Parameter: The initialized positional encodings.
        """
        # Create a zero tensor for the null positional encodings
        # Add batch dimension for broadcasting
        return torch.nn.Parameter(
            torch.zeros(num_positions, d_model, requires_grad=False).unsqueeze(0), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is the input tensor of shape (batch_size, seq_len, d_model)."""
        return self.encodings[:, : x.size(1)]
