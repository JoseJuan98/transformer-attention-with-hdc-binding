# -*- coding: utf-8 -*-
"""HDC positional embedding modules for Time Series.

Implements circular convolution binding methods for hyperdimensional computing (HDC) positional encodings.
"""
# Third party imports
import torch

# First party imports
from models.positional_encoding.base import PositionalEncoding


class RandomPositionalEncoding(PositionalEncoding):
    """Creates a random positional encoding using the uniform distribution."""

    name = "random"

    def __init__(self, d_model: int, num_positions: int = 5000, seed: int = 42, **kwargs):
        """Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            num_positions (int, optional): The maximum sequence length. Defaults to 5000.
        """
        super(RandomPositionalEncoding, self).__init__(
            d_model=d_model, num_positions=num_positions, seed=seed, **kwargs
        )
        self.learnable = kwargs.get("learnable", False)

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
        # Create random position vectors with the uniform distribution
        position_vectors = torch.distributions.uniform.Uniform(low=-1, high=1).sample((num_positions, d_model))

        # Normalize each position vector to have unit norm to add stability
        position_vectors = torch.nn.functional.normalize(position_vectors, p=2, dim=1)

        # By default, the positional encoding is not learnable unless specified
        return torch.nn.Parameter(position_vectors, requires_grad=kwargs.get("learnable", False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is the input tensor of shape (batch_size, seq_len, d_model)."""
        return self.encodings[:, : x.size(1)]
