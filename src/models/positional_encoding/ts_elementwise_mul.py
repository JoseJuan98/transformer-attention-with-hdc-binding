# -*- coding: utf-8 -*-
"""HDC positional embedding modules for time series data.

Implements circular convolution and element-wise multiplication binding methods
for hyperdimensional computing (HDC) positional encodings.
"""
# Third party imports
import torch


class TimeSeriesElementwiseMultiplicationPositionalEncoding(torch.nn.Module):
    """This module produces positional embeddings using element-wise multiplication binding."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions
        self.padding_idx = padding_idx
        self.position_vectors = self._init_position_vectors(num_positions, embedding_dim)

    @staticmethod
    def _init_position_vectors(num_positions: int, embedding_dim: int) -> torch.nn.Parameter:
        """Initialize random position vectors for element-wise multiplication binding."""
        # Create random position vectors centered around 1.0 with small variance
        # This helps ensure the original signal is preserved while adding positional information
        position_vectors = torch.ones(num_positions, embedding_dim) + 0.1 * torch.randn(num_positions, embedding_dim)
        return torch.nn.Parameter(position_vectors, requires_grad=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply element-wise multiplication binding to input tensor.

        Args:
            input_tensor: Input tensor of shape [bsz, seq_len, input_size]

        Returns:
            Position-encoded tensor of shape [bsz, seq_len, input_size]
        """
        bsz, seq_len, input_size = input_tensor.shape

        # Ensure we have enough position vectors
        if seq_len > self.num_positions:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum positions {self.num_positions}")

        # Get position vectors for the sequence
        pos_vectors = self.position_vectors[:seq_len].to(input_tensor.device)

        # Expand position vectors for batch size
        pos_vectors = pos_vectors.unsqueeze(0).expand(bsz, seq_len, input_size)

        # Apply element-wise multiplication
        output = input_tensor * pos_vectors

        return output
