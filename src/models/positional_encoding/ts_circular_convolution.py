# -*- coding: utf-8 -*-
"""HDC positional embedding modules for Time Series.

Implements circular convolution binding methods for hyperdimensional computing (HDC) positional encodings.
"""
# Third party imports
import torch


class TimeSeriesCircularConvolutionPositionalEncoding(torch.nn.Module):
    """This module produces positional embeddings using circular convolution binding."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions
        self.padding_idx = padding_idx
        self.position_vectors = self._init_position_vectors(num_positions, embedding_dim)

    @staticmethod
    def _init_position_vectors(num_positions: int, embedding_dim: int) -> torch.nn.Parameter:
        """Initialize random position vectors for circular convolution binding."""
        # Create random position vectors with unit norm
        position_vectors = torch.randn(num_positions, embedding_dim)
        # Normalize each position vector to have unit norm
        position_vectors = torch.nn.functional.normalize(position_vectors, p=2, dim=1)
        return torch.nn.Parameter(position_vectors, requires_grad=False)

    @staticmethod
    def circular_convolution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute circular convolution between two vectors using FFT."""
        # FFT of both vectors
        x_fft = torch.fft.rfft(x, dim=-1)
        y_fft = torch.fft.rfft(y, dim=-1)

        # Element-wise multiplication in frequency domain
        z_fft = x_fft * y_fft

        # Inverse FFT to get back to time domain
        z = torch.fft.irfft(z_fft, n=x.shape[-1], dim=-1)

        return z

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply circular convolution binding to input tensor.

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
        pos_vectors = pos_vectors.unsqueeze(0).expand(bsz, -1, -1)

        # Apply circular convolution to each position in the sequence
        output = torch.zeros_like(input_tensor)
        for i in range(seq_len):
            output[:, i] = self.circular_convolution(input_tensor[:, i], pos_vectors[:, i])

        return output
