# -*- coding: utf-8 -*-
"""Circular Convolutional binding method for hyperdimensional computing (HDC) positional encodings."""
# Third party imports
import torch

# First party imports
from models.binding.basic import EmbeddingBinding


class ConvolutionalBinding(EmbeddingBinding):
    """Binds embeddings and positional encodings by circular convolution."""

    name = "convolutional"

    def __init__(self, embedding_dim: int):
        super(ConvolutionalBinding, self).__init__()
        self.embedding_dim = embedding_dim
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, embeddings: torch.Tensor, positional_encodings: torch.Tensor) -> torch.Tensor:
        """Applies the circular convolutional binding method.

        Args:
            embeddings (torch.Tensor): The input embeddings.
            positional_encodings (torch.Tensor): The positional encodings.

        Returns:
            torch.Tensor: The circularly convolved embeddings.
        """
        # Support for complex numbers half precission is experimental and some features might not work
        # Compute the FFT of both tensors
        emb_fft = torch.fft.fft(embeddings.float())
        pos_fft = torch.fft.fft(positional_encodings.float())

        # Component-wise multiplication in the frequency domain
        C_fft = emb_fft * pos_fft

        # Compute the inverse FFT to get the circular convolution result
        c = torch.fft.ifft(C_fft)

        # Return the real part (imaginary part should be negligible)
        return self.layer_norm(c.real)
