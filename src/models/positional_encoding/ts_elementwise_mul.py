# -*- coding: utf-8 -*-
"""Time series HDC positional embedding module.

This module implements element-wise multiplication binding method for hyperdimensional computing (HDC) positional encodings for time series classification.
"""
# Third party imports
import torch


class TimeSeriesElementwiseMultiplicationPositionalEncoding(torch.nn.Module):
    """This module produces HDC positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        """Initializes the HDC positional encoding.

        Args:
            num_positions (int): The maximum sequence length.
            embedding_dim (int): The dimensionality of the HDC vectors.  Must be even.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions  # max_len

        # Ensure embedding_dim is even
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be an even number for HDC.")

        self.position_vectors = self._init_position_vectors(num_positions, embedding_dim)

    def _init_position_vectors(self, num_positions: int, embedding_dim: int) -> torch.nn.Parameter:
        """Initialize the HDC position vectors.

        Each position is represented by a unique, randomly initialized bipolar vector (+1 or -1).
        """
        # Initialize position vectors with random bipolar values (+1 or -1)
        position_vectors = torch.randint(0, 2, (num_positions, embedding_dim), dtype=torch.float) * 2 - 1
        return torch.nn.Parameter(position_vectors, requires_grad=False)

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Applies positional encoding to the input tensor using element-wise multiplication (binding).

        Args:
            input_tensor (torch.Tensor): Input tensor of shape [bsz x seqlen x input_size].

        Returns:
            torch.Tensor: Tensor with positional encoding applied, of shape [bsz x seqlen x embedding_dim].
        """
        bsz, seq_len, _ = input_tensor.shape

        # Get position vectors for the sequence length
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.position_vectors.device)

        # Shape: [seq_len x embedding_dim]
        position_encodings = self.position_vectors[positions]

        # Expand position encodings to match the batch size
        # Shape: [bsz x seq_len x embedding_dim]
        position_encodings = position_encodings.unsqueeze(0).expand(bsz, seq_len, self.embedding_dim)

        # Element-wise multiplication (binding)
        # We need to project the input_tensor to the embedding_dim before binding
        # Here, we assume that the input_size is smaller than embedding_dim, and we pad the input_tensor with zeros
        input_size = input_tensor.shape[-1]
        if input_size < self.embedding_dim:
            padding_size = self.embedding_dim - input_size
            padding = torch.zeros(bsz, seq_len, padding_size, device=input_tensor.device)
            projected_input_tensor = torch.cat([input_tensor, padding], dim=-1)

        # If input_size is larger than embedding_dim, the input_tensor is truncated
        elif input_size > self.embedding_dim:
            projected_input_tensor = input_tensor[:, :, : self.embedding_dim]

        else:
            projected_input_tensor = input_tensor

        encoded_tensor = projected_input_tensor * position_encodings

        return encoded_tensor
