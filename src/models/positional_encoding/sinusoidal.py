# -*- coding: utf-8 -*-
"""Positional encoding module."""
# Standard imports
import math

# Third party imports
import torch

# First party imports
from models.positional_encoding.base import PositionalEncoding


class SinusoidalPositionalEncoding(PositionalEncoding):
    r"""Sinusoidal positional encoding module.

    This class implements sinusoidal positional encoding as described in the "Attention is All You Need" paper. It adds
    information about the position of each token in the sequence to the token embeddings. The positional encoding is
    defined as:
    .. math::
        \begin{align}
        PE(pos, 2i) &= \sin(pos / 10000^{2i / d_{\text{model}}}) \\ \\
        PE(pos, 2i+1) &= \cos(pos / 10000^{2i / d_{\text{model}}})
        \end{align}
    where :math:`PE(pos, 2i)` and :math:`PE(pos, 2i+1)` represent the even and odd dimensions of the positional encoding
    at position :math:`pos` and dimension :math:`i`. :math:`d_{\text{model}}` is the embedding dimension.
    Paper: https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): The dimensionality of the embeddings.
        num_positions (int, optional): The maximum sequence length. Defaults to 5000.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Adds the positional encoding to the input tensor.
    """

    name = "sinusoidal"

    def __init__(self, d_model: int, num_positions: int = 5000, **kwargs):
        """Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            num_positions (int, optional): The maximum sequence length. Defaults to 5000.
        """
        super(SinusoidalPositionalEncoding, self).__init__(d_model=d_model, num_positions=num_positions, **kwargs)
        self.learnable = kwargs.get("learnable", False)

    @staticmethod
    def _init_weight(d_model: int, num_positions: int, **kwargs) -> torch.nn.Parameter:
        """Initializes the positional encodings.

        Args:
            d_model (int): The dimensionality of the embeddings.
            num_positions (int): The maximum sequence length.

        Returns:
            torch.nn.Parameter: The initialized positional encodings.
        """
        encodings = torch.zeros(num_positions, d_model, requires_grad=False)
        position = torch.arange(0, num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        # By default, the positional encoding is not learnable unless specified
        return torch.nn.Parameter(encodings.unsqueeze(0), requires_grad=kwargs.get("learnable", False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input tensor with the positional encoding added.
        """
        return self.encodings[:, : x.size(1)]
