# -*- coding: utf-8 -*-
"""Positional encoding module."""
# Standard imports
import math

# Third party imports
import torch


class SinusoidalPositionalEncoding(torch.nn.Module):
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
        max_len (int, optional): The maximum sequence length. Defaults to 5000.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Adds the positional encoding to the input tensor.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            max_len (int, optional): The maximum sequence length. Defaults to 5000.
        """
        super(SinusoidalPositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input tensor with the positional encoding added.
        """
        if self.encoding.device.type != x.device.type:
            self.encoding = self.encoding.to(x.device)
        return x + self.encoding[:, : x.size(1)]
