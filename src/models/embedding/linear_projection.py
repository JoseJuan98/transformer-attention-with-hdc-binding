# -*- coding: utf-8 -*-
"""Linear projection layer for time series data."""
# Third party imports
import torch

# First party imports
from models.embedding.base import BaseEmbedding


class LinearProjection(BaseEmbedding):
    """Linear projection layer for time series data.

    This layer projects the input features to a specified output dimension using a linear transformation.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term in the linear layer.
    """

    name = "linear_projection"

    def __init__(self, in_features: int, out_features: int, bias: bool = False, **kwargs):
        """Initializes the LinearProjection class.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to include a bias term in the linear layer.
            kwargs (dict): Additional keyword arguments, such as:
                - normalization (bool): Whether to apply layer normalization after the linear transformation. Defaults
                to True.
        """
        super(LinearProjection, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.normalization = kwargs.get("normalization", True)

        if self.normalization:
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=self.bias),
                torch.nn.LayerNorm(normalized_shape=self.out_features, eps=1e-5),
            )
        else:
            self.linear = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=self.bias)

        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the linear layer using the Xavier Normal initialization."""
        if self.normalization:
            torch.nn.init.xavier_normal_(self.linear[0].weight, gain=1.0)

            # Initialize bias to zero
            if self.linear[0].bias is not None:
                torch.nn.init.zeros_(self.linear[0].bias)
        else:
            torch.nn.init.xavier_normal_(self.linear.weight, gain=1.0)

            # Initialize bias to zero
            if self.linear.bias is not None:
                torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the linear projection to the input data.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, seq_len, in_features).

        Returns:
            torch.Tensor: Projected data of shape (batch_size, seq_len, out_features).
        """
        return self.linear(x)
