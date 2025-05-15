# -*- coding: utf-8 -*-
"""Linear projection layer for time series data."""
# Third party imports
import torch


class LinearProjection(torch.nn.Module):
    """Linear projection layer for time series data.

    This layer projects the input features to a specified output dimension using a linear transformation.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term in the linear layer.
    """

    name = "linear_projection"

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        """Initializes the LinearProjection class.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to include a bias term in the linear layer.
        """
        super(LinearProjection, self).__init__()

        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the linear layer using the Xavier Normal initialization."""
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
