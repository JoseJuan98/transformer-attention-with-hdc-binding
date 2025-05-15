# -*- coding: utf-8 -*-
"""Dense Layer module."""
# Third party imports
import torch


class FeedForward(torch.nn.Module):
    r"""Position-wise feed-forward network.

    This class implements the feed-forward network used in each encoder layer of the Transformer. It consists of two
    linear layers with a Gaussian Error Linear Unit (GeLU) activation in between.

    Args:
        d_model (int): The dimensionality of the input and output embeddings.
        d_ff (int): The dimensionality of the inner layer.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the feed-forward module.
    """

    def __init__(self, d_model: int, d_ff: int):
        """Initializes the FeedForward module.

        Args:
            d_model (int): The dimensionality of the input and output embeddings.
            d_ff (int): The dimensionality of the inner layer.
        """
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=d_model, out_features=d_ff)
        self.linear2 = torch.nn.Linear(in_features=d_ff, out_features=d_model)
        self.activation = torch.nn.GELU()
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the feed-forward layers using the Xavier Normal initialization."""
        torch.nn.init.xavier_normal_(self.linear1.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.linear2.weight, gain=1.0)

        # Initialize bias to zero
        if self.linear1.bias is not None:

            torch.nn.init.zeros_(self.linear1.bias)

        if self.linear2.bias is not None:
            torch.nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feed-forward module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.linear2(self.activation(self.linear1(x)))
