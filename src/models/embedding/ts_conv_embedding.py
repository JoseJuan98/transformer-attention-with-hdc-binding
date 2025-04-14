# -*- coding: utf-8 -*-
"""Time Series Convolutional Embedding Layer."""
# Third party imports
import torch


class TimeSeries1dConvEmbedding(torch.nn.Module):
    """Embeds time series data using a 1D convolutional layer.

    This layer learns to extract features from the time series and represent each time point (or a small window of time
     points) as a d_model-dimensional vector. This embedding is designed to capture local temporal dependencies.

    Args:
        c_in (int): Number of input channels (features) in the time series.
        d_model (int): Dimension of the output embeddings.
        kernel_size (int): Size of the convolutional kernel (window).
        padding_mode (str): Padding mode for the convolution ('circular', 'zeros', etc.).
        bias (bool): Whether to include a bias term in the convolutional layer.
    """

    name = "1d_conv"

    def __init__(
        self, c_in: int, d_model: int, kernel_size: int = 3, padding_mode: str = "circular", bias: bool = False
    ):
        super(TimeSeries1dConvEmbedding, self).__init__()

        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        self.conv = torch.nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initializes the weights of the convolutional layer using Kaiming Normal."""
        torch.nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

        if self.conv.bias is not None:
            # Initialize bias to zero
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x) -> torch.Tensor:
        """Applies the convolutional embedding to the input time series.

        Args:
            x (torch.Tensor): Input time series data of shape (batch_size, seq_len, c_in).

        Returns:
            torch.Tensor: Time series embeddings of shape (batch_size, seq_len, d_model).
        """
        # Transpose to (batch_size, c_in, seq_len) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)

        # Transpose back to (batch_size, seq_len, d_model)
        return x.transpose(1, 2)
