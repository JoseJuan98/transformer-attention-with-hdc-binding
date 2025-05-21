# -*- coding: utf-8 -*-
"""Time Series Convolutional Embedding Layer.

The TimeSeries1dConvEmbedding class in this module is adapted from the Time-Series-Library, licensed under the MIT License.

Original code:
https://github.com/thuml/Time-Series-Library/blob/40da06fef6e703b62c86bbea1c3d798f15a73fcb/layers/Embed.py#L29-L42


Copyright (c) 2021 THUML @ Tsinghua University

Licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
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
        self,
        c_in: int,
        d_model: int,
        kernel_size: int = 3,
        padding_mode: str = "circular",
        bias: bool = False,
        stride: int = 1,
    ):
        super(TimeSeries1dConvEmbedding, self).__init__()

        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        self.conv = torch.nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
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
        x = x.transpose(dim0=1, dim1=2)

        # Apply the 1D Convolution
        x = self.conv(x)

        # Transpose back to (batch_size, seq_len, d_model)
        return x.transpose(dim0=1, dim1=2)
