# -*- coding: utf-8 -*-
"""ConvTran Embedding Layer from Foumani et al. [1], renamed to SpatialTemporalEmbedding.

This module implements the embedding layer used in the ConvTran model, which consists of two sequential 2D
convolutional layers designed to capture temporal and spatial features from multivariate time series data.

Original code:
https://github.com/Navidfoumani/ConvTran/blob/148afb6ca549915b7e78b05e2ec3b4ba6e052341/Models/model.py#L86-L150

MIT License:

Copyright (c) 2022 Department of Data Science and Artificial Intelligence @Monash University

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONN

Modifications:

- The original code was adapted to fit the project's architecture, including the use of a base class for embeddings.

- The biases in the convolutional layers are disabled by default. In the original implementation they are enabled, but
  the effect of the bias is completely cancelled out by the Batch Normalization process, making it a redundant
  parameter.

References:
    [1] Foumani, N. M., Tan, C. W., Webb, G. I., & Salehi, M. (2023). "Improving position encoding of transformers for
    multivariate time series classification." Data Mining and Knowledge Discovery.
    https://link.springer.com/content/pdf/10.1007/s10618-023-00948-2.pdf
"""

# Third party imports
import torch

# First party imports
from models.embedding.base import BaseEmbedding


class SpatialTemporalEmbedding(BaseEmbedding):
    """Implements the factored Spatial Temporal embedding layer, found in the ConvTran architecture [1].

    This layer processes multivariate time series input through two main steps as described in the paper [1]:

    1. A 'temporal' convolution that expands the feature dimension. It uses a Conv2d layer with a kernel that
    spans the time dimension to capture local temporal patterns.

    2. A 'spatial' convolution that projects the expanded features back to the desired model dimension (d_model).
    It uses a Conv2d layer with a kernel that spans all input channels, mixing inter-variable information.

    This is analogous to an 'Inverted Bottleneck' structure.

    Args:
        in_features (int): The number of input features/variates in the time series (F).
        out_features (int): The target embedding dimension of the model.
        temporal_kernel_size (int): The kernel size for the temporal convolution. Defaults to 8.

    References:
        [1] Foumani, N. M., et al. (2023). "Improving position encoding of transformers for multivariate time series
        classification."
    """

    name = "convtran"

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, temporal_kernel_size: int = 8, **kwargs
    ):
        super(SpatialTemporalEmbedding, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias, **kwargs
        )

        # The first layer is a temporal convolution that expands the feature dimension.
        # It uses Conv2d to process the (batch_size, 1, num_channels, seq_len) input.
        # The kernel [1, temporal_kernel_size] slides along the time dimension.
        self.temporal_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=out_features * 4,
                kernel_size=(1, temporal_kernel_size),
                padding="same",
                bias=self.bias,
            ),
            torch.nn.BatchNorm2d(num_features=out_features * 4),
            torch.nn.GELU(),
        )

        # The second layer is a spatial convolution that mixes channel information and projects to d_model.
        # The kernel [num_channels, 1] slides across the input variates.
        self.spatial_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=out_features * 4,
                out_channels=out_features,
                kernel_size=(in_features, 1),
                padding="valid",
                bias=self.bias,
            ),
            torch.nn.BatchNorm2d(num_features=out_features),
            torch.nn.GELU(),
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initializes the weights of the convolutional layers using Kaiming Normal initialization."""
        # Initialize the weights of the temporal convolution
        torch.nn.init.kaiming_normal_(self.temporal_conv[0].weight, mode="fan_in", nonlinearity="leaky_relu")

        if self.temporal_conv[0].bias is not None:
            torch.nn.init.zeros_(self.temporal_conv[0].bias)

        # Initialize the weights of the spatial convolution
        torch.nn.init.kaiming_normal_(self.spatial_conv[0].weight, mode="fan_in", nonlinearity="leaky_relu")

        if self.spatial_conv[0].bias is not None:
            torch.nn.init.zeros_(self.spatial_conv[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the ConvTran embedding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, num_channels).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        # The input shape is (B, T, F). It is permuted to (B, F, T) for convolution.
        x = x.permute(0, 2, 1)

        # A channel dimension is added for Conv2d: (B, F, T) -> (B, 1, F, T).
        x = x.unsqueeze(1)

        # The temporal convolution is applied. Output shape: (B, d_model*4, F, T).
        x = self.temporal_conv(x)

        # The spatial convolution is applied. Output shape: (B, d_model, 1, T).
        x = self.spatial_conv(x)

        # The channel dimension is removed: (B, d_model, 1, T) -> (B, d_model, T).
        x = x.squeeze(2)

        # The final tensor is permuted back to the standard Transformer format: (B, T, d_model).
        x = x.permute(0, 2, 1)

        return x
