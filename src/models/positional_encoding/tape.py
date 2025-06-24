# -*- coding: utf-8 -*-
"""Time Absolute Position Encoding (tAPE) from Foumani et al. [1].

This module implements the Time Absolute Position Encoding (tAPE) as proposed in the paper [1]. tAPE adapts the
standard sinusoidal positional encoding for time series data by incorporating the model's embedding dimension (d_model)
and the sequence length (L) into the frequency calculation.

Original code:
https://github.com/Navidfoumani/ConvTran/blob/148afb6ca549915b7e78b05e2ec3b4ba6e052341/Models/AbsolutePositionalEncoding.py#L8-L44

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
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Modifications:

- The core logic for tAPE was extracted from the original implementation to fit the project's architecture.

- The `TimeAbsolutePositionalEncoding` class inherits from the base `PositionalEncoding` class to ensure architectural
consistency.

- The `_init_weight` method was modified to implement the tAPE formula as described in the paper.

- The `forward` method was adapted to return the positional encodings based on the input sequence length and the
addition of the embeddings was moved outside the class to maintain consistency with the project's design and be
compatible with different binding methods.


References:
    [1] Foumani, N. M., Tan, C. W., Webb, G. I., & Salehi, M. (2023). "Improving position encoding of transformers for
    multivariate time series classification." Data Mining and Knowledge Discovery.
    https://link.springer.com/content/pdf/10.1007/s10618-023-00948-2.pdf
"""

# Standard imports
import math

# Third party imports
import torch

# First party imports
from models.positional_encoding.base import PositionalEncoding


class TimeAbsolutePositionalEncoding(PositionalEncoding):
    r"""Implements Time Absolute Position Encoding (tAPE) from Foumani et al. [1].

    This method modifies the standard sinusoidal positional encoding by scaling the frequency term to make it more
    suitable for time series data, where embedding dimensions and sequence lengths can vary significantly.
    The formula for the argument inside the sine and cosine functions is modified as follows:
    .. math::
        \text{arg}(pos, i) = \left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right) \times \frac{d_{\text{model}}}{L}

    where :math:`pos` is the position, :math:`i` is the dimension index, :math:`d_{\text{model}}` is the embedding
    dimension, and :math:`L` is the maximum sequence length (`num_positions`).

    This scaling helps maintain the "distance awareness" and "isotropy" properties of the encodings, which can be
    lost in standard PEs when applied to time series with low embedding dimensions.

    References:
        [1] Foumani, N. M., Tan, C. W., Webb, G. I., & Salehi, M. (2023). "Improving position encoding of transformers for
        multivariate time series classification." Data Mining and Knowledge Discovery.
        https://link.springer.com/content/pdf/10.1007/s10618-023-00948-2.pdf

    Args:
        d_model (int): The dimensionality of the embeddings.
        num_positions (int, optional): The maximum sequence length (L in the paper). Defaults to 5000.
        learnable (bool, optional): If True, the positional encodings are learnable parameters. Defaults to False.
    """

    name = "tape"

    def __init__(self, d_model: int, num_positions: int = 5000, **kwargs):
        """Initializes the TimeAbsolutePositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            num_positions (int, optional): The maximum sequence length. Defaults to 5000.
        """
        # The parent __init__ calls _init_weight with all provided arguments.
        super(TimeAbsolutePositionalEncoding, self).__init__(d_model=d_model, num_positions=num_positions, **kwargs)

    @staticmethod
    def _init_weight(d_model: int, num_positions: int, **kwargs) -> torch.nn.Parameter:
        """Initializes the tAPE positional encodings based on Equation (13) from the paper.

        Args:
            d_model (int): The dimensionality of the embeddings.
            num_positions (int): The maximum sequence length.

        Returns:
            torch.nn.Parameter: The initialized, non-learnable positional encodings.
        """
        # A tensor of shape (num_positions, d_model) is created to store the encodings.
        encodings = torch.zeros(num_positions, d_model, requires_grad=False)

        # A tensor representing the positions (0, 1, ..., num_positions-1) is created.
        position = torch.arange(0, num_positions).unsqueeze(1).float()

        # The division term for the frequency is calculated, same as in standard Sinusoidal PE.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # The argument for the sin/cos functions is calculated and scaled by the factor from Foumani et al. [1],
        #  (d_model / L)
        # num_positions corresponds to L (sequence length).
        scaled_arg = (position * div_term) * (d_model / num_positions)

        # The even and odd dimensions of the encoding matrix are filled.
        encodings[:, 0::2] = torch.sin(scaled_arg)
        encodings[:, 1::2] = torch.cos(scaled_arg)

        # A batch dimension is added for broadcasting across the batch.
        # The encodings are wrapped in a Parameter, with learnability determined by the 'learnable' argument
        return torch.nn.Parameter(encodings.unsqueeze(0), requires_grad=kwargs.get("learnable", False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the pre-computed tAPE encodings for the input tensor's sequence length.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model). The content of x is used to
                              determine the sequence length.

        Returns:
            torch.Tensor: The Time Absolute Positional Encoding tensor of shape (1, seq_len, d_model).
        """
        # The pre-computed encodings are sliced to match the input sequence length.
        return self.encodings[:, : x.size(1)]
