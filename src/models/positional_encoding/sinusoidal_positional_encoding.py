# -*- coding: utf-8 -*-
# This project incorporates code from the following projects:
#
# Hugging Face Transformers
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original code:
# https://github.com/huggingface/transformers/blob/9e94801146ceeb3b215bbdb9492be74d7d7b7210/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1-L271
#
# Specifically, the following classes were adapted:
# - TimeSeriesSinusoidalPositionalEmbedding
"""Positional encoding module."""

# Standard imports
import math

# Third party imports
import numpy
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


class TimeSeriesSinusoidalPositionalEmbedding(torch.nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions  # max_len
        self.padding_idx = padding_idx
        self.weight = self._init_weight(num_positions, embedding_dim)

    @staticmethod
    def _init_weight(num_positions: int, embedding_dim: int) -> torch.nn.Parameter:
        """Initialize the sinusoidal positional embeddings.

        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        position_enc = numpy.array(
            [
                [pos / numpy.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)]
                for pos in range(num_positions)
            ]
        )
        # Convert numpy array to tensor and move requires_grad inside the function
        out = torch.zeros(num_positions, embedding_dim, requires_grad=False)
        sentinel = embedding_dim // 2 if embedding_dim % 2 == 0 else (embedding_dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(numpy.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(numpy.cos(position_enc[:, 1::2]))
        return torch.nn.Parameter(out, requires_grad=False)

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen x input_size]."""
        bsz, seq_len, input_size = input_tensor.shape
        # Use the maximum sequence length from input tensor
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device)
        # Expand for batch size and input size
        # positions = positions.unsqueeze(0).expand(bsz, -1)
        # return super().forward(positions)
        positions = positions.unsqueeze(0).expand(bsz, seq_len)  # Corrected line
        return self.weight[positions]
