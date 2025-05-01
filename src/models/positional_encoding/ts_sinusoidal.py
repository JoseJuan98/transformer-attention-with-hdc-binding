# -*- coding: utf-8 -*-
"""Time series sinusoidal positional embedding module.

The TimeSeriesSinusoidalPositionalEmbedding class in this module are adapted from the Hugging Face Transformers library,
licensed under the Apache License, Version 2.0.

Original code:
https://github.com/huggingface/transformers/blob/9e94801146ceeb3b215bbdb9492be74d7d7b7210/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1-L271

Copyright 2022 The HuggingFace Inc. team. All rights reserved.
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications:
- Adapted with a common interface for positional encodings.
"""
# Third party imports
import numpy
import torch

# First party imports
from models.positional_encoding.base import PositionalEncoding


class TimeSeriesSinusoidalPositionalEncoding(PositionalEncoding):
    """This module produces sinusoidal positional embeddings of any length."""

    name = "ts_sinusoidal"

    def __init__(self, d_model: int, num_positions: int, padding_idx: int | None = None) -> None:
        super().__init__(d_model=d_model, num_positions=num_positions)
        self.padding_idx = padding_idx

    @staticmethod
    def _init_weight(d_model: int, num_positions: int) -> torch.nn.Parameter:
        """Initialize the sinusoidal positional embeddings.

        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        position_enc = numpy.array(
            [[pos / numpy.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] for pos in range(num_positions)]
        )
        # Convert numpy array to tensor and move requires_grad inside the function
        out = torch.zeros(num_positions, d_model, requires_grad=False)
        sentinel = d_model // 2 if d_model % 2 == 0 else (d_model // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(numpy.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(numpy.cos(position_enc[:, 1::2]))
        return torch.nn.Parameter(out, requires_grad=False)

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """`input_ids_shape` is expected to be (batch_size, seq_len, d_model)."""
        batch_size, seq_len, _ = input_tensor.shape
        # Use the maximum sequence length from input tensor to broadcast the positional encodings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.encodings.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        return self.encodings[positions]
