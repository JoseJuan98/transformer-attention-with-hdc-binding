# -*- coding: utf-8 -*-
"""Time series sinusoidal positional embedding module.

The TimeSeriesSinusoidalPositionalEmbedding class in this module are adapted from the Hugging Face Transformers library,
licensed under the Apache License, Version 2.0.

Original code:
https://github.com/huggingface/transformers/blob/9e94801146ceeb3b215bbdb9492be74d7d7b7210/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1-L271

Copyright 2022 The HuggingFace Inc. team. All rights reserved.
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
# Third party imports
import numpy
import torch


class TimeSeriesSinusoidalPositionalEncoding(torch.nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    name = "ts_sinusoidal"

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
        bsz, seq_len, _ = input_tensor.shape
        # Use the maximum sequence length from input tensor
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device)
        positions = positions.unsqueeze(0).expand(bsz, seq_len)
        return self.weight[positions]
