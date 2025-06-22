# -*- coding: utf-8 -*-
"""Rotary Positional Encoding (RoPE) from Su Jianlin et al. [1]

This module generates the rotational embeddings required by the RoFormer architecture. It reuses the core logic from
:object:`models.positional_encoding.split_sinusoidal.SplitSinusoidalPositionalEncoding`, but adapts the forward
pass for use within the attention mechanism.

Original code:
https://github.com/huggingface/transformers/blob/0725cd6953803b8aacfc85288cbfb83dea30c469/src/transformers/models/roformer/modeling_roformer.py#L48-L79

References:
    [1] Su Jianlin, et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv, 2023,
    arxiv.org/abs/2104.09864.

Modifications:

- The core logic for generating the positional encodings is reused from the `SplitSinusoidalPositionalEncoding` class.
"""
# Third party imports
import torch

# First party imports
from models.positional_encoding.split_sinusoidal import SplitSinusoidalPositionalEncoding


class RotaryPositionalEncoding(SplitSinusoidalPositionalEncoding):
    """Generates rotational embeddings for RoPE [1]

    References:
        [1] Su Jianlin, et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv, 2023,
        arxiv.org/abs/2104.09864.

    Args:
        d_model (int): The dimensionality of the embeddings. This must be an even number.
        num_positions (int, optional): The maximum sequence length. Defaults to 5000.
    """

    name = "rotary"

    def __init__(self, d_model: int, num_positions: int = 5000, **kwargs):
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be an even number for Rotary Positional Encoding, but got {d_model}")

        self.learnable = kwargs.get("learnable", False)
        if self.learnable:
            raise ValueError("Rotary Positional Encoding is not designed to be learnable.")

        # Call the parent constructor, which already initializes self.encodings
        super().__init__(d_model=d_model, num_positions=num_positions, **kwargs)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the pre-computed rotational encodings for the given sequence length.

         The output shape is (1, seq_len, d_model) to be compatible with broadcasting during the attention computation.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model). This is used only to determine
            the sequence length.

        Returns:
            torch.Tensor: The rotational encodings of shape (1, seq_len, d_model).
        """
        seq_len = x.size(1)
        # self.encodings from the parent is (num_positions, d_model)
        # A slice is returned up to the required sequence length, with an added batch dimension.
        return self.encodings.unsqueeze(0)[:, :seq_len, :]
