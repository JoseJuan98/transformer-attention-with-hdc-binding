# -*- coding: utf-8 -*-
"""Rotary Multi-Head Attention module.

The `RotaryMultiHeadAttention` class and `apply_rotary_pos_emb` method in this module are adapted from the Hugging Face
Transformers library, licensed under the Apache License, Version 2.0.

The implementation is based on the RoFormer paper, which introduces Rotary Position Embedding (RoPE) [1] to apply
positional information directly within the self-attention mechanism. This version is fully vectorized compared to the
parent class `MultiHeadAttention` and is designed to work with the project's architecture.

Original code: https://github.com/huggingface/transformers/blob/0725cd6953803b8aacfc85288cbfb83dea30c469/src/transformers/models/roformer/modeling_roformer.py#L189-L324

Copyright 2021 The HuggingFace Inc. team. All rights reserved.

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

- The core rotation logic was extracted from the original `RoFormerSelfAttention` class into the method
`apply_rotary_pos_emb`.

- Created a `RotaryMultiHeadAttention` class that inherits from the project's `MultiHeadAttention` class to ensure
architectural consistency.

- The `forward` method was adapted to accept rotary embeddings as a separate argument.

References:
    [1] Su Jianlin, et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv, 2023,
    arxiv.org/abs/2104.09864.
"""
# Third party imports
import torch

# First party imports
from models.transformer.attention.multi_head_attention import MultiHeadAttention


class RotaryMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention with Rotary Positional Embeddings (RoPE).

    Attributes:
        name (str): The name of the attention type, set to "rotary".
        embed_dim (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimensionality of each attention head, calculated as embed_dim // num_heads.
        sqrt_head_dim (float): The square root of the head dimension, used for scaling attention scores.
        W_q (torch.nn.Linear): Linear layer for projecting queries.
        W_k (torch.nn.Linear): Linear layer for projecting keys.
        W_v (torch.nn.Linear): Linear layer for projecting values.
        W_o (torch.nn.Linear): Linear layer for the output projection.
    """

    name = "rotary"

    def __init__(self, embed_dim: int, num_heads: int):
        # Overrides __init__ to create the specific layers needed for this attention type
        super(MultiHeadAttention, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by the number of heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sqrt_head_dim = self.head_dim**0.5

        # Q, K, V projection layers
        self.W_q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection layer
        self.W_o = torch.nn.Linear(embed_dim, embed_dim)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the linear layers using the Xavier Normal initialization."""
        torch.nn.init.xavier_normal_(self.W_q.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.W_k.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.W_v.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.W_o.weight, gain=1.0)
        if self.W_o.bias is not None:
            torch.nn.init.zeros_(self.W_o.bias)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates the second half of the last dimension of the input tensor.

        This is used to apply the rotary positional embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (..., head_dim).

        Returns:
            torch.Tensor: The tensor with the second half of the last dimension rotated.
        """
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        return torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)

    def _apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, positional_encodings: torch.Tensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies rotary positional embedding to the query and key tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
            cos (torch.Tensor): Cosine component of the rotary embeddings.
            sin (torch.Tensor): Sine component of the rotary embeddings.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors.
        """
        if seq_len is None:
            seq_len = q.size(2)

        # The positional encodings are prepared for rotation.
        sin, cos = positional_encodings.chunk(2, dim=-1)
        sin = torch.stack([sin, sin], dim=-1).reshape_as(positional_encodings)
        cos = torch.stack([cos, cos], dim=-1).reshape_as(positional_encodings)

        # Reshape and transpose to match the expected shape for multi-head attention
        cos = cos.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        sin = sin.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotation
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

    def _split_for_attention_heads(self, tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshapes and transposes the input tensor to split it into multiple heads.

        Args:
            tensor (torch.Tensor): The input tensor to be transposed of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: The reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        positional_encodings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs the forward pass with RoPE.

        Args:
            q (torch.Tensor): The queries tensor of shape (batch_size, seq_len, embed_dim).
            k (torch.Tensor): The keys tensor of shape (batch_size, seq_len, embed_dim).
            v (torch.Tensor): The values tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): An optional mask tensor.
            positional_encodings (torch.Tensor): Rotary positional embeddings of shape (1, seq_len, embed_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim).
        """
        if positional_encodings is None:
            raise ValueError("'positional_encodings' must be provided for RotaryMultiHeadAttention.")

        batch_size, seq_len, _ = q.shape

        # Project and split into heads
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q_proj = self._split_for_attention_heads(tensor=self.W_q(q), batch_size=batch_size, seq_len=seq_len)
        k_proj = self._split_for_attention_heads(tensor=self.W_k(k), batch_size=batch_size, seq_len=seq_len)
        v_proj = self._split_for_attention_heads(tensor=self.W_v(v), batch_size=batch_size, seq_len=seq_len)

        # Apply Rotary Positional Embeddings
        q_rot, k_rot = self._apply_rotary_pos_emb(
            q=q_proj, k=k_proj, positional_encodings=positional_encodings, seq_len=seq_len
        )

        # Compute attention scores
        # Z = Q @ K^T / sqrt(head_dim)
        attention_scores = torch.matmul(q_rot, k_rot.transpose(-1, -2)) / self.sqrt_head_dim

        # Z = Z + mask
        if mask is not None:
            # mask shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.logical_not()
            # Get the minimum value for the data type used
            fill_value = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill_(mask=mask, value=fill_value)

        # Compute attention weights
        # Z = softmax(Z)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention to values
        # Z = Z @ V
        output = torch.matmul(attention_weights, v_proj)

        # Concatenate heads and apply final linear layer
        # Shape: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.W_o(output)
