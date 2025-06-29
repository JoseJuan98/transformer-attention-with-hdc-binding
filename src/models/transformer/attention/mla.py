# -*- coding: utf-8 -*-
"""Multi-Head Latent Attention (MLA) from DeepSeek-AI et al. [1].

This module implements the Multi-head Latent Attention (MLA) mechanism proposed by DeepSeek-AI et al. [1].
MLA is designed to reduce the Key-Value (KV) cache size for efficient inference by compressing keys and values into a
shared low-rank latent vector.

The implementation is adapted to fit the project's architecture, inheriting from `BaseMultiHeadAttention` and
utilizing the project's existing rotary embedding logic.

Source code: https:/    /github.com/deepseek-ai/DeepSeek-VL2/blob/main/deepseek_vl2/models/modeling_deepseek.py

Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.

This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
and OPT implementations in this library. It has been modified from its
original forms to accommodate minor architectural differences compared
to GPT-NeoX and OPT used by the Meta AI team that trained the model.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications from the source code and paper:

- The class inherits from the project's `BaseMultiHeadAttention` to ensure architectural consistency.

- Hyperparameters are scaled to be suitable for smaller models and are made configurable, e.g. `kv_lora_rank` and
  `q_lora_rank` are for a very large model.

- The logic for applying rotary embeddings is adapted to use the pre-computed encodings passed to the forward method.

- `torch.nn.LayerNorm` is used instead of the custom `RMSNorm` implementation, for simplicity.

References:
    [1] DeepSeek-AI. (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model."
    arXiv preprint arXiv:2405.04434.
"""
# Third party imports
import torch

# Local imports
from .base_multihead_attention import BaseMultiHeadAttention


class MultiHeadLatentAttention(BaseMultiHeadAttention):
    """Multi-Head Attention with Latent Compression and Decoupled RoPE.

    Attributes:
        name (str): The name of the attention type, set to "mla".
        q_lora_rank (int): The dimension of the compressed query's latent space.
        kv_lora_rank (int): The dimension of the compressed key-value latent space (d_c in the paper).
        qk_rope_head_dim (int): The dimension of the rotary part of the query and key heads (d_R_h in the paper).
        qk_nope_head_dim (int): The dimension of the non-rotary part of the query and key heads.
        v_head_dim (int): The dimension of the value head.
        q_head_dim (int): The total dimension of the query head (qk_rope_head_dim + qk_nope_head_dim).
        softmax_scale (float): The scaling factor for the attention scores.
    """

    name = "mla"

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        seq_len: int,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        **kwargs,
    ):
        """Initializes the MultiHeadLatentAttention module.

        Args:
            embed_dim (int): The dimensionality of the input embeddings (d_model).
            num_heads (int): The number of attention heads.
            seq_len (int): The maximum sequence length of the input.
            q_lora_rank (int, optional): Rank for query compression. Defaults to 3 * kv_lora_rank.
            kv_lora_rank (int, optional): Rank for key-value compression. Defaults to 4 * head_dim.
            qk_rope_head_dim (int, optional): Dimension for the rotary part of Q/K. Defaults to head_dim / 2.
            v_head_dim (int, optional): Dimension for the value head. Defaults to head_dim.
            **kwargs: Additional keyword arguments.
        """
        super(MultiHeadLatentAttention, self).__init__(
            embed_dim=embed_dim, num_heads=num_heads, seq_len=seq_len, **kwargs
        )

        # --- Hyperparameters ---
        # Default values are scaled down from the paper to be suitable for smaller models.
        self.qk_rope_head_dim = qk_rope_head_dim if qk_rope_head_dim is not None else self.head_dim // 2
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.q_head_dim = self.head_dim
        # The total dimension required for the rotary encodings
        self.total_rope_dim = self.num_heads * self.qk_rope_head_dim

        self.v_head_dim = v_head_dim if v_head_dim is not None else self.head_dim
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else 4 * self.head_dim
        self.q_lora_rank = q_lora_rank if q_lora_rank is not None else 3 * self.kv_lora_rank

        # --- Layers ---
        # Query projection layers
        self.q_a_proj = torch.nn.Linear(self.embed_dim, self.q_lora_rank, bias=False)
        self.q_a_layernorm = torch.nn.LayerNorm(self.q_lora_rank)
        self.q_b_proj = torch.nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        # Key-Value projection layers
        self.kv_a_proj_with_mqa = torch.nn.Linear(
            self.embed_dim, self.kv_lora_rank + self.num_heads * self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = torch.nn.LayerNorm(self.kv_lora_rank)
        self.kv_b_proj = torch.nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )

        # Output projection W_o is inherited from BaseMultiHeadAttention, but it's redefined with new dimensions.
        self.W_o = torch.nn.Linear(self.num_heads * self.v_head_dim, self.embed_dim, bias=False)

        # Scaling Factor
        self.softmax_scale = self.q_head_dim**-0.5

        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the linear layers using Xavier Normal initialization."""
        super(MultiHeadLatentAttention, self).init_weights()
        for layer in [self.q_a_proj, self.q_b_proj, self.kv_a_proj_with_mqa, self.kv_b_proj]:
            torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

    # TODO: make it a trait to be reused her and in RotaryMultiHeadAttention: use _split_for_attention_heads
    def _split_for_attention_heads(
        self, tensor: torch.Tensor, batch_size: int, seq_len: int, head_dim: int
    ) -> torch.Tensor:
        """Reshapes and transposes the input tensor to split it into multiple heads.

        Args:
            tensor (torch.Tensor): The input tensor to be transposed of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: The reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        return tensor.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

    # TODO: make it a trait to be reused her and in RotaryMultiHeadAttention
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates the second half of the last dimension of the input tensor."""
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)

    # TODO: make it a trait to be reused her and in RotaryMultiHeadAttention
    def _apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, positional_encodings: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies rotary positional embedding to the query and key tensors."""
        # The incoming positional_encodings (shape: 1, seq_len, d_model) are sliced to get only the part needed
        rope_encodings = positional_encodings[:, :seq_len, : self.total_rope_dim]

        # The positional encodings are split into sin and cos components.
        sin, cos = rope_encodings.chunk(2, dim=-1)

        # The sin and cos components are duplicated and reshaped to match the full embedding dimension.
        sin = torch.stack([sin, sin], dim=-1).reshape_as(rope_encodings)
        cos = torch.stack([cos, cos], dim=-1).reshape_as(rope_encodings)

        # The sin and cos tensors are reshaped to match the multi-head attention format.
        sin = self._split_for_attention_heads(tensor=sin, batch_size=1, seq_len=seq_len, head_dim=self.qk_rope_head_dim)
        cos = self._split_for_attention_heads(tensor=cos, batch_size=1, seq_len=seq_len, head_dim=self.qk_rope_head_dim)

        # The rotation is applied to the query and key tensors.
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

    def forward(  # type: ignore[override]
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        positional_encodings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs the forward pass for Multi-Head Latent Attention.

        Args:
            q (torch.Tensor): The queries tensor of shape (batch_size, seq_len, embed_dim).
            k (torch.Tensor): The keys tensor of shape (batch_size, seq_len, embed_dim).
            v (torch.Tensor): The values tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): An optional mask tensor of shape (batch_size, seq_len).
            positional_encodings (torch.Tensor, optional): Rotary positional embeddings of shape (1, seq_len, embed_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # In self-attention, q, k, and v are the same.
        hidden_states = q
        batch_size, seq_len, _ = hidden_states.shape

        # Project query and split into non-rotary (nope) and rotary (rope) parts.
        q_proj = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_proj = q_proj.view(batch_size, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q_proj, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Project hidden_states to get the compressed KV latent vector and the rotary key part.
        kv_proj = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(kv_proj, [self.kv_lora_rank, self.num_heads * self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)

        # Apply rotary embeddings to the rotary parts of Q and K.
        q_pe, k_pe = self._apply_rotary_pos_emb(
            q=q_pe, k=k_pe, positional_encodings=positional_encodings, seq_len=seq_len
        )

        # Compute attention scores.
        # The score is a sum of two parts: the rotary attention and the latent attention.
        # This is an optimization to avoid explicitly reconstructing the full keys and values.
        rotary_attn = torch.matmul(q_pe, k_pe.transpose(-1, -2))

        # The latent part involves absorbing the key/value up-projection matrices.
        kv_b_proj_w = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
        k_absorb, v_absorb = torch.split(kv_b_proj_w, [self.qk_nope_head_dim, self.v_head_dim], dim=1)

        # Add a singleton dimension to compressed_kv to represent the head dimension for broadcasting
        compressed_kv = compressed_kv.unsqueeze(1)
        latent_attn = torch.matmul(torch.matmul(q_nope, k_absorb), compressed_kv.transpose(-1, -2))

        attention_scores = (rotary_attn + latent_attn) * self.softmax_scale

        # Apply attention mask.
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).logical_not()
            fill_value = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill(mask=mask, value=fill_value)

        # Compute attention weights.
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Compute output.
        # The output is computed by applying attention weights to the latent vector and then up-projecting.
        attn_output = torch.matmul(attention_weights, compressed_kv)
        attn_output = torch.matmul(attn_output, v_absorb.transpose(-1, -2))

        # Concatenate heads and apply final linear layer.
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        )
        return self.W_o(attn_output)
