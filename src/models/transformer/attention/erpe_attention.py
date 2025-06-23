# -*- coding: utf-8 -*-
"""Multi-Head Attention with efficient Relative Position Encoding (eRPE) from Foumani et al. [1].

This module implements the attention mechanism from Foumani et al. [1], which incorporates a learnable, scalar-based
relative position bias into the attention scores.

Source code:
https://github.com/Navidfoumani/ConvTran/blob/148afb6ca549915b7e78b05e2ec3b4ba6e052341/Models/Attention.py#L45C1-L95C1

Modifications:

- The original code was adapted to fit the project's architecture, including the use of a base class for multi-head
  attention.

- The relative position index is a 2D matrix directly usable for the attention computation, avoiding the need for
  additional reshaping or gathering operations during the forward pass.

References:
    [1] Foumani, N. M., Tan, C. W., Webb, G. I., & Salehi, M. (2023). "Improving position encoding of transformers for
    multivariate time series classification." Data Mining and Knowledge Discovery.
    https://link.springer.com/content/pdf/10.1007/s10618-023-00948-2.pdf
"""

# Third party imports
import torch

# Local imports
from .base_multihead_attention import BaseMultiHeadAttention


class ERPEAttention(BaseMultiHeadAttention):
    """Multi-Head Attention with efficient Relative Position Encoding (eRPE).

    This attention mechanism extends the standard multi-head attention by adding a learnable relative position bias
    to the attention scores. The bias for each relative position is a learnable scalar, unique per attention head.

    The key steps are:
    1. Standard multi-head attention score calculation.
    2. A learnable bias table of shape (2*seq_len - 1, num_heads) is created.
    3. An index is used to gather the appropriate biases for each query-key pair, forming a (batch, heads, seq, seq) bias matrix.
    4. This bias is added to the attention scores *after* the softmax operation, as described in the paper.

    Args:
        embed_dim (int): The dimensionality of the input embeddings (d_model).
        num_heads (int): The number of attention heads.
        seq_len (int): The maximum sequence length of the input.
    """

    def __init__(self, embed_dim: int, num_heads: int, seq_len: int, **kwargs):
        super(ERPEAttention, self).__init__(embed_dim=embed_dim, num_heads=num_heads, seq_len=seq_len, **kwargs)

        # Linear projections for Q, K, V
        self.W_q = torch.nn.Linear(embed_dim, embed_dim)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim)
        # W_o is already defined in BaseMultiHeadAttention

        # Scaling factor for attention scores
        self.sqrt_head_dim = self.head_dim**0.5

        # eRPE specific parameters
        # A learnable table for relative position biases. One scalar per head for each possible relative position.
        self.relative_bias_table = torch.nn.Parameter(torch.zeros(2 * self.seq_len - 1, num_heads))

        # Pre-compute the relative position indices to be used for gathering biases.
        # This avoids re-computation in the forward pass.
        pos = torch.arange(self.seq_len)
        relative_pos = pos[None, :] - pos[:, None]  # Shape: (seq_len, seq_len)
        relative_pos += self.seq_len - 1  # Shift to be non-negative indices
        self.register_buffer("relative_position_index", relative_pos)

        self.init_weights()

    def init_weights(self):
        """Initializes weights with Xavier Normal."""
        super(ERPEAttention, self).init_weights()
        torch.torch.nn.init.xavier_normal_(self.W_q.weight, gain=1.0)
        torch.torch.nn.init.xavier_normal_(self.W_k.weight, gain=1.0)
        torch.torch.nn.init.xavier_normal_(self.W_v.weight, gain=1.0)

    def _split_for_attention_heads(self, tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshapes and transposes the input tensor to split it into multiple heads.

        Args:
            tensor (torch.Tensor): The input tensor to be transposed of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: The reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """Forward pass for eRPE Attention.

        Args:
            q (torch.Tensor): The queries tensor of shape (batch_size, seq_len, embed_dim).
            k (torch.Tensor): The keys tensor of shape (batch_size, seq_len, embed_dim).
            v (torch.Tensor): The values tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): An optional mask tensor.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = q.shape

        # Project and reshape Q, K, V for multi-head processing
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q_proj = self._split_for_attention_heads(tensor=self.W_q(q), batch_size=batch_size, seq_len=seq_len)
        k_proj = self._split_for_attention_heads(tensor=self.W_k(k), batch_size=batch_size, seq_len=seq_len)
        v_proj = self._split_for_attention_heads(tensor=self.W_v(v), batch_size=batch_size, seq_len=seq_len)

        # Calculate scaled dot-product attention scores
        # Z = Q @ K^T / sqrt(head_dim)
        attention_scores = torch.matmul(q_proj, k_proj.transpose(-1, -2)) / self.sqrt_head_dim

        # Apply mask if provided
        if mask is not None:
            # mask shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.logical_not()

            # Get the minimum value for the data type used
            fill_value = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill_(mask=mask, value=fill_value)

        # Apply softmax to get attention probabilities
        # Z = softmax(Z)
        attention_weights = torch.torch.nn.functional.softmax(attention_scores, dim=-1)

        # eRPE bias addition
        # Gather the relative position biases from the learnable table.
        # self.relative_position_index has shape (seq_len, seq_len)
        relative_biases = self.relative_bias_table[self.relative_position_index]  # Shape: (seq_len, seq_len, num_heads)
        relative_biases = relative_biases.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, num_heads, seq_len, seq_len)

        # Add the relative bias to the attention probabilities (post-softmax, as per paper's code)
        attn_weights_with_bias = attention_weights + relative_biases

        # Apply attention to values
        # Z = Z @ V
        output = torch.matmul(attn_weights_with_bias, v_proj)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.W_o(output)
