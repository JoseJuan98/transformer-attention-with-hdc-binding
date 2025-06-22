# -*- coding: utf-8 -*-
"""Multi-Head Attention module, composed of multiple SelfAttention modules."""
# Third party imports
import torch

# Local imports
from .base_multihead_attention import BaseMultiHeadAttention
from .self_attention import SelfAttention


class MultiHeadAttention(BaseMultiHeadAttention):
    r"""Multi-Head Attention module.

    This module implements multi-head attention by composing multiple SelfAttention modules.

    Notes:
        .. math::
            \text{MultiHeadAttention}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
        where :math:`\text{head}_i = \text{SelfAttention}(QW_i^Q, KW_i^K, VW_i^V)`.
        :math:`QW_i^Q`, :math:`KW_i^K`, and :math:`VW_i^V` are linear transformations of the queries, keys, and values
        respectively for the :math:`i`-th head.

    Methods:
        forward(token_encodings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            Performs the forward pass of the multi-head attention mechanism.

    Attributes:
        embed_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimensionality of each attention head.
        attention_heads (torch.nn.ModuleList): A list of SelfAttention modules.
        W_o (torch.nn.Linear): Linear transformation for the concatenated output.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """Initializes the MultiHeadAttention module.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__(embed_dim=embed_dim, num_heads=num_heads)

        # Create multiple SelfAttention heads.
        self.attention_heads = torch.nn.ModuleList([SelfAttention(embed_dim=self.head_dim) for _ in range(num_heads)])

        # Xavier Normal initialization as in the original paper.
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the linear layers using the Xavier Normal initialization."""
        super(MultiHeadAttention, self).init_weights()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """Performs the forward pass of the multi-head attention mechanism.

        Args:
            q (torch.Tensor): The queries tensor of shape (batch_size, seq_len, embed_dim).
            k (torch.Tensor): The keys tensor of shape (batch_size, seq_len, embed_dim).
            v (torch.Tensor): The values tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): An optional mask tensor.  See `SelfAttention.forward` for details.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, *__ = q.shape

        # Split the input tensor into multiple heads.
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        # Transpose to (batch_size, num_heads, seq_len, head_dim) for parallel processing of heads.
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(dim0=1, dim1=2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(dim0=1, dim1=2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(dim0=1, dim1=2)

        # Apply each attention head to its corresponding input slice.
        # List of (batch_size, seq_len, head_dim)
        attention_outputs = [
            head(q=q[:, i, :, :], k=k[:, i, :, :], v=v[:, i, :, :], mask=mask)
            for i, head in enumerate(self.attention_heads)
        ]

        # Concatenate the outputs of all heads.
        # (batch_size, seq_len, num_heads * head_dim) = (batch_size, seq_len, embed_dim)
        concatenated_outputs = torch.cat(attention_outputs, dim=-1)

        # Apply the output projection.
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        return self.W_o(concatenated_outputs)
