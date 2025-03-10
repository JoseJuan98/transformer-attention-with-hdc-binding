# -*- coding: utf-8 -*-
"""Self-attention module. This implementation uses the scaled dot-product."""
# Third party imports
import torch


class SelfAttention(torch.nn.Module):
    """Self-attention implementation using the scaled dot-product attention and supports optional masking.

    It does *not* include multi-head attention, positional encoding, or any optimizations.
    """

    def __init__(self, embed_dim: int):
        """Initializes the SelfAttention module.

        Args:
            embed_dim (int): The dimensions of the input embeddings.
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        # Linear transformations for queries, keys, and values.
        self.query_projection = torch.nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.key_projection = torch.nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.value_projection = torch.nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Performs the forward pass of the self-attention mechanism.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): An optional mask tensor of shape (batch_size, seq_len) or
                (batch_size, 1, seq_len) or (batch_size, seq_len, seq_len).
                If provided, masked positions will have attention scores set to -inf (or a very large negative number).
                A value of 1 in the mask indicates that the position should be *kept*, and 0 indicates it should be
                *masked*.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim).
        """

        # Project input to queries, keys, and values.
        # (batch_size, seq_len, embed_dim)
        queries = self.query_projection(x)
        keys: torch.Tensor = self.key_projection(x)
        values = self.value_projection(x)

        # Calculate attention scores (scaled dot-product).
        # (batch_size, seq_len, embed_dim) @ (batch_size, embed_dim, seq_len) -> (batch_size, seq_len, seq_len)
        attention_scores = torch.matmul(queries, keys.transpose(dim0=-2, dim1=-1))
        attention_scores = attention_scores / self.embed_dim**0.5

        # Apply mask (if provided). Mask should be of shape (batch_size, seq_len) or (batch_size, 1, seq_len)
        if mask is not None:
            if mask.dim() > 3:
                raise ValueError(
                    "Invalid mask shape. Must be (batch_size, seq_len), (batch_size, 1, seq_len) or "
                    "(batch_size, seq_len, seq_len)"
                )

            # (batch_size, seq_len) -> (batch_size, 1, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention scores.
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Weighted sum of values.
        # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        output = torch.matmul(attention_weights, values)

        return output
