# -*- coding: utf-8 -*-
"""Self-attention module. This implementation uses the scaled dot-product."""
# Third party imports
import torch


class SelfAttention(torch.nn.Module):
    r"""Self-attention implementation using the scaled dot-product attention and supports optional masking.

    It does *not* include multi-head attention, positional encoding, or any optimizations.

    Notes:
        .. math::
            $Attention(Q, K, V) = softmax(Q K^T / \sqrt{d_k}) V$
        where :math:`Q`, :math:`K`, and :math:`V` are the queries, keys, and values, respectively. :math:`d_k` is
        the dimension of the keys.

        If a mask is provided, it will be applied to the attention scores before the softmax operation.

        The embedding dimension "embed_dim" is used interchangeably with the dimension of the queries, keys, and values
        :math:`d_K`.

        The input tensor `token_encodings` should have shape (batch_size, seq_len, embed_dim).

    Methods:
        forward(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            Performs the forward pass of the self-attention mechanism.

    Attributes:
        d_k (int): The dimension of the keys.
        W_q (torch.nn.Linear): Linear transformation for queries.
        W_k (torch.nn.Linear): Linear transformation for keys.
        W_v (torch.nn.Linear): Linear transformation for values.
    """

    def __init__(self, embed_dim: int):
        """Initializes the SelfAttention module.

        Args:
            embed_dim (int): The dimensions of the input embeddings.
        """
        super(SelfAttention, self).__init__()
        self.d_k = embed_dim
        self.sqrt_d_k = self.d_k**0.5

        # Linear transformations for queries, keys, and values.
        # Some implementations use a single linear layer for all three transformations or not biases, but state-of-the-
        # art performance is not pursued here and the original paper uses biases.
        self.W_q = torch.nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.W_k = torch.nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.W_v = torch.nn.Linear(in_features=embed_dim, out_features=embed_dim)

        # Xavier Normal initialization as in the original paper.
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the linear layers using the Xavier Normal initialization."""
        torch.nn.init.xavier_normal_(self.W_q.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.W_k.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.W_v.weight, gain=1.0)

        # Initialize bias to zero
        if self.W_q.bias is not None:
            torch.nn.init.zeros_(self.W_q.bias)

        if self.W_k.bias is not None:
            torch.nn.init.zeros_(self.W_k.bias)

        if self.W_v.bias is not None:
            torch.nn.init.zeros_(self.W_v.bias)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Performs the forward pass of the self-attention mechanism.

        Args:
            q (torch.Tensor): The queries tensor of shape (batch_size, seq_len, embed_dim).
            k (torch.Tensor): The keys tensor of shape (batch_size, seq_len, embed_dim).
            v (torch.Tensor): The values tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): An optional mask tensor (boolean) of shape (batch_size, seq_len) or
                (batch_size, 1, seq_len) or (batch_size, seq_len, seq_len).
                If provided, masked positions will have attention scores set to -inf (or a very large negative number).
                A value of 1 in the mask indicates that the position should be *kept*, and 0 indicates it should be
                *masked*.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim).
        """

        # Project input to queries $Q$, keys $K$, and values $V$.
        # (batch_size, seq_len, embed_dim)
        queries = self.W_q(q)
        keys: torch.Tensor = self.W_k(k)
        values = self.W_v(v)

        # Calculate attention scores, $Z$ (scaled dot-product):
        # Z = Q @ K^T
        # (batch_size, seq_len, embed_dim) @ (batch_size, embed_dim, seq_len)^T -> (batch_size, seq_len, seq_len)
        # `keys.transpose(-2, -1)` swaps the seq_len and embed_dim dimensions of the keys tensor.
        attention_scores = torch.matmul(queries, keys.transpose(dim0=-2, dim1=-1))

        # Scale by the square root of the embedding dimension.
        # Z = Z / sqrt(embed_dim)
        attention_scores = attention_scores / self.sqrt_d_k

        # Apply mask (if provided). Mask should be of shape (batch_size, seq_len), (batch_size, 1, seq_len)
        # or (batch_size, seq_len, seq_len).
        # Z = Z + mask
        if mask is not None:
            if mask.dim() > 3:
                raise ValueError(
                    f"Invalid mask shape {mask.shape}. Must be (batch_size, seq_len), (batch_size, 1, seq_len) or "
                    "(batch_size, seq_len, seq_len)"
                )

            # (batch_size, seq_len) -> (batch_size, 1, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill_(
                mask=mask.logical_not(), value=float("-inf")
            )  # or value=-1e9

        # Apply softmax to get attention scores.
        # Z = softmax(Z)
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Weighted sum of values.
        # Z = Z @ V
        # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        attention_scores = torch.matmul(attention_scores, values)

        return attention_scores
