# -*- coding: utf-8 -*-
"""Abstract base class for attention mechanisms."""
# Standard imports
from abc import ABC, abstractmethod

# Third party imports
import torch


class BaseMultiHeadAttention(ABC, torch.nn.Module):
    """Abstract base class for attention mechanisms.

    This class defines the common interface that all attention mechanisms should implement.

    Attributes:
        embed_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimensionality of each attention head.
        W_o (torch.nn.Linear): Linear transformation for the output.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """Initializes the base attention module.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
        """
        super(BaseMultiHeadAttention, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by the number of heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim: int = embed_dim // num_heads

        # Linear transformation for the concatenated output.
        self.W_o = torch.nn.Linear(in_features=embed_dim, out_features=embed_dim)

    @abstractmethod
    def init_weights(self):
        """Initializes the weights of the attention module.

        This method must be implemented by subclasses to initialize their specific weights.
        """
        torch.nn.init.xavier_normal_(self.W_o.weight, gain=1.0)

        # Initialize bias to zero
        if self.W_o.bias is not None:
            torch.nn.init.zeros_(self.W_o.bias)

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """Performs the forward pass of the attention mechanism.

        Args:
            q (torch.Tensor): The queries tensor of shape (batch_size, seq_len, embed_dim).
            k (torch.Tensor): The keys tensor of shape (batch_size, seq_len, embed_dim).
            v (torch.Tensor): The values tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): An optional mask tensor.
            **kwargs: Additional keyword arguments specific to the attention type.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, embed_dim).
        """
        pass
