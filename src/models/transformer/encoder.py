# -*- coding: utf-8 -*-
"""Encoder module."""
# Third party imports
import torch

# Local imports
from .attention.factory import AttentionTypeStr, MultiHeadAttentionFactory
from .feed_forward import FeedForward


class EncoderLayer(torch.nn.Module):
    """Implements a single encoder layer.

    This class represents a single layer in the Transformer encoder.  It consists of a multi-head self-attention
    mechanism, followed by a position-wise feed-forward network.  Residual connections and layer normalization are
    applied around both sub-layers.

    Args:
        d_model (int): The dimensionality of the input and output embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the inner layer of the feed-forward network.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        mhsa_type (AttentionTypeStr, optional): The type of multi-head self-attention to use. Defaults to "standard".

    Methods:
        forward(x, mask) -> torch.Tensor:
            Applies the encoder layer to the input tensor.

    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, mhsa_type: AttentionTypeStr = "standard"
    ):
        """Initializes the EncoderLayer module.

        Args:
            d_model (int): The dimensionality of the input and output embeddings.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner layer of the feed-forward network.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
            mhsa_type (AttentionTypeStr, optional): The type of multi-head self-attention to use. Defaults to "standard".
        """
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttentionFactory.get_attention_module(
            attention_type=mhsa_type, embed_dim=d_model, num_heads=num_heads
        )
        self.mhsa_type = mhsa_type
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)

        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, positional_encodings: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Applies the encoder layer to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
            positional_encodings (torch.Tensor, optional): Positional encodings for rotary attention. Defaults to None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        # Multi-head attention and residual connection.
        if self.mhsa_type == "rotary":
            attn_output = self.self_attention(q=x, k=x, v=x, mask=mask, positional_encodings=positional_encodings)
        else:
            attn_output = self.self_attention(q=x, k=x, v=x, mask=mask)

        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward and residual connection.
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class Encoder(torch.nn.Module):
    """Implements the Transformer encoder.

    This class stacks multiple encoder layers to form the complete Transformer encoder.

    Args:
        num_layers (int): The number of encoder layers.
        d_model (int): The dimensionality of the input and output embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the inner layer of the feed-forward network.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        mhsa_type (AttentionTypeStr, optional): The type of multi-head self-attention to use. Defaults to "standard".

    Methods:
        forward(x, mask): Applies the encoder to the input tensor.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout=0.1,
        mhsa_type: AttentionTypeStr = "standard",
    ):
        """Initializes the Encoder module.

        Args:
            num_layers (int): The number of encoder layers.
            d_model (int): The dimensionality of the input and output embeddings.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner layer of the feed-forward network.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
            mhsa_type (AttentionTypeStr, optional): The type of multi-head self-attention to use. Defaults to "standard".
        """
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, mhsa_type=mhsa_type)
                for _ in range(num_layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, positional_encodings: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Applies the encoder to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
            positional_encodings (torch.Tensor, optional): Positional encodings for rotary attention. Defaults to None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask, positional_encodings)
        return self.norm(x)
