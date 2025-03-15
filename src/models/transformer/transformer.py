# -*- coding: utf-8 -*-
"""Transformer model implementation."""
# transformer.py
# Standard imports
import math

# Third party imports
import torch

# Local imports
from .encoder import Encoder
from .positional_encoding import SinusoidalPositionalEncoding


class EncoderOnlyTransformerClassifier(torch.nn.Module):
    """Implements the Encoder-only Transformer Classifier.

    This class implements an encoder-only Transformer model for classification. It consists of an embedding layer,
    positional encoding,a Transformer encoder, and a classification head.

    Args:
        num_layers (int): The number of encoder layers.
        d_model (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the inner layer of the feed-forward network.
        vocab_size (int): The size of the vocabulary.
        max_len (int): The maximum sequence length.
        dropout (float, optional): The dropout probability. Defaults to 0.1.

    Methods:
        forward(x, mask): Performs a forward pass through the model.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_len: int,
        positional_encoding: SinusoidalPositionalEncoding,
        num_classes: int,
        dropout: float = 0.1,
    ):
        """Initializes the EncoderOnlyTransformerClassifier model.

        Args:
            num_layers (int): The number of encoder layers.
            d_model (int): The dimensionality of the embeddings.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner layer of the feed-forward network.
            vocab_size (int): The size of the vocabulary.
            max_len (int): The maximum sequence length.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
        """
        super(EncoderOnlyTransformerClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = positional_encoding(d_model=d_model, max_len=max_len)
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
        self.fc = torch.nn.Linear(in_features=d_model, out_features=num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len).
            mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 2).
        """
        # Scale embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x, mask)
        # Global average pooling over the sequence length
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
