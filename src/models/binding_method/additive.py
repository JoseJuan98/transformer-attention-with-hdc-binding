# -*- coding: utf-8 -*-
"""Additive binding method for hyperdimensional computing (HDC) positional encodings."""
# Third party imports
import torch

# First party imports
from models.binding_method.basic import EmbeddingBinding


class AdditiveBinding(EmbeddingBinding):
    """Binds embeddings and positional encodings by addition."""

    name = "additive"

    def __init__(self, embedding_dim: int):
        super(AdditiveBinding, self).__init__()
        self.embedding_dim = embedding_dim
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, embeddings: torch.Tensor, positional_encodings: torch.Tensor) -> torch.Tensor:
        """Applies the additive binding method."""
        # Bind them
        combined = embeddings + positional_encodings

        return self.layer_norm(combined)
