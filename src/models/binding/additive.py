# -*- coding: utf-8 -*-
"""Additive binding method for hyperdimensional computing (HDC) positional encodings."""
# Third party imports
import torch

# First party imports
from models.binding.basic import EmbeddingBinding


class AdditiveBinding(EmbeddingBinding):
    """Binds embeddings and positional encodings by addition."""

    name = "additive"

    def __init__(self, embedding_dim: int):
        super(AdditiveBinding, self).__init__()
        self.embedding_dim = embedding_dim

    @torch.no_grad()
    def forward(self, embeddings: torch.Tensor, positional_encodings: torch.Tensor) -> torch.Tensor:
        """Applies the additive binding method."""
        return embeddings + positional_encodings
