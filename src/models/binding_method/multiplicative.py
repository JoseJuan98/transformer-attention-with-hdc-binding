# -*- coding: utf-8 -*-
"""Component-Wise Multiplicative binding method for hyperdimensional computing (HDC) positional encodings."""
# Third party imports
import torch

# First party imports
from models.binding_method.basic import EmbeddingBinding


class MultiplicativeBinding(EmbeddingBinding):
    """Binds embeddings and positional encodings by component-wise multiplication."""

    name = "multiplicative"

    def __init__(self, embedding_dim: int):
        super(MultiplicativeBinding, self).__init__()
        self.embedding_dim = embedding_dim
        self.layer_norm_after = torch.nn.LayerNorm(embedding_dim)

    def forward(self, embeddings: torch.Tensor, positional_encodings: torch.Tensor) -> torch.Tensor:
        """Applies the multiplicative binding method."""

        # Bind them
        combined = embeddings * positional_encodings

        # Apply layer norm after binding
        return self.layer_norm_after(combined)
