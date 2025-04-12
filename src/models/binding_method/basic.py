# -*- coding: utf-8 -*-
"""Basic binding methods for hyperdimensional computing (HDC) positional encodings.

It contains the addition and component-wise multiplication binding methods.
"""

# Third party imports
import torch


class EmbeddingBinding(torch.nn.Module):
    """Base class for binding methods between embeddings and positional encodings."""

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, embeddings: torch.Tensor, positional_encodings: torch.Tensor) -> torch.Tensor:
        """Applies the binding method.

        Args:
            embeddings (torch.Tensor): The embeddings' tensor.
            positional_encodings (torch.Tensor): The positional encodings' tensor.

        Returns:
            torch.Tensor: The combined tensor.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class AdditiveBinding(EmbeddingBinding):
    """Binds embeddings and positional encodings by addition."""

    @torch.no_grad()
    def forward(self, embeddings: torch.Tensor, positional_encodings: torch.Tensor) -> torch.Tensor:
        """Applies the additive binding method."""
        return embeddings + positional_encodings


class MultiplicativeBinding(EmbeddingBinding):
    """Binds embeddings and positional encodings by component-wise multiplication."""

    @torch.no_grad()
    def forward(self, embeddings: torch.Tensor, positional_encodings: torch.Tensor) -> torch.Tensor:
        """Applies the multiplicative binding method."""
        return embeddings * positional_encodings
