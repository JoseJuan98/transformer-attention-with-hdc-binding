# -*- coding: utf-8 -*-
"""Factory class for creating embedding based on configuration."""
# Standard imports
from typing import Literal, Union

# Third party imports
import torch

# First party imports
from models.embedding.ts_conv_embedding import TimeSeries1dConvEmbedding

EmbeddingType = Union[torch.nn.Module, TimeSeries1dConvEmbedding]
EmbeddingTypeStr = Literal["1d_conv", "2d_conv", "linear_projection"]


class EmbeddingFactory:
    """Factory class for creating embedding based on configuration."""

    @staticmethod
    def get_embedding(embedding_type: EmbeddingTypeStr, num_channels: int, d_model: int) -> torch.nn.Module:
        """Get the embedding based on the configuration.

        Args:
            embedding_type (str): The type of the embedding. Valid options are "linear_projection", "1d_conv", and
                "2d_conv".
            num_channels (int): The number of input channels.
            d_model (int): The dimension of the output embeddings.

        Returns:
            torch.nn.Module: The embedding instance.
        """
        if embedding_type == "linear_projection":

            return torch.nn.Linear(in_features=num_channels, out_features=d_model, bias=False)

        elif embedding_type == "1d_conv":

            return TimeSeries1dConvEmbedding(
                c_in=num_channels, d_model=d_model, kernel_size=3, padding_mode="circular", bias=False
            )

        elif embedding_type == "2d_conv":
            # TODO:
            raise NotImplementedError("2D Convolutional embedding is not implemented yet.")

        else:
            raise ValueError(f"Unknown binding method: {embedding_type}")
