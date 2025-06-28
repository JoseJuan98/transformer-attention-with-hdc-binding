# -*- coding: utf-8 -*-
"""Factory class for creating embedding based on configuration."""
# Standard imports
from typing import Literal, Union

# First party imports
from models.arg_formatter import ArgFormatter
from models.embedding.linear_projection import LinearProjection
from models.embedding.temporal_spatial import SpatialTemporalEmbedding
from models.embedding.ts_convolutional import TimeSeries1dConvEmbedding

EmbeddingType = Union[TimeSeries1dConvEmbedding, LinearProjection, SpatialTemporalEmbedding]
EmbeddingTypeStr = Literal["1d_conv", "linear_projection", "spatial_temporal"]


class EmbeddingFactory(ArgFormatter):
    """Factory class for creating embedding based on configuration."""

    catalog = {
        "linear_projection": LinearProjection,
        "1d_conv": TimeSeries1dConvEmbedding,
        "spatial_temporal": SpatialTemporalEmbedding,
    }

    @classmethod
    def get_embedding(cls, embedding_args: EmbeddingTypeStr | dict, num_channels: int, d_model: int) -> EmbeddingType:
        """Get the embedding based on the configuration.

        Args:
            embedding_args (str, Dict[str, Any]): The type of embedding if a string is provided, or a dictionary
            with the key "type" and any additional parameters for the embedding. Valid options are "linear_projection",
                "1d_conv", and "2d_conv".
            num_channels (int): The number of input channels.
            d_model (int): The dimension of the output embeddings.

        Returns:
            object: The embedding instance.
        """
        embedding_type, embedding_args = cls.format_arguments(arguments=embedding_args)
        embedding_type: EmbeddingTypeStr = embedding_type

        # Get the embedding class from the catalog
        embedding_class = cls.catalog[embedding_type]

        return embedding_class(in_features=num_channels, out_features=d_model, **embedding_args)
