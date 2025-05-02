# -*- coding: utf-8 -*-
"""Factory class for creating positional encodings based on configuration."""
# Standard imports
from typing import Literal, Union

# First party imports
from models.positional_encoding.random_pe import RandomPositionalEncoding
from models.positional_encoding.sinusoidal import SinusoidalPositionalEncoding
from models.positional_encoding.ts_sinusoidal import TimeSeriesSinusoidalPositionalEncoding

TSPositionalEncodingType = Union[
    TimeSeriesSinusoidalPositionalEncoding, SinusoidalPositionalEncoding, RandomPositionalEncoding
]

TSPositionalEncodingTypeStr = Literal["ts_sinusoidal", "sinusoidal", "random"]


class PositionalEncodingFactory:
    """Factory class for creating positional encodings based on configuration."""

    catalog = {
        "ts_sinusoidal": TimeSeriesSinusoidalPositionalEncoding,
        "sinusoidal": SinusoidalPositionalEncoding,
        "random": RandomPositionalEncoding,
    }

    @classmethod
    def get_positional_encoding(
        cls, positional_encoding_type: str, d_model: int, num_positions: int, seed: int, **kwargs
    ) -> TSPositionalEncodingType:
        """Get the positional encoding based on the configuration.

        Args:
            positional_encoding_type (str): The type of the positional encoding.
            d_model (int): The dimensionality of the embeddings.
            num_positions (int): The maximum sequence length.
            **kwargs: Additional parameters for the positional encoding.

        Returns:
            object: The positional encoding instance.
        """
        if positional_encoding_type not in cls.catalog:
            raise ValueError(f"Unknown positional encoding type: {positional_encoding_type}")

        # Get the positional encoding class based on the configuration and instantiate it
        positional_encoding_class = cls.catalog[positional_encoding_type]

        return positional_encoding_class(d_model=d_model, num_positions=num_positions, seed=seed, **kwargs)
