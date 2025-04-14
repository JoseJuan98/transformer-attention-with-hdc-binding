# -*- coding: utf-8 -*-
"""Factory class for creating positional encodings based on configuration."""
# Standard imports
from typing import Literal

# First party imports
from models.positional_encoding.ts_sinusoidal import TimeSeriesSinusoidalPositionalEncoding

TSPositionalEncodingType = TimeSeriesSinusoidalPositionalEncoding

TSPositionalEncodingTypeStr = Literal["ts_sinusoidal"]


class PositionalEncodingFactory:
    """Factory class for creating positional encodings based on configuration."""

    catalog = {
        "ts_sinusoidal": TimeSeriesSinusoidalPositionalEncoding,
    }

    @classmethod
    def get_positional_encoding(cls, positional_encoding_type: str, **kwargs) -> TSPositionalEncodingType:
        """Get the positional encoding based on the configuration.

        Args:
            positional_encoding_type (str): The type of the positional encoding.
            **kwargs: Additional parameters for the positional encoding.

        Returns:
            object: The positional encoding instance.
        """
        if positional_encoding_type not in cls.catalog:
            raise ValueError(f"Unknown positional encoding type: {positional_encoding_type}")

        # Get the positional encoding class based on the configuration and instantiate it
        positional_encoding_class = cls.catalog[positional_encoding_type]

        return positional_encoding_class(**kwargs)
