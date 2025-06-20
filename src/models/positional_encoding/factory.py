# -*- coding: utf-8 -*-
"""Factory class for creating positional encodings based on configuration."""
# Standard imports
from typing import Literal, Union

# First party imports
from models.positional_encoding.adatative_sinusoidal import AdaptiveSinusoidalPositionalEncoding
from models.positional_encoding.fractional import FPEOrigPositionalEncoding
from models.positional_encoding.null_pe import NullPositionalEncoding
from models.positional_encoding.random import RandomPositionalEncoding
from models.positional_encoding.sinusoidal import SinusoidalPositionalEncoding
from models.positional_encoding.split_sinusoidal import SplitSinusoidalPositionalEncoding

TSPositionalEncodingType = Union[
    SplitSinusoidalPositionalEncoding,
    SinusoidalPositionalEncoding,
    RandomPositionalEncoding,
    NullPositionalEncoding,
    FPEOrigPositionalEncoding,
    AdaptiveSinusoidalPositionalEncoding,
]

TSPositionalEncodingTypeStr = Literal[
    "split_sinusoidal", "sinusoidal", "random", "null", "fractional_power", "adaptive_sinusoidal"
]


class PositionalEncodingFactory:
    """Factory class for creating positional encodings based on configuration."""

    catalog = {
        "split_sinusoidal": SplitSinusoidalPositionalEncoding,
        "sinusoidal": SinusoidalPositionalEncoding,
        "random": RandomPositionalEncoding,
        "null": NullPositionalEncoding,
        "fractional_power": FPEOrigPositionalEncoding,
        "adaptive_sinusoidal": AdaptiveSinusoidalPositionalEncoding,
    }

    @classmethod
    def get_positional_encoding(
        cls,
        positional_encoding_arguments: TSPositionalEncodingTypeStr | dict,
        d_model: int,
        num_positions: int,
    ) -> TSPositionalEncodingType:
        """Get the positional encoding based on the configuration.

        Args:
            positional_encoding_arguments (str, Dict[str, Any]): The type of positional encoding if a string is
                provided, or a dictionary with the key "type" and any additional parameters for the positional encoding.
            d_model (int): The dimensionality of the embeddings.
            num_positions (int): The maximum sequence length.

        Returns:
            object: The positional encoding instance.
        """

        if isinstance(positional_encoding_arguments, dict):
            positional_encoding_type: TSPositionalEncodingTypeStr = positional_encoding_arguments["type"]
            positional_encoding_arguments = {k: v for k, v in positional_encoding_arguments.items() if k != "type"}
        else:
            positional_encoding_type = positional_encoding_arguments
            positional_encoding_arguments = {}

        if positional_encoding_type not in cls.catalog:
            raise ValueError(f"Unknown positional encoding type: {positional_encoding_type}")

        # Get the positional encoding class based on the configuration and instantiate it
        positional_encoding_class = cls.catalog[positional_encoding_type]

        return positional_encoding_class(d_model=d_model, num_positions=num_positions, **positional_encoding_arguments)
