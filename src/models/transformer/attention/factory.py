# -*- coding: utf-8 -*-
"""Factory for creating Multi-Head Attention modules."""
# Standard imports
from typing import Literal, Union

# Local imports
from .multi_head_attention import MultiHeadAttention
from .rotary_multi_head_attention import RotaryMultiHeadAttention

AttentionType = Union[MultiHeadAttention, RotaryMultiHeadAttention]
AttentionTypeStr = Literal["standard", "rotary"]


class MultiHeadAttentionFactory:
    """Factory class for obtaining attention modules."""

    catalog = {
        "standard": MultiHeadAttention,
        "rotary": RotaryMultiHeadAttention,
    }

    @classmethod
    def get_attention_module(cls, attention_type: AttentionTypeStr, embed_dim: int, num_heads: int) -> AttentionType:
        """Returns the attention module class based on the given name.

        Args:
            attention_type (AttentionTypeStr): The name of the attention mechanism.
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.

        Returns:
            torch.nn.Module: An instance of the requested attention module.
        """
        if attention_type not in cls.catalog:
            raise ValueError(
                f"Attention type '{attention_type}' is not supported.\nSupported types: {list(cls.catalog.keys())}"
            )

        attention_class = cls.catalog[attention_type]
        return attention_class(embed_dim=embed_dim, num_heads=num_heads)
