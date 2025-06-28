# -*- coding: utf-8 -*-
"""Factory for creating Multi-Head Attention modules."""
# Standard imports
from typing import Literal, Union

# First party imports
from models.arg_formatter import ArgFormatter

# Local imports
from .erpe_attention import ERPEAttention
from .mla import MultiHeadLatentAttention
from .multi_head_attention import MultiHeadAttention
from .rotary_multi_head_attention import RotaryMultiHeadAttention

AttentionType = Union[MultiHeadAttention, RotaryMultiHeadAttention, ERPEAttention, MultiHeadLatentAttention]
AttentionTypeStr = Literal["standard", "rotary", "erpe", "mla"]


class MultiHeadAttentionFactory(ArgFormatter):
    """Factory class for obtaining attention modules."""

    component_name = "attention"
    catalog = {
        "standard": MultiHeadAttention,
        "rotary": RotaryMultiHeadAttention,
        "erpe": ERPEAttention,
        "mla": MultiHeadLatentAttention,
    }

    @classmethod
    def get_attention_module(
        cls, attention_args: AttentionTypeStr | dict, embed_dim: int, num_heads: int, seq_len: int
    ) -> tuple[AttentionType, AttentionTypeStr]:
        """Returns the attention module class based on the given name.

        Args:
            attention_args (str | dict): The type of attention module to create. If a string is provided, it should be
                one of the keys in the catalog. If a dictionary is provided, it should contain the key "type" with the
                attention type.
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.

        Returns:
            torch.nn.Module: An instance of the requested attention module.
            str: The type of attention module created.
        """
        attention_type, attention_args = cls.format_arguments(arguments=attention_args)
        attention_type: AttentionTypeStr = attention_type

        attention_class = cls.catalog[attention_type]
        return (
            attention_class(embed_dim=embed_dim, num_heads=num_heads, seq_len=seq_len, **attention_args),
            attention_type,
        )
