# -*- coding: utf-8 -*-
"""Factory for creating Multi-Head Attention modules."""
# Standard imports
from typing import Literal, Union

# Local imports
from .erpe_attention import ERPEAttention
from .mla import MultiHeadLatentAttention
from .multi_head_attention import MultiHeadAttention
from .rotary_multi_head_attention import RotaryMultiHeadAttention

AttentionType = Union[MultiHeadAttention, RotaryMultiHeadAttention, ERPEAttention, MultiHeadLatentAttention]
AttentionTypeStr = Literal["standard", "rotary", "erpe", "mla"]


class MultiHeadAttentionFactory:
    """Factory class for obtaining attention modules."""

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
            attention_args (str, dict): The type of attention module to create. If a string is provided, it should be
                one of the keys in the catalog. If a dictionary is provided, it should contain the key "type" with the
                attention type.
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.

        Returns:
            torch.nn.Module: An instance of the requested attention module.
        """
        if isinstance(attention_args, dict):
            attention_type: AttentionTypeStr = attention_args["type"]
            attention_args = {k: v for k, v in attention_args.items() if k != "type"}
        else:
            attention_type = attention_args
            attention_args = {}

        if attention_type not in cls.catalog:
            raise ValueError(
                f"Attention type '{attention_type}' is not supported.\nSupported types: {list(cls.catalog.keys())}"
            )

        attention_class = cls.catalog[attention_type]
        return (
            attention_class(embed_dim=embed_dim, num_heads=num_heads, seq_len=seq_len, **attention_args),
            attention_type,
        )
