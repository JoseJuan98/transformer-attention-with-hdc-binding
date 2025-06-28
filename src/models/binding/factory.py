# -*- coding: utf-8 -*-
"""Binding Method Factory for obtaining the available binding methods."""

# Standard imports
from typing import Literal, Union

# First party imports
from models.arg_formatter import ArgFormatter
from models.binding.additive import AdditiveBinding
from models.binding.convolutional import ConvolutionalBinding
from models.binding.multiplicative import MultiplicativeBinding

BindingMethodType = Union[AdditiveBinding, ConvolutionalBinding, MultiplicativeBinding, None]
BindingMethodTypeStr = Literal["additive", "multiplicative", "convolutional", "identity"]


class BindingMethodFactory(ArgFormatter):
    """Factory class for obtaining the available binding methods."""

    component_name = "binding_method"
    catalog = {
        "additive": AdditiveBinding,
        "multiplicative": MultiplicativeBinding,
        "convolutional": ConvolutionalBinding,
        # Created for the RoPE positional embeddings, which does not require a binding method
        "identity": None,
    }

    @classmethod
    def get_binding_method(cls, binding_method_name: str, embedding_dim: int) -> BindingMethodType:
        """Returns the binding method class based on the given name.

        Args:
            binding_method_name (str): The name of the binding method.
            embedding_dim (int): The dimension of the embeddings.

        Returns:
            object: The binding method object.
        """
        binding_method_name, _ = cls.format_arguments(arguments=binding_method_name)

        if binding_method_name == "identity":
            # Return None for identity binding method, which means no binding is applied
            return None

        return cls.catalog[binding_method_name](embedding_dim=embedding_dim)  # type: ignore [misc]
