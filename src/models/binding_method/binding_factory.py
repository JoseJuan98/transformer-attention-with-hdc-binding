# -*- coding: utf-8 -*-
"""Binding Method Factory for obtaining the available binding methods."""

# Standard imports
from typing import Literal, Union

# First party imports
from models.binding_method.basic import AdditiveBinding
from models.binding_method.multiplicative import MultiplicativeBinding

BindingMethodType = Union[AdditiveBinding, MultiplicativeBinding]
BindingMethodTypeStr = Literal["additive", "multiplicative"]


class BindingMethodFactory:
    """Factory class for obtaining the available binding methods."""

    catalog = {
        "additive": AdditiveBinding,
        "multiplicative": MultiplicativeBinding,
    }

    @classmethod
    def get_binding_method(cls, binding_method_name: str, embedding_dim: int) -> BindingMethodType:
        """Returns the binding method class based on the given name.

        Args:
            binding_method_name (str): The name of the binding method.
            embedding_dim (int): The dimension of the embeddings.

        Returns:
            class: The binding method class.
        """

        if binding_method_name not in cls.catalog:
            raise ValueError(
                f"Binding method '{binding_method_name}' is not supported.\nSupported methods: {cls.catalog.keys()}"
            )

        if binding_method_name == "multiplicative":
            return cls.catalog[binding_method_name](embedding_dim=embedding_dim)

        return cls.catalog[binding_method_name]()
