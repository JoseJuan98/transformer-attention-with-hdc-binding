# -*- coding: utf-8 -*-
"""Binding Method Factory for obtaining the available binding methods."""

# Standard imports
from typing import Literal, Union

# First party imports
from models.binding_method.basic import AdditiveBinding, MultiplicativeBinding

BindingMethodType = Union[AdditiveBinding, MultiplicativeBinding]
BindingMethodTypeStr = Literal["additive", "multiplicative"]


class BindingMethodFactory:
    """Factory class for obtaining the available binding methods."""

    @staticmethod
    def get_binding_method(binding_method_name: str) -> BindingMethodType:
        """Returns the binding method class based on the given name.

        Args:
            binding_method_name (str): The name of the binding method.

        Returns:
            class: The binding method class.
        """
        binding_methods = {
            "additive": AdditiveBinding,
            "multiplicative": MultiplicativeBinding,
        }

        if binding_method_name not in binding_methods:
            raise ValueError(f"Binding method '{binding_method_name}' is not supported.")

        return binding_methods[binding_method_name]()
