# -*- coding: utf-8 -*-
"""Argument formatter for model configuration parameters."""
from typing import Any, Dict


class ArgFormatter:
    """Argument formatter for model configuration parameters."""
    catalog: Dict[str, Any]
    component_name: str

    @classmethod
    def format_arguments(cls, arguments: str | dict) -> tuple[str, dict]:
        """Format arguments from the model configuration.

        Args:
            arguments (str | dict): The component type module to create. If a string is provided, it should be
                one of the keys in the catalog. If a dictionary is provided, it should contain the key "type" with the
                attention type. The rest of arguments in the dictionary are kept the same.

        Returns:
            str: the type of the module
            dict: the arguments' dictionary. It's an empty dictionary if no arguments are provided.

        Raises:
            ValueError: If the type provided is not in the catalog.
        """
        if isinstance(arguments, dict):
            arg_type = arguments["type"]
            formatted_args = {k: v for k, v in arguments.items() if k != "type"}
        else:
            arg_type = arguments
            formatted_args = {}

        if arg_type not in cls.catalog:
            raise ValueError(
                f"{cls.component_name.replace('_', ' ').title()} type '{arg_type}' is not supported.\nSupported types: {list(cls.catalog.keys())}"
            )

        return arg_type, formatted_args
