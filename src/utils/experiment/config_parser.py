# -*- coding: utf-8 -*-
"""This module is responsible for parsing the configurations for the experiments."""

# Standard imports
import json
import pathlib


class ConfigParser:
    """This class is responsible for parsing the configuration files."""

    @classmethod
    def parse_config(cls, path: pathlib.Path | str) -> dict:
        """Parse the configuration file and return the configuration as a dictionary.

        Args:
            path (pathlib.Path | str): Path to the configuration file.

        Returns:
            dict: Parsed configuration as a dictionary.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)

        if path.suffix not in [".json"]:
            raise ValueError(f"Invalid configuration file type: {path.suffix}")

        with open(path, "r") as file:
            config_dict = json.load(file)

        return config_dict
