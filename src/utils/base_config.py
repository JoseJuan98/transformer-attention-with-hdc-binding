# -*- coding: utf-8 -*-
"""Base Configuration class to be inherited by other configuration classes."""
# Standard imports
import json
import pathlib
from dataclasses import dataclass


@dataclass
class BaseConfig:
    """Base Configuration class to be inherited by other configuration classes."""

    def to_dict(self) -> dict:
        """Converts the configuration to a dictionary."""
        return self.__dict__

    def dump(self, path: str | pathlib.Path):
        """Dumps the configuration to a JSON file."""
        if isinstance(path, str):
            path = pathlib.Path(path)

        # Create the directories if it does not exist
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as file:
            json.dump(obj=self.to_dict(), fp=file, indent=4)

    def pretty_str(self):
        """List the attributes of the class."""
        return "\n".join([f"\t{k}: {v}" for k, v in self.to_dict().items()])
