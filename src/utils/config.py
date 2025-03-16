# -*- utf-8 -*-
"""Configuration module.

This module contains the configuration class for the project.
"""
# Standard imports
import pathlib


class Config:
    """Configuration settings for the project.

    Attributes:
        data_dir (pathlib.Path): data directory
        model_dir (pathlib.Path): model directory
        log_dir (pathlib.Path): log directory
    """

    root_dir = pathlib.Path(__file__).parent.parent.parent / "artifacts"

    data_dir = root_dir / "data"
    model_dir = root_dir / "model"
    log_dir = root_dir / "log"
    plot_dir = root_dir.parent / "docs" / "img"

    # Create directories
    @classmethod
    def create_dirs(cls):
        """Creates directories for the project."""
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        cls.model_dir.mkdir(parents=True, exist_ok=True)
        cls.log_dir.mkdir(parents=True, exist_ok=True)
        cls.plot_dir.mkdir(parents=True, exist_ok=True)
