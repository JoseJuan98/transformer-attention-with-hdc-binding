# -*- utf-8 -*-
"""Configuration module.

This module contains the configuration class for the project.
"""
import pathlib


class Config:
    """Configuration settings for the project.

    Attributes:
        data_dir (pathlib.Path): data directory
        model_dir (pathlib.Path): model directory
        log_dir (pathlib.Path): log directory
    """

    __base_dir = pathlib.Path(__file__).parent.parent.parent / "artifacts"

    data_dir = __base_dir / "data"
    model_dir = __base_dir / "model"
    log_dir = __base_dir / "log"
    plot_dir = __base_dir.parent / "docs" / "img"
