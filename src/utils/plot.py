# -*- coding: utf-8 -*-
"""Utility functions."""

# Third party imports
from matplotlib import pyplot

# First party imports
from utils.config import Config


def save_plot(filename: str) -> None:
    """Function to save the plots"""
    plot_path = Config.plot_dir / filename

    # make dir if it doesn't exist yet
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    pyplot.savefig(plot_path, bbox_inches="tight")
