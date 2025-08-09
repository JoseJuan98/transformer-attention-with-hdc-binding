# -*- coding: utf-8 -*-
"""Utility functions."""

# Standard imports

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


def set_plot_style() -> None:
    """Set the plot style for matplotlib."""
    # Set the default figure size
    pyplot.rcParams["figure.figsize"] = (19, 10)

    # Increase text size
    pyplot.rcParams.update({"font.size": 14})

    # Set the grid style
    # pyplot.rcParams["grid.linestyle"] = "--"
    # pyplot.rcParams["grid.linewidth"] = 0.5
    # pyplot.rcParams["grid.alpha"] = 0.7

    # Set the legend font size
    pyplot.rcParams["legend.fontsize"] = 14

    # Set the axes label size
    pyplot.rcParams["axes.labelsize"] = 16
