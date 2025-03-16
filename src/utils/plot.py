# -*- coding: utf-8 -*-
"""Utility functions."""

# Standard imports
import logging
import os
import pathlib

# Third party imports
import pandas
import seaborn
from matplotlib import pyplot

# First party imports
from utils.config import Config


def save_plot(filename: str) -> None:
    """Function to save the plots"""
    plot_path = Config.plot_dir / filename

    # make dir if it doesn't exist yet
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    pyplot.savefig(plot_path, bbox_inches="tight")


def plot_csv_logger_metrics(
    plots_path: pathlib.Path, csv_dir: str, experiment: str, logger: logging.Logger | None = None
) -> None:
    """Plot the metrics."""
    metrics = pandas.read_csv(filepath_or_buffer=os.path.join(csv_dir, "metrics.csv"))

    metrics.drop(columns=["step", "n_samples"], axis=1, inplace=True)
    metrics.set_index("epoch", inplace=True)

    test_loss = metrics["test_loss"].dropna(how="all").mean()
    test_acc = metrics["test_acc"].dropna(how="all").mean()

    if logger is None:
        print(f"\nExperiment {experiment}\n\tTest loss: {test_loss}.\n\tTest accuracy: {test_acc}.\n\n")
    else:
        logger.info(f"\nExperiment {experiment}\n\tTest loss: {test_loss}.\n\tTest accuracy: {test_acc}.\n\n")

    plots_path.mkdir(parents=True, exist_ok=True)

    metrics.drop(columns=["test_loss", "test_acc"], axis=1, inplace=True)
    seaborn.relplot(data=metrics, kind="line")
    pyplot.savefig(fname=plots_path / f"metrics_{experiment}.png")
    pyplot.show()
