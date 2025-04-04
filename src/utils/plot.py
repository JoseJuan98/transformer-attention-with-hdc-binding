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


def get_train_metrics_and_plot(
    plots_path: pathlib.Path,
    csv_dir: str,
    experiment: str,
    logger: logging.Logger | None = None,
    plotting: bool = False,
) -> None:
    """Save the metrics plot.

    Args:
        plots_path (pathlib.Path): Path to save the plot.
        csv_dir (str): Path to the directory containing the metrics.csv file.
        experiment (str): Name of the experiment.
        logger (logging.Logger, optional): Logger object. Defaults to None.
        plotting (bool, optional): Whether to display the plot. Defaults to False.
    """
    metrics = pandas.read_csv(filepath_or_buffer=os.path.join(csv_dir, "metrics.csv"))

    metrics.drop(columns=["step", "n_samples"], axis=1, inplace=True, errors="ignore")
    metrics.set_index("epoch", inplace=True)

    test_loss = metrics["test_loss"].dropna(how="all").mean().round(4)
    test_acc = metrics["test_acc"].dropna(how="all").mean().round(4)

    if logger is None:
        print(f"\nExperiment {experiment}\n\tTest loss: {test_loss}.\n\tTest accuracy: {test_acc}.\n\n")
    else:
        logger.info(f"\nExperiment {experiment}\n\tTest loss: {test_loss}.\n\tTest accuracy: {test_acc}.\n\n")

    metrics.drop(columns=["test_loss", "test_acc"], axis=1, inplace=True, errors="ignore")
    seaborn.relplot(data=metrics, kind="line")

    plots_path.parent.mkdir(parents=True, exist_ok=True)
    pyplot.savefig(fname=plots_path)

    if plotting:
        pyplot.show()

    pyplot.close()
