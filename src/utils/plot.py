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
    csv_dir: str,
    experiment: str,
    logger: logging.Logger | None = None,
    plots_path: pathlib.Path | None = None,
    show_plot: bool = False,
) -> pandas.DataFrame:
    """Save the metrics plot.

    Args:
        plots_path (pathlib.Path): Path to save the plot.
        csv_dir (str): Path to the directory containing the metrics.csv file.
        experiment (str): Name of the experiment.
        logger (logging.Logger, optional): Logger object. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        pandas.DataFrame: Pandas DataFrame containing the final metrics of the plot.
    """
    metrics = pandas.read_csv(filepath_or_buffer=os.path.join(csv_dir, "metrics.csv"))

    metrics.drop(columns=["step", "n_samples"], axis=1, inplace=True, errors="ignore")
    metrics.set_index("epoch", inplace=True)

    test_loss = metrics["test_loss"].dropna(how="all").mean()

    if pandas.isna(test_loss):
        test_loss = None
    else:
        test_loss = round(test_loss, 4)
    test_acc = metrics["test_acc"].dropna(how="all").mean().round(4)

    if logger is None:
        print(f"\nExperiment {experiment}\n\tTest loss: {test_loss}.\n\tTest accuracy: {test_acc}.\n\n")
    else:
        logger.info(f"\nExperiment {experiment}\n\tTest loss: {test_loss}.\n\tTest accuracy: {test_acc}.\n\n")

    plotting_data = metrics.drop(columns=["test_loss", "test_acc"], axis=1, errors="ignore").copy()
    if plots_path is not None and not plotting_data.empty:
        seaborn.relplot(data=plotting_data, kind="line")

        plots_path.parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(fname=plots_path)

        if show_plot:
            pyplot.show()

        pyplot.close()

    metrics = (
        metrics.drop(["test_acc", "test_loss"], axis=1)
        # Drop if the row is full of NaN values
        .dropna(how="all")
    )
    # Get the last row of the metrics
    metrics = metrics[metrics.index == metrics.index[-1]].mean(axis=0).round(4)
    metrics["test_loss"] = test_loss
    metrics["test_acc"] = test_acc
    metrics["size_MB"] = round(pathlib.Path(f"{csv_dir}/model.pth").stat().st_size / (1024**2), 4)

    return metrics.to_frame().T
