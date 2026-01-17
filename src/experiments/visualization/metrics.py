# -*- coding: utf-8 -*-
"""Utility functions for experiment visualization."""
# Standard imports
import pathlib

# Third party imports
import pandas


def get_metrics(metrics_path: str | pathlib.Path) -> pandas.DataFrame:
    """Load experiment metrics from a CSV file into a pandas DataFrame.

    Args:
        metrics_path (str | pathlib.Path): Path to the CSV file containing experiment metrics.

    Returns:
        pandas.DataFrame: DataFrame containing the experiment metrics.
    """
    # Set pandas options for better display of DataFrames
    pandas.set_option("display.max_columns", None)

    # Set pandas to not truncate DataFrame output
    pandas.set_option("display.width", 0)

    # Define output directory
    output_dir = metrics_path.parent
    output_dir.mkdir(exist_ok=True)

    # Validate paths
    if not metrics_path.exists():
        raise FileNotFoundError(f"Experiment metrics file not found at: {metrics_path}")

    return pandas.read_csv(metrics_path, header=0)
