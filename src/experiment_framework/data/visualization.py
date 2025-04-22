# -*- coding: utf-8 -*-
"""Visualization for time series data."""
# Standard imports
import pathlib

# Third party imports
import numpy
from matplotlib import pyplot


def _plot_time_series_sample(
    dsid: str,
    sample: numpy.ndarray,
    num_dimensions: int,
    plot_path: pathlib.Path | None = None,
    show_plot: bool = False,
):
    pyplot.figure(figsize=(16, 9))

    # plot all dimensions in a heatmap
    data_row = sample
    if data_row.shape[0] >= data_row.shape[1]:
        data_row = data_row.T

    pyplot.imshow(data_row)
    pyplot.title(f"{dsid} Sample")

    if show_plot:
        pyplot.show()

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(plot_path)

    pyplot.close()

    # plot dimensions separately in a line plot
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(16, 9))
    for i in range(num_dimensions):
        ax.plot(data_row[:, i], label=f"Dimension {i}")
    pyplot.title(f"{dsid} Sample")
    pyplot.legend()

    if show_plot:
        pyplot.show()

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(plot_path.parent / plot_path.name.replace(".png", "_line_plot.png"))

    pyplot.close()
