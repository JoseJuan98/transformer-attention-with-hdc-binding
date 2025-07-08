# -*- coding: utf-8 -*-
"""Layer Visualizations for Multivariate and Univariate Signals.

This script visualizes the output of the embedding layer for both multivariate and univariate signals.

Notes:
    This is script is not intended to be run as a standalone script. It's a draft for visualizing the output of layers,
    and it was used in debugging mode meanwhile training for ease of understanding the model's behavior.
"""
# Standard imports
import pathlib
from typing import Literal

# Third party imports
import torch
from matplotlib import pyplot


def visualize_input_and_layer_output(
    x: torch.Tensor,
    model: torch.nn.Module,
    dataset: str,
    layer_name: str,
    kind_of_signal: Literal["multivariate", "univariate"],
    plot_dir: str | pathlib.Path = "",
) -> None:
    """Visualizes the input signal and the output of the embedding layer for both multivariate and univariate signals.

    Notes:
        The model is better to not be trained already, to see the first epoch without any changes, but if you want to
        visualize the changes during training, you can use load a trained model using:
        >>> torch.load("<...>.pth", weights_only=False, map_location=torch.device('cpu'))

    Args:
        x (torch.Tensor): The input signal tensor. It should be of shape (batch_size, channels, time_steps) for
            multivariate signals or (batch_size, time_steps) for univariate signals.
        model (torch.nn.Module): The model containing the embedding layer to visualize.
        dataset (str): The name of the dataset being visualized, e.g., "ArticularyWordRecognition" or "UWaveGesture".
        layer_name (str): The name of the layer to visualize, e.g., "Linear Projection".
        kind_of_signal (Literal["multivariate", "univariate"]): Specifies whether the signal is multivariate or
            univariate.
        plot_dir (str | pathlib.Path): The directory where the plots will be saved. If not provided, it defaults to
            the current working directory.
    """
    # Current directory if plot_dir is not specified
    if not plot_dir:
        plot_dir = pathlib.Path(".")

    # Ensure plot_dir is a Path object
    elif not isinstance(plot_dir, pathlib.Path):
        plot_dir = pathlib.Path(plot_dir)

    # Multivariate signal
    if kind_of_signal == "multivariate":

        x = x[0].T.cpu().detach().numpy()

        n_channels = x.shape[0]

        for channel in range(0, n_channels):
            pyplot.plot([t for t in range(0, 144)], x[channel], label=f"Channel {channel}")

        pyplot.title(f"{dataset} Multivariate Signal")
        pyplot.legend()
        pyplot.savefig(plot_dir / f"Multivariate_Signal_2_{dataset}_multivariate_raw.png", dpi=300)
        pyplot.close()

        # Result of embedding layer
        z = model.embedding(x)
        z = z[0].T.cpu().detach().long().numpy()

        for channel in range(0, n_channels):
            pyplot.plot([t for t in range(0, 144)], z[channel], label=f"Channel {channel}")

        pyplot.title(f"{dataset} {layer_name} Output")
        pyplot.legend()
        pyplot.savefig(plot_dir / f"Multivariate_Signal_2_{dataset}_{layer_name.replace(' ', '_')}_output.png", dpi=300)
        pyplot.close()

    else:

        x = x[0].T[0].cpu().detach().numpy()

        n_channels = 64
        time_steps = x.shape[0]

        pyplot.plot([t for t in range(0, time_steps)], x, label="Signal")

        pyplot.title(f"{dataset} Univariate Signal")
        pyplot.legend()
        pyplot.savefig(plot_dir / f"Univariate_Signal_1_{dataset}_Univariate_raw.png", dpi=300)
        pyplot.close()

        # Result of embedding layer
        z = model.embedding(x)
        z = z[0].T.cpu().detach().long().numpy()
        time_steps = z.shape[1]

        for channel in range(0, n_channels):
            pyplot.plot([t for t in range(0, time_steps)], z[channel], label=f"Channel {channel}")

        pyplot.title(f"{dataset} {layer_name} Output")
        pyplot.legend()
        pyplot.savefig(plot_dir / f"Univariate_Signal_1_{dataset}_{layer_name.replace(' ', '_')}_output.png", dpi=300)
        pyplot.close()
