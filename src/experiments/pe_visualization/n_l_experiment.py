# -*- coding: utf-8 -*-
"""Visualizations of Experiment 2 for different size $N_L$ binding operations."""
# Standard imports
import pathlib

# Third party imports
import numpy
import pandas
from matplotlib import pyplot
from scipy.interpolate import make_interp_spline

# First party imports
from utils import Config
from utils.plot import set_plot_style


def plot_metrics_by_binding(metrics: pandas.DataFrame, plot_path: pathlib.Path) -> None:
    """Plot metrics by binding operation.

    Args:
        metrics (pandas.DataFrame): DataFrame containing the metrics.
        plot_path (pathlib.Path): Directory to save the plots.
    """
    set_plot_style()

    # Create a figure and axis for the plot
    fig, ax = pyplot.subplots(figsize=(19, 10))

    models = ["additive", "component_wise", "circular_conv"]
    n_ls = [1, 2, 4, 8]
    x_ticks = list(range(len(n_ls)))
    lower_limit = 0.50

    for model in models:
        # Original accuracy data
        acc = [metrics[metrics["model"] == f"linear_{model}_N_L_{n_l}"]["mean_acc"].values[0] for n_l in n_ls]

        # Create a set of smooth x-coordinates for the spline
        x_smooth = numpy.linspace(min(x_ticks), max(x_ticks), 300)

        # Create the spline interpolation function
        # k=3 means cubic spline, which is great for smooth curves
        spline = make_interp_spline(x_ticks, acc, k=3)

        # Calculate the smoothed y-coordinates
        y_smooth = spline(x_smooth)

        # Plot the smoothed line
        ax.plot(x_smooth, y_smooth, label=model.replace("_", " ").title(), linewidth=2)

        # Plot the original data points as markers on top of the smooth line
        ax.plot(x_ticks, acc, marker="o", linestyle="none", color=ax.lines[-1].get_color())

        # Annotation loop remains the same, as it annotates the original points
        for i, v in enumerate(acc):
            ax.annotate(f"{v * 100:.2f}", xy=(i, v), xytext=(-7, 7), textcoords="offset points")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(n_ls)
    ax.set_ylim([lower_limit, 0.6525])
    ax.set_xlabel("d_model")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Mean Accuracy by $N_L$ for Different Binding Operations")

    # Add a legend
    ax.legend(loc="lower left")

    # Save the plot to the specified directory
    pyplot.savefig(plot_path)

    # Show the plot
    pyplot.show()
    pyplot.close()


if __name__ == "__main__":
    # Set pandas options for better display of DataFrames
    pandas.set_option("display.max_columns", None)

    # Set pandas to not truncate DataFrame output
    pandas.set_option("display.width", 0)

    # Path to experiment's raw metrics CSV file
    experiment_metrics_path = (
        pathlib.Path(__file__).parents[3]
        / "docs/experiment_results/2_N_L_version_1/summary_model_metrics_N_L_version_1.csv"
    )
    metrics_by_model = pandas.read_csv(filepath_or_buffer=experiment_metrics_path, header=0)

    # Plot path
    plot_path = Config.plot_dir / "experiment" / "2_N_L_version_1" / "n_l_v1_accuracy_plot.png"

    # Create the plot directory if it doesn't exist
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plot_metrics_by_binding(metrics=metrics_by_model, plot_path=plot_path)
