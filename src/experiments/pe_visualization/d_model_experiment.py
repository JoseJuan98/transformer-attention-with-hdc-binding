# -*- coding: utf-8 -*-
"""Visualizations of Experiment 5 for different size $d_model$ binding operations."""
# Standard imports
import pathlib

# Third party imports
import pandas
from matplotlib import pyplot

# First party imports
from utils import Config


def plot_metrics_by_binding(metrics: pandas.DataFrame, plot_path: pathlib.Path) -> None:
    """Plot metrics by binding operation.

    Args:
        metrics (pandas.DataFrame): DataFrame containing the metrics.
        plot_path (pathlib.Path): Directory to save the plots.
    """
    # Create a figure and axis for the plot
    fig, ax = pyplot.subplots(figsize=(19, 10))

    models = ["additive", "component_wise", "circular_conv"]
    d_models = [32, 64, 128, 256, 512, 1024]

    for model in models:
        acc = [metrics[metrics["model"] == f"linear_{model}_{d_model}"]["mean_acc"].values for d_model in d_models]

        ax.plot(d_models, acc, label=f"{model.replace('_', ' ').title()} binding", marker="o")

    ax.set_xticks(d_models)
    ax.set_xticklabels(d_models, rotation=45)
    ax.set_xlabel("d_model")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Mean Accuracy by d_model for Different Binding Operations")

    # Add a legend
    ax.legend()

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
        pathlib.Path(__file__).parents[3] / "docs/experiment_results/3_d_model_v1/summary_model_metrics_d_model_v1.csv"
    )
    metrics_by_model = pandas.read_csv(filepath_or_buffer=experiment_metrics_path, header=0)

    # Plot path
    plot_path = Config.plot_dir / "experiment" / "3_d_model_v1" / "d_model_v1_accuracy_plot.png"

    # Create the plot directory if it doesn't exist
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plot_metrics_by_binding(metrics=metrics_by_model, plot_path=plot_path)
