# -*- coding: utf-8 -*-
"""Critical Difference (CD) Diagram Visualization Module."""

# Standard imports
import pathlib

# Third party imports
import autorank
import pandas
from matplotlib import pyplot

# First party imports
from utils import Config
from utils.plot import set_plot_style


def get_average_acc(value):
    """Convert a string representing accuracy (with or without confidence interval) to a float."""
    # Ensure the value is a string before trying to split it
    if not isinstance(value, str):
        return value

    try:
        if " ± " in value:
            # Handle 'accuracy ± confidence' format
            parts = value.split(" ± ")
            return float(parts[0])
        else:
            # Handle single number format (like in the last row)
            return float(value)
    except (ValueError, IndexError):
        # If conversion to float fails or split doesn't work, return original
        return value


def plot_cd_diagram(metrics: pandas.DataFrame, output_path: pathlib.Path | None = None) -> None:
    """Plot a Critical Difference (CD) diagram from the given metrics DataFrame.

    Args:
        metrics (pandas.DataFrame): DataFrame containing experiment metrics.
        output_path (pathlib.Path): Path to save the CD diagram.
    """
    # Format the metrics DataFrame
    metrics.drop(columns=["train_samples", "sequence_length", "num_classes"], inplace=True)
    metrics.set_index("dataset", inplace=True)
    metrics.columns = metrics.columns.str.replace("_", " ").str.title()

    # Get only the accuracy values
    metrics = metrics.map(get_average_acc)

    # Autorank is used to run the Friedman test and the Nemenyi post-hoc test.
    # 'alpha=0.05' is the standard significance level.
    # 'order="descending"' because higher accuracy is better.
    result = autorank.autorank(
        data=metrics, alpha=0.05, verbose=True, order="descending", force_mode="nonparametric", approach="frequentist"
    )

    print("Statistical Analysis Report:")
    print(result)
    print("-" * 30)

    # Create and Save the CD Diagram
    set_plot_style()
    pyplot.rcParams["grid.linewidth"] = 3
    fig, ax = pyplot.subplots(nrows=1, ncols=1)

    # The plot_stats function generates the CD diagram.
    autorank.plot_stats(result, ax=ax)

    # Save the plot to a file if output_path is provided
    if output_path is not None:
        pyplot.savefig(plot_path, bbox_inches="tight")

    pyplot.show()


if __name__ == "__main__":
    # Set parameters for the plot
    # relative_file_path = "1_binding_version_1/binding_v1_CD.png"
    # experiment = "Experiment 1"
    relative_file_path = "5_sota_version_1/sota_v1_CD.png"
    experiment = "Experiment 5"

    # Set pandas options for better display of DataFrames
    pandas.set_option("display.max_columns", None)

    # Set pandas to not truncate DataFrame output
    pandas.set_option("display.width", 0)

    # Path to experiment's raw metrics CSV file
    experiment_metrics_path = (
        pathlib.Path(__file__).parents[3]
        # / "docs/experiment_results/1_binding_version_1/summary_dataset_results.csv"
        / "docs/experiment_results/5_sota_version_1/summary_dataset_results.csv"
    )

    # Define output directory
    output_dir = experiment_metrics_path.parent
    output_dir.mkdir(exist_ok=True)

    # Validate paths
    if not experiment_metrics_path.exists():
        raise FileNotFoundError(f"Experiment metrics file not found at: {experiment_metrics_path}")

    # Load the experiment metrics CSV file into a DataFrame
    metrics = pandas.read_csv(experiment_metrics_path, header=0)

    # For Experiment 1, the `split_sinusoidal` variants are not included in the CD diagram, as explained in the README for experiment 1 results directory.
    if "Experiment 1" in experiment:
        # Remove the columns that contains 'split_sinusoidal' in their names.
        split_sin_columns = metrics.columns[metrics.columns.str.contains("split_sinusoidal")].tolist()

        if split_sin_columns:
            print(f"Removing columns: {split_sin_columns} from the metrics DataFrame.")
            metrics.drop(columns=split_sin_columns, inplace=True)

    # Define the output path for the CD diagram
    plot_path = Config.plot_dir / "experiment" / relative_file_path
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot the CD diagram
    plot_cd_diagram(metrics=metrics, output_path=plot_path)
