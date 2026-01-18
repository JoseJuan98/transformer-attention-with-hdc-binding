# -*- coding: utf-8 -*-
"""Critical Difference (CD) Diagram Visualization Module."""

# Standard imports
import pathlib

# Third party imports
import autorank
import matplotlib
import pandas
import numpy
from matplotlib import pyplot

# First party imports
from experiments.visualization.metrics import get_metrics
from utils import Config
from utils.plot import set_plot_style
from experiments.visualization.vis_config import rc_config, modern_palette


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
        pyplot.savefig(output_path, bbox_inches="tight")

    pyplot.show()

def plot_bar_mean_accuracies(metrics: pandas.DataFrame, output_path: pathlib.Path) -> None:
    """Plot a bar chart of mean accuracies for different models.

    Args:
        metrics (pandas.DataFrame): DataFrame containing experiment metrics.
        output_path (pathlib.Path): Path to save the bar plot.
    """
    from experiments.visualization.vis_config import rc_config, modern_palette
    matplotlib.rcParams.update(rc_config)

    # Get accuracy columns
    mean_accuracies = metrics[["model", "mean_acc"]].copy().sort_values(by="mean_acc", ascending=False)

    # Convert to percentage
    mean_accuracies["mean_acc"] = mean_accuracies["mean_acc"] * 100

    # Create bar plot
    fig, ax = pyplot.subplots(nrows=1, ncols=1)
    ax.bar(mean_accuracies["model"], mean_accuracies["mean_acc"], label=mean_accuracies["model"], color=modern_palette)

    # Set labels and title
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_title("Mean Accuracies of Different Models")
    ax.set_xticks(range(len(mean_accuracies["model"])))
    ax.set_xticklabels(mean_accuracies["model"].tolist())#, rotation=45, ha="right")

    # Annotate bars with accuracy values
    for i, v in enumerate(mean_accuracies["mean_acc"]):
        ax.text(i, v + 0.5, f"{v:.2f}%", ha="center")

    # Adjust ylimit
    ax.set_ylim(mean_accuracies["mean_acc"].min() - 4.37, mean_accuracies["mean_acc"].max() + 2)

    pyplot.tight_layout()

    if output_path is not None:
        pyplot.savefig(output_path)

    pyplot.show()


def plot_bar_dataset_acc(metrics: pandas.DataFrame, output_path: pathlib.Path) -> None:
    """Plot a grouped bar chart of accuracies for different datasets and models.

    Args:
        metrics (pandas.DataFrame): DataFrame containing experiment metrics.
        output_path (pathlib.Path): Path to save the bar plot.
    """

    # Apply style configuration
    matplotlib.rcParams.update(rc_config)

    # Create a copy to avoid modifying the original dataframe outside this function
    metrics_ = metrics.copy()

    # Convert "mean ± std" strings to float values and to percentage
    metrics_ = metrics_.map(get_average_acc) * 100

    # Setup Plot Dimensions
    num_datasets = len(metrics_)
    num_models = len(metrics_.columns)

    # Calculate figure size: ensure it's wide enough if there are many datasets
    fig_width = max(10, num_datasets)
    fig, ax = pyplot.subplots(figsize=(fig_width, 14))

    # Calculate Bar Positions
    # X locations for the groups
    x = numpy.arange(0, num_datasets)
    # Total width allocated for one group (dataset)
    total_width = 0.8
    # Width of an individual bar
    bar_width = total_width / num_models

    # Plot Bars
    for i, model_name in enumerate(metrics_.columns):
        # Calculate offset to center the group of bars on the tick
        # (i - num_models/2) centers the group around 0
        # + 0.5 shifts it to center (if even number of bars)
        offset = (i - num_models / 2) * bar_width + (bar_width / 2)

        # Cycle through palette if we have more models than colors
        color = modern_palette[i % len(modern_palette)]

        # Draw bars
        rects = ax.bar(
            x + offset,
            metrics_[model_name],
            width=bar_width,
            label=model_name,
            color=color
        )

        # # Annotate bars
        # for rect in rects:
        #     height = rect.get_height()
        #     # Only annotate if height is visible
        #     if height > 0:
        #         ax.text(
        #             rect.get_x() + rect.get_width() / 2,  # X: Center of bar
        #             height + 1,  # Y: Just above bar
        #             f'{height:.1f}',  # Text: 1 decimal place
        #             ha='center',
        #             va='bottom',
        #             # rotation=90,  # Vertical rotation
        #             fontsize=8,  # Smaller font
        #             color='black'
        #         )

    # Styling and Labels
    ax.set_ylabel("Accuracy (%)")

    # Set X-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_.index, rotation=45, ha="right")

    # Add Legend (placed outside top or bottom usually better for many models)
    # Here we place it above the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -1),
        ncol=min(num_models, 3),
        frameon=False
    )

    # Set Y-axis limits (0 to 100 with some headroom)
    # ax.set_ylim(0, 105)

    # Add grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    pyplot.tight_layout()

    if output_path is not None:
        pyplot.savefig(output_path)

    pyplot.show()


def plot_relative_accuracy_scatter(metrics: pandas.DataFrame, output_path: pathlib.Path) -> None:
    """Plot a scatter plot of accuracy relative to the dataset mean.

    Each point represents a dataset. The Y-value is (Model Accuracy - Dataset Mean Accuracy).
    Points are grouped by model on the X-axis.

    Args:
        metrics (pandas.DataFrame): DataFrame containing experiment metrics.
        output_path (pathlib.Path): Path to save the plot.
    """
    # Apply style
    matplotlib.rcParams.update(rc_config)

    # Create a copy to avoid modifying the original dataframe outside this function
    metrics_ = metrics.copy()

    # Convert strings to floats
    metrics_ = metrics_.map(get_average_acc)

    # Convert to percentage
    metrics_ = metrics_ * 100

    # Calculate Relative Accuracy
    # Calculate the mean accuracy for each dataset (row-wise)
    dataset_means = metrics_.mean(axis=1)

    # Subtract the dataset mean from the model accuracy
    # Positive value = Model performed better than average on this dataset
    # Negative value = Model performed worse than average
    relative_acc = metrics_.sub(dataset_means, axis=0)

    # Plotting
    num_models = len(relative_acc.columns)
    fig, ax = pyplot.subplots(figsize=(max(10, num_models), 10))

    # Create X coordinates
    x_positions = numpy.arange(num_models)

    # Plot each model's cloud
    for i, model in enumerate(relative_acc.columns):
        values = relative_acc[model].values

        # Add "Jitter" to X to spread the points out horizontally
        # np.random.normal creates a distribution around the center 'i'
        jitter = numpy.random.normal(0, 0.08, size=len(values))

        color = modern_palette[i % len(modern_palette)]

        # Plot the scatter points (The Cloud)
        ax.scatter(
            x_positions[i] + jitter,
            values,
            alpha=0.6,
            s=40,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            label=model if i == -1 else ""  # Don't add to legend, labels are on X-axis
        )

        # Optional: Add a marker for the Mean of the model's relative performance
        mean_val = values.mean()
        ax.scatter(
            x_positions[i],
            mean_val,
            s=200,
            marker='_',
            color='black',
            linewidth=3,
            zorder=10
        )

    # Styling
    # Add a reference line at 0 (The Dataset Average)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Accuracy Average')

    ax.set_ylabel("Accuracy Difference from Mean (%)")
    # ax.set_title("Relative Performance of Models across Datasets")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(relative_acc.columns, rotation=45, ha="right")

    # Add a small legend just for the reference line
    ax.legend(loc='upper right', frameon=True)

    # Adjust Y limits to make sure clouds aren't cut off
    y_max = relative_acc.max().max()
    y_min = relative_acc.min().min()
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)

    pyplot.tight_layout()

    if output_path is not None:
        pyplot.savefig(output_path, bbox_inches="tight")

    pyplot.show()



def plot_cd_diagram_of_experiment(experiment_name: str, plot_name: str, exp_dataset_metrics: pathlib.Path, exp_model_metrics: pathlib.Path) -> None:
    """Plot the CD diagram for a given experiment.

    Args:
        experiment_name (str): Name of the experiment.
        plot_name (pathlib.Path): Relative path to save the CD diagram plot.
        exp_dataset_metrics (pathlib.Path): Path to the CSV file containing experiment metrics.
        exp_model_metrics (pathlib.Path): Path to the CSV file containing model metrics.
    """
    # Metrics by dataset
    metrics_by_dataset = get_metrics(exp_dataset_metrics)

    # Metrics by model
    metrics_by_model = get_metrics(exp_model_metrics)

    # Data Preparation
    # Drop metadata columns if they exist
    cols_to_drop = ["train_samples", "sequence_length", "num_classes"]
    metrics_by_dataset.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Set index for dataset metrics
    metrics_by_dataset.set_index("dataset", inplace=True)

    metrics_by_model["model"] = (
        metrics_by_model["model"]
        .str.replace("_", " ")
        .str.title()
        .str.replace("Sinusoidal", "")
        .str.replace("Component", "Comp.")
    )

    # Format Model Names
    metrics_by_dataset.columns = (
        metrics_by_dataset.columns
        .str.replace("_", " ")
        .str.title()
        .str.replace("Sinusoidal", "")
        .str.replace("Component", "Comp.")
        .str.strip()
    )

    # For Experiment 1, the `split_sinusoidal` variants are not included in the CD diagram, as explained in the README for experiment 1 results directory.
    if "Experiment 1" in experiment_name:
        # Remove the columns that contains 'split_sinusoidal' in their names.
        split_sin_columns = metrics_by_dataset.columns[metrics_by_dataset.columns.str.contains("Split")].tolist()

        if split_sin_columns:
            print(f"Removing columns: {split_sin_columns} from the metrics DataFrame.")
            metrics_by_dataset.drop(columns=split_sin_columns, inplace=True)

    # Define the output path for the CD diagram
    plot_path = Config.plot_dir / "experiment" / plot_name
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_suffix = plot_name.split("/")[-1].replace("_CD.png", "")

    # Plot the CD diagram
    plot_cd_diagram(metrics=metrics_by_dataset, output_path=plot_path)

    # Create bar plot of mean accuracies
    plot_bar_mean_accuracies(metrics=metrics_by_model, output_path=plot_path.parent / f"{plot_suffix}_mean_accuracies.png")

    # Create bar plot of dataset accuracies
    plot_bar_dataset_acc(metrics=metrics_by_dataset, output_path=plot_path.parent / f"{plot_suffix}_dataset_accuracies.png")

    # Create scatter plot of relative accuracies
    plot_relative_accuracy_scatter(metrics=metrics_by_dataset, output_path=plot_path.parent / f"{plot_suffix}_relative_accuracies.png")


if __name__ == "__main__":

    # Path to experiment's raw metrics CSV file
    exp_results_dir = pathlib.Path(__file__).parents[3] / "docs" / "experiment_results"

    # Set parameters for the plot
    exp1_dir_name = "1_binding_version_1"
    # exp4_comp_dir_name = "..."
    # exp4_cconv_dir_name = "..."
    exp_to_plot: list[dict[str, str | pathlib.Path]] = [
        {
            "experiment_name": "Experiment 1",
            "plot_name": f"{exp1_dir_name}/binding_v1_CD.png",
            "exp_dataset_metrics": exp_results_dir / exp1_dir_name / "summary_dataset_results.csv",
            "exp_model_metrics": exp_results_dir / exp1_dir_name / "summary_model_metrics_binding_version_1.csv",
        },
        # {
        #     "experiment_name": "Experiment 4 Component Wise",
        #     "plot_name": f"{exp4_comp_dir_name}/component_wise_1_CD.png",
        #     "exp_dataset_metrics": exp_results_dir / exp4_comp_dir_name / "summary_dataset_results.csv",
        #     "exp_model_metrics": exp_results_dir / exp4_comp_dir_name / "",
        # },
        # {
        #     "experiment_name": "Experiment 4 CConv",
        #     "plot_name": f"{exp4_cconv_dir_name}/cconv_1_CD.png",
        #     "exp_dataset_metrics": exp_results_dir / exp4_cconv_dir_name / "summary_dataset_results.csv",
        #     "exp_model_metrics": exp_results_dir / exp4_cconv_dir_name / "",
        # },
    ]

    for exp in exp_to_plot:
        plot_cd_diagram_of_experiment(
            # experiment_name=str(exp["experiment_name"]),
            # plot_name=str(exp["plot_path"]),
            # exp_dataset_metrics=pathlib.Path(exp["exp_dataset_metrics"]),
            # exp_model_metrics=pathlib.Path(exp["exp_model_metrics"]),
            **exp
        )
