# -*- coding: utf-8 -*-
"""Critical Difference (CD) Diagram Visualization Module."""

# Standard imports
import pathlib

# Third party imports
import autorank
import matplotlib
import numpy
import pandas
from matplotlib import pyplot

# First party imports
from experiments.visualization.metrics import format_model_names, get_metrics
from experiments.visualization.vis_config import modern_palette, rc_config
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
    matplotlib.rcParams.update(rc_config)
    set_plot_style()
    pyplot.rcParams["grid.linewidth"] = 10
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
    # First party imports
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
    # ax.set_title("Mean Accuracies of Different Models")
    ax.set_xticks(range(len(mean_accuracies["model"])))
    ax.set_xticklabels(mean_accuracies["model"].tolist())  # , rotation=45, ha="right")

    # Annotate bars with accuracy values
    for i, v in enumerate(mean_accuracies["mean_acc"]):
        ax.text(i, v + 0.5, f"{v:.2f}%", ha="center")

    # Adjust ylimit
    ax.set_ylim(mean_accuracies["mean_acc"].min() - 4.37, mean_accuracies["mean_acc"].max() + 2)

    pyplot.tight_layout()

    if output_path is not None:
        pyplot.savefig(output_path)

    pyplot.show()


def plot_bar_dataset_acc(
    metrics: pandas.DataFrame, output_path: pathlib.Path, top_n: int = -1, target_models: tuple = ("none", "none")
) -> None:
    """Plot a grouped bar chart of accuracies for different datasets and models.

    Args:
        metrics (pandas.DataFrame): DataFrame containing experiment metrics.
        output_path (pathlib.Path): Path to save the bar plot.
        top_n (int): Number of top datasets to plot based on divergence between two target models. Defaults to -1 (all
            datasets).
        target_models (tuple): Tuple of two model names to calculate divergence for selecting top datasets. Defaults to
            ("none", "none"), which means no sorting by divergence.
    """

    # Apply style configuration
    matplotlib.rcParams.update(rc_config)

    # Create a copy to avoid modifying the original dataframe outside this function
    metrics_ = metrics.copy()

    # Convert "mean ± std" strings to float values and to percentage
    metrics_ = metrics_.map(get_average_acc) * 100

    # Select Top N Datasets with Highest Divergence between the Two Target Models
    top_n = top_n if top_n > 0 else len(metrics_)
    target_model_1, target_model_2 = target_models

    # Check if these columns exist to avoid errors
    if target_model_1 in metrics_.columns and target_model_2 in metrics_.columns:

        # Separate out the Average row for later re-insertion
        avg = metrics_[metrics_.index == "Average"]
        metrics_ = metrics_[metrics_.index != "Average"]

        # Calculate the difference
        metrics_["_diff"] = metrics_[target_model_1] - metrics_[target_model_2]

        # Sort descending
        metrics_ = metrics_.sort_values(by="_diff", ascending=False)

        # Select top N datasets + Average
        metrics_ = metrics_.head(top_n)
        metrics_ = pandas.concat([metrics_, avg])

        # Remove the temporary diff column so it doesn't get plotted
        metrics_.drop(columns=["_diff"], inplace=True)
        print(f"Plotting top {top_n} datasets with highest divergence between Circular and Additive.")
    else:
        print(f"Warning: Target models for sorting not found. Plotting first {top_n} datasets.")
        metrics_ = metrics_.head(top_n)

    # Setup Plot Dimensions
    num_datasets = len(metrics_)
    num_models = len(metrics_.columns)

    # Calculate figure size: ensure it's wide enough if there are many datasets
    fig_width = max(top_n, num_datasets)
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
        ax.bar(x + offset, metrics_[model_name], width=bar_width, label=model_name, color=color)

        # # Annotate bars
        # # rects is what is returned by ax.bar(...) before
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

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -1), ncol=min(num_models, 3), frameon=False)

    # Set Y-axis limits (0 to 100 with some headroom)
    # ax.set_ylim(0, 105)

    # Add grid for readability
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    pyplot.tight_layout()

    if output_path is not None:
        pyplot.savefig(output_path, bbox_inches="tight")

    pyplot.show()


def plot_relative_accuracy_scatter(
    dataset_metrics: pandas.DataFrame,
    output_path: pathlib.Path,
    baseline_conf: dict,
    legend_label: str | None = None,
    absolute: bool = False,
    ylim: tuple | None = None,
) -> pandas.DataFrame:
    """Plot a scatter plot of accuracy relative to specific the given Baselines.

    Each point represents a dataset. The Y-value can be either relative accuracy (improvement over baseline) or absolute
     accuracy (difference from baseline in percentage points), depending on the `absolute` flag.

    Args:
        dataset_metrics (pandas.DataFrame): DataFrame containing experiment metrics.
        output_path (pathlib.Path): Path to save the plot.
        baseline_conf (dict): Configuration dict defining which models to use as baselines and their corresponding target models.
        legend_label (str | None): Label for the legend entry of the baseline line. If None, a default label is used.
        absolute (bool): If True, plot absolute accuracies instead of relative. Defaults to False.
        ylim (tuple | None): Y-axis limits for the plot. If None, automatic limits are used.

    Returns:
        pandas.DataFrame: DataFrame containing the calculated accuracies difference.
    """
    # Apply style
    matplotlib.rcParams.update(rc_config)

    # Create a copy to avoid modifying the original dataframe
    metrics_ = dataset_metrics.copy()

    # Remove the `Average` row if it exists to avoid duplication
    metrics_ = metrics_[metrics_.index != "Average"]

    # Convert strings to floats
    metrics_ = metrics_.map(get_average_acc)

    # Convert to percentage
    metrics_ = metrics_ * 100

    # Initialize a dataframe for relative accuracies
    relative_acc = pandas.DataFrame(index=metrics_.index)

    # Group Calculation of Relative Accuracies
    for baseline_name, cols in baseline_conf.items():
        baseline = metrics_[baseline_name]

        # Calculate diffs (Target - Baseline)
        for col in cols:
            if col in metrics_.columns:
                if absolute:
                    relative_acc[col] = metrics_[col] - baseline
                else:
                    relative_acc[col] = (metrics_[col] - baseline) / baseline

    # Plotting
    num_models = len(relative_acc.columns)
    fig, ax = pyplot.subplots(figsize=(19, 10))

    # Create X coordinates
    x_positions = numpy.arange(num_models)

    # Plot each model's cloud
    for i, model in enumerate(relative_acc.columns):
        values = relative_acc[model].values

        # Add "Jitter" to X to spread the points out horizontally
        jitter = numpy.random.normal(0, 0.08, size=len(values))

        # Assign colors
        color = modern_palette[i % len(modern_palette)]

        # Plot the scatter points
        ax.scatter(
            x_positions[i] + jitter,
            values,
            alpha=0.6,
            s=100,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=model if i == -1 else "",  # Don't add to legend, labels are on X-axis
        )

        # Add a marker for the Mean of the model's relative performance
        mean_val = values.mean()
        ax.scatter(x_positions[i], mean_val, s=400, marker="_", color="black", linewidth=3, zorder=10)

    # Styling
    # Reference line at 0 now represents the Additive Baseline
    legend_label = r"Baseline ($\Delta=0$)" if legend_label is None else legend_label + r" ($\Delta=0$)"
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=legend_label)

    kind_of_increase = "Absolute" if absolute else "Relative"
    ylabel = r"$\Delta$ " + f"{kind_of_increase} Accuracy"
    ylabel += r" (%)" if absolute else ""
    ax.set_ylabel(ylabel)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(relative_acc.columns, rotation=45, ha="right")

    # Add a small legend just for the reference line
    ax.legend(loc="upper right", frameon=True)

    # Adjust Y limits
    y_max = relative_acc.max().max()
    y_min = relative_acc.min().min()
    padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 1.0
    ylim = ylim if ylim is not None else (y_min - padding, y_max + padding)
    ax.set_ylim(ylim)

    pyplot.tight_layout()

    if output_path is not None:
        pyplot.savefig(output_path, bbox_inches="tight")

    pyplot.show()

    return relative_acc


def plot_paper_diagrams(
    experiment_name: str,
    plot_name: str,
    exp_dataset_metrics_path: pathlib.Path,
    exp_model_metrics_path: pathlib.Path,
    baseline_conf: dict,
    naming_mapping: dict | None = None,
    models: list | None = None,
    target_models: tuple = ("none", "none"),
    top_n: int = 9,
    relative_acc_legend_label: str | None = None,
    relative_ylim: tuple | None = None,
    abs_ylim: tuple | None = None,
) -> None:
    """Plot the CD diagram for a given experiment.

    Args:
        experiment_name (str): Name of the experiment.
        plot_name (pathlib.Path): Relative path to save the CD diagram plot.
        exp_dataset_metrics_path (pathlib.Path): Path to the CSV file containing experiment metrics.
        exp_model_metrics_path (pathlib.Path): Path to the CSV file containing model metrics.
        baseline_conf (dict): Configuration dict defining which models to use as baselines and their corresponding
            target models.
        naming_mapping (dict | None): Optional mapping to rename models for clarity in plots. Defaults to None.
        models (list | None): List of model names after naming formatting to include in the plots. If None, include all
            models. Defaults to None.
        target_models (tuple): Tuple of two model names to calculate divergence for selecting top datasets. Defaults to
            ("none", "none"), which means no sorting by divergence.
        top_n (int): Number of top datasets to plot based on divergence between two target models. Defaults to 9. It
            adds the "Average" of all datasets automatically.
        relative_acc_legend_label (str | None): Label for the legend entry of the baseline line in the relative accuracy
         scatter plot. If None, a default label is used.
        relative_ylim (tuple | None): Y-axis limits for the relative accuracy scatter plot. If None, automatic limits
            are used.
        abs_ylim (tuple | None): Y-axis limits for the absolute accuracy scatter plot. If None, automatic limits are
         used.
    """
    print(f"{'-'*20}\nPlotting {experiment_name}\n{'-'*20}\n")

    # Metrics by dataset
    metrics_by_dataset = get_metrics(exp_dataset_metrics_path)

    # Metrics by model
    metrics_by_model = get_metrics(exp_model_metrics_path)

    # Data Preparation
    # Drop metadata columns if they exist
    cols_to_drop = ["train_samples", "sequence_length", "num_classes"]
    metrics_by_dataset.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Set index for dataset metrics
    metrics_by_dataset.set_index("dataset", inplace=True)

    # Format Model Names
    metrics_by_model["model"] = format_model_names(metrics_by_model["model"], additional_mapping=naming_mapping)
    metrics_by_dataset.columns = format_model_names(metrics_by_dataset.columns, additional_mapping=naming_mapping)

    if models is not None:
        # Filter to include only specified models
        metrics_by_model = metrics_by_model[metrics_by_model["model"].isin(models)]
        metrics_by_dataset = metrics_by_dataset[[col for col in models if col in metrics_by_dataset.columns]]

    # Define the output path for the CD diagram
    plot_path = Config.plot_dir / "experiment" / plot_name
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_suffix = plot_name.split("/")[-1].replace("_CD.png", "")

    # Plot the CD diagram
    plot_cd_diagram(metrics=metrics_by_dataset, output_path=plot_path)

    # Create bar plot of mean accuracies
    plot_bar_mean_accuracies(
        metrics=metrics_by_model, output_path=plot_path.parent / f"{plot_suffix}_mean_accuracies.png"
    )

    # Create bar plot of dataset accuracies
    plot_bar_dataset_acc(
        metrics=metrics_by_dataset,
        output_path=plot_path.parent / f"{plot_suffix}_dataset_accuracies.png",
    )

    # top10 bar plot of dataset
    plot_bar_dataset_acc(
        metrics=metrics_by_dataset,
        output_path=plot_path.parent / f"{plot_suffix}_top10_dataset_accuracies.png",
        top_n=top_n,
        target_models=target_models,
    )

    # Create scatter plot of relative accuracies
    relative_acc = plot_relative_accuracy_scatter(
        dataset_metrics=metrics_by_dataset,
        output_path=plot_path.parent / f"{plot_suffix}_relative_accuracies.png",
        baseline_conf=baseline_conf,
        legend_label=relative_acc_legend_label,
        ylim=relative_ylim
    )

    abs_acc = plot_relative_accuracy_scatter(
        dataset_metrics=metrics_by_dataset,
        output_path=plot_path.parent / f"{plot_suffix}_absolute_accuracies.png",
        baseline_conf=baseline_conf,
        legend_label=relative_acc_legend_label,
        absolute=True,
        ylim=abs_ylim
    )

    # Print Summary Statistics
    for baseline_col, cols in baseline_conf.items():
        baseline = metrics_by_model[metrics_by_model["model"] == baseline_col]["mean_acc"].values.item()

        summary_diff = (
            metrics_by_model[metrics_by_model["model"].isin(cols)][["model", "mean_acc"]]
            .copy()
            .set_index("model")
        )
        summary_diff["Absolute Difference (%)"] = ((summary_diff["mean_acc"] - baseline) * 100).round(2)
        summary_diff["Relative Difference"] = ((summary_diff["mean_acc"] - baseline) / baseline).round(4)
        summary_diff["mean_acc"] = (summary_diff["mean_acc"] * 100).round(4)
        summary_diff["Max Absolute Difference (%)"] = (abs_acc[cols].max()).round(2)
        summary_diff["Max Relative Difference"] = (relative_acc[cols].max()).round(4)

        print(f"\n{'-'*10} Summary for Exp. {experiment_name} Baseline {baseline_col} {'-'*10}\n")
        print(f"Summary of Accuracies Difference per Model:\n\n{summary_diff.to_latex()}\n\n\n")
        print(f"{'-'* 40}\n")


if __name__ == "__main__":

    # Path to experiment's raw metrics CSV file
    exp_results_dir = pathlib.Path(__file__).parents[3] / "docs" / "experiment_results"

    # Set parameters for the plot
    exp1_dir_name = "1_binding_version_1"
    exp4_comp_dir_name = "4_comp_wise_pe_version_1"
    exp4_cconv_dir_name = "4_conv_pe_version_1"
    plotting_config = [
        # {
        #     "experiment_name": "Experiment 1",
        #     "plot_name": f"{exp1_dir_name}/binding_v1_CD.png",
        #     "exp_dataset_metrics_path": exp_results_dir / exp1_dir_name / "summary_dataset_results.csv",
        #     "exp_model_metrics_path": exp_results_dir / exp1_dir_name / "summary_model_metrics_binding_version_1.csv",
        #     "top_n": 9,
        #     "naming_mapping": {"Sinusoidal": ""},
        #     "models": [
        #         "Linear Comp. Wise",
        #         "Linear Circular Conv.",
        #         "Linear Additive",
        #         "1D Conv. Comp. Wise",
        #         "1D Conv. Circular Conv.",
        #         "1D Conv. Additive",
        #     ],
        #     "target_models": ("1D Conv. Circular Conv.", "1D Conv. Additive"),
        #     "baseline_conf": {
        #         "Linear Additive": ["Linear Comp. Wise", "Linear Circular Conv."],
        #         "1D Conv. Additive": ["1D Conv. Comp. Wise", "1D Conv. Circular Conv."],
        #     },
        #     "relative_acc_legend_label": "Linear & 1D Conv. Additive Baselines",
        # },
        # {
        #     "experiment_name": "Experiment 4 Component Wise",
        #     "plot_name": f"{exp4_comp_dir_name}/component_wise_1_CD.png",
        #     "exp_dataset_metrics_path": exp_results_dir / exp4_comp_dir_name / "summary_dataset_results.csv",
        #     "exp_model_metrics_path": exp_results_dir / exp4_comp_dir_name / "summary_model_metrics_pe_a_version_1.csv",
        #     "top_n": 9,
        #     "naming_mapping": {
        #         "Linear": "",
        #         "Comp. Wise": "",
        #         "No Pe": "Null",
        #         "1 Sinc Fpe": "Sinc 1 FPE",
        #         "2 Sinc Fpe": "Sinc 2 FPE",
        #         "5 Sinc Fpe": "Sinc 5 FPE",
        #         "Random Pe": "Random",
        #     },
        #     "models": [
        #         "Null",
        #         "Random",
        #         "Sinusoidal",
        #         "Sinc 1 FPE",
        #         "Sinc 2 FPE",
        #         "Sinc 5 FPE",
        #     ],
        #     "target_models": ("Sinc 1 FPE", "Null"),
        #     "baseline_conf": {
        #         "Sinusoidal": ["Null", "Random", "Sinc 1 FPE", "Sinc 2 FPE", "Sinc 5 FPE"],
        #     },
        #     "relative_acc_legend_label": "Sinusoidal Baseline",
        #     # y_min=-25, no padding
        # },
        {
            "experiment_name": "Experiment 4 CConv",
            "plot_name": f"{exp4_cconv_dir_name}/cconv_1_CD.png",
            "exp_dataset_metrics_path": exp_results_dir / exp4_cconv_dir_name / "summary_dataset_results.csv",
            "exp_model_metrics_path": exp_results_dir / exp4_cconv_dir_name / "summary_model_metrics_pe_version_1.csv",
            "top_n": 9,
            "naming_mapping": {
                "Linear": "",
                "Conv.": "",
                "No Pe": "Null",
                "1 Sinc Fpe": "FPE $\u03B2=1$",
                "2 Sinc Fpe": "FPE $\u03B2=2$",
                "5 Sinc Fpe": "FPE $\u03B2=5$",
                "Random Pe": "Random",
            },
            "models": [
                "Null",
                "Random",
                "Sinusoidal",
                "FPE $\u03B2=1$",
                "FPE $\u03B2=2$",
                "FPE $\u03B2=5$",
            ],
            "target_models": ("FPE $\u03B2=5$", "Null"),
            "baseline_conf": {
                "Sinusoidal": ["Null", "Random", "FPE $\u03B2=1$", "FPE $\u03B2=2$", "FPE $\u03B2=5$"],
            },
            "relative_acc_legend_label": "Sinusoidal Baseline",
            "relative_ylim": (-0.45, 0.4),
            "abs_ylim": (-25, 10),
            # y_max=10, y_min=-25, no padding
        },
    ]

    # Generate plots for each experiment
    for exp in plotting_config:
        plot_paper_diagrams(**exp)  # type: ignore[arg-type]

    # Exp 5 plot only the CD diagram
    metrics_by_dataset = get_metrics(exp_results_dir / "5_sota_version_1" / "summary_dataset_results.csv")
    cols_to_drop = ["train_samples", "sequence_length", "num_classes"]
    metrics_by_dataset.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Set index for dataset metrics
    metrics_by_dataset.set_index("dataset", inplace=True)
    metrics_by_dataset.columns = format_model_names(
        metrics_by_dataset.columns,
        additional_mapping={
            "Linear": "",
            "Rope": "RoPE",
            "Mla": "MLA",
            "Conv.tran": "ConvTran",
            "Fpe": "FPE",
            "5 Sinc Fpe": "FPE $\u03B2=5$",
            "0 8 Gaussian": "Gaussian $\u03B2=0.8$",
        },
    )

    plot_path = Config.plot_dir / "experiment" / "5_sota_version_1" / "sota_v1b_CD.png"
    plot_cd_diagram(metrics=metrics_by_dataset, output_path=plot_path)
