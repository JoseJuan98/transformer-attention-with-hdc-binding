# -*- coding: utf-8 -*-
"""Metrics Aggregation Script.

This script aggregates the results from the experiments using the MetricsHandler.

Notes:
    Useful to re-calculate the metrics after modifying any metrics-related code or after adding new metrics.
    It processes the metrics CSV file, aggregates the results, and outputs a DataFrame with the aggregated results.
"""

# Standard imports
import json
import pathlib

# Third party imports
import pandas

# First party imports
from experiment_framework.runner.metrics_handler import MetricsHandler


# Deprecated: This function is no longer used in the current implementation.
def parse_baseline_results(json_path: pathlib.Path) -> pandas.DataFrame:
    """Parses the JSON file containing baseline model results into a DataFrame.

    The JSON is expected to have a "Datasets" key with a list of dataset names, and other keys corresponding to model
     names with lists of accuracies. The order of accuracies must match the order of datasets.

    Args:
        json_path (pathlib.Path): Path to the baseline results JSON file.

    Returns:
        pandas.DataFrame: A DataFrame with per-dataset results for baseline models, formatted to match the
            MetricsHandler output.
    """
    with open(json_path, "r") as file:
        data = json.load(file)

    # The list of datasets is the key to mapping accuracies
    dataset_names = data.pop("Datasets")

    # Remove other metadata keys
    data.pop("Source", None)

    baseline_rows = []
    for model_name, accuracies in data.items():
        if len(accuracies) != len(dataset_names):
            print(
                f"Warning: Skipping model '{model_name}'. Mismatch between number of datasets "
                f"({len(dataset_names)}) and accuracies ({len(accuracies)})."
            )
            continue

        for dataset_name, acc in zip(dataset_names, accuracies):
            # Since these are single-run results, std and MOE are 0.
            # Format this to match the output of `aggregate_test_acc_per_dataset_and_model`
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "mean_acc": round(acc, 4),
                "std_acc": 0.0,
                "num_runs": 1,
                "margin_of_error": 0.0,
                "confidence_interval": f"{acc:.4f} ± 0.0000",
            }
            baseline_rows.append(row)

    if not baseline_rows:
        raise ValueError("No valid baseline data could be parsed from the JSON file.")

    return pandas.DataFrame(baseline_rows)


if __name__ == "__main__":
    # Set pandas options for better display of DataFrames
    pandas.set_option("display.max_columns", None)

    # Set pandas to not truncate DataFrame output
    pandas.set_option("display.width", 0)

    # Path to experiment's raw metrics CSV file
    my_experiment_metrics_path = (
        pathlib.Path(__file__).parents[2]
        # / "docs/experiment_results/1_binding_version_1/metrics_binding_version_1.csv"
        / "docs/experiment_results/2_N_L_version_1/metrics_N_L_version_1.csv"
        # / "docs/experiment_results/3_pe_version_1/metrics_pe_version_1.csv"
        # / "docs/experiment_results/4_sota_version_1/metrics_sota_version_1.csv"
        # / "docs/experiment_results/5_d_model_v1/metrics_d_model_v1.csv"
    )

    # Define output directory
    output_dir = my_experiment_metrics_path.parent
    output_dir.mkdir(exist_ok=True)

    # Validate paths
    if not my_experiment_metrics_path.exists():
        raise FileNotFoundError(f"Experiment metrics file not found at: {my_experiment_metrics_path}")

    # Aggregate experiment's results
    print(f"--> Processing experiment results from: {my_experiment_metrics_path}")
    handler = MetricsHandler(metrics_path=my_experiment_metrics_path, metrics_mode="append")
    handler.aggregate_metrics()
