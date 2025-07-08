# -*- coding: utf-8 -*-
"""Metrics Aggregation and SOTA Comparison Script.

This script performs the following tasks:
1. It aggregates the results from my experiments using the MetricsHandler.
2. It parses the ConvTran_results.json file, which contains single-run accuracy scores for several baseline models
    (ConvTran, TST, etc.) on various datasets.
3. It combines the aggregated results with the baseline results into a unified DataFrame.
4. From this combined data, it generates two key comparison outputs:
   a) A pivot table (Dataset vs. Model) showing the performance of all models side-by-side.
   b) A table of average ranks, providing a robust overall comparison of all models across all datasets.
"""

# Standard imports
import json
import pathlib

# Third party imports
import pandas

# First party imports
from experiment_framework.runner.metrics_handler import MetricsHandler


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
                "mean_acc_filtered": round(acc, 4),
                "std_acc_filtered": 0.0,
                "num_runs_filtered": 1,
                "margin_of_error": 0.0,
                "ci_filtered": f"{acc:.4f} ± 0.0000",
            }
            baseline_rows.append(row)

    if not baseline_rows:
        raise ValueError("No valid baseline data could be parsed from the JSON file.")

    return pandas.DataFrame(baseline_rows)


if __name__ == "__main__":
    # --- Step 1: Define paths ---
    # Path to experiment's raw metrics CSV file
    my_experiment_metrics_path = (
        pathlib.Path(__file__).parents[2]
        # / "docs/experiment_results/binding_version_1/metrics_binding_version_1.csv"
        # / "docs/experiment_results/binding_N_L_4/metrics_binding_N_l_4.csv"
        # / "docs/experiment_results/pe_version_1/metrics_pe_version_1.csv"
        / "docs/experiment_results/sota_version_1/metrics_sota_version_1.csv"
    )
    # Path to the JSON file with baseline (ConvTran, TST, etc.) results
    baseline_results_path = pathlib.Path("ConvTran_results.json")

    # Define output directory
    output_dir = my_experiment_metrics_path.parent
    output_dir.mkdir(exist_ok=True)

    # Validate paths
    if not my_experiment_metrics_path.exists():
        raise FileNotFoundError(f"Experiment metrics file not found at: {my_experiment_metrics_path}")
    if not baseline_results_path.exists():
        raise FileNotFoundError(f"Baseline results file not found at: {baseline_results_path}")

    # --- Step 2: Aggregate experiment's results ---
    print(f"--> Processing experiment results from: {my_experiment_metrics_path}")
    handler = MetricsHandler(metrics_path=my_experiment_metrics_path, metrics_mode="append")
    handler.aggregate_metrics()

    # Uncomment the following line if you want to compare the test accuracies with the results of the ConvTran paper.
    # my_results_df = handler.aggregate_test_acc_per_dataset_and_model()
    # print(f"Found {len(my_results_df)} aggregated results for models.")
    #
    # # --- Step 3: Parse the baseline model results ---
    # print(f"\n--> Parsing baseline results from: {baseline_results_path}")
    # baseline_df = parse_baseline_results(baseline_results_path)
    # print(f"Parsed {len(baseline_df)} results for {baseline_df['model'].nunique()} baseline models.")
    #
    # # --- Step 4: Combine results with the baseline results ---
    # combined_df = pandas.concat([my_results_df, baseline_df], ignore_index=True)
    # print(f"\n--> Combined results contain {len(combined_df)} total entries.")
    #
    # # --- Step 5: Generate final comparison outputs from the combined data ---
    #
    # # 5a. Create a comprehensive pivot table (Dataset vs. Model)
    # print("\n--> Generating comparison pivot table...")
    # pivot_table = combined_df.pivot_table(
    #     index="dataset",
    #     columns="model",
    #     values="confidence_interval",  # Using the confidence interval string for a clear view
    #     aggfunc='first'  # Use 'first' as there should be only one entry per dataset/model
    # )
    # pivot_output_path = output_dir / "dataset_comparison_pivot_table.csv"
    # pivot_table.to_csv(pivot_output_path)
    # print(f"    Saved pivot table to: {pivot_output_path}")
    #
    # # 5b. Calculate and save the average ranks for an overall comparison
    # print("\n--> Generating overall model comparison by rank...")
    # # Rank models within each dataset (lower rank is better)
    # combined_df["rank"] = combined_df.groupby("dataset")["mean_acc_filtered"].rank(
    #     method="average", ascending=False
    # )
    # # Calculate average rank for each model across all datasets
    # rank_comparison_df = combined_df.groupby("model")["rank"].agg(
    #     mean_rank="mean",
    #     std_rank="std",
    #     num_datasets="count"
    # ).round(3).sort_values("mean_rank")
    #
    # rank_output_path = output_dir / "model_comparison_by_rank.csv"
    # rank_comparison_df.to_csv(rank_output_path)
    # print(f"    Saved rank comparison to: {rank_output_path}")
    #
    # print("\nComparison script finished successfully.")
