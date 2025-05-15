# -*- coding: utf-8 -*-
"""Metrics handler module for experiments."""
# Standard imports
import logging
import os
import pathlib
from typing import Literal

# Third party imports
import numpy
import pandas
import seaborn
from matplotlib import pyplot
from scipy import stats

METRICS_MODE_STR = Literal["append", "write"]


class MetricsHandler:
    """Handles metrics updates and aggregation."""

    def __init__(
        self,
        metrics_path: pathlib.Path | str,
        aggregated_metrics_path: pathlib.Path | str | None = None,
        model_metrics_path: pathlib.Path | str | None = None,
        metrics_mode: METRICS_MODE_STR = "append",
        significance_level: float = 0.05,
        min_runs_for_filter: int = 4,
        metrics_precision: int = 4,
    ):
        """Initializes the MetricsHandler class.

        Args:
            metrics_path (pathlib.Path, str): The path to save the metrics.
            aggregated_metrics_path (pathlib.Path): The path to save the aggregated metrics. If None, it will be set to
                the parent directory of the metrics path with the prefix "aggregated_".
            model_metrics_path (pathlib.Path): The path to save the model metrics. If None, it will be set to the parent
                directory of the metrics path with the prefix "model_".
            metrics_mode (str): The mode to aggregate metrics to. Values are ['append', 'write']. If 'append', and the
                metrics file already exist, the new metrics will be appended to the existing file. If 'write', the new
                metrics will overwrite the existing file.
            significance_level (float): The significance level for statistical tests. Defaults to 0.05.
            metrics_precision (int): The number of decimal places to round the metrics to. Defaults to 4.
            min_runs_for_filter (int): The minimum number of runs in a dataset/model group to apply filtering. Defaults to 4.
        """
        self.metrics_path = metrics_path if isinstance(metrics_path, pathlib.Path) else pathlib.Path(metrics_path)
        self.mode = metrics_mode
        # Divide significance level by 2 for two-tailed tests
        self.significance_level = significance_level / 2
        self.metrics_precision = metrics_precision
        # Minimum runs in a dataset/model group to apply filtering
        self.min_runs_for_filter = min_runs_for_filter

        if isinstance(aggregated_metrics_path, str):
            aggregated_metrics_path = pathlib.Path(aggregated_metrics_path)

        if isinstance(model_metrics_path, str):
            model_metrics_path = pathlib.Path(model_metrics_path)

        self.aggregated_metrics_path = (
            aggregated_metrics_path
            if aggregated_metrics_path
            else self.metrics_path.parent / f"aggregated_{self.metrics_path.name}"
        )
        self.model_metrics_path = (
            model_metrics_path if model_metrics_path else self.metrics_path.parent / f"model_{self.metrics_path.name}"
        )

        # Ensure directory exists
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if self.mode == "append" and self.metrics_path.exists():
            self.results = pandas.read_csv(filepath_or_buffer=self.metrics_path, header=0)
        else:
            self.results = pandas.DataFrame()

    def update_metrics(
        self,
        metrics: pandas.DataFrame,
        run: int,
        dataset: str,
        model: str,
        num_dimensions: int,
        num_classes: int,
        sequence_length: int,
        train_samples: int,
        test_samples: int,
        validation_samples: int,
        n_train_epochs: int,
        training_time: float = 0.0,
    ) -> None:
        """Update the global metrics with the new metrics.

        Args:
            metrics (pandas.DataFrame): The new metrics to update.
            run (int): The run number.
            dataset (str): The name of the dataset.
            model (str): The name of the model.
            num_dimensions (int): The number of dimensions in the dataset.
            num_classes (int): The number of classes in the dataset.
            sequence_length (int): The length of the sequences in the dataset.
            train_samples (int): The number of training samples.
            test_samples (int): The number of testing samples.
            validation_samples (int): The number of validation samples.
            n_train_epochs (int): The number of training epochs.
            training_time (float): The training time in seconds.
        """
        if metrics.empty:
            raise ValueError(f"Metrics DataFrame is empty. Cannot update metrics.\n{metrics}")

        # Create a copy to avoid modifying the original DataFrame passed to the function
        metrics_copy = metrics.copy()

        # Add other information to metrics
        metrics_copy["num_dimensions"] = num_dimensions
        metrics_copy["num_classes"] = num_classes
        metrics_copy["sequence_length"] = sequence_length
        metrics_copy["train_samples"] = train_samples
        metrics_copy["test_samples"] = test_samples
        metrics_copy["validation_samples"] = validation_samples
        metrics_copy["n_train_epochs"] = n_train_epochs
        metrics_copy["training_time_seconds"] = round(training_time, 2)

        metric_cols = metrics_copy.columns.tolist()

        metrics_copy["run"] = run
        metrics_copy["dataset"] = dataset
        metrics_copy["model"] = model

        # Reordering the columns
        # Ensure 'dataset', 'model', 'run' are first, handle potential missing cols gracefully
        core_cols = ["dataset", "model", "run"]
        other_cols = [col for col in metric_cols if col not in core_cols]
        final_order = core_cols + other_cols
        metrics_copy = metrics_copy[final_order]

        if self.results.empty:
            self.results = metrics_copy
        else:
            self.results = pandas.concat([self.results, metrics_copy], axis=0, sort=False)

        # Save updated metrics
        self.results.to_csv(path_or_buf=self.metrics_path, index=False, header=True)

    def aggregate_test_acc_per_dataset_and_model(self) -> pandas.DataFrame:
        """Aggregates test accuracy per model and dataset with 95% confidence interval and stores it in a CSV file."""
        return self._aggregate_metrics(aggregate_by="dataset_model")

    def aggregate_test_acc_per_model(self) -> pandas.DataFrame:
        """Aggregates test accuracy per model with 95% confidence interval and stores it in a CSV file."""
        return self._aggregate_metrics(aggregate_by="model")

    def _calculate_moe(self, row):
        """Calculate the margin of error (MOE) for the given row."""
        n = row["num_runs_filtered"]
        std = row["std_filtered"]

        if n > 1 and pandas.notna(std) and std > 0:
            t_score = stats.t.ppf((1 - self.significance_level * 2), n - 1)
            return t_score * std / numpy.sqrt(n)

        elif n == 1 or std == 0:
            return 0.0

        # n=0 or std is NaN/0
        else:
            return numpy.nan

    def _aggregate_metrics(self, aggregate_by: Literal["dataset_model", "model"]) -> pandas.DataFrame:
        # --- Initial Loading ---
        # Use the in-memory DataFrame if available and not empty, otherwise read from file
        if not self.results.empty:
            metrics = self.results.copy()
            print(f"Using in-memory metrics (found {len(metrics)} rows).")

        else:
            metrics = pandas.read_csv(
                filepath_or_buffer=self.metrics_path,
                header=0,
            )
            print(f"Read metrics from {self.metrics_path} ({len(metrics)} rows).")

        if metrics.empty:
            raise ValueError("Error: No metrics data available to aggregate.")

        # --- Essential Column Checks ---
        required_cols = ["dataset", "model", "test_acc"]
        if not all(col in metrics.columns for col in required_cols):
            missing = [col for col in required_cols if col not in metrics.columns]
            raise KeyError(f"Error: Missing required columns in metrics data: {missing}")

        # --- Data Cleaning ---
        # Ensure 'test_acc' is numeric, coerce errors to NaN
        metrics["test_acc"] = pandas.to_numeric(metrics["test_acc"], errors="coerce")
        original_rows = len(metrics)
        metrics.dropna(subset=["test_acc"], inplace=True)
        if len(metrics) < original_rows:
            print(f"Warning: Dropped {original_rows - len(metrics)} rows with non-numeric or NaN 'test_acc'.")

        if metrics.empty:
            raise ValueError("Error: No valid 'test_acc' data remaining after cleaning.")

        # --- Step 1: Filter Data Per Dataset/Model ---
        filter_group_keys = ["dataset", "model"]
        metric_col = "test_acc"

        # Calculate group size, q05, q95 using transform to broadcast results back to the original DataFrame shape
        metrics["group_count"] = metrics.groupby(filter_group_keys)[metric_col].transform("count")
        metrics["q05"] = metrics.groupby(filter_group_keys)[metric_col].transform(
            lambda x: x.quantile(self.significance_level)
        )
        metrics["q95"] = metrics.groupby(filter_group_keys)[metric_col].transform(
            lambda x: x.quantile(1 - self.significance_level)
        )

        # Keep row if:
        # 1. The group has fewer than min_runs_for_filter OR
        # 2. The group has enough runs AND the value is within [q05, q95]
        filter_mask = (metrics["group_count"] < self.min_runs_for_filter) | (
            (metrics[metric_col] >= metrics["q05"]) & (metrics[metric_col] <= metrics["q95"])
        )

        filtered_metrics = metrics[filter_mask].copy()  # Create the filtered DataFrame

        # Optional: Report how many rows were filtered out
        rows_filtered_out = len(metrics) - len(filtered_metrics)
        if rows_filtered_out > 0:
            print(
                f"Filtered out {rows_filtered_out} runs falling outside the {self.significance_level}th-"
                f"{1 - self.significance_level}th percentile range within their dataset/model group (for groups with "
                f">= {self.min_runs_for_filter} runs)."
            )
        else:
            print("No runs were filtered out (or all groups were too small to filter).")

        # Drop temporary columns
        filtered_metrics = filtered_metrics.drop(columns=["group_count", "q05", "q95"])

        if filtered_metrics.empty:
            print("Warning: All rows were filtered out. No data left for aggregation.")
            return pandas.DataFrame()

        # --- Step 2: Aggregate the Filtered Data ---
        # Define the final grouping keys based on the desired aggregation level
        final_grouping_keys = ["model"] if aggregate_by == "model" else ["dataset", "model"]
        grouped_filtered = filtered_metrics.groupby(final_grouping_keys)

        # Calculate statistics on the filtered data
        stats_filtered = grouped_filtered[metric_col].agg(["mean", "std", "count"]).reset_index()
        stats_filtered.rename(
            columns={"count": "num_runs_filtered", "mean": "mean_filtered", "std": "std_filtered"}, inplace=True
        )

        # Calculate original counts for the same groups using the *unfiltered* data
        original_counts = metrics.groupby(final_grouping_keys).size().reset_index(name="num_runs_original")

        # Merge filtered stats and original counts
        result = pandas.merge(stats_filtered, original_counts, on=final_grouping_keys, how="left")

        # --- Calculate Margin of Error on Filtered Data ---
        result["margin_of_error"] = result.apply(self._calculate_moe, axis=1)

        # --- Formatting ---
        result["mean_test_acc_filtered"] = result["mean_filtered"].round(self.metrics_precision)
        result["std_test_acc_filtered"] = result["std_filtered"].round(self.metrics_precision)
        result["margin_of_error"] = result["margin_of_error"].round(self.metrics_precision)

        result["confidence_interval_filtered"] = result.apply(
            lambda row: (
                f"{row['mean_test_acc_filtered']:.4f} ± {row['margin_of_error']:.4f}"
                if pandas.notna(row["mean_test_acc_filtered"]) and pandas.notna(row["margin_of_error"])
                else "N/A"
            ),
            axis=1,
        )

        # Ensure integer types for counts
        result["num_runs_filtered"] = result["num_runs_filtered"].astype(int)
        result["num_runs_original"] = result["num_runs_original"].astype(int)

        # --- Select and Reorder Final Columns ---
        cols = [
            "model",
            "confidence_interval_filtered",
            "mean_test_acc_filtered",
            "std_test_acc_filtered",
            "margin_of_error",
            "num_runs_filtered",
            "num_runs_original",
        ]
        if aggregate_by == "dataset_model":
            cols = ["dataset"] + cols

        # Select only the desired columns that actually exist in the result
        final_cols = [col for col in cols if col in result.columns]
        aggregated_metrics = result[final_cols]

        # --- Save Results ---
        output_path = self.model_metrics_path if aggregate_by == "model" else self.aggregated_metrics_path

        aggregated_metrics = aggregated_metrics.rename(
            columns={
                "mean_test_acc_filtered": "mean_test_acc",
                "std_test_acc_filtered": "std",
                "num_runs_filtered": "num_runs",
                "confidence_interval_filtered": "confidence_interval",
            }
        )

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        aggregated_metrics.to_csv(
            path_or_buf=output_path,
            index=False,
            header=True,
            float_format="%.4f",
        )

        return aggregated_metrics

    def aggregate_per_dataset_with_model_as_cols(self) -> pandas.DataFrame:
        """Aggregates test accuracy per dataset with models as columns and stores it in a CSV file."""
        # Use the already aggregated (and filtered) data
        agg_dataset_model_metrics = self.aggregate_test_acc_per_dataset_and_model()

        if agg_dataset_model_metrics.empty:
            print("Cannot generate dataset_results.csv as dataset/model aggregation is empty.")
            return pandas.DataFrame()

        # Check if the required columns exist before pivoting
        required_pivot_cols = ["dataset", "model", "confidence_interval"]
        if not all(col in agg_dataset_model_metrics.columns for col in required_pivot_cols):
            missing = [col for col in required_pivot_cols if col not in agg_dataset_model_metrics.columns]
            print(f"Cannot pivot for dataset_results.csv. Missing columns: {missing}")
            return pandas.DataFrame()

        # Pivot over the models using the filtered confidence interval
        try:
            # Use the filtered confidence interval column
            pivot_table = agg_dataset_model_metrics.pivot(
                index="dataset", columns="model", values="confidence_interval"
            )
        except Exception as e:
            print(f"Error pivoting data: {e}")
            # Handle potential duplicate dataset/model entries if they weren't aggregated correctly
            # Or if the aggregation step failed silently.
            # Let's try pivot_table instead which handles aggregation
            print("Attempting pivot_table with aggregation...")
            try:
                pivot_table = pandas.pivot_table(
                    agg_dataset_model_metrics,
                    values="confidence_interval",
                    index=["dataset"],
                    columns=["model"],
                    aggfunc="first",  # Should only be one value per dataset/model after aggregation
                )
            except Exception as e2:
                print(f"Error creating pivot_table: {e2}")
                return pandas.DataFrame()

        output_path = self.metrics_path.parent / "dataset_results.csv"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pivot_table.to_csv(path_or_buf=output_path, index=True, header=True)
            print(f"Dataset results (pivot table) saved to {output_path}")
        except Exception as e:
            print(f"Error saving pivot table results to {output_path}: {e}")

        return pivot_table

    def aggregate_metrics(self) -> None:
        """Alias for all aggregate methods."""
        print("\n\t=> Aggregating Test Accuracy per Dataset and Model")
        self.aggregate_test_acc_per_dataset_and_model()
        print("\n\t=> Aggregating Test Accuracy per Model (using filtered data pooled across datasets)")
        self.aggregate_test_acc_per_model()
        print("\n\t=> Generating Pivot Table (Dataset vs Model)")
        self.aggregate_per_dataset_with_model_as_cols()
        print("\n\t=> Aggregation complete")

    def get_train_metrics_and_plot(
        self,
        csv_dir: str,
        experiment: str,
        logger: logging.Logger | None = None,
        plots_path: pathlib.Path | None = None,
        show_plot: bool = False,
        dataset_name: str = "",
        model_name: str = "",
        run_version: str = "",
    ) -> pandas.DataFrame:
        """Save the metrics plot.

        Args:
            plots_path (pathlib.Path): Path to save the plot.
            csv_dir (str): Path to the directory containing the metrics.csv file.
            experiment (str): Name of the experiment.
            logger (logging.Logger, optional): Logger object. Defaults to None.
            show_plot (bool, optional): Whether to display the plot. Defaults to False.
            dataset_name (str, optional): Name of the dataset.
            model_name (str, optional): Name of the model.
            run_version (str, optional): Version of the run.

        Returns:
            pandas.DataFrame: Pandas DataFrame containing the final metrics of the plot.
        """
        metrics = pandas.read_csv(filepath_or_buffer=os.path.join(csv_dir, "metrics.csv"))

        metrics.drop(columns=["step", "n_samples"], axis=1, inplace=True, errors="ignore")
        metrics.set_index("epoch", inplace=True)

        test_loss = metrics["test_loss"].dropna(how="all").mean()

        if pandas.isna(test_loss):
            test_loss = None
        else:
            test_loss = round(test_loss, 4)
        test_acc = metrics["test_acc"].dropna(how="all").mean().round(4)

        if logger:
            logger.info(f"\nExperiment {experiment}\n\tTest loss: {test_loss}.\n\tTest accuracy: {test_acc}.\n\n")

        # Prepare data for plotting (exclude test metrics)
        plot_cols_to_drop = ["test_loss", "test_acc"]
        plotting_data = metrics.drop(columns=plot_cols_to_drop, axis=1, errors="ignore").copy()
        # Drop columns that are entirely NaN for plotting
        plotting_data.dropna(axis=1, how="all", inplace=True)

        if plots_path is not None and not plotting_data.empty:
            graph = seaborn.relplot(data=plotting_data, kind="line")
            graph.figure.suptitle(f"{dataset_name} - {model_name} run {run_version}")
            graph.figure.subplots_adjust(top=0.9)
            graph.tight_layout()

            plots_path.parent.mkdir(parents=True, exist_ok=True)
            pyplot.savefig(fname=plots_path)

            if show_plot:
                pyplot.show()

            pyplot.close()

        # --- Extract Final Metrics ---
        # Drop test columns again if they existed, keep only train/val
        metrics = metrics.drop(columns=plot_cols_to_drop, axis=1, errors="ignore").copy()
        # Drop rows where all train/val metrics are NaN
        metrics.dropna(how="all", inplace=True)

        # Get the values of the row with the best validation accuracy, if it is not empty, "val_acc" is in the columns,
        #   and it has at least one value
        if not metrics.empty and "val_acc" in metrics.columns.tolist() and metrics["val_acc"].notna().any():
            best_val_acc_index = metrics["val_acc"].idxmax()
            metrics = metrics[metrics.index == best_val_acc_index].mean(axis=0).round(4)

        else:
            # Most likely, due to exploding or vanishing gradients
            metrics = pandas.Series()
            metrics["train_loss"] = numpy.nan
            metrics["train_acc"] = numpy.nan
            metrics["val_loss"] = numpy.nan
            metrics["val_acc"] = numpy.nan

        metrics["test_loss"] = test_loss
        metrics["test_acc"] = test_acc
        metrics["size_MB"] = round(pathlib.Path(f"{csv_dir}/model.pth").stat().st_size / (1024**2), 4)

        return metrics.to_frame().T


if __name__ == "__main__":
    # Standard imports
    import pathlib

    metrics_path = (
        pathlib.Path(__file__).parents[3]
        / "artifacts/metrics/version_2/metrics_version_2_until_characters_trajectory.csv"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    handler = MetricsHandler(metrics_path=metrics_path)
    handler.aggregate_test_acc_per_model()
    handler.aggregate_test_acc_per_dataset_and_model()
