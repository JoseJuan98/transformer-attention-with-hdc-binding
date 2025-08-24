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
            metrics_path (pathlib.Path, str): The path to save the raw trial metrics.
            aggregated_metrics_path (pathlib.Path): Path to save aggregated metrics per dataset/model.
            model_metrics_path (pathlib.Path): Path to save aggregated metrics per model (based on ranks).
            metrics_mode (str): 'append' to add to existing metrics file, 'write' to overwrite.
            significance_level (float): The significance level for statistical tests (e.g., 0.05 for 95% CI).
            min_runs_for_filter (int): The minimum number of runs in a group to apply percentile-based filtering.
            metrics_precision (int): The number of decimal places to round the metrics to.
        """
        self.metrics_path = metrics_path if isinstance(metrics_path, pathlib.Path) else pathlib.Path(metrics_path)
        self.mode = metrics_mode
        self.significance_level = significance_level
        self.metrics_precision = metrics_precision
        self.min_runs_for_filter = min_runs_for_filter

        # Define output paths with clear naming
        self.aggregated_dataset_model_path = (
            pathlib.Path(aggregated_metrics_path)
            if aggregated_metrics_path
            else self.metrics_path.parent / f"aggregated_by_dataset_{self.metrics_path.name}"
        )
        self.aggregated_model_rank_path = (
            pathlib.Path(model_metrics_path)
            if model_metrics_path
            else self.metrics_path.parent / f"summary_model_{self.metrics_path.name}"
        )

        # Ensure directory exists
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if self.mode == "append" and self.metrics_path.exists():
            self.results = pandas.read_csv(filepath_or_buffer=self.metrics_path, header=0)
        else:
            self.results = pandas.DataFrame()

        # Aggregated metrics by model and dataset (to avoid re-computing it multiple times)
        self.dataset_and_model_metrics = pandas.DataFrame()

        # Filtered metrics DataFrame to avoid re-computing it multiple times
        self._filtered_metrics = pandas.DataFrame()

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
        best_train_epoch: int,
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
            best_train_epoch (int): The best training epoch choosen by the Pocket Algorithm.
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
        metrics_copy["best_epoch"] = best_train_epoch
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

        self.results = pandas.concat([self.results, metrics_copy], ignore_index=True, axis=0, sort=False)

        # Save updated metrics
        self.results.to_csv(path_or_buf=self.metrics_path, index=False, header=True)

    def _calculate_moe(self, row: pandas.Series, std_col: str, count_col: str) -> float:
        """Calculate the Margin of Error (MOE) for a given row."""
        n = row[count_col]
        std = row[std_col]

        # If n > 1 and std is not NaN and greater than 0, calculate MOE using t-distribution
        if n > 1 and pandas.notna(std) and std > 0:

            t_score = stats.t.ppf(1 - self.significance_level / 2, n - 1)
            return t_score * std / numpy.sqrt(n)

        # If n <= 1 or std is NaN or 0, return 0.0 (no uncertainty)
        elif n <= 1 or std == 0:
            return 0.0
        else:
            return numpy.nan

    def _get_filtered_metrics(self) -> pandas.DataFrame:
        """Loads, cleans, and filters the raw metrics data.

        Notes:
            The Interquartile Range (IQR) method is used to filter out outliers, because it is robust, and it's a
            well-established non-parametric method for identifying outliers.

        """

        if not self._filtered_metrics.empty:
            print("Using cached filtered metrics.")
            return self._filtered_metrics

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

        required_cols = ["dataset", "model", "test_acc"]
        if not all(col in metrics.columns for col in required_cols):
            missing = [col for col in required_cols if col not in metrics.columns]
            raise KeyError(f"Error: Missing required columns in metrics data: {missing}")

        metrics["test_acc"] = pandas.to_numeric(metrics["test_acc"], errors="coerce")
        original_rows = len(metrics)
        metrics.dropna(subset=["test_acc"], inplace=True)

        if len(metrics) < original_rows:
            logging.warning(f"Dropped {original_rows - len(metrics)} rows with non-numeric or NaN 'test_acc'.")

        if metrics.empty:
            raise ValueError("Error: No valid 'test_acc' data remaining after cleaning.")

        # --- Outlier Filtering Step ---
        group_keys = ["dataset", "model"]
        metric_col = "test_acc"

        # Use transform to broadcast group-wise calculations
        metrics["group_count"] = metrics.groupby(group_keys)[metric_col].transform("count")

        # Calculate Q1, Q3, and IQR for filtering
        Q1 = metrics.groupby(group_keys)[metric_col].transform("quantile", 0.25)
        Q3 = metrics.groupby(group_keys)[metric_col].transform("quantile", 0.75)
        IQR = Q3 - Q1

        # Define the outlier bounds
        metrics["lower_bound"] = Q1 - 1.5 * IQR
        metrics["upper_bound"] = Q3 + 1.5 * IQR

        # Keep a row if the group is too small to filter OR if the value is within the bounds
        filter_mask = (metrics["group_count"] < self.min_runs_for_filter) | (
            (metrics[metric_col] >= metrics["lower_bound"]) & (metrics[metric_col] <= metrics["upper_bound"])
        )

        filtered_metrics = metrics[filter_mask].copy()

        rows_filtered_out = metrics.shape[0] - filtered_metrics.shape[0]
        if rows_filtered_out > 0:
            print(
                f"Filtered out {rows_filtered_out}/{metrics.shape[0]} rows as outliers falling outside the IQR bounds "
                f"(for groups with >= {self.min_runs_for_filter} runs)."
            )

        self._filtered_metrics = filtered_metrics.drop(columns=["group_count", "lower_bound", "upper_bound"])
        return self._filtered_metrics

    def aggregate_test_acc_per_dataset_and_model(self) -> pandas.DataFrame:
        """Aggregates test accuracy per model and dataset after filtering outliers.

        Calculates mean, std, and a 95% confidence interval.
        """
        if not self.dataset_and_model_metrics.empty:
            print("Using cached dataset and model metrics.")
            return self.dataset_and_model_metrics

        filtered_metrics = self._get_filtered_metrics()
        if filtered_metrics.empty:
            print("Warning: No data remains after filtering. Cannot aggregate.")
            return pandas.DataFrame()

        group_keys = ["dataset", "model", "train_samples", "sequence_length", "num_classes"]

        # Aggregate the filtered data
        agg_filtered = (
            filtered_metrics.groupby(group_keys)["test_acc"].agg(mean="mean", std="std", count="count").reset_index()
        )

        # Calculate MOE on the filtered data
        agg_filtered["margin_of_error"] = agg_filtered.apply(
            self._calculate_moe, axis=1, std_col="std", count_col="count"
        )

        # Formatting
        for col in ["mean", "std", "margin_of_error"]:
            agg_filtered[col] = agg_filtered[col].round(self.metrics_precision)

        agg_filtered["confidence_interval"] = agg_filtered.apply(
            lambda row: (
                f"{row['mean']:.4f} ± {row['margin_of_error']:.4f}" if pandas.notna(row["margin_of_error"]) else "N/A"
            ),
            axis=1,
        )

        agg_filtered.rename(
            columns={
                "mean": "mean_acc",
                "std": "std_acc",
                "count": "num_runs_filtered",
            },
            inplace=True,
        )

        # Save and return
        self.aggregated_dataset_model_path.parent.mkdir(parents=True, exist_ok=True)
        agg_filtered.to_csv(
            self.aggregated_dataset_model_path, index=False, float_format=f"%.{self.metrics_precision}f"
        )
        print(f"Aggregated metrics per dataset/model saved to {self.aggregated_dataset_model_path}")
        self.dataset_and_model_metrics = agg_filtered

        return agg_filtered

    def aggregate_test_acc_per_model_by_rank(self) -> pandas.DataFrame:
        """Aggregates model performance across datasets using ranks, which is more robust than averaging raw accuracy"""
        # Use the per-dataset aggregated results as the starting point
        per_dataset_results = self.aggregate_test_acc_per_dataset_and_model()

        # Rank models within each dataset based on their mean filtered accuracy (higher is better)
        per_dataset_results["rank"] = per_dataset_results.groupby("dataset")["mean_acc"].rank(
            method="average", ascending=False
        )

        # Aggregate the ranks for each model across all datasets
        rank_aggregation = (
            per_dataset_results.groupby("model")["rank"]
            .agg(mean_rank="mean", std_rank="std", num_datasets="count")
            .reset_index()
        )

        # Sort by the most important metric: mean_rank (lower is better)
        rank_aggregation.sort_values("mean_rank", inplace=True)

        # Formatting
        for col in ["mean_rank", "std_rank"]:
            rank_aggregation[col] = rank_aggregation[col].round(self.metrics_precision)

        # Get the mean accuracy for each model across all datasets
        model_acc = (
            self.aggregate_test_acc_per_model()
            .rename(columns={"mean": "mean_acc"})
            .drop(columns=["std", "count"], errors="ignore")
        )

        rank_aggregation = (
            rank_aggregation
            # Merge the mean accuracy with the rank aggregation
            .merge(model_acc, on="model", how="left")
            # Reorder columns to have a clear view
            [["model", "mean_acc", "mean_rank", "std_rank", "num_datasets"]]
        )

        # Save and return
        self.aggregated_model_rank_path.parent.mkdir(parents=True, exist_ok=True)
        rank_aggregation.to_csv(
            self.aggregated_model_rank_path, index=False, float_format=f"%.{self.metrics_precision}f"
        )
        print(f"Aggregated model performance by rank saved to {self.aggregated_model_rank_path}")

        return rank_aggregation

    def aggregate_test_acc_per_model(self) -> pandas.DataFrame:
        """Aggregates test accuracy per model from the filtered results."""
        return (
            self._get_filtered_metrics()
            .groupby(["model"])["test_acc"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .round(self.metrics_precision)
        )

    def aggregate_per_dataset_with_model_as_cols(self) -> pandas.DataFrame:
        """Creates a pivot table of (Dataset x Model) showing the confidence interval."""
        agg_dataset_model_metrics = self.aggregate_test_acc_per_dataset_and_model()

        # Get the average accuracy for each model across all datasets
        model_metrics = self.aggregate_test_acc_per_model()
        avg_dataset_name = "Average"
        model_metrics["dataset"] = avg_dataset_name

        # Pivot the model metrics to have models as columns
        row_model_metrics = model_metrics.pivot(index="dataset", columns="model", values="mean").reset_index()

        # Create a new "dataset" that is the average of each model across all datasets
        # Pivot using a multi-level index to preserve the dataset columns
        dataset_columns = ["dataset", "train_samples", "sequence_length", "num_classes"]
        pivot_table = agg_dataset_model_metrics.pivot(
            index=dataset_columns, columns="model", values="confidence_interval"
        ).reset_index()

        # Calculate the mean of each dataset column except "dataset"
        for col in dataset_columns[1:]:
            row_model_metrics[col] = pivot_table[col].mean().round(0).astype(int)

        # Concatenate the average model metrics row with the dataset pivoted table
        pivot_table = pandas.concat([pivot_table, row_model_metrics])

        # Reorder the columns to have the dataset columns first
        pivot_table = pivot_table[
            dataset_columns + sorted(list(set(pivot_table.columns.to_list()) - set(dataset_columns)))
        ]

        output_path = self.metrics_path.parent / "summary_dataset_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pivot_table.to_csv(output_path, index=False, header=True)
        print(f"Dataset results pivot table saved to {output_path}")
        return pivot_table

    def aggregate_metrics(self) -> None:
        """Runs all aggregation methods."""
        print("\n\t=> Aggregating Test Accuracy per Dataset and Model...")
        self.aggregate_test_acc_per_dataset_and_model()

        print("\n\t=> Aggregating Model Performance Across Datasets by Rank...")
        self.aggregate_test_acc_per_model_by_rank()

        print("\n\t=> Generating Pivot Table (Dataset vs Model)...")
        self.aggregate_per_dataset_with_model_as_cols()
        print("\n\t=> Aggregation complete")

    @staticmethod
    def get_train_metrics_and_plot(
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
