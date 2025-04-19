# -*- coding: utf-8 -*-
"""Metrics handler module for experiments."""
# Standard imports
import pathlib
from typing import Literal

# Third party imports
import numpy
import pandas
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
    ):
        """Initializes the MetricsHandler class.

        Args:
            metrics_path (pathlib.Path): The path to save the metrics.
            aggregated_metrics_path (pathlib.Path): The path to save the aggregated metrics. If None, it will be set to
                the parent directory of the metrics path with the prefix "aggregated_".
            model_metrics_path (pathlib.Path): The path to save the model metrics. If None, it will be set to the parent
                directory of the metrics path with the prefix "model_".
            metrics_mode (str): The mode to aggregate metrics to. Values are ['append', 'write']. If 'append', and the
                metrics file already exist, the new metrics will be appended to the existing file. If 'write', the new
                metrics will overwrite the existing file.
        """
        self.metrics_path = metrics_path if isinstance(metrics_path, pathlib.Path) else pathlib.Path(metrics_path)
        self.mode = metrics_mode

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

        # Add other information to metrics
        metrics["num_dimensions"] = num_dimensions
        metrics["num_classes"] = num_classes
        metrics["sequence_length"] = sequence_length
        metrics["train_samples"] = train_samples
        metrics["test_samples"] = test_samples
        metrics["validation_samples"] = validation_samples
        metrics["n_train_epochs"] = n_train_epochs
        metrics["training_time_seconds"] = round(training_time, 2)

        metric_cols = metrics.columns.tolist()

        metrics["run"] = run
        metrics["dataset"] = dataset
        metrics["model"] = model

        # Reordering the columns
        metrics = metrics[["dataset", "model", "run"] + metric_cols]

        if self.results.empty:
            self.results = metrics
        else:
            self.results = pandas.concat([self.results, metrics], axis=0)

        # Save updated metrics
        self.results.to_csv(path_or_buf=self.metrics_path, index=False, header=True)

    def aggregate_test_acc_per_dataset_and_model(self) -> None:
        """Aggregates test accuracy per model and dataset with 95% confidence interval and stores it in a CSV file."""
        # metrics = pandas.read_csv(
        #     filepath_or_buffer=self.metrics_path,
        #     header=0,
        # )
        #
        # # Group by dataset and model
        # grouped = metrics.groupby(["dataset", "model"])
        #
        # # Calculate statistics
        # result = grouped["test_acc"].agg(["mean", "std", "count"]).reset_index()
        #
        # # Calculate margin of error for 95% confidence interval
        # # Using t-distribution since we have small sample sizes
        # result["margin_of_error"] = result.apply(
        #     lambda row: stats.t.ppf(0.95, row["count"] - 1) * row["std"] / numpy.sqrt(row["count"]), axis=1
        # ).round(3)
        #
        # # Format the results for better readability
        # result["mean_test_acc"] = result["mean"].round(3)
        # result["std_test_acc"] = result["std"].round(3)
        # result["confidence_interval"] = result.apply(
        #     lambda row: f"{row['mean']:.3f} ± {row['margin_of_error']:.3f}", axis=1
        # )
        #
        # # Select and reorder columns
        # aggregated_metrics = result[
        #     ["dataset", "model", "mean_test_acc", "std_test_acc", "margin_of_error", "confidence_interval"]
        # ]
        #
        # aggregated_metrics.to_csv(path_or_buf=self.aggregated_metrics_path, index=False, header=True)
        self._aggregate_metrics(aggregate_by="dataset_model")

    def aggregate_test_acc_per_model(self) -> None:
        """Aggregates test accuracy per model with 95% confidence interval and stores it in a CSV file."""
        self._aggregate_metrics(aggregate_by="model")

    def _aggregate_metrics(self, aggregate_by: Literal["dataset_model", "model"]) -> None:
        metrics = pandas.read_csv(
            filepath_or_buffer=self.metrics_path,
            header=0,
        )

        # Group by model
        grouped = metrics.groupby(["model"] if aggregate_by == "model" else ["dataset", "model"])

        # Calculate statistics
        result = grouped["test_acc"].agg(["mean", "std", "count"]).reset_index()

        # Calculate margin of error for 95% confidence interval
        # Using t-distribution since we have small sample sizes
        result["margin_of_error"] = result.apply(
            lambda row: stats.t.ppf(0.95, row["count"] - 1) * row["std"] / numpy.sqrt(row["count"]), axis=1
        ).round(3)

        # Format the results for better readability
        result["mean_test_acc"] = result["mean"].round(3)
        result["std_test_acc"] = result["std"].round(3)
        result["confidence_interval"] = result.apply(
            lambda row: f"{row['mean']:.3f} ± {row['margin_of_error']:.3f}", axis=1
        )

        # Select and reorder columns
        cols = ["model", "mean_test_acc", "std_test_acc", "margin_of_error", "confidence_interval"]
        if aggregate_by == "dataset_model":
            cols = ["dataset"] + cols
        aggregated_metrics = result[cols]

        aggregated_metrics.to_csv(
            path_or_buf=self.model_metrics_path if aggregate_by == "model" else self.aggregated_metrics_path,
            index=False,
            header=True,
        )
