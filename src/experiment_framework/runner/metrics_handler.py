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

    def aggregate_test_acc_per_dataset_and_model(self) -> pandas.DataFrame:
        """Aggregates test accuracy per model and dataset with 95% confidence interval and stores it in a CSV file."""
        return self._aggregate_metrics(aggregate_by="dataset_model")

    def aggregate_test_acc_per_model(self) -> pandas.DataFrame:
        """Aggregates test accuracy per model with 95% confidence interval and stores it in a CSV file."""
        return self._aggregate_metrics(aggregate_by="model")

    def _aggregate_metrics(self, aggregate_by: Literal["dataset_model", "model"]) -> pandas.DataFrame:
        metrics = pandas.read_csv(
            filepath_or_buffer=self.metrics_path,
            header=0,
        )

        # Group by model
        grouped = metrics.groupby(["model"] if aggregate_by == "model" else ["dataset", "model"])

        # Calculate statistics
        # TODO: calculate the average training time and epochs
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
        result["num_runs"] = result["count"].astype(int)

        # Select and reorder columns
        cols = ["model", "mean_test_acc", "std_test_acc", "margin_of_error", "confidence_interval", "num_runs"]
        if aggregate_by == "dataset_model":
            cols = ["dataset"] + cols
        aggregated_metrics = result[cols]

        aggregated_metrics.to_csv(
            path_or_buf=self.model_metrics_path if aggregate_by == "model" else self.aggregated_metrics_path,
            index=False,
            header=True,
        )

        return aggregated_metrics

    def aggregate_per_dataset_with_model_as_cols(self) -> pandas.DataFrame:
        """Aggregates test accuracy per dataset with models as columns and stores it in a CSV file."""
        # Pivot over the models
        agg_metrics = self.aggregate_test_acc_per_dataset_and_model()[
            ["dataset", "model", "confidence_interval"]
        ].pivot(index="dataset", columns="model", values="confidence_interval")

        agg_metrics.to_csv(path_or_buf=handler.metrics_path.parent / "dataset_results.csv", index=True, header=True)

        return agg_metrics

    def aggregate_metrics(self) -> None:
        """Alias for all aggregate methods."""
        self.aggregate_test_acc_per_dataset_and_model()
        self.aggregate_test_acc_per_model()
        self.aggregate_per_dataset_with_model_as_cols()

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

        plotting_data = metrics.drop(columns=["test_loss", "test_acc"], axis=1, errors="ignore").copy()
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

        metrics = (
            metrics.drop(["test_acc", "test_loss"], axis=1)
            # Drop if the row is full of NaN values
            .dropna(how="all")
        )

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
