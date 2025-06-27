# -*- coding: utf-8 -*-
"""Metrics Aggregation Script."""

# Standard imports
import json
import pathlib

# Third party imports
import numpy
import pandas

# First party imports
from experiment_framework.runner.metrics_handler import MetricsHandler

if __name__ == "__main__":
    # Get the metrics from the ConvTran experiments and append them to the metrics file
    file_path = (
        pathlib.Path(__file__).parents[2]
        # / "docs/experiment_results/binding_version_1/metrics_binding_version_1.csv"
        # / "docs/experiment_results/binding_N_L_4/metrics_binding_N_l_4.csv"
        # / "docs/experiment_results/pe_version_1/metrics_pe_version_1.csv"
        / "docs/experiment_results/sota_version_1/metrics_sota_version_1.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    handler = MetricsHandler(metrics_path=file_path, metrics_mode="append")
    handler.aggregate_metrics()
    model_metrics = handler.aggregate_test_acc_per_model()

    # Load the ConvTran experiment results
    with open("ConvTran_results.json", "r") as file:
        convtran_experiment_results = json.load(file)

    # Remove unnecessary keys
    convtran_experiment_results.pop("Datasets")
    convtran_experiment_results.pop("Source")

    # Compare with ConvTran experiments
    for model, results in convtran_experiment_results.items():
        # Get the mean and standard deviation
        new_metrics = pandas.DataFrame(
            columns=[
                "model",
                "mean_test_acc",
                "margin_of_error",
                "confidence_interval",
                "num_runs_original",
                "std",
                "num_runs_total",
            ],
            data=[
                [
                    model,
                    round(numpy.mean(results), 4),
                    0,
                    f"{numpy.mean(results):.4f} ± 0",
                    int(len(results)),
                    0,
                    int(len(results)),
                ]
            ],
        ).rename(columns={"num_runs_original": "num_runs"})

        # Append new_metrics to the model_metrics DataFrame
        model_metrics = pandas.concat([model_metrics, new_metrics], axis=0)

    model_metrics.to_csv(path_or_buf=handler.metrics_path.parent / "experiment_results.csv", index=False, header=True)
