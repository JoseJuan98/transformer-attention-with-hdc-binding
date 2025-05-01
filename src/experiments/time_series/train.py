# -*- coding: utf-8 -*-
"""Training script for the Transformer model."""
# Standard imports
import pathlib

# First party imports
from experiment_framework import ExperimentConfigFactory, ExperimentRunner, ModelConfigFactory


def run_time_series_experiments() -> None:
    """Run time series experiments for the Transformer model with different positional encodings."""

    current_dir = pathlib.Path(__file__).resolve().parent / "binding_methods"

    model_configs = ModelConfigFactory().create_model_configs(model_cfg_path=current_dir / "models_cfg.json")

    # Load the models and experiment configurations
    experiment_config = ExperimentConfigFactory().create_experiment_config(
        experiment_cfg_path=current_dir / "experiment_cfg.json", model_configs=model_configs
    )

    # Define the experiment runner, responsible for running the experiments
    experiment_runner = ExperimentRunner(
        experiment_cfg=experiment_config,
        # Set the random seed for reproducibility
        seed=42,
    )

    # Run the experiments
    experiment_runner.run()


if __name__ == "__main__":
    run_time_series_experiments()
