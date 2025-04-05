# -*- coding: utf-8 -*-
"""Training script for the Transformer model."""
# Standard imports
import pathlib

# First party imports
from utils.experiments import ExperimentConfigFactory, ExperimentRunner


def run_time_series_experiments() -> None:
    """Run time series experiments for the Transformer model with different positional encodings."""

    current_dir = pathlib.Path(__file__).resolve().parent

    # Load the models and experiment configurations
    experiment_config = ExperimentConfigFactory().create_experiment_config(
        model_cfg_path=current_dir / "models_cfg.json", experiment_cfg_path=current_dir / "experiment_cfg.json"
    )

    # FIXME: when implemented use all models, for now only the base one
    experiment_config.model_configs.pop("Transformer Ciruclar Convolution PE")
    experiment_config.model_configs.pop("Transformer Element-Wise PE")

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
