# -*- coding: utf-8 -*-
"""Training script for the Transformer model."""
# Standard imports
import pathlib

# First party imports
from experiment_framework import ExperimentConfigFactory, ExperimentRunner, ModelConfigFactory


def choose_experiment_to_run() -> str:
    """Prompt the user to choose an experiment to run."""
    print("Choose an experiment to run:")
    print("1. TS Classification with different binding methods")
    print("2. TS Classification with different binding methods with N_L=4")

    experiments = {
        1: "binding_methods",
        2: "binding_methods_N_L_4",
    }

    choice_msg = "> Enter the number of the experiment you want to run: "
    choice = int(input(choice_msg))

    while choice not in experiments.keys():
        choice = int(input(f"> Invalid choice. Please enter {list(experiments.keys())}: "))

    return experiments[choice]


def run_time_series_experiments() -> None:
    """Run time series experiments for the Transformer model with different positional encodings."""

    current_dir = pathlib.Path(__file__).resolve().parent / choose_experiment_to_run()

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
