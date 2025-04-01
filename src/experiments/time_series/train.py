# -*- coding: utf-8 -*-
"""Training script for the Transformer model."""
# First party imports
from utils import ExperimentConfigFactory


def run_time_series_experiments() -> None:
    """Run time series experiments for the Transformer model with different positional encodings."""
    # Load the models and experiment configurations
    experiment_configs = ExperimentConfigFactory().create_experiment_config(
        model_cfg_path="models_cfg.json", experiment_cfg_path="experiment_cfg.json"
    )

    # # Define the experiment runner
    # experiment_runner = ExperimentRunner()
    #
    # # Run the experiments
    # experiment_runner.run()
    print(experiment_configs.pretty_str())


if __name__ == "__main__":
    run_time_series_experiments()
