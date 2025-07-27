# -*- coding: utf-8 -*-
"""Training script for the Transformer model."""
# Standard imports
import argparse
import pathlib

# First party imports
from experiment_framework import ExperimentConfigFactory, ExperimentRunner, ModelConfigFactory

experiments = {
    1: "1_binding_methods",
    2: "2_N_L",
    3: "3_positional_encodings",
    4: "4_sota",
    5: "5_d_model",
}


def parse_experiment_choice() -> int:
    """Parse command line arguments to select the experiment to run."""

    args = argparse.ArgumentParser("Experiment Selection")
    args.add_argument(
        "--experiment",
        "-exp",
        dest="exp",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=0,
        help="Choose the experiment to run (1-5).",
    )
    return args.parse_args().exp


def ask_user_for_experiment_choice() -> int:
    """Prompt the user to choose an experiment to run."""
    print("Choose an experiment to run:")
    print("1. TS Classification with different binding methods")
    print("2. TS Classification with different binding methods with different N_L values")
    print("3. TS Classification with different positional encodings")
    print("4. TS Classification with State-of-the-Art (SOTA) models")
    print("5. TS Classification with different d_model values")

    choice_msg = "> Enter the number of the experiment you want to run: "
    choice = int(input(choice_msg))

    while choice not in experiments.keys():
        choice = int(input(f"> Invalid choice. Please enter {list(experiments.keys())}: "))

    return choice


def choose_experiment_to_run() -> str:
    """Choose the experiment to run based on user input or command line argument."""

    # Check if a command line argument is provided
    exp_choice = parse_experiment_choice()

    if exp_choice == 0:
        # If no command line argument, ask the user for their choice
        exp_choice = ask_user_for_experiment_choice()

    return experiments[exp_choice]


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
