# -*- coding: utf-8 -*-
"""This module is responsible for running the experiments."""
# Standard imports
import json
import pathlib
import traceback

# Third party imports
import lightning
import pandas
import torch
from torch.utils.data import DataLoader

# First party imports
from experiments.data_factory import DataFactory
from models import (
    ModelFactory,
)
from utils import Config, get_logger, get_train_metrics_and_plot, msg_task
from utils.experiments.dataset_config import DatasetConfig
from utils.experiments.model_config import ModelConfig

# Local imports
from .experiment_config import ExperimentConfig


class ExperimentRunner:
    """This class is responsible for running the experiments."""

    def __init__(self, experiment_cfg: ExperimentConfig):
        """Initializes the ExperimentRunner class.

        Args:
            experiment_cfg (ExperimentConfig): The experiment configuration.
        """
        self.experiment_cfg = experiment_cfg
        self.results = pandas.Series()
        self.errors: dict[str, list] = {}
        self.model_factory = ModelFactory()
        self.data_factory = DataFactory()

        # Binding the logger to the lightning module logger to avoid conflicts
        self.task_fmt = self.experiment_cfg.task.replace(" ", "_")
        self._task_exp_path = f"{self.task_fmt}/{self.experiment_cfg.run_version}/"

        # Create the directories for the experiment
        self.experiment_logs_path = Config.log_dir / self._task_exp_path
        self.results_path: pathlib.Path = pathlib.Path()
        self.data_dir = Config.data_dir / self.task_fmt

        self.experiment_logs_path.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(
            name="lightning.pytorch.core", log_filename=self.experiment_logs_path / "main.log", propagate=False
        )

        self.logger.info(f"\n{self.experiment_cfg.pretty_str()}")

        self.logger.info(
            f"Saving experiment configuration to {Config.model_dir / self._task_exp_path}/experiment_config.json"
        )
        self.experiment_cfg.dump(path=Config.model_dir / self._task_exp_path / "experiment_config.json")

        self.logger.info("Starting experiment ...\n")

    def set_random_seed(self, seed: int = 42) -> None:
        """Sets the random seed for reproducibility."""
        lightning.seed_everything(seed, workers=True, verbose=False)
        torch.cuda.manual_seed(seed)
        self.logger.info(
            f"Seeds from Python's random module, NumPy, PyTorch, and CuDNN set to {seed} for reproducibility."
        )

    def run(self):
        """Run the experiment.

        This method runs the experiment for the specified number of runs and datasets.
        """
        for dataset in self.experiment_cfg.dataset_names:
            # TODO: create dataloaders here
            self.results_path = Config.model_dir / self._task_exp_path / dataset

            msg_task(msg=f" Dataset {dataset} ", logger=self.logger)

            self.logger.info(f"\n\n\t{f'{"=" * 24}{" Loading Data ":^40}{"=" * 24}': ^100}\n")

            # --- Data Loaders and Dataset Configuration ---
            dataset_cfg, train_dataloader, test_dataloader, val_dataloader = (
                self.data_factory.get_data_loaders_and_config(
                    dataset_name=dataset,
                    # TODO: redefine how the batch size is defined based in experiment config or a relative % of memory
                    # with mini-batching gradient accumulation
                    batch_size=64,
                    extract_path=self.data_dir,
                    logger=self.logger,
                    # If defined, it will plot the first sample of the dataset
                    plot_path=None,  # Config.plot_dir / self.task_fmt / f"{dataset}_sample.png"
                )
            )

            self.logger.info(f"Saving dataset configuration to {self.results_path}/dataset_config.json")
            dataset_cfg.dump(path=self.results_path / "dataset_config.json")

            self.single_run(
                dataset=dataset,
                dataset_cfg=dataset_cfg,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                validation_dataloader=val_dataloader,
            )

            self.logger.info(f"\n\nAll models for {dataset} trained successfully!\n{'':_^100}\n\n")

        self.logger.info("Experiment completed!")

    def single_run(
        self,
        dataset: str,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        dataset_cfg: DatasetConfig,
    ):
        """Run the experiment.

        Args:
            dataset (str): The name of the dataset.
            train_dataloader (DataLoader): The training dataloader.
            test_dataloader (DataLoader): The testing dataloader.
            validation_dataloader (DataLoader): The validation dataloader.
            dataset_cfg (DatasetConfig): The dataset configuration.
        """

        for run in range(1, self.experiment_cfg.runs_per_experiment + 1):

            # Create the results directory if it doesn't exist
            self.results_path.mkdir(parents=True, exist_ok=True)

            self.logger.add_component_handler(
                component_name=f"run_{run}",
                log_filename=self.experiment_logs_path / dataset / f"run_{run}.log",
            )
            self.logger.info(f"\n\n{'*'*40} Run {run} {'*'*40}\n")

            for model_name, model_cfg in self.experiment_cfg.model_configs.items():

                self.logger.info(f"\n\n\t{f'{"="*24}{f" {model_name} (Run {run:<2})":^40}{"="*24}': ^100}\n")
                # Add component-specific handler
                component_name = f"{model_name}_{dataset}"
                self.logger.add_component_handler(
                    component_name=component_name,
                    log_filename=self.experiment_logs_path / dataset / f"{model_name}.log",
                    log_file_mode="a",
                )

                try:
                    # Train the model for the dataset
                    self.logger.info(
                        f"Training {model_name} for {dataset} for {self.experiment_cfg.run_version} in run {run}"
                    )
                    self._train_model_for_dataset(
                        task=self.experiment_cfg.task,
                        dataset_name=dataset,
                        model_name=model_cfg.model_name,
                        run=run,
                        run_version=self.experiment_cfg.run_version,
                        model_cfg=model_cfg,
                        dataset_cfg=dataset_cfg,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        validation_dataloader=validation_dataloader,
                    )

                    self.logger.info(f"Model {model_name} for {dataset} trained successfully")

                except Exception as e:
                    err_msg = f"\n\n\t{f'{"x" * 24}'f' {dataset} | {model_name} | Run {run} 'f'{"x" * 24}': ^100}\n\n"
                    err_msg += f"Error for {dataset} training {model_name}:\n\n{str(e)}\n\n"
                    tb_msg = f"Traceback:\n{traceback.format_exc()}"
                    self.logger.error(err_msg)
                    self.logger.error(tb_msg)
                    # Update errors for model
                    model_errors = self.errors.get(model_name, [])
                    model_errors.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "run": run,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    self.errors[model_name] = model_errors
                    with open(self.experiment_logs_path / "errors.log", "a") as f:
                        f.write(err_msg)
                        f.write(tb_msg)
                finally:
                    # Remove component handler when done
                    self.logger.remove_component_handler(component_name=component_name)

        if self.errors:
            self.logger.error(f"Errors occurred during the experiment:\n{json.dumps(self.errors, indent=4)}")

    def _train_model_for_dataset(
        self,
        task: str,
        dataset_name: str,
        model_name: str,
        run_version: str,
        run: int,
        model_cfg: ModelConfig,
        dataset_cfg: DatasetConfig,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """Trains a model for a dataset."""
        model_run_path = self.results_path / model_name

        model = self.model_factory.get_model(
            model_config=model_cfg,
            dataset_cfg=dataset_cfg,
            profiler_path=(model_run_path / f"run_{run}").as_posix() if self.experiment_cfg.profiler else "",
        )

        trainer = self.model_factory.get_trainer(
            default_root_dir=Config.root_dir,
            experiment_cfg=self.experiment_cfg,
            num_epochs=model_cfg.num_epochs,
            model_relative_path=(model_run_path / f"run_{run}" / "model.pth").as_posix(),
            save_dir=model_run_path.parent,
            save_dir_name=model_name,
            save_version=f"run_{run}",
        )

        # --- Train and Test ---
        msg_task(msg="Train and Test", logger=self.logger)
        self.logger.info(f"Training {model_name} for {model_cfg.num_epochs} epochs...")
        if model_cfg.num_epochs > 0:
            # TODO: torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: AttributeError: 'float' object has no attribute 'meta'
            # model = torch.compile(model)
            trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

        self.logger.info(f"{model_name.title()} training finished!")
        trainer.test(model, dataloaders=test_dataloader)

        # --- Plot Metrics ---
        if model_cfg.num_epochs > 0:
            metrics = get_train_metrics_and_plot(
                csv_dir=trainer.log_dir,
                experiment=f"{model_name.replace("_", " ").title()} for {dataset_name.replace('_', ' ').title()} in {run_version}",
                logger=self.logger,
                plots_path=Config.plot_dir / task / dataset_name / f"epoch_metrics_{model_name}_{run_version}.png",
            )
            self.update_global_metrics(
                metrics=metrics,
                run=run,
                dataset=dataset_name,
                model=model_name,
                version=run_version,
            )

    def update_global_metrics(self, metrics: pandas.Series, run: int, dataset: str, model: str, version: str) -> None:
        """Update the global metrics with the new metrics.

        This method updates the global metrics with the new metrics and saves them to a CSV file.

        Args:
            metrics (pandas.Series): The new metrics to update.
            run (int): The run number.
            dataset (str): The name of the dataset.
            model (str): The name of the model.
            version (str): The version of the experiment.
        """
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

        self.results.to_csv(
            path_or_buf=self.results_path.parent / f"global_metrics_{version}.csv", index=False, header=True
        )
