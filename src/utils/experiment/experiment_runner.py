# -*- coding: utf-8 -*-
"""This module is responsible for running the experiments."""
# Standard imports
import json
import pathlib
import traceback
from typing import Dict

# Third party imports
import lightning
import pandas
import torch
from torch.utils.data import DataLoader

# First party imports
from experiments.data_factory import DataFactory
from models import ModelFactory
from utils import Config, get_logger, get_train_metrics_and_plot, msg_task
from utils.experiment.dataset_config import DatasetConfig
from utils.experiment.model_config import ModelConfig

# Local imports
from .experiment_config import ExperimentConfig


class ExperimentRunner:
    """This class is responsible for running the experiment."""

    def __init__(self, experiment_cfg: ExperimentConfig, seed: int = 42):
        """Initializes the ExperimentRunner class.

        Args:
            experiment_cfg (ExperimentConfig): The experiment configuration.
            seed (int): Random seed for reproducibility.
        """
        self.experiment_cfg = experiment_cfg
        self.seed = seed

        # Initialize data structures for results and errors
        self.results = pandas.DataFrame()
        self.errors: Dict[str, list] = {}

        # Initialize factories
        self.model_factory = ModelFactory()
        self.data_factory = DataFactory()

        # Cache for dataloaders to reuse across models
        self.dataset_cache: Dict[str, Dict[str, DataLoader]] = {}
        self.dataset_configs: Dict[str, DatasetConfig] = {}

        # Set up paths and logging
        self._setup_paths_and_logging()

        # Set random seed for reproducibility
        self._set_random_seed()

        self.logger.info(f"\n{self.experiment_cfg.pretty_str()}")
        self.logger.info(f"Saving experiment configuration to model/{self._task_exp_path}/experiment_config.json")
        self.experiment_cfg.dump(path=Config.model_dir / self._task_exp_path / "experiment_config.json")
        self.logger.info("Starting experiment ...\n")

    def _setup_paths_and_logging(self) -> None:
        """Set up paths and logging for the experiment."""
        # Format task name for path creation
        self.task_fmt = self.experiment_cfg.task.replace(" ", "_")
        self._task_exp_path = f"{self.task_fmt}/{self.experiment_cfg.run_version}/"

        # Create the directories for the experiment
        self.experiment_logs_path = Config.log_dir / self._task_exp_path
        self.results_path: pathlib.Path = pathlib.Path()
        self.data_dir = Config.data_dir / self.task_fmt

        self.experiment_logs_path.mkdir(parents=True, exist_ok=True)

        # Set up logger
        self.logger = get_logger(
            name="lightning.pytorch.core", log_filename=self.experiment_logs_path / "main.log", propagate=False
        )

    def _set_random_seed(self) -> None:
        """Sets the random seed for reproducibility."""
        lightning.seed_everything(self.seed, workers=True, verbose=False)
        torch.cuda.manual_seed(self.seed)
        self.logger.info(
            f"Seeds from Python's random module, NumPy, PyTorch, and CuDNN set to {self.seed} for reproducibility."
        )

    def run(self):
        """Run the experiment.

        This method runs the experiment for the specified number of runs and datasets.
        """
        for dataset in self.experiment_cfg.dataset_names:
            self.results_path = Config.model_dir / self._task_exp_path / dataset

            msg_task(msg=f" Dataset {dataset} ", logger=self.logger)
            self.logger.info(f"\n\n\t{f'{"=" * 24}{" Loading Data ":^40}{"=" * 24}': ^100}\n")

            # Load dataset and create dataloaders only once per dataset
            self._load_dataset(dataset_name=dataset)

            # Run the experiment for this dataset
            self.single_run(dataset=dataset)

            # Free memory after processing each dataset
            if dataset in self.dataset_cache:
                del self.dataset_cache[dataset]
                torch.cuda.empty_cache()

            self.logger.info(f"\n\nAll models for {dataset} trained successfully!\n{'':_^100}\n\n")

        self.logger.info("Experiment completed!")

    def _load_dataset(self, dataset_name: str) -> None:
        """Load dataset and create dataloaders.

        Args:
            dataset_name (str): The name of the dataset.
        """
        # Skip if already loaded
        if dataset_name in self.dataset_cache:
            self.logger.info(f"Dataset {dataset_name} already loaded, reusing existing dataloaders")
            return

        # Load dataset and create dataloaders
        dataset_cfg, train_dataloader, test_dataloader, val_dataloader = self.data_factory.get_data_loaders_and_config(
            dataset_name=dataset_name,
            batch_size=self.experiment_cfg.batch_size,
            seed=self.seed,
            extract_path=self.data_dir,
            logger=self.logger,
            # If defined, it will plot the first sample of the dataset
            plot_path=None,  # Config.plot_dir / self.task_fmt / f"{dataset_name}_sample.png"
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        # Cache the dataloaders and config for reuse
        self.dataset_cache[dataset_name] = {"train": train_dataloader, "test": test_dataloader, "val": val_dataloader}
        self.dataset_configs[dataset_name] = dataset_cfg

        # Save dataset configuration
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving dataset configuration to {self._task_exp_path}/{dataset_name}/dataset_config.json")
        dataset_cfg.dump(path=self.results_path / "dataset_config.json")

    def single_run(self, dataset: str):
        """Run the experiment for a single dataset.

        Args:
            dataset (str): The name of the dataset.
        """
        # Get cached dataloaders and config
        dataloaders = self.dataset_cache[dataset]
        dataset_cfg = self.dataset_configs[dataset]

        for run in range(1, self.experiment_cfg.runs_per_experiment + 1):
            # Create the results directory if it doesn't exist
            self.results_path.mkdir(parents=True, exist_ok=True)

            # Add run-specific logger
            self.logger.add_component_handler(
                component_name=f"run_{run}",
                log_filename=self.experiment_logs_path / dataset / f"run_{run}.log",
            )
            self.logger.info(f"\n\n{'*' * 40} Run {run} {'*' * 40}\n")

            # Train each model using the same dataloaders
            for model_name, model_cfg in self.experiment_cfg.model_configs.items():
                self.logger.info(f"\n\n\t{f'{"=" * 24}{f" {model_name} (Run {run:<2})":^40}{"=" * 24}': ^100}\n")

                # Add model-specific logger
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
                        train_dataloader=dataloaders["train"],
                        test_dataloader=dataloaders["test"],
                        validation_dataloader=dataloaders["val"],
                    )

                    self.logger.info(f"Model {model_name} for {dataset} trained successfully")

                    # Explicitly free memory after training each model
                    torch.cuda.empty_cache()

                except Exception as e:
                    self._handle_training_error(dataset, model_name, run, e)
                finally:
                    # Remove component handler when done
                    self.logger.remove_component_handler(component_name=component_name)

            # Remove run-specific logger
            self.logger.remove_component_handler(component_name=f"run_{run}")

        if self.errors:
            self.logger.error(f"Errors occurred during the experiment:\n{json.dumps(self.errors, indent=4)}")

    def _handle_training_error(self, dataset: str, model_name: str, run: int, exception: Exception) -> None:
        """Handle errors during model training.

        Args:
            dataset (str): The dataset name.
            model_name (str): The model name.
            run (int): The run number.
            exception (Exception): The exception that occurred.
        """
        err_msg = f"\n\n\t{f'{"x" * 24}'f' {dataset} | {model_name} | Run {run} 'f'{"x" * 24}': ^100}\n\n"
        err_msg += f"Error for {dataset} training {model_name}:\n\n{str(exception)}\n\n"
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
                "error": str(exception),
                "traceback": traceback.format_exc(),
            }
        )
        self.errors[model_name] = model_errors

        # Write to error log file
        with open(self.experiment_logs_path / "errors.log", "a") as f:
            f.write(err_msg)
            f.write(tb_msg)

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
        """Trains a model for a dataset.

        Args:
            task (str): The task name.
            dataset_name (str): The dataset name.
            model_name (str): The model name.
            run_version (str): The run version.
            run (int): The run number.
            model_cfg (ModelConfig): The model configuration.
            dataset_cfg (DatasetConfig): The dataset configuration.
            train_dataloader (DataLoader): The training dataloader.
            test_dataloader (DataLoader): The testing dataloader.
            validation_dataloader (DataLoader): The validation dataloader.
        """
        model_run_path = self.results_path / model_name

        # Create model
        model = self.model_factory.get_model(
            model_config=model_cfg,
            dataset_cfg=dataset_cfg,
            profiler_path=(model_run_path / f"run_{run}").as_posix() if self.experiment_cfg.profiler else "",
        )

        # Create trainer
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
            # Train the model
            trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

        self.logger.info(f"{model_name.title()} training finished!")

        # Test the model
        trainer.test(model, dataloaders=test_dataloader)

        # --- Plot Metrics ---
        if model_cfg.num_epochs > 0:
            metrics = get_train_metrics_and_plot(
                csv_dir=trainer.log_dir,
                experiment=f"{model_name.replace('_', ' ').title()} for {dataset_name.replace('_', ' ').title()} in {run_version}",
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

    def update_global_metrics(
        self, metrics: pandas.DataFrame, run: int, dataset: str, model: str, version: str
    ) -> None:
        """Update the global metrics with the new metrics.

        Args:
            metrics (pandas.DataFrame): The new metrics to update.
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

        # Save updated metrics
        self.results.to_csv(
            path_or_buf=self.results_path.parent / f"global_metrics_{version}.csv", index=False, header=True
        )
