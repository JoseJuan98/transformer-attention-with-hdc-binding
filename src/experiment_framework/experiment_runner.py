# -*- coding: utf-8 -*-
"""This module is responsible for running the experiments."""
# Standard imports
import json
import os
import pathlib
import time
from typing import Dict

# Third party imports
import lightning
import pandas
import torch
from lightning.pytorch.tuner import Tuner

# First party imports
from experiment_framework.data_factory import DataFactory
from experiment_framework.dataset_config import DatasetConfig
from experiment_framework.error_handler import ErrorHandler
from experiment_framework.experiment_config import ExperimentConfig
from experiment_framework.metrics_handler import MetricsHandler
from experiment_framework.model_config import ModelConfig
from models.model_factory import ModelFactory
from utils import Config, get_logger, get_train_metrics_and_plot, msg_task


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

        # Initialize data structures for results
        self.results = pandas.DataFrame()

        # Initialize factories
        self.model_factory = ModelFactory()
        self.data_factory = DataFactory()

        # Cache for dataloaders to reuse across models
        self.dataset_cache: Dict[str, lightning.LightningDataModule] = {}
        self.dataset_configs: Dict[str, DatasetConfig] = {}

        # Set up paths and logging: `self.logger` and `self.experiment_logs_path`
        self.exp_name_title = self.experiment_cfg.experiment_name.replace("_", " ").title()
        self._setup_paths_and_logging()

        # Set random seed for reproducibility
        self._set_random_seed()

        # Initialize MetricsHandler
        self.metrics_handler = MetricsHandler(
            metrics_path=self.metrics_path,
            aggregated_metrics_path=self.metrics_path.parent / f"aggregated_{self.metrics_path.name}",
        )

        # Initialize ErrorHandler (AFTER logger and paths are set)
        self.error_handler = ErrorHandler(
            logger=self.logger,
            error_log_path=self.experiment_logs_path / "errors.log",
            development_mode=self.experiment_cfg.development,
        )

        self.logger.info(f"\n{self.experiment_cfg.pretty_str()}")
        self.logger.info(f"Saving experiment configuration to model/{self._task_exp_path}experiment_config.json")
        self.experiment_cfg.dump(path=Config.model_dir / self._task_exp_path / "experiment_config.json")
        self.exp_start_time = time.perf_counter()

    def _setup_paths_and_logging(self) -> None:
        """Set up paths and logging for the experiment."""
        # Format task name for path creation
        self.task_fmt = self.experiment_cfg.task.replace(" ", "_")

        # Development mode: reduce the number of epochs for development purposes
        if self.experiment_cfg.development:
            self.experiment_cfg.run_version = "dev"

        self._task_exp_path = f"{self.task_fmt}/{self.experiment_cfg.run_version}/"

        # Create the directories for the experiment
        self.experiment_logs_path = Config.log_dir / self._task_exp_path
        self.results_path: pathlib.Path = pathlib.Path()  # Will be set per dataset
        self.data_dir = Config.data_dir / self.task_fmt
        self.metrics_path = Config.model_dir / self._task_exp_path / f"metrics_{self.experiment_cfg.run_version}.csv"

        self.experiment_logs_path.mkdir(parents=True, exist_ok=True)

        # Set up logger
        self.logger = get_logger(
            name="lightning.pytorch.core", log_filename=self.experiment_logs_path / "main.log", propagate=False
        )

        self.logger.info(f"Starting experiment {self.exp_name_title} ...\n")

        # Development mode: reduce the number of epochs for development purposes
        if self.experiment_cfg.development:
            self.logger.info("\t => Development mode is enabled.")
            self.experiment_cfg.runs_per_experiment = 2
            for model_config in self.experiment_cfg.model_configs.values():
                model_config.num_epochs = 2

    def _set_random_seed(self) -> None:
        """Sets the random seed for reproducibility."""
        lightning.seed_everything(self.seed, workers=True, verbose=False)

        # CUDA and ROCM: ROCM shows up as CUDA in PyTorch
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed(self.seed)

        self.logger.info(
            f"Seeds from Python's random module, NumPy, PyTorch, and any backend in used set to {self.seed} for "
            "reproducibility."
        )

    def run(self):
        """Run the experiment.

        This method runs the experiment for the specified number of runs and datasets.
        """
        for dataset in self.experiment_cfg.dataset_names:
            self.results_path = Config.model_dir / self._task_exp_path / dataset

            msg_task(msg=f" Dataset {dataset} ", logger=self.logger)
            self.logger.info(f"\n\n\t{f'{"=" * 24}{" Loading Data ":^40}{"=" * 24}': ^100}\n")

            try:
                # Load dataset and create dataloaders only once per dataset
                self._load_dataset(dataset_name=dataset)
                # Run the experiment for this dataset
                self.single_run(dataset=dataset)

            except Exception as e:
                self.error_handler.handle_error(dataset=dataset, exception=e)

            finally:
                # Free memory after processing each dataset
                if dataset in self.dataset_cache:
                    del self.dataset_cache[dataset]
                self._clean_backend_cache()

            self.logger.info(f"\n\nAll models for {dataset} trained successfully!\n{'':_^100}\n\n")

        total_time = time.perf_counter() - self.exp_start_time

        # Calculate aggregated metrics
        self.metrics_handler.aggregate_test_acc_per_dataset_and_model()
        self.metrics_handler.aggregate_test_acc_per_model()

        # Log final errors if any occurred
        if self.error_handler.errors:
            self.logger.error(
                f"Summary of errors occurred during the experiment:\n{json.dumps(self.error_handler.errors, indent=4)}"
            )

        self.logger.info(
            f"Experiment {self.exp_name_title} completed in {total_time // 60:.2f} minutes {total_time % 60:.2f} "
            "seconds!"
        )

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
        dataset_cfg, data_module = self.data_factory.get_data_loaders_and_config(
            dataset_name=dataset_name,
            batch_size=self.experiment_cfg.default_batch_size,
            seed=self.seed,
            extract_path=self.data_dir,
            logger=self.logger,
            # If defined, it will plot the first sample of the dataset
            plot_path=(
                Config.plot_dir / self.task_fmt / dataset_name / "sample.png" if self.experiment_cfg.plots else None
            ),
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            val_split=self.experiment_cfg.validation_split,
        )

        # Cache the dataloaders and config for reuse
        self.dataset_cache[dataset_name] = data_module
        self.dataset_configs[dataset_name] = dataset_cfg

        # Save dataset configuration
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving dataset configuration to {self._task_exp_path}/{dataset_name}/dataset_config.json")
        self.dataset_configs[dataset_name].dump(path=self.results_path / "dataset_config.json")

    def single_run(self, dataset: str):
        """Run the experiment for a single dataset.

        Args:
            dataset (str): The name of the dataset.
        """
        # Get cached dataloaders and config
        data_module = self.dataset_cache[dataset]
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
            for cfg_name, model_cfg in self.experiment_cfg.model_configs.items():

                banner = f"{'=' * 24}{f' {cfg_name} (Run {run:<2})':^40}{'=' * 24}"
                self.logger.info(f"\n\n\t{banner:^100}\n")

                # Add model-specific logger
                component_name = f"{model_cfg.model_name}_{dataset}"
                self.logger.add_component_handler(
                    component_name=component_name,
                    log_filename=self.experiment_logs_path / dataset / f"{model_cfg.model_name}.log",
                    log_file_mode="a",
                )

                try:
                    # Train the model for the dataset
                    self.logger.info(
                        f"Training {cfg_name} for {dataset} for {self.experiment_cfg.run_version} in run {run}"
                    )
                    self._train_model_for_dataset(
                        dataset_name=dataset,
                        model_name=model_cfg.model_name,
                        run=run,
                        run_version=self.experiment_cfg.run_version,
                        model_cfg=model_cfg,
                        dataset_cfg=dataset_cfg,
                        data_module=data_module,
                    )

                    self.logger.info(f"Model {cfg_name} for {dataset} trained successfully")

                except Exception as e:
                    self.error_handler.handle_error(
                        dataset=dataset, model_name=model_cfg.model_name, run=run, exception=e
                    )

                finally:
                    # Explicitly free memory after training each model
                    self._clean_backend_cache()

                    # Remove component handler when done
                    self.logger.remove_component_handler(component_name=component_name)

            # Remove run-specific logger
            self.logger.remove_component_handler(component_name=f"run_{run}")

    def _train_model_for_dataset(
        self,
        dataset_name: str,
        model_name: str,
        run_version: str,
        run: int,
        model_cfg: ModelConfig,
        dataset_cfg: DatasetConfig,
        data_module: lightning.LightningDataModule,
    ) -> None:
        """Trains a model for a dataset.

        Args:
            dataset_name (str): The dataset name.
            model_name (str): The model name.
            run_version (str): The run version.
            run (int): The run number.
            model_cfg (ModelConfig): The model configuration.
            dataset_cfg (DatasetConfig): The dataset configuration.
            data_module (lightning.LightningDataModule): The LightningDataModule.
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
            default_root_dir=model_run_path,
            experiment_cfg=self.experiment_cfg,
            num_epochs=model_cfg.num_epochs,
            model_relative_path=(model_run_path / f"run_{run}" / "model.pth").as_posix(),
            save_dir=self.results_path,
            save_dir_name=model_name,
            save_version=f"run_{run}",
            logger=self.logger,
            gradient_clip_val=(
                self.experiment_cfg.gradient_clip_val if hasattr(self.experiment_cfg, "gradient_clip_val") else 0
            ),
            gradient_clip_algorithm=(
                self.experiment_cfg.gradient_clip_algorithm
                if hasattr(self.experiment_cfg, "gradient_clip_algorithm")
                else "norm"
            ),
            lr_iterations=model_cfg.num_epochs,
        )

        # --- Tune Batch Size ---
        if self.experiment_cfg.auto_scale_batch_size:

            self.logger.info("\t=> Tuning batch size")
            tuner = Tuner(trainer=trainer)

            # Turning off the memory growth for the GPU for the batch size tuning
            # Note: This env var might not be the most reliable way across systems/versions
            original_alloc_conf = None
            if torch.cuda.is_available():
                original_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                expandable_segments = "expandable_segments:False"
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                    expandable_segments
                    if original_alloc_conf is None
                    else original_alloc_conf + "," + expandable_segments
                )

            try:
                # if default batchsize is bigger than the dataset size, set it to the dataset size for faster tuning
                n_train_samples = len(data_module.train_dataset)
                new_batch_size = min(self.experiment_cfg.default_batch_size, n_train_samples // 2)
                # it will use the batch size in the datamodule as initial value
                data_module.batch_size = new_batch_size

                if self.experiment_cfg.default_batch_size > n_train_samples:
                    self.logger.warning(
                        f"Default batch size {self.experiment_cfg.default_batch_size} is larger than the dataset "
                        f"size {n_train_samples // 2}, setting it to the dataset size."
                    )

                # Auto-scale batch size by growing it exponentially (default)
                # You can change the mode to "binsearch" for a binary search approach
                new_batch_size = tuner.scale_batch_size(model, datamodule=data_module, mode="power", max_trials=25)

                # Big number batch sizes gives memory problems, so better to reduce it 10%
                if new_batch_size > 1024:
                    self.logger.warning(f"Tuned batch size {new_batch_size} > 1024, capping at 1024.")
                    new_batch_size = 1024
                    # new_batch_size = int(new_batch_size * 0.9) # Alternative reduction

                # |Bug fix| for small datasets when the batch size is too small or bigger than the train sampoles
                # `binsearch` mode doesn't finish and power mode finds one too big
                if new_batch_size > n_train_samples:
                    # Find the closest exponent of 2 to the number of train samples if the batch size is bigger than the
                    #   dataset
                    exponent = 0
                    while 2**exponent < n_train_samples:
                        exponent += 1
                    new_batch_size = 2 ** (exponent - 1)

                self.logger.info(f"Optimal batch size found: {new_batch_size}")

                # Update the datamodule with the new batch size
                data_module.batch_size = new_batch_size

            finally:
                # Restore original alloc conf and turning on the memory growth for the GPU for the training
                expandable_segments = "expandable_segments:True"
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                    original_alloc_conf + "," + expandable_segments if original_alloc_conf else expandable_segments
                )

        # --- Train and Test ---
        self.logger.info(f"=> Train and Test (Run {run})")
        self.logger.info(f"Training {model_name} for {model_cfg.num_epochs} epochs...")

        training_time = 0.0
        if model_cfg.num_epochs > 0:
            # Train the model
            start_time = time.perf_counter()
            trainer.fit(model, datamodule=data_module)

            training_time = time.perf_counter() - start_time

            self.logger.info(f"{model_name.title()} training finished in {training_time:.2f} seconds!")
        else:
            self.logger.info("Skipping training as num_epochs is 0.")

        # Test the model (always test, even if not trained in this run, e.g., loading checkpoint)
        self.logger.info("Testing model...")
        trainer.test(model=model, datamodule=data_module)

        # --- Plot Metrics ---
        if model_cfg.num_epochs > 0:
            metrics = get_train_metrics_and_plot(
                csv_dir=trainer.log_dir,
                experiment=f"{model_name.replace('_', ' ').title()} for "
                + f"{dataset_name.replace('_', ' ').title()} in {run_version}",
                logger=self.logger,
                plots_path=(
                    Config.plot_dir
                    / self.task_fmt
                    / dataset_name
                    / f"epoch_metrics_{model_name}_run_{run}_{run_version}.png"
                    if self.experiment_cfg.plots
                    else None
                ),
                show_plot=False,
            )

            self.metrics_handler.update_metrics(
                metrics=metrics,
                run=run,
                dataset=dataset_name,
                model=model_name,
                num_dimensions=dataset_cfg.input_size,
                num_classes=dataset_cfg.num_classes,
                sequence_length=dataset_cfg.context_length,
                train_samples=len(data_module.train_dataset),  # Use DataModule
                test_samples=len(data_module.test_dataset),  # Use DataModule
                validation_samples=len(data_module.val_dataset),  # Use DataModule
                training_time=training_time,
                n_train_epochs=trainer.current_epoch,
            )

        # --- Teardown ---
        self.teardown(trainer_default_dir=trainer.default_root_dir)

    def teardown(self, trainer_default_dir: str):
        """Teardown steps after a model run, like cleaning temporary files.

        Removes tuner temporary files if they exist.

        Args:
            trainer_default_dir (str): The default directory used by the trainer for this run.
        """
        try:
            base_dir = pathlib.Path(trainer_default_dir)

            # Check for files starting with .lr_find and ending with .ckpt
            for file in base_dir.glob(".lr_find*.ckpt"):
                if file.is_file():
                    file.unlink()  # Remove the file

        except Exception as e:
            # Log error if cleanup fails but don't stop the experiment
            self.logger.warning(f"[Warning] Failed to clean up tuner files in {trainer_default_dir}: {e}")

    @staticmethod
    def _clean_backend_cache() -> None:
        """Clean up the cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
