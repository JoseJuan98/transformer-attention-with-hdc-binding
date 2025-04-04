# -*- coding: utf-8 -*-
"""This module is responsible for running the experiments."""
# Standard imports
import json
import multiprocessing
import traceback

# Third party imports
import lightning
import torch
from torch.utils.data import DataLoader

# First party imports
from experiments.time_series.dataset import get_ucr_datasets
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
        self.results: dict[str, dict] = {}
        self.errors: dict[str, list] = {}

        # Binding the logger to the lightning module logger to avoid conflicts
        _task_exp_path = f"{self.experiment_cfg.task.replace(' ', '_')}/{self.experiment_cfg.run_version}/"

        # Create the directories for the experiment
        self.experiment_logs_path = Config.log_dir / _task_exp_path
        self.results_path = Config.model_dir / _task_exp_path

        # Create the results directory if it doesn't exist
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.experiment_logs_path.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(
            name="lightning.pytorch.core", log_filename=self.experiment_logs_path / "main.log", propagate=False
        )

        self.logger.info(f"\n{self.experiment_cfg.pretty_str()}")
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
        for run in range(1, self.experiment_cfg.runs_per_experiment):
            self.logger.info(f"\n\n{'*'*40} Run {run} {'*'*40}\n")
            self.single_run(run=run)

    def single_run(self, run: int):
        """Run the experiment.

        Args:
            run (int): The current run number index. In total there are `ExperimentConfig.runs_per_experiment` runs.
        """

        for dataset in self.experiment_cfg.dataset_names:
            msg_task(msg=f" Dataset {dataset}", logger=self.logger)

            # TODO: create dataloaders here

            for model_name, model_cfg in self.experiment_cfg.model_configs.items():

                self.logger.info(f"\n\n\t{f'{"="*24}'f'{model_name:^40}'f'{"="*24}': ^100}\n")
                # Add component-specific handler
                component_name = f"{model_name}_{dataset}"
                self.logger.add_component_handler(
                    component_name=component_name,
                    log_filename=self.experiment_logs_path / model_name / f"{dataset}.log",
                )

                try:
                    raise Exception(f"Experiment {dataset} failed.")

                    # Train the model for the dataset
                    self._train_model_for_dataset(
                        task=self.experiment_cfg.task,
                        dataset_name=dataset,
                        model_name=model_cfg.model_name,
                        run_version=self.experiment_cfg.run_version,
                        model_cfg=model_cfg,
                        profiler=False,
                        plot_first_sample=False,
                    )

                    self.logger.info(f"Model {model_name} trained successfully")
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

        self.logger.info("Experiment completed.")

        if self.errors:
            self.logger.error(f"Errors occurred during the experiment:\n{json.dumps(self.errors, indent=4)}")
        # TODO:
        # with open(results_path / "results.csv", "w") as f:

    def _train_model_for_dataset(
        self,
        task: str,
        dataset_name: str,
        model_name: str,
        run_version: str,
        model_cfg: ModelConfig,
        profiler: bool = False,
        plot_first_sample: bool = False,
    ) -> None:
        """Trains a model for a dataset."""
        run_path = Config.model_dir / "runs" / task / model_name
        test_only = False  # Set to True to only test the model

        if not torch.cuda.is_available():
            raise Exception("CUDA not available. No GPU found.")

        # TODO: move it to method above for reusing it for several models
        msg_task(msg="Loading data", logger=self.logger)
        train_dataset, test_dataset, max_len, num_classes, num_channels = get_ucr_datasets(
            dsid=dataset_name,
            extract_path=Config.data_dir / task,
            logger=self.logger,
            plot_first_sample=plot_first_sample,
            plot_path=Config.plot_dir / task / f"{dataset_name}_sample.png",
        )

        # --- Dataset Configuration ---
        dataset_cfg = DatasetConfig(
            dataset_name=dataset_name, num_classes=num_classes, input_size=num_channels, context_length=max_len
        )

        self.logger.info(f"Saving experiment configuration to {run_path}/{run_version}")
        self.experiment_cfg.dump(path=run_path / dataset_name / run_version / "experiment_config.json")

        # --- Data Loaders ---
        cpu_count = multiprocessing.cpu_count()
        num_workers = cpu_count - 2 if cpu_count > 4 else 1

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=model_cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=model_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model = ModelFactory.get_model(
            model_config=model_cfg,
            dataset_cfg=dataset_cfg,
            profiler_path=(run_path / run_version).as_posix() if profiler else "",
        )

        csv_logger_args = {"save_dir": Config.log_dir / task / model_name, "name": dataset_name, "version": run_version}
        tensorboard_args = {"save_dir": run_path, "name": dataset_name, "version": run_version}
        trainer = ModelFactory().get_trainer(
            default_root_dir=Config.root_dir,
            experiment_cfg=self.experiment_cfg,
            num_epochs=model_cfg.num_epochs,
            model_relative_path=f"runs/{task}/{model_name.lower()}/{dataset_name}/{run_version}/model.pth",
            csv_logger_args=csv_logger_args,
            tensorboard_args=tensorboard_args,
            profiler=profiler,
        )

        # --- Train and Test ---
        msg_task(msg="Train and Test", logger=self.logger)
        self.logger.info(
            f"Training {model_name} for {dataset_name} in {run_version} for {model_cfg.num_epochs} epochs..."
        )
        if model_cfg.num_epochs > 0 and not test_only:
            # TODO: torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: AttributeError: 'float' object has no attribute 'meta'
            # model = torch.compile(model)
            trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
        self.logger.info(f"{model_name.title()} training finished!")
        trainer.test(model, dataloaders=test_dataloader)

        # --- Plot Metrics ---
        if model_cfg.num_epochs > 0:
            print(f"Log dir: {trainer.log_dir}")
            get_train_metrics_and_plot(
                csv_dir=trainer.log_dir,
                experiment=f"{model_name.replace("_", " ").title()} for {dataset_name.replace('_', ' ').title()} in {run_version}",
                logger=self.logger,
                plots_path=Config.plot_dir / task / model_name / f"epoch_metrics_{dataset_name}_{run_version}.png",
            )
