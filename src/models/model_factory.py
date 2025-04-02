# -*- coding: utf-8 -*-
"""Model factory for creating models based on configuration."""
# Standard imports
import pathlib

# Third party imports
import lightning
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

# First party imports
from models import EncoderOnlyTransformerTSClassifier
from models.pocket_algorithm import PocketAlgorithm
from models.positional_encoding.ts_sinusoidal_positional_embedding import TimeSeriesSinusoidalPositionalEmbedding
from utils import Config
from utils.experiments.dataset_config import DatasetConfig
from utils.experiments.experiment_config import ExperimentConfig
from utils.experiments.model_config import ModelConfig


class ModelFactory:
    """Factory class for creating models based on configuration."""

    @staticmethod
    def get_model(
        model_config: ModelConfig, dataset_cfg: DatasetConfig, profiler_path: str = ""
    ) -> EncoderOnlyTransformerTSClassifier:
        """Get the model based on the configuration.

        Args:
            model_config (ModelConfig): The model configuration.
            dataset_cfg (DatasetConfig): The dataset configuration.
            profiler_path (str): [Optional] Path to save the profiler logs.

        Returns:
            EncoderOnlyTransformerTSClassifier: The model instance.
        """
        positional_encoding_catalog = {
            "transformer_absolute_sinusoidal_pe": TimeSeriesSinusoidalPositionalEmbedding,
            # "transformer_elementwise_pe": NotImplementedError,
            # "transformer_ciruclarconvolution_pe": NotImplementedError,
        }

        # Get the positional encoding class based on the configuration
        positional_encoding = positional_encoding_catalog[model_config.model_name]

        return EncoderOnlyTransformerTSClassifier(
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            input_size=dataset_cfg.input_size,
            context_length=dataset_cfg.context_length,
            positional_encoding=positional_encoding(
                embedding_dim=model_config.d_model, num_positions=dataset_cfg.context_length
            ),
            num_classes=dataset_cfg.num_classes,
            dropout=model_config.dropout,
            learning_rate=model_config.learning_rate,
            scaling="mean",
            mask_input=True,
            loss_fn=torch.nn.CrossEntropyLoss() if dataset_cfg.num_classes > 2 else torch.nn.BCELoss(),
            # Torch Profiler: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
            torch_profiling=(
                torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=profiler_path),
                    record_shapes=True,
                    with_stack=True,
                )
                if profiler_path
                else None
            ),
        )

    @staticmethod
    def get_trainer(
        default_root_dir: pathlib.Path,
        experiment_cfg: ExperimentConfig,
        num_epochs: int,
        model_relative_path: str,
        csv_logger_args: dict,
        tensorboard_args: dict,
        profiler: bool = False,
    ) -> lightning.Trainer:
        """Get the trainer based on the configuration.

        Args:
            default_root_dir (pathlib.Path): The default root directory for the trainer.
            experiment_cfg (ExperimentConfig): The experiment configuration.
            num_epochs (int): The number of epochs to train.
            model_relative_path (str): The relative path to save the model.
            csv_logger_args (dict): Arguments for the CSV logger.
            tensorboard_args (dict): Arguments for the TensorBoard logger.
            profiler (bool): Whether to enable the profiler.

        Returns:
            lightning.Trainer: The trainer instance.
        """
        # --- Callbacks ---
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        pocket_algorithm = PocketAlgorithm(
            monitor="val_acc",
            mode="max",
            ckpt_filepath=Config.model_dir / pathlib.Path(model_relative_path).with_suffix(".ckpt"),
            model_file_path=Config.model_dir / model_relative_path,
        )

        return lightning.Trainer(
            default_root_dir=default_root_dir,
            max_epochs=num_epochs,
            accelerator="auto",
            devices="auto",
            precision=experiment_cfg.precision,
            logger=[
                CSVLogger(**csv_logger_args),
                TensorBoardLogger(**tensorboard_args),
            ],
            log_every_n_steps=1,
            callbacks=[ModelSummary(max_depth=-1), early_stopping, pocket_algorithm],
            # measures all the key methods across Callbacks, DataModules and the LightningModule in the training loop.
            profiler=SimpleProfiler(filename="simple_profiler") if profiler else None,
            # If True, runs 1 batch of train, test and val to find any bugs. Also, it can be specified the number of
            # batches to run as an integer
            fast_dev_run=False if num_epochs > 0 else True,
        )
