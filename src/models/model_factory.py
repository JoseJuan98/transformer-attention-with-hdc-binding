# -*- coding: utf-8 -*-
"""Model factory for creating models based on configuration."""
# Standard imports
import logging
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
from models.positional_encoding import (
    TimeSeriesCircularConvolutionPositionalEncoding,
    TimeSeriesComponentwiseMultiplicationPositionalEncoding,
    TimeSeriesSinusoidalPositionalEncoding,
    TSPositionalEncodingType,
)
from utils import Config
from utils.experiment.dataset_config import DatasetConfig
from utils.experiment.experiment_config import ExperimentConfig
from utils.experiment.model_config import ModelConfig


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
        positional_encoding_catalog: dict[str, type[TSPositionalEncodingType]] = {
            "transformer_absolute_sinusoidal_pe": TimeSeriesSinusoidalPositionalEncoding,
            "transformer_component_wise_pe": TimeSeriesComponentwiseMultiplicationPositionalEncoding,
            "transformer_ciruclarconvolution_pe": TimeSeriesCircularConvolutionPositionalEncoding,
        }

        # Get the positional encoding class based on the configuration and instantiate it
        positional_encoding = positional_encoding_catalog[model_config.model_name](
            embedding_dim=model_config.d_model, num_positions=dataset_cfg.context_length
        )

        return EncoderOnlyTransformerTSClassifier(
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            input_size=dataset_cfg.input_size,
            context_length=dataset_cfg.context_length,
            positional_encoding=positional_encoding,
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
            embedding_binding=model_config.embedding_binding,
        )

    @staticmethod
    def get_trainer(
        default_root_dir: pathlib.Path,
        experiment_cfg: ExperimentConfig,
        num_epochs: int,
        model_relative_path: str,
        save_dir: str | pathlib.Path,
        save_dir_name: str,
        save_version: str,
        logger: logging.Logger | None = None,
    ) -> lightning.Trainer:
        """Get the trainer based on the configuration.

        Args:
            default_root_dir (pathlib.Path): The default root directory for the trainer.
            experiment_cfg (ExperimentConfig): The experiment configuration.
            num_epochs (int): The number of epochs to train.
            model_relative_path (str): The relative path to save the model.
            save_dir (str | pathlib.Path): The directory to save the model.
            save_dir_name (str): The name of the directory to save the model.
            save_version (str): The version of the model to save.
            logger (logging.Logger | None, optional): Logger object. Defaults to None.

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
            logger=logger,
        )

        callbacks = [early_stopping, pocket_algorithm]
        if experiment_cfg.summary:
            callbacks.append(ModelSummary(max_depth=-1))

        return lightning.Trainer(
            default_root_dir=default_root_dir,
            max_epochs=num_epochs,
            accelerator=experiment_cfg.accelerator,
            devices="auto",
            precision=experiment_cfg.precision,
            logger=[
                CSVLogger(save_dir=save_dir, name=save_dir_name, version=save_version),
                TensorBoardLogger(save_dir=save_dir, name=save_dir_name, version=save_version),
            ],
            log_every_n_steps=1,
            callbacks=callbacks,
            # measures all the key methods across Callbacks, DataModules and the LightningModule in the training loop.
            profiler=SimpleProfiler(filename="simple_profiler") if experiment_cfg.profiler else None,
            # If True, runs 1 batch of train, test and val to find any bugs. Also, it can be specified the number of
            # batches to run as an integer
            fast_dev_run=False if num_epochs > 0 else True,
            # PyTorch operations are non-deterministic by default. This means that the results of the operations may
            #  vary from run to run.
            deterministic=True,
        )
