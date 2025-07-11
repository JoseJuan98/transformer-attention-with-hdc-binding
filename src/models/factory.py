# -*- coding: utf-8 -*-
"""Model factory for creating models based on configuration."""
# Standard imports
import logging
import pathlib
from typing import Dict, Type

# Third party imports
import lightning
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelSummary,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

# First party imports
from experiment_framework.config.dataset import DatasetConfig
from experiment_framework.config.experiment import ExperimentConfig
from experiment_framework.config.model import ModelConfig
from models import EncoderOnlyTransformerTSClassifier
from models.binding import BindingMethodFactory
from models.callbacks.fine_tune_lr_finder import FineTuneLearningRateFinder
from models.callbacks.pocket_algorithm import PocketAlgorithm
from models.embedding.factory import EmbeddingFactory
from models.positional_encoding import PositionalEncodingFactory
from utils import Config

ModelType = EncoderOnlyTransformerTSClassifier
ModelTypeStr = "encoder-only-transformer"


class ModelFactory:
    """Factory class for creating models based on configuration."""

    model_catalog: Dict[str, Type[ModelType]] = {
        "encoder-only-transformer": EncoderOnlyTransformerTSClassifier,
    }

    @classmethod
    def get_model(
        cls, model_config: ModelConfig, dataset_cfg: DatasetConfig, profiler_path: str = ""
    ) -> EncoderOnlyTransformerTSClassifier:
        """Get the model based on the configuration.

        Args:
            model_config (ModelConfig): The model configuration.
            dataset_cfg (DatasetConfig): The dataset configuration.
            profiler_path (str): [Optional] Path to save the profiler logs.

        Returns:
            EncoderOnlyTransformerTSClassifier: The model instance.
        """
        if model_config.model not in cls.model_catalog:
            raise ValueError(
                f"Model '{model_config.model}' is not supported.\nSupported models: {cls.model_catalog.keys()}"
            )

        model = cls.model_catalog[model_config.model]

        # Get the positional encoding instance based on the configuration
        positional_encoding = PositionalEncodingFactory.get_positional_encoding(
            positional_encoding_arguments=model_config.positional_encoding,
            d_model=model_config.d_model,
            num_positions=dataset_cfg.context_length,
        )

        # Get the embedding instance based on the configuration
        embedding = EmbeddingFactory.get_embedding(
            embedding_args=model_config.embedding, num_channels=dataset_cfg.num_channels, d_model=model_config.d_model
        )

        # Get the Embedding Binding Method instance based on the configuration
        embedding_binding = BindingMethodFactory.get_binding_method(
            binding_method_name=model_config.embedding_binding, embedding_dim=model_config.d_model
        )

        return model(
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            in_features=dataset_cfg.num_channels,
            context_length=dataset_cfg.context_length,
            embedding=embedding,
            embedding_binding=embedding_binding,
            positional_encoding=positional_encoding,
            mhsa_type=model_config.multihead_attention,
            num_classes=dataset_cfg.num_classes,
            dropout=model_config.dropout,
            learning_rate=model_config.learning_rate,
            mask_input=True,
            loss_fn=torch.nn.CrossEntropyLoss() if dataset_cfg.num_classes > 2 else torch.nn.BCEWithLogitsLoss(),
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

    @classmethod
    def get_trainer(
        cls,
        default_root_dir: pathlib.Path,
        experiment_cfg: ExperimentConfig,
        num_epochs: int,
        model_relative_path: str,
        save_dir: str | pathlib.Path,
        save_dir_name: str,
        save_version: str,
        logger: logging.Logger,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = "norm",
        lr_iterations: int = 10,
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
            logger (logging.Logger): The logger object.
            gradient_clip_val (float): The value to clip gradients to.
            gradient_clip_algorithm (str): The algorithm to use for gradient clipping.
            logger (logging.Logger | None, optional): Logger object. Defaults to None.
            lr_iterations (int, optional): Number of iterations for learning rate finder. Defaults to 10.

        Returns:
            lightning.Trainer: The trainer instance.
        """

        # --- Configure Trainer Callbacks ---
        # Default callback (Pocket Algorithm)
        pocket_algorithm = PocketAlgorithm(
            monitor="val_acc",
            mode="max",
            ckpt_filepath=None,  # Config.model_dir / pathlib.Path(model_relative_path).with_suffix(".ckpt"),
            model_file_path=Config.model_dir / model_relative_path,
            logger=logger,
        )

        callbacks: list[lightning.pytorch.callbacks.Callback] = [pocket_algorithm]

        # Model Deep Summary
        if experiment_cfg.summary:
            callbacks.append(ModelSummary(max_depth=-1))

        # Best-practices Callbacks
        # 1. Early Stopping
        if experiment_cfg.early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=experiment_cfg.early_stopping_patience,
                verbose=False,
                strict=False,
            )
            callbacks.append(early_stopping)
            logger.info(f"\t=> Using EarlyStopping with patience={experiment_cfg.early_stopping_patience}")

        # 2. Gradient Accumulation
        if isinstance(experiment_cfg.accumulate_grad_batches, dict):
            accumulator = GradientAccumulationScheduler(scheduling=experiment_cfg.accumulate_grad_batches)
            callbacks.append(accumulator)
            logger.info(
                f"\t=> Using GradientAccumulationScheduler with scheduling: {experiment_cfg.accumulate_grad_batches}"
            )

        # 3. Stochastic Weight Averaging
        if experiment_cfg.use_swa:
            swa_lrs = experiment_cfg.swa_learning_rate if hasattr(experiment_cfg, "swa_learning_rate") else 1e-2
            callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lrs))
            logger.info(f"\t=> Using Stochastic Weight Averaging with swa_lrs={swa_lrs}")

        # 4. Learning Rate Finder
        if experiment_cfg.use_lr_finder:
            if hasattr(experiment_cfg, "lr_finder_milestones"):
                lr_finder_callback = FineTuneLearningRateFinder(
                    milestones=experiment_cfg.lr_finder_milestones,
                    update_attr=True,
                    num_training_steps=lr_iterations,
                    early_stop_threshold=3.0,
                )
                callbacks.append(lr_finder_callback)
                logger.info(f"\t=> Using Learning Rate Finder with milestones={experiment_cfg.lr_finder_milestones}")

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
            # 4. Gradient Clipping
            gradient_clip_val=gradient_clip_val if gradient_clip_val > 0 else None,
            gradient_clip_algorithm=gradient_clip_algorithm,
            # measures all the key methods across Callbacks, DataModules and the LightningModule in the training loop.
            profiler=SimpleProfiler(filename="simple_profiler") if experiment_cfg.profiler else None,
            # If True, runs 1 batch of train, test and val to find any bugs. Also, it can be specified the number of
            # batches to run as an integer
            fast_dev_run=False if num_epochs > 0 else True,
            # PyTorch operations are non-deterministic by default. This means that the results of the operations may
            #  vary from run to run.
            deterministic=True,
            # benchmark=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
