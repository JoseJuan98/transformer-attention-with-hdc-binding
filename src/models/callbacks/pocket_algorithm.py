# -*- coding: utf-8 -*-
"""Pocket Algorithm callback for PyTorch Lightning."""
# Standard imports
import logging
import pathlib
from typing import Any

# Third party imports
import lightning
import torch
from lightning.pytorch.callbacks import Callback


class PocketAlgorithm(Callback):
    """Implements the Pocket Algorithm, which saves the model weights with the best validation accuracy seen so far.

    Notes:
        It restores the best model for testing.

    Args:
        monitor (str): Metric to monitor (e.g., 'val_acc').
        mode (str): 'max' or 'min' (whether to maximize or minimize the metric).
        ckpt_filepath (str, pathlib.Path):  Path to save the best model weights.
        model_file_path (str, pathlib.Path): Path to save the best model.
        logger (logging.Logger, optional): Logger object. Defaults to None.
        store_on_cpu (bool): If True, store the best model state on CPU. Defaults to False. If memory is a concern,
            especially for large models it's recommended to set this to True.
    """

    def __init__(
        self,
        monitor: str = "val_acc",
        mode: str = "max",
        ckpt_filepath: str | pathlib.Path | None = "best_model.ckpt",
        model_file_path: str | pathlib.Path = "best_model.pt",
        logger: logging.Logger | None = None,
        store_on_cpu: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.store_on_cpu = store_on_cpu
        self.ckpt_filepath = pathlib.Path(ckpt_filepath) if isinstance(ckpt_filepath, str) else ckpt_filepath
        self.model_file_path = (
            model_file_path if isinstance(model_file_path, pathlib.Path) else pathlib.Path(model_file_path)
        )
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.best_model_state: dict[Any, Any] = {}
        self.logger = logger if logger else logging.getLogger(__name__)

    def on_validation_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        """Save the model weights with the best validation accuracy seen so far."""
        super().on_validation_end(trainer=trainer, pl_module=pl_module)

        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)

        # Metric not found
        if current_score is None:
            return

        if (self.mode == "max" and current_score > self.best_score) or (
            self.mode == "min" and current_score < self.best_score
        ):
            self.best_score = current_score

            if self.ckpt_filepath:
                trainer.save_checkpoint(filepath=self.ckpt_filepath)

            torch.save(obj=trainer.model, f=self.model_file_path)

            # Store the best model state dict
            if self.store_on_cpu:
                self.best_model_state = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
            else:
                self.best_model_state = {k: v.clone() for k, v in pl_module.state_dict().items()}

    def on_train_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        """Restore the best model at the end of training."""
        super().on_train_end(trainer=trainer, pl_module=pl_module)

        if self.best_model_state is not None:
            # Restore the best model state
            pl_module.load_state_dict(self.best_model_state)
            self.logger.info(
                f"Pocket Algorithm: Restored best model with {self.monitor}={self.best_score:.4f} for testing"
            )
        else:
            self.logger.info("Pocket Algorithm: No best model state found to restore")
