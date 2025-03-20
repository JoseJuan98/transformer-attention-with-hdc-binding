# -*- coding: utf-8 -*-
"""Pocket Algorithm callback for PyTorch Lightning."""
# Standard imports
import pathlib

# Third party imports
import lightning
import torch
from lightning.pytorch.callbacks import Callback


class PocketAlgorithm(Callback):
    """Implements the Pocket Algorithm, which saves the model weights with the best validation accuracy seen so far.

    Args:
        monitor (str): Metric to monitor (e.g., 'val_acc').
        mode (str): 'max' or 'min' (whether to maximize or minimize the metric).
        ckpt_filepath (str, pathlib.Path):  Path to save the best model weights.
        model_file_path (str, pathlib.Path): Path to save the best model.
    """

    def __init__(
        self,
        monitor: str = "val_acc",
        mode: str = "max",
        ckpt_filepath: str | pathlib.Path = "best_model.ckpt",
        model_file_path: str | pathlib.Path = "best_model.pt",
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.ckpt_filepath = ckpt_filepath if isinstance(ckpt_filepath, pathlib.Path) else pathlib.Path(ckpt_filepath)
        self.model_file_path = (
            model_file_path if isinstance(model_file_path, pathlib.Path) else pathlib.Path(model_file_path)
        )
        self.best_score = -float("inf") if mode == "max" else float("inf")

    def on_validation_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        """Save the model weights with the best validation accuracy seen so far."""
        super().on_validation_end(trainer=trainer, pl_module=pl_module)

        # TODO: flush logs so that the progress bar is updated instead of just printed line after line
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)

        if current_score is None:
            return  # Metric not found

        if (self.mode == "max" and current_score > self.best_score) or (
            self.mode == "min" and current_score < self.best_score
        ):
            self.best_score = current_score
            trainer.save_checkpoint(filepath=self.ckpt_filepath)
            torch.save(obj=trainer.model, f=self.model_file_path)
            print(
                f"Pocket Algorithm: Saved best model with {self.monitor}={current_score:.4f} to {self.model_file_path}"
            )
