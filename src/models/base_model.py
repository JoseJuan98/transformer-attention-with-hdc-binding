# -*- coding: utf-8 -*-
"""Base Model Abstract Class."""
# Standard imports
from abc import ABC, abstractmethod
from typing import Optional

# Third party imports
import torch


class BaseModel(ABC):
    """Base Model Abstract Class which defines the interface for all models."""

    @abstractmethod
    def evaluate(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, progress_bar: bool = True) -> dict:
        """Evaluates the model on a batch of data.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch of data (x,y) of shape (batch_size, seq_len, input_size)
            and (batch_size,).
            stage (str): The stage of the evaluation (train, val, test).
            progress_bar (bool): Whether to display the progress bar.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_size).
            mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 2).
        """
        ...
