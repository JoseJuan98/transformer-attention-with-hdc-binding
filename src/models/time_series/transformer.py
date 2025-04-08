# -*- coding: utf-8 -*-
"""Transformer model implementation.

The TimeSeriesFeatureEmbedder class in this module are adapted from the Hugging Face Transformers library, licensed
under the Apache License, Version 2.0.

Original code:
https://github.com/huggingface/transformers/blob/9e94801146ceeb3b215bbdb9492be74d7d7b7210/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py

Copyright 2022 The HuggingFace Inc. team. All rights reserved.
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

The following modifications were made:
- Use PyTorch Lightning for training and evaluation.
- Adapted for time series classification.
- Added TimeSeriesSinusoidalPositionalEmbedding for continuous time series data.
- Integrated scaling (mean, std, or none) before the embedding layer.
- Improved masking to handle multivariate inputs and missing values.
- Simplified the model to be encoder-only.
- Added clear docstrings and type hints.
"""

# Standard imports
import math
from typing import Literal, Optional

# Third party imports
import lightning
import torch
import torchmetrics

# First party imports
from models.base_model import BaseModel
from models.binding_method.binding_method_factory import BindingMethodFactory, BindingMethodType, BindingMethodTypeStr
from models.positional_encoding import TSPositionalEncodingType
from models.time_series.scaler import TimeSeriesScalerFactory, TimeSeriesScalerType, TimeSeriesScalerTypeStr
from models.transformer.encoder import Encoder


class EncoderOnlyTransformerTSClassifier(BaseModel, lightning.LightningModule):
    """Implements the Encoder-only Transformer Time Series Classifier.

    This class implements an encoder-only Transformer model for time series classification. It consists of an embedding
    layer, positional encoding,a Transformer encoder, and a classification head.

    Notes:
        Embedding layer is a Linear layer, because it deals with continuous time series data, instead of discrete
        tokens. The linear layer projects the input features to the model's embedding dimension (d_model).

    Args:
        num_layers (int): The number of encoder layers.
        d_model (int): The dimensionality of the embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimensionality of the inner layer of the feed-forward network.
        input_size (int): The size of the input (number of variates in the time series).
        context_length (int): The length of the input sequence.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        learning_rate (float, optional): The learning rate. Defaults to 1e-3.
        scaling (Literal["mean", "std", "none"] | None, optional): The scaling method. Defaults to "mean".
        mask_input (bool, optional): Whether to mask the input. Defaults to False.

    Methods:
        forward(x, mask): Performs a forward pass through the model.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        input_size: int,
        context_length: int,
        positional_encoding: TSPositionalEncodingType,
        loss_fn: torch.nn.Module | torch.nn.CrossEntropyLoss | torch.nn.BCELoss,
        num_classes: int,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        embedding_binding: BindingMethodTypeStr = "additive",
        scaling: TimeSeriesScalerTypeStr | None = "mean",
        mask_input: bool = False,
        torch_profiling: torch.profiler.profile | None = None,
    ):
        """Initializes the EncoderOnlyTransformerClassifier model.

        Args:
            num_layers (int): The number of encoder layers.
            d_model (int): The dimensionality of the embeddings.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner layer of the feed-forward network.
            input_size (int): The dimensionality of the input features.
            context_length (int): The length of the input sequence.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
            learning_rate (float, optional): The learning rate. Defaults to 1e-3.
            scaling (Literal["mean", "std", "none"] | None, optional): The scaling method. Defaults to "mean".
            mask_input (bool, optional): Whether to mask the input. Defaults to False.
        """
        super(EncoderOnlyTransformerTSClassifier, self).__init__()
        # Layers
        self.embedding = torch.nn.Linear(in_features=input_size, out_features=d_model, bias=False)
        self.positional_encoding = positional_encoding
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
        self.fc = torch.nn.Linear(in_features=d_model, out_features=num_classes)
        self.dropout = torch.nn.Dropout(dropout)

        # Hyperparameters
        self.input_size = input_size
        self.context_length = context_length
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.num_heads = num_heads
        self.scaling = scaling
        self.mask_input = mask_input

        # Others
        self.loss_fn = loss_fn
        self.sqrt_d_model = math.sqrt(d_model)
        self.num_classes = num_classes
        self.positional_encoding_name = positional_encoding.__class__.__name__
        self.classification_task: Literal["binary", "multiclass", "multilabel"] = (
            "multiclass" if num_classes > 1 else "binary"
        )
        self.profiler = torch_profiling

        self._example_input_array = torch.zeros(size=(1, self.context_length, self.input_size))

        self.scaler: TimeSeriesScalerType = TimeSeriesScalerFactory().get_ts_scaler(scaling_method=scaling)

        self.embedding_binding_name = embedding_binding
        self.embedding_binding: BindingMethodType = BindingMethodFactory().get_binding_method(
            binding_method_name=embedding_binding
        )

        self.save_hyperparameters(
            ignore=[
                "classification_task",
                "embedding",
                "positional_encoding",
                "encoder",
                "fc",
                "dropout",
                "scaler",
                "profiler",
                "loss_fn",
                "embedding_binding",
            ]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_size).
            mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 2).
        """
        # Create observed mask if not provided and mask_input is True
        if mask is None and self.mask_input:
            mask = ~torch.isnan(x)  # True for observed, False for NaN
            if mask.ndim == 3:
                mask = mask.all(dim=2)  # Reduce to (batch_size, seq_len) if multivariate
        elif mask is None:
            mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)

        observed_mask = ~torch.isnan(x)
        if observed_mask.ndim == 3:
            observed_mask = observed_mask.all(dim=2)

        # Scaling
        x_scaled, _, _ = self.scaler(x, observed_mask.unsqueeze(-1).expand_as(x))

        # Embedding
        x_embed = self.embedding(x_scaled) * self.sqrt_d_model

        # Positional Encoding
        x_pos_enc = self.positional_encoding(x_scaled)

        # Binding embeddings and positional encodings
        x = self.embedding_binding(x_embed, x_pos_enc)
        x = self.dropout(x)

        # Encoder
        x = self.encoder(x, mask)

        # Global average pooling over the sequence length
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

    def evaluate(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, progress_bar: bool = True) -> dict:
        """Evaluates the model on a batch of data.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch of data (x,y) of shape (batch_size, seq_len, input_size)
            and (batch_size,).
            stage (str): The stage of the evaluation (train, val, test).
            progress_bar (bool): Whether to display the progress bar.
        """
        x, y = batch
        logits = self(x)  # No mask needed, handled in forward
        loss = self.loss_fn(logits, y)

        # Calculate and log accuracy
        accuracy = torchmetrics.functional.accuracy(
            preds=logits, target=y, task=self.classification_task, num_classes=self.num_classes
        )

        metrics = {f"{stage}_loss": loss, f"{stage}_acc": accuracy.round(decimals=4)}
        self.log_dict(
            dictionary=metrics, prog_bar=progress_bar, logger=True, reduce_fx="mean", on_epoch=True, on_step=False
        )

        # It's needed for the gradient accumulation
        if stage == "train":
            metrics["loss"] = loss

        return metrics

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Compute and return the training loss and some additional metrics for e.g. the progress bar or logger.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The output of your data iterable, normally a
            :class:`~torch.utils.data.DataLoader`.
            batch_idx (int): The index of this batch.

        Returns:
            torch.Tensor: The loss tensor
            dict: A dictionary which can include any keys, but must include the key ``'loss'`` in the case of
            automatic optimization.
            None: In automatic optimization, this will skip to the next batch (but is not supported for
            multi-GPU, TPU, or DeepSpeed). For manual optimization, this has no special meaning, as returning
            the loss is not required.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.
        """
        return self.evaluate(batch, stage="train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Defines the validation step."""
        return self.evaluate(batch, stage="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Defines the test step."""
        return self.evaluate(batch, stage="test")

    def configure_optimizers(self):
        """Configures the optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_start(self) -> None:
        """Called when the train epoch begins."""
        if self.profiler is not None:
            self.profiler.start()
        super().on_train_epoch_start()

    def on_train_epoch_end(self):
        """Called when the train epoch ends."""
        if self.profiler is not None:
            self.profiler.stop()
        super().on_train_epoch_end()
