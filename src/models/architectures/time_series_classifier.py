# -*- coding: utf-8 -*-
"""Transformer model implementation.

This module implements an encoder-only Transformer model for time series classification.

The following features are included:
- Use of PyTorch Lightning for training and evaluation.
- Adapted the Transformer Architecture for time series classification (replace embeddings layer for a linear layer
    and added a classification head `fc`).
- Applied the Factory design pattern to create the binding methods.
- Improved masking to handle multivariate inputs and missing values.
- Simplified the model to be encoder-only.
- Clear docstrings and type hints.
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
from models.binding_method import BindingMethodType
from models.embedding.factory import EmbeddingType
from models.positional_encoding import TSPositionalEncodingType
from models.transformer.attention.factory import AttentionTypeStr
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
        in_features (int): The size of the input (number of variates in the time series).
        context_length (int): The length of the input sequence.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        learning_rate (float, optional): The learning rate. Defaults to 1e-3.
        mask_input (bool, optional): Whether to mask the input. Defaults to False.

    Methods:
        forward(x, mask): Performs a forward pass through the model.
    """

    name = "encoder-only-transformer-classifier"

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        in_features: int,
        context_length: int,
        positional_encoding: TSPositionalEncodingType,
        loss_fn: torch.nn.Module | torch.nn.CrossEntropyLoss | torch.nn.BCEWithLogitsLoss,
        num_classes: int,
        embedding: EmbeddingType,
        embedding_binding: BindingMethodType,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        mask_input: bool = False,
        torch_profiling: torch.profiler.profile | None = None,
        mhsa_type: AttentionTypeStr = "standard",
    ):
        """Initializes the EncoderOnlyTransformerClassifier model.

        Args:
            num_layers (int): The number of encoder layers.
            d_model (int): The dimensionality of the embeddings.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner layer of the feed-forward network.
            in_features (int): The size of the input (number of variates/channels in the time series).
            context_length (int): The length of the input sequence.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
            learning_rate (float, optional): The learning rate. Defaults to 1e-3.
            loss_fn (torch.nn.Module | torch.nn.CrossEntropyLoss | torch.nn.BCELoss): The loss function.
            num_classes (int): The number of classes for classification.
            mask_input (bool, optional): Whether to mask the input. Defaults to False.
            positional_encoding (TSPositionalEncodingType): The positional encoding layer.
            embedding_binding (BindingMethodType): The binding method for the embeddings.
            torch_profiling (torch.profiler.profile | None): The PyTorch profiler for performance profiling.
            mhsa_type (str): The type of multi-head self-attention to use. Defaults to "standard". Options are
                "standard" or "rotary".
        """
        super(EncoderOnlyTransformerTSClassifier, self).__init__()
        # Layers
        self.embedding = embedding
        self.positional_encoding = positional_encoding
        self.embedding_binding = embedding_binding
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            mhsa_type=mhsa_type,
            seq_len=context_length,
        )

        # Classification head
        self.fc = torch.nn.Linear(in_features=d_model, out_features=num_classes if num_classes > 2 else 1)
        self.dropout = torch.nn.Dropout(dropout)

        # Hyperparameters
        self.in_features = in_features
        self.context_length = context_length
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.num_heads = num_heads
        self.mask_input = mask_input
        self.mhsa_type = mhsa_type

        # Others
        self.loss_fn = loss_fn
        self.sqrt_d_model = math.sqrt(d_model)
        self.num_classes = num_classes
        self.positional_encoding_name = positional_encoding.name
        self.embedding_binding_name = embedding_binding.name if embedding_binding else "identity"
        self.classification_task: Literal["binary", "multiclass", "multilabel"] = (
            "multiclass" if num_classes > 2 else "binary"
        )
        self.profiler = torch_profiling

        # Used by PyTorch Lightning for sanity checks
        self._example_input_array = torch.zeros(size=(1, self.context_length, self.in_features))

        self.save_hyperparameters(
            ignore=[
                "classification_task",
                "embedding",
                "positional_encoding",
                "encoder",
                "fc",
                "dropout",
                "profiler",
                "loss_fn",
                "embedding_binding",
            ]
        )

        # Xavier Normal initialization for the linear layer
        torch.nn.init.xavier_normal_(self.fc.weight, gain=1.0)

        # Initialize bias of the linear layer to zero
        if self.fc.bias is not None:
            torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, in_features).
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

        # Embedding
        x_embed = self.embedding(x) * self.sqrt_d_model

        # Positional Encoding
        x_pos_enc = self.positional_encoding(x)

        if self.mhsa_type in ["standard", "erpe"]:
            # Binding
            x = self.embedding_binding(x_embed, x_pos_enc)  # type: ignore [misc]
            x = self.dropout(x)

            # Encoder
            x = self.encoder(x, mask)
        # RoPE Attention
        elif self.mhsa_type == "rotary":
            # For RoPE, there is no binding at the input level. The embeddings are passed directly to the encoder.
            x = self.dropout(x_embed)

            # Encoder: The encoder is called with the positional encodings passed as an argument.
            x = self.encoder(x, mask, positional_encodings=x_pos_enc)

        # Global average pooling over the sequence length
        x = x.mean(dim=1)
        return self.fc(x)

    def evaluate(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, progress_bar: bool = True) -> dict:
        """Evaluates the model on a batch of data.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch of data (x,y) of shape (batch_size, seq_len, in_features)
                and (batch_size,).
            stage (str): The stage of the evaluation (train, val, test).
            progress_bar (bool): Whether to display the progress bar.
        """
        x, y = batch

        # Get logits from the model
        logits = self(x)  # Shape: (batch_size (N), 1) for binary, (batch_size (N), num_classes (C)) for multiclass

        # Squeeze the *last* dimension ONLY if it's 1 (i.e., binary classification).
        # This converts (N, 1) -> (N,) for BCEWithLogitsLoss compatibility, including the case (1, 1) -> (1,).
        # It leaves multiclass output (N, C) unchanged.
        if self.classification_task == "binary":
            logits = logits.squeeze(dim=-1)  # Squeeze the feature dimension

            # Casting y to the same type as logits for BCEWithLogitsLoss. This should already be (N,) or (1,),
            #   matching the squeezed logits
            y = y.type(logits.dtype)

        # If logits is 1D (shape [C]) for multiclass, add a batch dimension
        if self.classification_task == "multiclass" and logits.ndim == 1:
            # logits are (C,). Unsqueeze adds batch dim -> (1, C).
            logits = logits.unsqueeze(dim=0)

        # Calculate loss
        loss = self.loss_fn(logits, y)

        # Calculate and log accuracy
        # For binary accuracy with BCEWithLogitsLoss, preds should be probabilities or logits.
        # For multiclass accuracy with CrossEntropyLoss, preds should be logits or probabilities.
        # torchmetrics handles this based on the 'task'.
        accuracy = torchmetrics.functional.accuracy(
            preds=logits,
            target=y.long() if self.classification_task == "binary" else y,  # Accuracy often expects long targets
            task=self.classification_task,
            num_classes=(
                self.num_classes if self.classification_task == "multiclass" else None
            ),  # num_classes only for multiclass/multilabel
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
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)

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
