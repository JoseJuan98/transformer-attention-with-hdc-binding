# -*- coding: utf-8 -*-
"""Transformer model implementation."""

# Standard imports
import math
from typing import Literal

# Third party imports
import lightning
import torch
import torchmetrics

# First party imports
from models.positional_encoding.sinusoidal_positional_encoding import SinusoidalPositionalEncoding

# Local imports
from .encoder import Encoder


class EncoderOnlyTransformerTSClassifier(lightning.LightningModule):
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
        embed_dim (int): The size of the vocabulary.
        max_len (int): The maximum sequence length.
        dropout (float, optional): The dropout probability. Defaults to 0.1.

    Methods:
        forward(x, mask): Performs a forward pass through the model.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        embed_dim: int,
        batch_size: int,
        positional_encoding: SinusoidalPositionalEncoding,
        num_classes: int,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,  # Add learning_rate as an argument
    ):
        """Initializes the EncoderOnlyTransformerClassifier model.

        Args:
            num_layers (int): The number of encoder layers.
            d_model (int): The dimensionality of the embeddings.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner layer of the feed-forward network.
            embed_dim (int): The dimensionality of the input features.
            batch_size (int): The batch size.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
        """
        super(EncoderOnlyTransformerTSClassifier, self).__init__()
        # Layers
        # Use Linear instead of Embedding
        self.embedding = torch.nn.Linear(in_features=embed_dim * batch_size, out_features=d_model, bias=False)
        self.positional_encoding = positional_encoding
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
        self.fc = torch.nn.Linear(in_features=d_model, out_features=num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        # Hyperparameters
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.num_heads = num_heads
        # Others
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sqrt_d_model = math.sqrt(d_model)
        self.num_classes = num_classes
        self.positional_encoding_name = positional_encoding.__class__.__name__
        self.classification_task: Literal["binary", "multiclass", "multilabel"] = (
            "multiclass" if num_classes > 1 else "binary"
        )

        self._example_input_array = torch.zeros(size=(self.batch_size, self.embed_dim, self.d_model))

        self.save_hyperparameters(
            ignore=["classification_task", "embedding", "positional_encoding", "encoder", "fc", "dropout"]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len).
            mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 2).
        """
        # Scale embedding
        x = self.embedding(x) * self.sqrt_d_model
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x, mask)
        # Global average pooling over the sequence length
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

    def evaluate(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, progress_bar: bool = True) -> dict:
        """Evaluates the model on a batch of data."""
        x, y = batch
        mask = None
        logits = self(x, mask)
        loss = self.loss_fn(logits, y)

        # Calculate and log accuracy
        # preds = torch.argmax(logits, dim=1)
        accuracy = torchmetrics.functional.accuracy(
            preds=logits, target=y, task=self.classification_task, num_classes=self.num_classes
        )

        metrics = {f"{stage}_loss": loss, f"{stage}_acc": accuracy, "n_samples": len(y)}
        self.log_dict(dictionary=metrics, prog_bar=progress_bar, logger=True, reduce_fx="mean")

        # It's needed for the train step
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
