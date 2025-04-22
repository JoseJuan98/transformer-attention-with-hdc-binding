# -*- coding: utf-8 -*-
"""Unit tests for the EncoderOnlyTransformerTSClassifier class."""
# Pytest imports
import pytest

# Third party imports
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy

# First party imports
from models.architectures.time_series_classifier import EncoderOnlyTransformerTSClassifier

# --- Mock Components ---
# Simple mock for embedding, positional encoding, and binding


class MockEmbedding(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        self.linear = nn.Linear(input_size, d_model)

    def forward(self, x):
        return self.linear(x)


class MockPositionalEncoding(nn.Module):
    name = "mock_pos_enc"

    def __init__(self, d_model, context_length):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length

    def forward(self, x):
        # Return zeros of the expected shape
        batch_size = x.shape[0]
        return torch.zeros(batch_size, self.context_length, self.d_model, device=x.device)


class MockBinding(nn.Module):
    name = "mock_binding"

    def __init__(self):
        super().__init__()

    def forward(self, x_embed, x_pos_enc):
        # Simple addition for testing shape compatibility
        return x_embed + x_pos_enc


# --- Test Fixtures ---


@pytest.fixture(params=[2, 3])  # Test both binary (2) and multiclass (3)
def num_classes(request):
    return request.param


@pytest.fixture
def model_config(num_classes):
    """Provides common model configuration."""
    return {
        "num_layers": 2,
        "d_model": 16,
        "num_heads": 4,
        "d_ff": 32,
        "input_size": 5,
        "context_length": 10,
        "num_classes": num_classes,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "mask_input": False,
    }


@pytest.fixture
def model_instance(model_config):
    """Creates a model instance based on the config."""
    num_classes = model_config["num_classes"]
    input_size = model_config["input_size"]
    d_model = model_config["d_model"]
    context_length = model_config["context_length"]

    # Select appropriate loss function
    loss_fn = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

    # Create mock dependencies
    embedding = MockEmbedding(input_size, d_model)
    positional_encoding = MockPositionalEncoding(d_model, context_length)
    embedding_binding = MockBinding()

    model = EncoderOnlyTransformerTSClassifier(
        **model_config,
        loss_fn=loss_fn,
        embedding=embedding,
        positional_encoding=positional_encoding,
        embedding_binding=embedding_binding,
    )
    # Set to evaluation mode for testing evaluate/forward
    model.eval()
    return model


@pytest.fixture(params=[1, 4])  # Test batch sizes 1 and >1
def batch_size(request):
    return request.param


@pytest.fixture
def batch_data(batch_size, model_instance):
    """Creates dummy batch data."""
    config = model_instance.hparams
    context_length = config.context_length
    input_size = config.input_size
    num_classes = config.num_classes

    x = torch.randn(batch_size, context_length, input_size)

    if num_classes == 2:
        # Binary classification: targets should be float (0.0 or 1.0) for BCEWithLogitsLoss
        y = torch.randint(0, 2, (batch_size,)).float()
    else:
        # Multiclass classification: targets should be long (class indices) for CrossEntropyLoss
        y = torch.randint(0, num_classes, (batch_size,)).long()

    return x, y


# --- Test Cases ---


def test_evaluate_runs_without_error(model_instance, batch_data):
    """
    Tests if the evaluate method runs without raising unexpected errors,
    particularly the ValueError related to tensor shapes in BCEWithLogitsLoss
    when batch_size is 1. This test calls the actual evaluate method.
    """
    batch = batch_data
    stage = "test"

    try:
        metrics = model_instance.evaluate(batch, stage=stage)
        # Basic checks on the output
        assert isinstance(metrics, dict)
        assert f"{stage}_loss" in metrics
        assert f"{stage}_acc" in metrics
        assert isinstance(metrics[f"{stage}_loss"], torch.Tensor)
        assert isinstance(metrics[f"{stage}_acc"], (torch.Tensor, float))

    except ValueError as e:
        # Explicitly fail if the specific ValueError occurs
        if "Target size" in str(e) and "must be the same as input size" in str(e):
            pytest.fail(f"evaluate() method still raised the ValueError: {e}")
        else:
            pytest.fail(f"evaluate() raised an unexpected ValueError: {e}")
    except Exception as e:
        pytest.fail(f"evaluate() raised an unexpected exception: {e}")


def test_evaluate_output_shapes_and_types(model_instance, batch_data):
    """Tests the shapes and types involved in the evaluate method more directly, simulating the internal steps to
    verify the logic.
    """
    x, y = batch_data
    batch_size = x.shape[0]
    num_classes = model_instance.num_classes
    classification_task = model_instance.classification_task

    # --- Simulate steps within evaluate ---
    # 1. Forward pass
    logits_raw = model_instance(x)

    # Expected shape after forward pass
    expected_logit_shape = (batch_size, 1) if classification_task == "binary" else (batch_size, num_classes)
    assert (
        logits_raw.shape == expected_logit_shape
    ), f"Raw logits shape mismatch. Expected {expected_logit_shape}, Got {logits_raw.shape}"

    # 2. Apply the CORRECT squeeze logic
    if classification_task == "binary":
        logits = logits_raw.squeeze(dim=-1)  # Use the fix: squeeze only the last dim
    else:
        logits = logits_raw  # No squeeze for multiclass

    # 3. Adjust target type for binary
    if classification_task == "binary":
        y = y.type(logits.dtype)
        # After specific squeeze for binary:
        # Logits shape should now always be (batch_size,)
        assert logits.shape == torch.Size(
            [batch_size]
        ), f"Logits shape for binary should be [{batch_size}] after squeeze(-1). Got {logits.shape}"
        assert y.shape == torch.Size([batch_size]), f"Target shape for binary should be [{batch_size}]. Got {y.shape}"

    # 4. Handle potential unsqueeze for multiclass
    elif classification_task == "multiclass":
        # Logits shape is (batch_size, num_classes) unless BS=1
        if logits.ndim == 1 and batch_size == 1:  # Condition from evaluate method (logits shape is (C,) here)
            logits = logits.unsqueeze(dim=0)
            assert logits.shape == (
                1,
                num_classes,
            ), f"Logits shape for BS=1 multiclass after unsqueeze should be (1, {num_classes}). Got {logits.shape}"
            assert y.shape == torch.Size([1]), f"Target shape for BS=1 multiclass should be [1]. Got {y.shape}"
        else:  # batch_size > 1
            assert logits.shape == (
                batch_size,
                num_classes,
            ), f"Logits shape for BS>1 multiclass should be ({batch_size}, {num_classes}). Got {logits.shape}"
            assert y.shape == torch.Size(
                [batch_size]
            ), f"Target shape for BS>1 multiclass should be [{batch_size}]. Got {y.shape}"

    # 5. Loss calculation (check types and that it runs without error)
    try:
        loss = model_instance.loss_fn(logits, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Loss should be a scalar tensor
    except ValueError as e:
        # Fail explicitly if the target error occurs *within the test's simulation*
        if "Target size" in str(e) and "must be the same as input size" in str(e):
            pytest.fail(f"Test logic still produced the ValueError: {e}. Logits: {logits.shape}, Target: {y.shape}")
        else:
            pytest.fail(f"Unexpected ValueError during loss calculation in test: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected exception during loss calculation in test: {e}")

    # 6. Accuracy calculation
    acc = accuracy(
        preds=logits,
        target=y.long() if classification_task == "binary" else y,  # Accuracy often expects long targets
        task=classification_task,
        num_classes=num_classes if classification_task == "multiclass" else None,
    )
    assert isinstance(acc, torch.Tensor)
    assert acc.numel() == 1
