# -*- coding: utf-8 -*-
"""Unit tests for the MultiHeadAttention module."""
# Pytest imports
import pytest

# Third party imports
import torch

# First party imports
from models.transformer.multi_head_attention import MultiHeadAttention
from models.transformer.self_attention import SelfAttention


# Fixture for creating a MultiHeadAttention instance
@pytest.fixture(params=[(64, 2), (128, 4), (256, 8)])
def multihead_attention(request):
    embed_dim, num_heads = request.param
    return MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)


# Test initialization
def test_multihead_attention_init(multihead_attention):
    assert multihead_attention.embed_dim % multihead_attention.num_heads == 0
    assert multihead_attention.head_dim == multihead_attention.embed_dim // multihead_attention.num_heads
    assert len(multihead_attention.attention_heads) == multihead_attention.num_heads
    for head in multihead_attention.attention_heads:
        assert isinstance(head, SelfAttention)
    assert isinstance(multihead_attention.output_projection, torch.nn.Linear)
    assert multihead_attention.output_projection.in_features == multihead_attention.embed_dim
    assert multihead_attention.output_projection.out_features == multihead_attention.embed_dim


# Test invalid initialization (embed_dim not divisible by num_heads)
def test_multihead_attention_init_invalid():
    with pytest.raises(ValueError):
        MultiHeadAttention(embed_dim=63, num_heads=2)


# Test forward pass with valid input
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [10, 20, 30])
def test_multihead_attention_forward_valid(multihead_attention, batch_size, seq_len):
    embed_dim = multihead_attention.embed_dim
    token_encodings = torch.randn(batch_size, seq_len, embed_dim)
    output = multihead_attention(token_encodings)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert output.dtype == torch.float32  # Check default dtype


# Test forward pass with mask
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [10, 32])
def test_multihead_attention_forward_with_mask(multihead_attention, batch_size, seq_len):
    embed_dim = multihead_attention.embed_dim
    token_encodings = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)  # Boolean mask
    output = multihead_attention(token_encodings, mask=mask)
    assert output.shape == (batch_size, seq_len, embed_dim)

    # Also test with a float mask (e.g., -inf for masked positions)
    float_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    float_mask[mask] = float("-inf")
    output_float_mask = multihead_attention(token_encodings, mask=float_mask)
    assert output_float_mask.shape == (batch_size, seq_len, embed_dim)
    assert output_float_mask.dtype == torch.float32


# Test forward pass with different data types
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_multihead_attention_forward_dtype(multihead_attention, dtype):
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("Half precision tests require a CUDA-enabled GPU.")
    batch_size, seq_len, embed_dim = 2, 10, multihead_attention.embed_dim
    token_encodings = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype)

    # Convert the model to the specified dtype
    multihead_attention.to(dtype=dtype)  # This line is crucial for float64

    output = multihead_attention(token_encodings)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert output.dtype == dtype


# Test forward pass with edge cases (seq_len=1, batch_size=1)
def test_multihead_attention_forward_edge_cases(multihead_attention):
    embed_dim = multihead_attention.embed_dim
    # Single token
    token_encodings = torch.randn(1, 1, embed_dim)
    output = multihead_attention(token_encodings)
    assert output.shape == (1, 1, embed_dim)

    # Single batch
    token_encodings = torch.randn(1, 10, embed_dim)
    output = multihead_attention(token_encodings)
    assert output.shape == (1, 10, embed_dim)


# Test that the output projection is actually used
def test_multihead_attention_output_projection(multihead_attention):
    batch_size, seq_len, embed_dim = 2, 10, multihead_attention.embed_dim
    token_encodings = torch.randn(batch_size, seq_len, embed_dim)

    # Disable the output projection by setting its weight to identity and bias to zero
    with torch.no_grad():
        identity_matrix = torch.eye(embed_dim)
        multihead_attention.output_projection.weight.copy_(identity_matrix)
        multihead_attention.output_projection.bias.zero_()

    output = multihead_attention(token_encodings)

    # Manually compute the expected output (without output projection)
    batch_size, seq_len, *__ = token_encodings.shape
    token_encodings = token_encodings.view(
        batch_size, seq_len, multihead_attention.num_heads, multihead_attention.head_dim
    )
    token_encodings = token_encodings.transpose(dim0=1, dim1=2)
    attention_outputs = [
        head(token_encodings=token_encodings[:, i, :, :]) for i, head in enumerate(multihead_attention.attention_heads)
    ]
    expected_output = torch.cat(attention_outputs, dim=-1)

    # Check if the actual output matches the expected output
    torch.testing.assert_close(output, expected_output)


# Test with a mask of all ones (should be equivalent to no mask)
def test_multihead_attention_forward_all_ones_mask(multihead_attention):
    batch_size, seq_len, embed_dim = 2, 10, multihead_attention.embed_dim
    token_encodings = torch.randn(batch_size, seq_len, embed_dim)
    all_ones_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

    output_with_mask = multihead_attention(token_encodings, mask=all_ones_mask)
    output_without_mask = multihead_attention(token_encodings)

    torch.testing.assert_close(output_with_mask, output_without_mask)


# Test with a mask of all zeros (should attend to nothing, but still produce output)
def test_multihead_attention_forward_all_zeros_mask(multihead_attention):
    batch_size, seq_len, embed_dim = 2, 10, multihead_attention.embed_dim
    token_encodings = torch.randn(batch_size, seq_len, embed_dim)
    all_zeros_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    output_with_mask = multihead_attention(token_encodings, mask=all_zeros_mask)
    assert output_with_mask.shape == (batch_size, seq_len, embed_dim)
    # We don't check for specific values, as the behavior with an all-zeros mask
    # might be implementation-dependent (e.g., might still attend to a special token).
    # We just ensure it doesn't crash and produces the correct shape.


# Test on CPU and, if available, GPU
@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        (
            torch.device("cuda")
            if torch.cuda.is_available()
            else pytest.param(torch.device("cpu"), marks=pytest.mark.skip(reason="CUDA not available"))
        ),
    ],
)
def test_multihead_attention_device(multihead_attention, device):
    multihead_attention.to(device)
    batch_size, seq_len, embed_dim = 2, 10, multihead_attention.embed_dim
    token_encodings = torch.randn(batch_size, seq_len, embed_dim, device=device)
    output = multihead_attention(token_encodings)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert output.device.type == device.type
