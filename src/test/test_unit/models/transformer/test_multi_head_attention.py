# -*- coding: utf-8 -*-
"""Unit tests for the MultiHeadAttention module."""
# Pytest imports
import pytest

# Third party imports
import torch

# First party imports
from models.transformer.attention.multi_head_attention import MultiHeadAttention
from models.transformer.attention.self_attention import SelfAttention


# Fixture for creating a MultiHeadAttention instance
@pytest.fixture(params=[(64, 2, 32), (128, 4, 64), (256, 8, 128)])
def multihead_attention(request):
    embed_dim, num_heads, seq_len = request.param
    return MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, seq_len=seq_len)


# Test initialization
def test_multihead_attention_init(multihead_attention):
    assert multihead_attention.embed_dim % multihead_attention.num_heads == 0
    assert multihead_attention.head_dim == multihead_attention.embed_dim // multihead_attention.num_heads
    assert len(multihead_attention.attention_heads) == multihead_attention.num_heads
    for head in multihead_attention.attention_heads:
        assert isinstance(head, SelfAttention)
    assert isinstance(multihead_attention.W_o, torch.nn.Linear)
    assert multihead_attention.W_o.in_features == multihead_attention.embed_dim
    assert multihead_attention.W_o.out_features == multihead_attention.embed_dim


# Test invalid initialization (embed_dim not divisible by num_heads)
def test_multihead_attention_init_invalid():
    with pytest.raises(ValueError):
        MultiHeadAttention(embed_dim=63, num_heads=2, seq_len=1)


# Test forward pass with valid input
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [10, 20, 30])
def test_multihead_attention_forward_valid(multihead_attention, batch_size, seq_len):
    embed_dim = multihead_attention.embed_dim
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = multihead_attention(q=x, k=x, v=x)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert output.dtype == torch.float32  # Check default dtype


# Test forward pass with mask
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [10, 32])
def test_multihead_attention_forward_with_mask(multihead_attention, batch_size, seq_len):
    embed_dim = multihead_attention.embed_dim
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)  # Boolean mask
    output = multihead_attention(q=x, k=x, v=x, mask=mask)
    assert output.shape == (batch_size, seq_len, embed_dim)

    # Also test with a float mask (e.g., -inf for masked positions)
    float_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    float_mask[mask] = float("-inf")
    output_float_mask = multihead_attention(q=x, k=x, v=x, mask=float_mask)
    assert output_float_mask.shape == (batch_size, seq_len, embed_dim)
    assert output_float_mask.dtype == torch.float32


# Test forward pass with different data types
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_multihead_attention_forward_dtype(multihead_attention, dtype):
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("Half precision tests require a CUDA-enabled GPU.")
    batch_size, seq_len, embed_dim = 2, 10, multihead_attention.embed_dim
    x = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype)

    # Convert the model to the specified dtype
    multihead_attention.to(dtype=dtype)  # This line is crucial for float64

    output = multihead_attention(q=x, k=x, v=x)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert output.dtype == dtype


# Test forward pass with edge cases (seq_len=1, batch_size=1)
def test_multihead_attention_forward_edge_cases(multihead_attention):
    embed_dim = multihead_attention.embed_dim
    # Single token
    x = torch.randn(1, 1, embed_dim)
    output = multihead_attention(q=x, k=x, v=x)
    assert output.shape == (1, 1, embed_dim)

    # Single batch
    x = torch.randn(1, 10, embed_dim)
    output = multihead_attention(q=x, k=x, v=x)
    assert output.shape == (1, 10, embed_dim)


# Test that the output projection is actually used
def test_multihead_attention_output_projection(multihead_attention):
    batch_size, seq_len, embed_dim = 2, 10, multihead_attention.embed_dim
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Disable the output projection by setting its weight to identity and bias to zero
    with torch.no_grad():
        identity_matrix = torch.eye(embed_dim)
        multihead_attention.W_o.weight.copy_(identity_matrix)
        multihead_attention.W_o.bias.zero_()

    output = multihead_attention(q=x, k=x, v=x)

    # Manually compute the expected output (without output projection)
    batch_size, seq_len, *__ = x.shape
    q = x.view(batch_size, seq_len, multihead_attention.num_heads, multihead_attention.head_dim).transpose(1, 2)
    k = x.view(batch_size, seq_len, multihead_attention.num_heads, multihead_attention.head_dim).transpose(1, 2)
    v = x.view(batch_size, seq_len, multihead_attention.num_heads, multihead_attention.head_dim).transpose(1, 2)

    attention_outputs = [
        head(q=q[:, i, :, :], k=k[:, i, :, :], v=v[:, i, :, :])
        for i, head in enumerate(multihead_attention.attention_heads)
    ]
    expected_output = torch.cat(attention_outputs, dim=-1)

    # Check if the actual output matches the expected output
    torch.testing.assert_close(output, expected_output)


# Test with a mask of all ones (should be equivalent to no mask)
def test_multihead_attention_forward_all_ones_mask(multihead_attention):
    batch_size, seq_len, embed_dim = 2, 10, multihead_attention.embed_dim
    x = torch.randn(batch_size, seq_len, embed_dim)
    all_ones_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

    output_with_mask = multihead_attention(q=x, k=x, v=x, mask=all_ones_mask)
    output_without_mask = multihead_attention(q=x, k=x, v=x)

    torch.testing.assert_close(output_with_mask, output_without_mask)


# Test with a mask of all zeros (should attend to nothing, but still produce output)
def test_multihead_attention_forward_all_zeros_mask(multihead_attention):
    batch_size, seq_len, embed_dim = 2, 10, multihead_attention.embed_dim
    x = torch.randn(batch_size, seq_len, embed_dim)
    all_zeros_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    output_with_mask = multihead_attention(q=x, k=x, v=x, mask=all_zeros_mask)
    assert output_with_mask.shape == (batch_size, seq_len, embed_dim)
    # Don't check for specific values, as the behavior with an all-zeros mask
    # might be implementation-dependent (e.g., might still attend to a special token).
    # Just ensure it doesn't crash and produces the correct shape.


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
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    output = multihead_attention(q=x, k=x, v=x)
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert output.device.type == device.type
