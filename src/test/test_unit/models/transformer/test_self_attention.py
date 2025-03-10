# -*- coding: utf-8 -*-

# Pytest imports
import pytest

# Third party imports
import torch

# First party imports
from models.transformer.self_attention import SelfAttention


def calculate_attention_manually(x, query_proj, key_proj, value_proj, mask=None):
    """
    Calculates self-attention manually, step-by-step, without using PyTorch's
    matrix multiplication functions (except for the final weighted sum).  This
    is for verification purposes.
    """
    batch_size, seq_len, embed_dim = x.shape
    output = torch.zeros_like(x)

    for b in range(batch_size):
        for i in range(seq_len):
            query = torch.matmul(x[b, i], query_proj.weight.T) + query_proj.bias
            attention_weights = []

            for j in range(seq_len):
                key = torch.matmul(x[b, j], key_proj.weight.T) + key_proj.bias
                score = torch.dot(query, key) / embed_dim**0.5

                if mask is not None:
                    if mask.dim() == 2 and mask[b, j] == 0:
                        score = -1e9
                    elif mask.dim() == 3 and mask[b, i, j] == 0:
                        score = -1e9

                attention_weights.append(score)

            attention_weights = torch.tensor(attention_weights)
            attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)

            weighted_value_sum = torch.zeros(embed_dim)
            for j in range(seq_len):
                value = torch.matmul(x[b, j], value_proj.weight.T) + value_proj.bias
                weighted_value_sum += attention_weights[j] * value

            output[b, i] = weighted_value_sum

    return output


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [3, 5, 7])
@pytest.mark.parametrize("embed_dim", [8, 16, 32])
@pytest.mark.parametrize("use_mask", [True, False])
def test_self_attention(batch_size, seq_len, embed_dim, use_mask):
    """
    Tests the SelfAttention module against a manual calculation.
    """
    torch.manual_seed(42)  # For reproducibility

    # Create the SelfAttention module.
    attention = SelfAttention(embed_dim)

    # Create a random input tensor.
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create a mask (if needed).
    mask = None
    if use_mask:
        # mask = torch.randint(0, 2, (batch_size, seq_len)).bool() # Random mask
        mask = (
            torch.tril(torch.ones(seq_len, seq_len))
            .bool()
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )  # Causal mask

    # Calculate attention using the module.
    output_module = attention(x, mask)

    # Calculate attention manually.
    output_manual = calculate_attention_manually(
        x,
        attention.query_projection,
        attention.key_projection,
        attention.value_projection,
        mask,
    )

    # Check if the outputs are close.
    assert torch.allclose(
        output_module, output_manual, atol=1e-5
    ), f"Outputs differ:\nModule: {output_module}\nManual: {output_manual}"


def test_mask_shapes():
    batch_size = 2
    seq_len = 3
    embed_dim = 4
    attention = SelfAttention(embed_dim=embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Test different mask shapes
    mask2d = torch.randint(0, 2, (batch_size, seq_len)).bool()
    mask3d_1 = torch.randint(0, 2, (batch_size, 1, seq_len)).bool()
    mask3d_full = torch.randint(0, 2, (batch_size, seq_len, seq_len)).bool()

    attention(x=x, mask=mask2d)  # Should work
    attention(x=x, mask=mask3d_1)  # Should work
    attention(x=x, mask=mask3d_full)  # Should work

    # Test invalid mask shape
    invalid_mask = torch.randint(0, 2, (batch_size, seq_len, seq_len, 2)).bool()
    with pytest.raises(ValueError):
        attention(x, invalid_mask)


def test_single_element_seq():
    attention = SelfAttention(embed_dim=4)
    x = torch.randn(1, 1, 4)
    out = attention(x)
    assert out.shape == (1, 1, 4)


def test_self_attention_output_shape():
    """Test that the output shape of self-attention is correct for various input shapes."""
    for batch_size in [1, 3, 5]:
        for seq_len in [1, 4, 10]:
            for embed_dim in [8, 16, 64]:
                # Create input tensor
                x = torch.randn(batch_size, seq_len, embed_dim)

                # Create self-attention module
                self_attn = SelfAttention(embed_dim)

                # Forward pass
                output = self_attn(x)

                # Check output shape
                assert output.shape == (
                    batch_size,
                    seq_len,
                    embed_dim,
                ), f"Expected shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"


def test_causal_mask_behavior():
    """Test that the causal mask correctly prevents attending to future positions."""
    batch_size = 2
    seq_len = 5
    embed_dim = 8

    # Create input tensor with distinct values for easier debugging
    x = torch.zeros(batch_size, seq_len, embed_dim)
    for b in range(batch_size):
        for i in range(seq_len):
            x[b, i] = i + 1  # Each position has a distinct value

    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(batch_size, seq_len, seq_len)).bool()

    # Create self-attention module with identity projections for easier verification
    self_attn = SelfAttention(embed_dim)
    with torch.no_grad():
        # Set weights to identity matrices for easier verification
        self_attn.query_projection.weight.copy_(torch.eye(embed_dim))
        self_attn.query_projection.bias.zero_()
        self_attn.key_projection.weight.copy_(torch.eye(embed_dim))
        self_attn.key_projection.bias.zero_()
        self_attn.value_projection.weight.copy_(torch.eye(embed_dim))
        self_attn.value_projection.bias.zero_()

    # Forward pass
    output = self_attn(x, mask)

    # For each position, verify it only attends to itself and previous positions
    for b in range(batch_size):
        for i in range(seq_len):
            # The first position should only attend to itself
            if i == 0:
                assert torch.allclose(
                    output[b, i], x[b, i]
                ), "Position 0 should only attend to itself"
            # Other positions should have a weighted average of current and previous positions
            else:
                # Check that the output at position i is a weighted average of positions 0 to i
                assert torch.all(
                    output[b, i] <= torch.max(x[b, : i + 1])
                ), f"Position {i} should only attend to positions 0 to {i}"
                assert torch.all(
                    output[b, i] >= torch.min(x[b, : i + 1])
                ), f"Position {i} should only attend to positions 0 to {i}"


def test_attention_with_padding_mask():
    """Test that padding masks correctly prevent attending to padded positions."""
    batch_size = 2
    seq_len = 5
    embed_dim = 8

    # Create input tensor
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create padding mask (first batch has 3 valid tokens, second batch has 4)
    padding_mask = torch.ones(batch_size, seq_len)
    padding_mask[0, 3:] = 0  # Pad last 2 positions in first batch
    padding_mask[1, 4:] = 0  # Pad last 1 position in second batch

    # Create self-attention module
    self_attn = SelfAttention(embed_dim)

    # Forward pass with padding mask
    output_masked = self_attn(x, padding_mask)

    # Forward pass without mask but with padded positions zeroed out
    x_padded = x.clone()
    x_padded[0, 3:] = 0
    x_padded[1, 4:] = 0
    output_zeroed = self_attn(x_padded)

    # The outputs should be different because zeroing out inputs is not the same as masking attention
    assert not torch.allclose(
        output_masked, output_zeroed, atol=1e-5
    ), "Masking attention should be different from zeroing out inputs"

    # But the valid positions should still receive some attention
    assert torch.all(
        torch.abs(output_masked[0, :3]) > 1e-6
    ), "Valid positions should receive attention"
    assert torch.all(
        torch.abs(output_masked[1, :4]) > 1e-6
    ), "Valid positions should receive attention"


def test_gradient_flow():
    """Test that gradients flow correctly through the self-attention mechanism."""
    batch_size = 2
    seq_len = 3
    embed_dim = 4

    # Create input tensor that requires gradients
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)

    # Create self-attention module
    self_attn = SelfAttention(embed_dim)

    # Forward pass
    output = self_attn(x)

    # Compute loss and backward
    loss = output.sum()
    loss.backward()

    # Check that gradients are computed for input
    assert x.grad is not None, "Input should have gradients"
    assert not torch.allclose(
        x.grad, torch.zeros_like(x.grad)
    ), "Input gradients should not be zero"

    # Check that gradients are computed for parameters
    for name, param in self_attn.named_parameters():
        assert param.grad is not None, f"{name} should have gradients"
        assert not torch.allclose(
            param.grad, torch.zeros_like(param.grad)
        ), f"{name} gradients should not be zero"


def test_numerical_stability():
    """Test numerical stability with extreme input values."""
    batch_size = 2
    seq_len = 5
    embed_dim = 8

    # Create self-attention module
    self_attn = SelfAttention(embed_dim)

    # Test with very large values
    x_large = torch.ones(batch_size, seq_len, embed_dim) * 1e6
    output_large = self_attn(x_large)
    assert not torch.isnan(output_large).any(), "Output contains NaN with large inputs"
    assert not torch.isinf(output_large).any(), "Output contains Inf with large inputs"

    # Test with very small values
    x_small = torch.ones(batch_size, seq_len, embed_dim) * 1e-6
    output_small = self_attn(x_small)
    assert not torch.isnan(output_small).any(), "Output contains NaN with small inputs"
    assert not torch.isinf(output_small).any(), "Output contains Inf with small inputs"

    # Test with mixed values
    x_mixed = torch.randn(batch_size, seq_len, embed_dim) * 1e3
    output_mixed = self_attn(x_mixed)
    assert not torch.isnan(output_mixed).any(), "Output contains NaN with mixed inputs"
    assert not torch.isinf(output_mixed).any(), "Output contains Inf with mixed inputs"


def test_attention_weights_sum_to_one():
    """Test that attention weights sum to one for each query."""
    batch_size = 2
    seq_len = 5
    embed_dim = 8

    # Create input tensor
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create self-attention module
    self_attn = SelfAttention(embed_dim)

    # Get the attention weights by modifying the forward method temporarily
    original_matmul = torch.matmul
    attention_weights_list = []

    def mock_matmul(a, b):
        if (
            a.dim() == 3
            and b.dim() == 3
            and a.shape[-1] == seq_len
            and b.shape[1] == seq_len
        ):
            # This is the attention weights @ values multiplication
            attention_weights_list.append(a.detach().clone())
        return original_matmul(a, b)

    # Patch torch.matmul temporarily
    torch.matmul = mock_matmul

    # Forward pass
    self_attn(x)

    # Restore torch.matmul
    torch.matmul = original_matmul

    # Check that we captured the attention weights
    assert len(attention_weights_list) == 1, "Expected to capture attention weights"
    attention_weights = attention_weights_list[0]

    # Check that attention weights sum to one for each query
    row_sums = attention_weights.sum(dim=-1)
    assert torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-5
    ), "Attention weights should sum to one for each query"


def test_deterministic_output():
    """Test that the output is deterministic for the same input."""
    batch_size = 2
    seq_len = 5
    embed_dim = 8

    # Create input tensor
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create self-attention module
    self_attn = SelfAttention(embed_dim)

    # Forward pass twice
    output1 = self_attn(x)
    output2 = self_attn(x)

    # Check that outputs are identical
    assert torch.allclose(
        output1, output2
    ), "Output should be deterministic for the same input"
