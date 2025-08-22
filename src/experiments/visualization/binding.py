# -*- coding: utf-8 -*-
"""Generates a high-quality visualization comparing the structural impact of different HDC bindings."""

# Standard imports
import pathlib

# Third party imports
import torch
from matplotlib import pyplot

# First party imports
from models.binding.factory import BindingMethodFactory
from models.positional_encoding.factory import PositionalEncodingFactory
from utils import Config
from utils.plot import set_plot_style


def create_binding_visualization(
    d_model: int = 128, num_positions: int = 256, output_path: pathlib.Path | None = None
) -> None:
    """Generates and saves a visualization comparing the effects of different embedding binding methods.

    Args:
        d_model (int): The embedding dimension.
        num_positions (int): The sequence length.
        output_path (pathlib.Path): The path to save the final image.
    """
    # --- Setup and Style ---
    pyplot.style.use("dark_background")
    torch.manual_seed(42)
    set_plot_style()

    # --- Create Mock Token Embedding and Positional Encoding ---
    # Create a structured, wave-like mock token embedding for clear visualization
    # This creates a pattern that varies smoothly across both time and dimension
    dim_pattern = torch.linspace(0, 4 * torch.pi, d_model)
    pos_pattern = torch.linspace(0, 2 * torch.pi, num_positions).unsqueeze(1)
    mock_token_embedding = torch.sin(pos_pattern + dim_pattern)

    # Generate Sinusoidal PE using your factory
    pe_factory = PositionalEncodingFactory()
    pe_config = {"type": "sinusoidal"}
    pe_encoder = pe_factory.get_positional_encoding(
        positional_encoding_arguments=pe_config,
        d_model=d_model,
        num_positions=num_positions,
    )
    sinusoidal_pe = pe_encoder.encodings.detach().squeeze(0)

    # --- Apply Binding Methods ---
    binding_factory = BindingMethodFactory()

    # Get binders
    additive_binder = binding_factory.get_binding_method("additive", d_model)
    multiplicative_binder = binding_factory.get_binding_method("multiplicative", d_model)
    convolutional_binder = binding_factory.get_binding_method("convolutional", d_model)

    # Apply each binding operation
    # Note: Your forward pass shows you don't scale by sqrt(d_model) before binding,
    # so we follow that logic here for a faithful representation.
    additive_result = additive_binder(mock_token_embedding, sinusoidal_pe).detach()  # type: ignore[misc]
    multiplicative_result = multiplicative_binder(mock_token_embedding, sinusoidal_pe).detach()  # type: ignore[misc]
    convolutional_result = convolutional_binder(mock_token_embedding, sinusoidal_pe).detach()  # type: ignore[misc]

    # --- Plot the Results ---
    fig, axes = pyplot.subplots(nrows=1, ncols=3, figsize=(18, 6), constrained_layout=True)

    # Colormap
    cmap = "magma"

    # Plot 1: Additive Binding
    axes[0].imshow(additive_result.numpy(), cmap=cmap, aspect="auto", interpolation="nearest")
    axes[0].set_title(r"$E_{token} + E_{pos}$", fontsize=30, pad=20)
    axes[0].axis("off")

    # Plot 2: Multiplicative Binding
    axes[1].imshow(multiplicative_result.numpy(), cmap=cmap, aspect="auto", interpolation="nearest")
    axes[1].set_title(r"$E_{token} \odot E_{pos}$", fontsize=30, pad=20)
    axes[1].axis("off")

    # Plot 3: Circular Convolution Binding
    axes[2].imshow(convolutional_result.numpy(), cmap=cmap, aspect="auto", interpolation="nearest")
    axes[2].set_title(r"$E_{token} \circledast E_{pos}$", fontsize=30, pad=20)
    axes[2].axis("off")

    # Save the figure
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(output_path, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    pyplot.show()
    pyplot.close()


if __name__ == "__main__":
    plot_path = Config.plot_dir / "binding" / "binding_comparison.png"

    create_binding_visualization(output_path=plot_path, d_model=128, num_positions=128)
