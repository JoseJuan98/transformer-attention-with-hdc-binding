# -*- coding: utf-8 -*-
"""Generates a high-quality visualization comparing the structural impact of different HDC bindings."""

# Standard imports
import pathlib

# Third party imports
import numpy
import torch
from matplotlib import pyplot

# First party imports
from models.binding.factory import BindingMethodFactory
from models.positional_encoding.factory import PositionalEncodingFactory
from utils import Config
from utils.plot import set_plot_style


def create_binding_visualization(
    d_model: int = 128, num_positions: int = 256, output_path: pathlib.Path | None = None, dimension_to_plot: int = 0
) -> None:
    """Generates and saves a visualization comparing the effects of different embedding binding methods.

    Args:
        d_model (int): The embedding dimension.
        num_positions (int): The sequence length.
        output_path (pathlib.Path): The path to save the final image.
        dimension_to_plot (int): The specific embedding dimension to plot in the 1D signal comparison.
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

    # --- Plot the 2D Heatmaps of Each Binding Result ---
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
        output_path.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(output_path / "binding_heatmap_comparison.png", bbox_inches="tight")
        print(f"Heatmap plot saved to: {output_path}")

    pyplot.show()
    pyplot.close()

    # Set default background style
    pyplot.style.use("default")

    # --- Extract the 1D Signals for Plotting ---

    positions = numpy.arange(num_positions)
    input_signal_1d = mock_token_embedding[:, dimension_to_plot].numpy()
    pe_signal_1d = sinusoidal_pe[:, dimension_to_plot].numpy()
    additive_1d = additive_result[:, dimension_to_plot].numpy()
    multiplicative_1d = multiplicative_result[:, dimension_to_plot].numpy()
    convolutional_1d = convolutional_result[:, dimension_to_plot].numpy()

    # --- Plot the 1D Signals ---
    fig, axs = pyplot.subplots(nrows=2, ncols=3, figsize=(19, 10))

    # Plot original signals (the "ingredients") with lighter, dashed styles
    axs[0][1].plot(
        positions, input_signal_1d, label="Input Signal (Token)", linestyle="--", alpha=0.9
    )  # , color="darkblue")
    axs[0][1].plot(
        positions, pe_signal_1d, label="Positional Encoding Signal", linestyle=":", alpha=0.9
    )  # , color="darkgreen")

    # Plot resulting signals (the "products") with thicker, solid lines
    axs[1][0].plot(positions, additive_1d, label=r"Additive ($Input + PE$)", linewidth=2.5, color="red")
    axs[1][1].plot(
        positions, multiplicative_1d, label=r"Multiplicative ($Input \odot PE$)", linewidth=2.5, color="purple"
    )
    axs[1][2].plot(
        positions,
        convolutional_1d,
        label=r"Convolutional ($Input \circledast PE$)",
        linewidth=2.5,
        color="black",
    )

    # Hide unused subplot (top-left)
    axs[0][0].axis("off")
    axs[0][2].axis("off")

    # --- Final Adjustments ---
    fig.suptitle(
        f"Effect of Binding Operations on a Single Signal Dimension (dim={dimension_to_plot})",
        fontsize=18,
        fontweight="bold",
    )
    axs[1][1].set_xlabel("Position / Time Step", fontsize=14)
    axs[0][1].set_ylabel("Signal Value", fontsize=14)
    axs[1][0].set_ylabel("Signal Value", fontsize=14)
    for ax in axs.flat:
        # if the ax is off, skip it
        if not ax.has_data():
            continue

        ax.legend(loc="upper right", fontsize=12)
        ax.axhline(0, color="gray", linewidth=0.5)

    pyplot.tight_layout()

    if output_path:
        pyplot.savefig(output_path / "binding_1d_signal_comparison.png", bbox_inches="tight")
        print(f"1D signal comparison plot saved to: {output_path}")

    pyplot.show()
    pyplot.close()


if __name__ == "__main__":
    # Define output path
    plot_path = Config.plot_dir / "binding"

    # Create the visualizations
    print("Generating binding method visualizations...")
    create_binding_visualization(
        output_path=plot_path,
        d_model=128,
        num_positions=128,
    )
