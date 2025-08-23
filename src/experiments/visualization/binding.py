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


def calculate_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Calculates the pairwise cosine similarity matrix for a set of vectors.

    Args:
        vectors (torch.Tensor): A tensor of shape (num_vectors, dim).

    Returns:
        torch.Tensor: A similarity matrix of shape (num_vectors, num_vectors).
    """
    # Normalize each vector to unit length
    vectors_norm = torch.nn.functional.normalize(vectors, p=2, dim=1)

    # Compute the similarity matrix using matrix multiplication
    return torch.matmul(vectors_norm, vectors_norm.T)


def plot_binding_heatmaps(
    additive_result: torch.Tensor,
    multiplicative_result: torch.Tensor,
    convolutional_result: torch.Tensor,
    output_path: pathlib.Path | None = None,
) -> None:
    """Plots 2D heatmaps of the results of different binding operations.

    Args:
        additive_result (torch.Tensor): The result of additive binding.
        multiplicative_result (torch.Tensor): The result of multiplicative binding.
        convolutional_result (torch.Tensor): The result of convolutional binding.
        output_path (pathlib.Path | None): The path to save the final image.
    """
    fig, axes = pyplot.subplots(nrows=1, ncols=3, figsize=(18, 6), constrained_layout=True)

    # Colormap
    cmap = "magma"

    for idx, symbol, result in zip(
        range(3),
        ["+", "\\odot", "\\circledast"],
        [additive_result, multiplicative_result, convolutional_result],
    ):
        axes[idx].imshow(result.numpy(), cmap=cmap, aspect="auto", interpolation="nearest")
        axes[idx].set_title(f"$E_{{token}} {symbol} E_{{pos}}$", fontsize=30, pad=20)
        axes[idx].axis("off")

    # Save the figure
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(output_path / "binding_heatmap_comparison.png", bbox_inches="tight")
        print(f"Heatmap plot saved to: {output_path}")

    pyplot.show()
    pyplot.close()


def plot_1d_signal_comparison(
    mock_token_embedding: torch.Tensor,
    sinusoidal_pe: torch.Tensor,
    additive_result: torch.Tensor,
    multiplicative_result: torch.Tensor,
    convolutional_result: torch.Tensor,
    dimension_to_plot: int,
    num_positions: int,
    output_path: pathlib.Path | None = None,
) -> tuple[pyplot.Figure, pyplot.Axes]:
    """Plots a comparison of 1D signals from a single dimension of the embeddings.

    Args:
        mock_token_embedding (torch.Tensor): The original token embeddings.
        sinusoidal_pe (torch.Tensor): The positional encodings.
        additive_result (torch.Tensor): The result of additive binding.
        multiplicative_result (torch.Tensor): The result of multiplicative binding.
        convolutional_result (torch.Tensor): The result of convolutional binding.
        dimension_to_plot (int): The specific embedding dimension to plot.
        num_positions (int): The sequence length.
        output_path (pathlib.Path | None): The path to save the final image.
    """
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
    axs[0][1].plot(positions, input_signal_1d, label="Input Signal (Token)", linestyle="--", alpha=0.9)
    axs[0][1].plot(positions, pe_signal_1d, label="Positional Encoding Signal", linestyle=":", alpha=0.9)

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
    # pyplot.close()

    # Close the specific figure
    pyplot.close(fig)
    return fig, axs


def replot_1d_signal_clean(fig: pyplot.Figure, axs: pyplot.Axes, output_path: pathlib.Path | None = None) -> None:
    """Replots a given figure after removing titles, axes, and legends.

    Args:
        fig (Figure): The figure object to modify.
        axs (Axes): The axes object to modify.
        output_path (pathlib.Path): The path to save the cleaned plot.
    """
    # Remove the super title
    fig.suptitle("")

    # Turn off all axes, which removes labels, ticks, and spines
    # for ax in axs.flat:
    #     ax.axis("off")

    # Reactivate the figure
    pyplot.figure(fig.number)

    # Reactivate each axis and turn off the axes
    for ax in axs.flat:
        ax.figure = fig
        ax.axis("off")

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Cleaned 1D signal plot saved to: {output_path}")

    fig.show()
    pyplot.close(fig)


def plot_similarity_matrices(
    mock_token_embedding: torch.Tensor,
    sinusoidal_pe: torch.Tensor,
    additive_result: torch.Tensor,
    multiplicative_result: torch.Tensor,
    convolutional_result: torch.Tensor,
    output_path: pathlib.Path | None = None,
) -> None:
    """Computes and plots the cosine similarity matrices for original and bound embeddings.

    Args:
        mock_token_embedding (torch.Tensor): The original token embeddings.
        sinusoidal_pe (torch.Tensor): The positional encodings.
        additive_result (torch.Tensor): The result of additive binding.
        multiplicative_result (torch.Tensor): The result of multiplicative binding.
        convolutional_result (torch.Tensor): The result of convolutional binding.
        output_path (pathlib.Path | None): The path to save the final image.
    """
    # --- Compute and Plot Similarity Matrices ---
    sim_token = calculate_similarity_matrix(mock_token_embedding)
    sim_pe = calculate_similarity_matrix(sinusoidal_pe)
    sim_additive = calculate_similarity_matrix(additive_result)
    sim_multiplicative = calculate_similarity_matrix(multiplicative_result)
    sim_convolutional = calculate_similarity_matrix(convolutional_result)

    # --- Plot the Similarity Heatmaps ---
    fig, axes = pyplot.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
    cmap = "magma"
    vmin, vmax = -1.0, 1.0  # Cosine similarity range

    # --- Top Row: Inputs ---
    axes[0, 0].imshow(sim_token.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0, 0].set_title("(a) $E_{token}$", fontsize=24, loc="left", fontweight="bold")

    im1 = axes[0, 1].imshow(sim_pe.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0, 1].set_title("(b) $E_{pos}$", fontsize=24, loc="left", fontweight="bold")

    axes[0, 2].axis("off")  # Hide the unused subplot

    # --- Bottom Row: Bound Results ---
    axes[1, 0].imshow(sim_additive.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1, 0].set_title(r"(c) $E_{token} + E_{pos}$", fontsize=24, loc="left", fontweight="bold")

    axes[1, 1].imshow(sim_multiplicative.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1, 1].set_title(r"(d) $E_{token} \odot E_{pos}$", fontsize=24, loc="left", fontweight="bold")

    axes[1, 2].imshow(sim_convolutional.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1, 2].set_title(r"(e) $E_{token} \circledast E_{pos}$", fontsize=24, loc="left", fontweight="bold")

    # --- Final Adjustments ---
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # fig.suptitle("Impact of Binding Operations on Embedding Similarity Structure", fontsize=30, fontweight="bold")

    # Add a single colorbar for the entire figure
    cbar = fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.05, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Save the figure
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(output_path / "binding_similarity_heatmap.png", bbox_inches="tight")
        print(f"Similarity heatmap plot saved to: {output_path}")

    pyplot.show()
    pyplot.close()


def create_binding_visualization(
    d_model: int = 128, num_positions: int = 256, output_path: pathlib.Path | None = None, dimension_to_plot: int = 0
) -> None:
    """Generates and saves a visualization comparing the effects of different embedding binding methods.

    This function orchestrates the data generation and calls specific plotting functions.

    Notes:
        The dot product of two unit vectors is mathematically equivalent to their cosine similarity.

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
    # This creates a pattern that varies smoothly across both time and mebedding dimension
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
    additive_result = additive_binder(mock_token_embedding, sinusoidal_pe).detach()  # type: ignore[misc]
    multiplicative_result = multiplicative_binder(mock_token_embedding, sinusoidal_pe).detach()  # type: ignore[misc]
    convolutional_result = convolutional_binder(mock_token_embedding, sinusoidal_pe).detach()  # type: ignore[misc]

    # --- Plot the 2D Heatmaps of Each Binding Result ---
    plot_binding_heatmaps(
        additive_result=additive_result,
        multiplicative_result=multiplicative_result,
        convolutional_result=convolutional_result,
        output_path=output_path,
    )

    # Set default background style for subsequent plots
    pyplot.style.use("default")

    # --- Plot the 1D Signals ---
    fig_1d, axs_1d = plot_1d_signal_comparison(
        mock_token_embedding=mock_token_embedding,
        sinusoidal_pe=sinusoidal_pe,
        additive_result=additive_result,
        multiplicative_result=multiplicative_result,
        convolutional_result=convolutional_result,
        dimension_to_plot=dimension_to_plot,
        num_positions=num_positions,
        output_path=output_path,
    )

    # --- Replot the 1D signal plot without titles/axes for a cover image ---
    # This reuses the figure and axes objects from the previous call.
    if output_path:
        replot_1d_signal_clean(
            fig=fig_1d,
            axs=axs_1d,
            output_path=output_path / "cover_binding_1d_signal_comparison.png",
        )

    # --- Compute and Plot Similarity Matrices ---
    plot_similarity_matrices(
        mock_token_embedding=mock_token_embedding,
        sinusoidal_pe=sinusoidal_pe,
        additive_result=additive_result,
        multiplicative_result=multiplicative_result,
        convolutional_result=convolutional_result,
        output_path=output_path,
    )

    # Reset to default style after all plots are done
    pyplot.style.use("default")


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
