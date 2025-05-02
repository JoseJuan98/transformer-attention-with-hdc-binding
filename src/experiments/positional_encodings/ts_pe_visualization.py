# -*- coding: utf-8 -*-
"""Experiment to visualize TimeSeriesSinusoidalPositionalEncoding."""

# Third party imports
import matplotlib.colors as mcolors
import numpy
from matplotlib import pyplot

# First party imports
from models.positional_encoding import PositionalEncodingFactory, TSPositionalEncodingTypeStr
from utils import Config


def main(  # noqa: C901
    pe_type: TSPositionalEncodingTypeStr, num_positions: int = 100, embedding_dim: int = 128
) -> None:
    """Main function to visualize the Time Series Sinusoidal Positional Encoding."""
    dir_pe_type = Config.plot_dir / pe_type
    dir_pe_type.mkdir(parents=True, exist_ok=True)

    # --- Instantiate the Encoder ---
    pos_encoder = PositionalEncodingFactory.get_positional_encoding(
        positional_encoding_type=pe_type, d_model=embedding_dim, num_positions=num_positions, seed=42
    )

    # --- Get the Positional Encoding Weights ---
    # The .weight attribute holds the precomputed encodings
    # Shape: (num_positions, embedding_dim)
    pe_weights = pos_encoder.encodings.detach().cpu().numpy().squeeze()

    # Ensure we only plot up to num_positions_to_visualize if the encoder was initialized larger
    pe_weights = pe_weights[:num_positions, :]

    # --- Visualization 1: Heatmap of Positional Encodings ---
    print("Plotting Heatmap...")
    pyplot.figure(figsize=(12, 8))
    # Use a diverging colormap centered at 0
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    pyplot.imshow(pe_weights, aspect="auto", cmap="coolwarm_r", norm=norm)
    pyplot.colorbar(label="Encoding Value")
    pyplot.xlabel("Embedding Dimension")
    pyplot.ylabel("Position in Sequence")
    pyplot.title(f"TimeSeriesSinusoidalPositionalEncoding (Dim={embedding_dim})")

    # Add a vertical line to show the split between sin and cos
    if pe_type == "ts_sinusoidal":
        sentinel = embedding_dim // 2 if embedding_dim % 2 == 0 else (embedding_dim // 2) + 1
        pyplot.axvline(
            x=sentinel - 0.5, color="black", linestyle="--", linewidth=1.5, label=f"Sin/Cos Split (Dim {sentinel})"
        )
        pyplot.legend(loc="upper right", framealpha=0.9)
    pyplot.tight_layout()
    pyplot.savefig(dir_pe_type / f"{pe_type}_heatmap.png")
    pyplot.show()

    print("\nHeatmap Explanation:")
    print("- Each row represents the positional encoding vector for a specific time step (position).")
    print("- Each column represents a specific dimension within the embedding vector.")
    print("- The color indicates the value of the encoding (-1 to 1).")
    print("- Notice the distinct pattern change at the dashed line: ")
    if pe_type == "ts_sinusoidal":
        print(f"  - Dimensions 0 to {sentinel-1} use SIN functions.")
        print(f"  - Dimensions {sentinel} to {embedding_dim-1} use COS functions.")
    print("- Vertical stripes show how values change slowly across positions for low dimensions (low frequency).")
    print(
        "- Horizontal stripes (more frequent changes) show how values change rapidly across positions for high dimensions (high frequency)."
    )

    # --- Visualization 2: Encoding Vectors for Specific Positions ---
    print("\nPlotting Encoding Vectors for Specific Positions...")
    positions_to_plot = [0, 1, 5, 10, 50, 99]  # Select a few positions
    dims = numpy.arange(embedding_dim)

    pyplot.figure(figsize=(12, 6))
    for pos in positions_to_plot:
        if pos < num_positions_to_visualize:
            pyplot.plot(dims, pe_weights[pos, :], label=f"Position {pos}", alpha=0.8)

    pyplot.xlabel("Embedding Dimension")
    pyplot.ylabel("Encoding Value")
    pyplot.title("Positional Encoding Vectors at Specific Positions")
    if pe_type == "ts_sinusoidal":
        pyplot.axvline(x=sentinel - 0.5, color="black", linestyle="--", linewidth=1, label="Sin/Cos Split")
    pyplot.legend()
    pyplot.grid(True, linestyle="--", alpha=0.6)
    pyplot.tight_layout()
    pyplot.savefig(dir_pe_type / f"{pe_type}_encoding_vectors.png")
    pyplot.show()

    print("\nSpecific Position Vectors Explanation:")
    print("- Each line shows the complete encoding vector (across all dimensions) for one specific position.")
    print("- You can see the sinusoidal patterns along the dimension axis.")
    print("- The shape of the wave changes depending on the position.")
    print("- The split between sin (left) and cos (right) parts is visible.")

    # --- Visualization 3: Encoding Values Across Positions for Specific Dimensions ---
    print("\nPlotting Encoding Values Across Positions for Specific Dimensions...")
    dimensions_to_plot = [
        0,
        1,
        2,
        3,
        embedding_dim - 2,
        embedding_dim - 1,
    ]  # Select a few dimensions
    if pe_type == "ts_sinusoidal":
        dimensions_to_plot = dimensions_to_plot + [sentinel - 1, sentinel, sentinel + 1]
    positions = numpy.arange(num_positions)

    pyplot.figure(figsize=(12, 6))
    for dim in dimensions_to_plot:
        if dim < embedding_dim:
            label = f"Dim {dim}"
            if pe_type == "ts_sinusoidal":
                if dim < sentinel:
                    label += " (Sin)"
                else:
                    label += " (Cos)"

            if pe_type == "sinusoidal":
                if dim % 2 == 0:
                    label += " (Sin)"
                else:
                    label += " (Cos)"
            pyplot.plot(positions, pe_weights[:, dim], label=label, alpha=0.8)

    pyplot.xlabel("Position in Sequence")
    pyplot.ylabel("Encoding Value")
    pyplot.title("Positional Encoding Values Across Positions for Specific Dimensions")
    pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    pyplot.grid(True, linestyle="--", alpha=0.6)
    pyplot.tight_layout()
    pyplot.subplots_adjust(right=0.8)  # Adjust layout to make space for legend
    pyplot.savefig(dir_pe_type / f"{pe_type}_encoding_values.png")
    pyplot.show()

    print("\nSpecific Dimension Values Explanation:")
    print("- Each line shows how the encoding value for a *single dimension* changes across sequence positions.")
    print("- Dimensions 0, 1 (low indices) correspond to low-frequency sinusoids (change slowly across positions).")
    print(
        "- Dimensions near the end (high indices) correspond to high-frequency sinusoids (change rapidly across positions)."
    )
    print(
        "- Dimensions before the split use SIN, dimensions after use COS, based on the same underlying frequency calculation."
    )
    print("-" * 30)


if __name__ == "__main__":
    # --- Experiment Setup ---
    num_positions_to_visualize = 100  # How many sequence positions to show
    embedding_dim = 128  # Embedding dimension (must match model)

    # main(pe_type="sinusoidal", embedding_dim=embedding_dim, num_positions=num_positions_to_visualize)
    # main(pe_type="ts_sinusoidal", embedding_dim=embedding_dim, num_positions=num_positions_to_visualize)
    main(pe_type="random", embedding_dim=embedding_dim, num_positions=num_positions_to_visualize)
