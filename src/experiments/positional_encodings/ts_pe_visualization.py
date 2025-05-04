# -*- coding: utf-8 -*-
"""Visualize Positional Encoding Values, Vectors and Heatmap."""

# Standard imports
import math
import warnings
from pathlib import Path

# Third party imports
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot

# First party imports
from models.positional_encoding.pe_factory import PositionalEncodingFactory, TSPositionalEncodingTypeStr
from utils import Config

DEFAULT_ARGS = {
    "fractional_power": {"beta": 15.0, "kernel": "gaussian"},
    "sinusoidal": {},
}


def main(  # noqa: C901
    pe_type: TSPositionalEncodingTypeStr,
    num_positions: int = 100,
    embedding_dim: int = 128,
    seed: int = 42,
    output_dir: Path = Path("plots"),
    **kwargs,  # Allow overriding default args like beta, bandwidth, kernel
) -> None:
    """Main function to visualize the specified Positional Encoding."""

    pos_encoder = PositionalEncodingFactory.get_positional_encoding(
        positional_encoding_type=pe_type, num_positions=num_positions, d_model=embedding_dim, seed=seed
    )

    dir_pe_type = output_dir / pe_type
    dir_pe_type.mkdir(parents=True, exist_ok=True)
    print(f"--- Visualizing: {pe_type} ---")
    print(f"Parameters: d_model={embedding_dim}, num_positions={num_positions}, seed={seed}")

    # --- Get the Positional Encoding Weights ---
    # Shape: (1, num_positions, embedding_dim) -> (num_positions, embedding_dim)
    pe_weights = pos_encoder.encodings.detach().cpu().numpy()

    if pe_type not in ["random", "ts_sinusoidal"]:
        pe_weights = pe_weights.squeeze(0)

    # Ensure we only plot up to num_positions if the encoder was initialized larger
    # (Shouldn't happen with current setup but good practice)
    pe_weights = pe_weights[:num_positions, :]

    # --- Determine Plotting Specifics based on Type ---
    plot_title_prefix = f"{pos_encoder.name}"
    has_split = pe_type in ["fpe_sinusoid", "fpe_binary"]
    is_binary = pe_type == "fpe_binary"
    is_orig = pe_type == "fpe_orig"
    is_cosine = pe_type == "fpe_cosine"
    is_classic_sin = pe_type == "sinusoidal"

    # Adjust normalization based on expected range
    if is_binary:
        norm = mcolors.Normalize(vmin=-1.1, vmax=1.1)  # Give slight margin for visualization
        cmap = "coolwarm"  # Binary might look better with distinct colors
    elif is_cosine:
        # Range is [-sqrt(2), sqrt(2)]
        sqrt2 = math.sqrt(2.0)
        norm = mcolors.Normalize(vmin=-sqrt2, vmax=sqrt2)
        cmap = "coolwarm_r"
    elif is_orig:
        # Standardized (mean 0, std 1), range can vary. Let matplotlib decide or set manually.
        # Let's try letting matplotlib decide by setting norm=None
        norm = None
        # Or set a reasonable range like +/- 3 std devs:
        # norm = mcolors.Normalize(vmin=-3, vmax=3)
        cmap = "coolwarm_r"
    else:  # fpe_sinusoid and classic sinusoidal are [-1, 1]
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        cmap = "coolwarm_r"

    sentinel = -1
    if has_split:
        # Split is after the cos part, index starts at 0
        sentinel = math.ceil(embedding_dim / 2.0)
    elif is_classic_sin:
        # Split is after sin part for classic
        sentinel = embedding_dim // 2

    # --- Visualization 1: Heatmap of Positional Encodings ---
    print("Plotting Heatmap...")
    pyplot.figure(figsize=(12, 8))

    # Handle potential NaN/Inf values gracefully before plotting
    if np.isnan(pe_weights).any() or np.isinf(pe_weights).any():
        warnings.warn(f"NaN or Inf values found in '{pe_type}' weights. Clamping for visualization.")
        pe_weights = np.nan_to_num(
            pe_weights, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min
        )
        # If norm wasn't set (e.g., for fpe_orig), set one now after clamping
        if norm is None:
            val_min, val_max = np.min(pe_weights), np.max(pe_weights)
            norm = mcolors.Normalize(vmin=val_min, vmax=val_max)

    im = pyplot.imshow(pe_weights, aspect="auto", cmap=cmap, norm=norm)
    pyplot.colorbar(im, label="Encoding Value")
    pyplot.xlabel("Embedding Dimension")
    pyplot.ylabel("Position in Sequence")
    pyplot.title(f"{plot_title_prefix} (Dim={embedding_dim})")

    # Add a vertical line to show the split if applicable
    if has_split:
        split_label = (
            f"Cos/Sin Split (Dim {sentinel})" if not is_binary else f"Sign(Cos)/Sign(Sin) Split (Dim {sentinel})"
        )
        pyplot.axvline(x=sentinel - 0.5, color="black", linestyle="--", linewidth=1.5, label=split_label)
        pyplot.legend(loc="upper right", framealpha=0.9)
    elif is_classic_sin:
        pyplot.axvline(
            x=sentinel - 0.5, color="black", linestyle="--", linewidth=1.5, label=f"Sin/Cos Split (Dim {sentinel})"
        )
        pyplot.legend(loc="upper right", framealpha=0.9)

    pyplot.tight_layout()
    pyplot.savefig(dir_pe_type / f"{pe_type}_heatmap.png")
    pyplot.show()
    pyplot.close()

    print("\nHeatmap Explanation:")
    print("- Each row represents the positional encoding vector for a specific time step (position).")
    print("- Each column represents a specific dimension within the embedding vector.")
    print(
        f"- The color indicates the value of the encoding ({'approx +/-1' if is_binary else 'standardized' if is_orig else 'value range depends on type'})."
    )
    if has_split:
        print("- Notice the distinct pattern change at the dashed line: ")
        part1 = "Sign(Cos)" if is_binary else "Cos"
        part2 = "Sign(Sin)" if is_binary else "Sin"
        print(f"  - Dimensions 0 to {sentinel-1} use {part1} functions.")
        print(f"  - Dimensions {sentinel} to {embedding_dim-1} use {part2} functions.")
    elif is_classic_sin:
        print("- Notice the distinct pattern change at the dashed line: ")
        print(f"  - Dimensions 0 to {sentinel-1} use Sin functions.")
        print(f"  - Dimensions {sentinel} to {embedding_dim-1} use Cos functions.")
    elif is_orig:
        print("- Patterns arise from the inverse FFT of exponentiated base vector FFT.")
        print("- Similarity between rows depends on the 'beta' parameter and kernel.")
    elif is_cosine:
        print("- Patterns arise from cosine functions with random frequencies and phase biases.")
        print("- Similarity between rows depends on the 'bandwidth' parameter.")
    # General comments applicable to most
    print("- Vertical stripes/patterns show how values change across positions for different dimensions/frequencies.")

    # --- Visualization 2: Encoding Vectors for Specific Positions ---
    print("\nPlotting Encoding Vectors for Specific Positions...")
    positions_to_plot = [0, 1, 5, 10, num_positions // 2, num_positions - 1]  # Select a few positions
    dims = np.arange(embedding_dim)

    pyplot.figure(figsize=(12, 6))
    for pos in positions_to_plot:
        if pos < num_positions:
            pyplot.plot(dims, pe_weights[pos, :], label=f"Position {pos}", alpha=0.8)

    pyplot.xlabel("Embedding Dimension")
    pyplot.ylabel("Encoding Value")
    pyplot.title(f"{plot_title_prefix}: Vectors at Specific Positions")
    if has_split:
        split_label = "Cos/Sin Split" if not is_binary else "Sign(Cos)/Sign(Sin) Split"
        pyplot.axvline(x=sentinel - 0.5, color="black", linestyle="--", linewidth=1, label=split_label)
        pyplot.legend()
    elif is_classic_sin:
        pyplot.axvline(x=sentinel - 0.5, color="black", linestyle="--", linewidth=1, label="Sin/Cos Split")
        pyplot.legend()
    else:
        pyplot.legend()  # Show legend even without split line

    pyplot.grid(True, linestyle="--", alpha=0.6)
    pyplot.tight_layout()
    pyplot.savefig(dir_pe_type / f"{pe_type}_encoding_vectors.png")
    pyplot.show()
    pyplot.close()

    print("\nSpecific Position Vectors Explanation:")
    print("- Each line shows the complete encoding vector (across all dimensions) for one specific position.")
    if has_split or is_classic_sin or is_cosine:
        print("- You can observe wave-like patterns along the dimension axis.")
    elif is_orig:
        print("- The vector shape results from the IFFT process specific to that position's exponent.")
    print("- The shape/values of the vector change depending on the position.")
    if has_split:
        print(
            f"- The split between {'Sign(Cos)' if is_binary else 'Cos'} (left) and {'Sign(Sin)' if is_binary else 'Sin'} (right) parts might be visible."
        )
    elif is_classic_sin:
        print("- The split between Sin (left) and Cos (right) parts is visible.")

    # --- Visualization 3: Encoding Values Across Positions for Specific Dimensions ---
    print("\nPlotting Encoding Values Across Positions for Specific Dimensions...")
    dimensions_to_plot = sorted(
        list(
            set(
                [
                    0,
                    1,
                    2,
                    embedding_dim // 4,
                    embedding_dim // 4 + 1,
                    embedding_dim // 2,
                    embedding_dim // 2 + 1,
                    embedding_dim * 3 // 4,
                    embedding_dim * 3 // 4 + 1,
                    embedding_dim - 2,
                    embedding_dim - 1,
                ]
            )
        )
    )  # Select a spread of dimensions

    # Add dimensions around the split point if applicable
    if has_split or is_classic_sin:
        dimensions_to_plot.extend([sentinel - 1, sentinel, sentinel + 1])
        dimensions_to_plot = sorted(list(set(dimensions_to_plot)))  # Remove duplicates and sort

    positions = np.arange(num_positions)

    pyplot.figure(figsize=(12, 6))
    for dim in dimensions_to_plot:
        if dim < embedding_dim and dim >= 0:  # Ensure dim is valid
            label = f"Dim {dim}"
            if has_split:
                part = "Sign(Cos)" if is_binary else "Cos"
                if dim >= sentinel:
                    part = "Sign(Sin)" if is_binary else "Sin"
                label += f" ({part})"
            elif is_classic_sin:
                label += " (Sin)" if dim < sentinel else " (Cos)"

            pyplot.plot(positions, pe_weights[:, dim], label=label, alpha=0.8)

    pyplot.xlabel("Position in Sequence")
    pyplot.ylabel("Encoding Value")
    pyplot.title(f"{plot_title_prefix}: Values Across Positions for Specific Dimensions")
    pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    pyplot.grid(True, linestyle="--", alpha=0.6)
    pyplot.tight_layout()
    pyplot.subplots_adjust(right=0.75)  # Adjust layout to make space for legend
    pyplot.savefig(dir_pe_type / f"{pe_type}_encoding_values.png")
    pyplot.show()
    pyplot.close()

    print("\nSpecific Dimension Values Explanation:")
    print("- Each line shows how the encoding value for a *single dimension* changes across sequence positions.")
    if has_split or is_classic_sin or is_cosine:
        print("- Lower dimensions often correspond to lower effective frequencies (slower change across positions).")
        print("- Higher dimensions often correspond to higher effective frequencies (faster change across positions).")
    elif is_orig:
        print("- The change across positions reflects the fractional power binding via FFT.")
    if has_split:
        print(
            f"- Dimensions before the split use {'Sign(Cos)' if is_binary else 'Cos'}, dimensions after use {'Sign(Sin)' if is_binary else 'Sin'}."
        )
    elif is_classic_sin:
        print("- Dimensions before the split use Sin, dimensions after use Cos.")
    print("-" * 30 + "\n")


if __name__ == "__main__":
    # --- Experiment Setup ---
    num_positions_to_visualize = 100  # How many sequence positions to show
    embedding_dim = 128  # Embedding dimension

    # --- Run Visualization for Each Type ---
    all_types: list[TSPositionalEncodingTypeStr] = [
        "sinusoidal",  # Classic Vaswani et al.
        "fractional_power",
        "random",
        "ts_sinusoidal",
    ]

    for pe_type in all_types:
        main(
            pe_type=pe_type,
            embedding_dim=embedding_dim,
            num_positions=num_positions_to_visualize,
            output_dir=Config.plot_dir,
            seed=42,  # Use same seed for comparability
        )

    print(f"All plots saved in '{Config.plot_dir.resolve()}'")
