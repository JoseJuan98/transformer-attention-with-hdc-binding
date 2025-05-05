# -*- coding: utf-8 -*-
"""Visualize Positional Encoding Similarity relative to the Center Position."""

# Standard imports
import pathlib
import warnings
from typing import Dict, List, Tuple

# Third party imports
import numpy
import torch
import torch.nn.functional
from matplotlib import pyplot

# First party imports
from models.positional_encoding.pe_factory import PositionalEncodingFactory, TSPositionalEncodingTypeStr
from utils import Config


def calculate_similarity_from_center(pe_weights: torch.Tensor, pos_ref: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Calculates the cosine similarity between the center position's encoding and all other positions' encodings.

    Args:
        pe_weights (torch.Tensor): The positional encoding matrix (num_positions, d_model).
        pos_ref (int): The reference position (center) for similarity calculation.

    Returns:
        numpy.ndarray: Array of relative positions (p - p_ref).
        numpy.ndarray: Array of cosine similarities relative to the center.
    """
    num_positions, d_model = pe_weights.shape

    # Determine the reference position (center)
    print(f"Calculating similarities relative to center position p_ref={pos_ref}...")

    # Ensure weights are on CPU and float
    pe_weights = pe_weights.cpu().float()

    # Get the reference vector, unsqueeze for broadcasting
    vec_ref = pe_weights[pos_ref, :].unsqueeze(0)  # Shape: (1, d_model)

    # Calculate cosine similarity between the reference vector and all vectors
    # Result shape: (num_positions,)
    similarities = torch.nn.functional.cosine_similarity(vec_ref, pe_weights, dim=1)

    # Handle potential NaNs if vectors were zero
    if torch.isnan(similarities).any():
        warnings.warn("NaN detected in similarities relative to center. Replacing with 0.")
        similarities = torch.nan_to_num(similarities, nan=0.0)

    # Create relative positions
    relative_positions = torch.arange(num_positions) - pos_ref

    return relative_positions.numpy(), similarities.numpy()


def plot_similarity_from_center(
    pe_types: List[TSPositionalEncodingTypeStr],
    plot_path: pathlib.Path | None = None,
    pos_ref: int | None = None,
    num_positions: int = 201,
    d_model: int = 128,
    seed: int = 42,
    custom_args: Dict[str, Dict] | None = None,
    plot: bool = True,
    title: str | None = None,
) -> None:
    """Generates and plots the cosine similarity relative to the center position for multiple PE types."""
    similarity_results = {}

    # Calculate center position if not provided
    pos_ref = pos_ref if pos_ref is not None else num_positions // 2

    print("--- Generating Encodings and Calculating Similarities from Center ---")
    for pe_type in pe_types:
        print(f"\nProcessing type: {pe_type}")

        # Instantiate Encoder
        pos_encoder = PositionalEncodingFactory.get_positional_encoding(
            positional_encoding_type=pe_type,
            d_model=d_model,
            num_positions=num_positions,
            seed=seed,
            **custom_args.get(pe_type, {}) if custom_args is not None else {},
        )

        # Get Weights
        # Shape: (1, num_positions, embedding_dim) -> (num_positions, embedding_dim)
        pe_weights = pos_encoder.encodings.detach().squeeze(0)

        # Calculate Similarity Profile
        relative_positions, similarities = calculate_similarity_from_center(pe_weights=pe_weights, pos_ref=pos_ref)
        if relative_positions.size > 0:  # Only store if calculation was successful
            similarity_results[pe_type] = (relative_positions, similarities)

    print("\n--- Plotting Results ---")
    pyplot.figure(figsize=(12, 7))

    for pe_type, (relative_positions, similarities) in similarity_results.items():
        # Add the parameters to the label for clarity
        if custom_args:
            args_str_parts = [f"{k}={v}" for k, v in custom_args.get(pe_type, {}).items()]
        label = f"{pe_type}"
        if args_str_parts:
            label += f" ({', '.join(args_str_parts)})"

        # Add a shift to the y-axis for better visibility
        if pe_type == "split_sinusoidal":
            similarities = similarities - 0.01
            label += " (shifted vertically by 0.01)"

        pyplot.plot(relative_positions, similarities, label=label, alpha=0.8, linewidth=1.5)

    pyplot.xlabel("Relative Position (p - p_ref)")
    pyplot.ylabel("Cosine Similarity to Center Vector")
    title = (
        f"Similarity relative to Center Position (p_ref={pos_ref}, Dim={d_model}, pos={num_positions})"
        if title is None
        else title
    )
    pyplot.title(title)
    pyplot.axvline(0, color="red", linestyle=":", linewidth=1.5, label=f"Center (Ref Pos {pos_ref})")
    pyplot.legend(loc="lower right", fontsize="small")
    pyplot.grid(True, linestyle="--", alpha=0.6)
    # Set y-axis limits for cosine similarity
    # pyplot.ylim(-1.05, 1.05)
    # Add horizontal line at 0
    pyplot.axhline(0, color="black", linewidth=0.5, linestyle="--")

    pyplot.tight_layout()

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(plot_path)
        print(f"Similarity plot saved to: {plot_path.as_posix()}")

    if plot:
        pyplot.show()
    pyplot.close()


if __name__ == "__main__":
    # --- Experiment Setup ---
    # Use an odd number for a distinct center position
    num_positions_for_sim = 251
    d_model = 128
    seed = 42

    # Specify which types to include in the plot (ensure these are valid in your factory)
    types_to_plot: List[TSPositionalEncodingTypeStr] = [
        "sinusoidal",
        # "split_sinusoidal",
        "fractional_power",
    ]

    # These will override the defaults defined in the PE classes
    # TODO: change to try several combinations of fractional_power
    custom_arguments: Dict[str, Dict] = {
        "fractional_power": {"beta": 15, "kernel": "gaussian"},
    }

    # --- Define Output Directory using Config ---
    # Ensure Config.plot_dir is a Path object pointing where you want plots saved
    output_directory = Config.plot_dir / "similarity_center"  # Subdirectory for these plots

    # --- Run the plotting function ---
    plot_similarity_from_center(
        pe_types=types_to_plot,
        num_positions=65,
        d_model=d_model,
        seed=seed,
        plot_path=output_directory / f"similarity_d{d_model}_n{num_positions_for_sim}.png",
        custom_args=custom_arguments,
        plot=True,
    )

    types_to_plot.remove("sinusoidal")
    for pe_type in types_to_plot + ["random", "split_sinusoidal"]:
        pos_ref = num_positions_for_sim // 2
        title = f"{pe_type.title()} Similarity relative to pos={pos_ref} (Dim={d_model}"
        plot_similarity_from_center(
            pe_types=["sinusoidal", pe_type],
            num_positions=num_positions_for_sim,
            pos_ref=pos_ref,
            d_model=d_model,
            seed=seed,
            plot_path=output_directory / f"similarity_{pe_type}.png",
            custom_args=custom_arguments,
            plot=True,
            title=title,
        )

    # Check other centers
    types_to_plot.append("sinusoidal")
    for pos_ref in [0, num_positions_for_sim // 4, num_positions_for_sim * 3 // 4]:
        plot_similarity_from_center(
            pe_types=types_to_plot,
            num_positions=num_positions_for_sim,
            pos_ref=pos_ref,
            d_model=d_model,
            seed=seed,
            plot_path=output_directory / f"similarity_d{d_model}_pos_ref={pos_ref}.png",
            custom_args=custom_arguments,
            plot=True,
        )

    # TODO: calculate with the dot product only
