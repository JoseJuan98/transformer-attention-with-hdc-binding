# -*- coding: utf-8 -*-
"""Visualize Positional Encoding Similarity relative to the Center Position."""

# Standard imports
import pathlib
import random
import warnings
from typing import Literal, Tuple

# Third party imports
import numpy
import torch
import torch.nn.functional
from matplotlib import pyplot

# First party imports
from models.positional_encoding.factory import PositionalEncodingFactory
from utils import Config
from utils.plot import set_plot_style

MetricStr = Literal["cosine", "product"]

# Set the plot style globally
set_plot_style()


def calculate_similarity_from_position(
    pe_weights: torch.Tensor, pos_ref: int, metric: MetricStr
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Calculates the similarity between the reference position's encoding and all other positions' encodings.

    Args:
        pe_weights (torch.Tensor): The positional encoding matrix (num_positions, d_model).
        pos_ref (int): The reference position for similarity calculation.
        metric (MetricStr): The similarity metric to use ('cosine' or 'product').

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Array of relative positions (p - p_ref) and array of similarities.
    """
    num_positions, d_model = pe_weights.shape

    print(f"Calculating similarities relative to position p_ref={pos_ref} using {metric} metric...")

    # Ensure weights are on CPU and float
    pe_weights = pe_weights.cpu().float()

    # Get the reference vector, unsqueeze for broadcasting
    vec_ref = pe_weights[pos_ref, :].unsqueeze(0)  # Shape: (1, d_model)

    # Calculate similarity between the reference vector and all vectors
    # Result shape: (num_positions,)
    if metric == "cosine":
        similarities = torch.nn.functional.cosine_similarity(vec_ref, pe_weights, dim=1)

    elif metric == "product":
        # Dot product similarity
        similarities = torch.matmul(pe_weights, vec_ref.T).squeeze(1)
        # Normalize dot product to be comparable to cosine
        # norm_ref = torch.linalg.norm(vec_ref)
        # norm_all = torch.linalg.norm(pe_weights, dim=1)
        # similarities = similarities / (norm_ref * norm_all + 1e-8) # Add epsilon for stability
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'product'.")

    # Handle potential NaNs (e.g., from 0/0 in cosine similarity if both vectors are zero)
    if torch.isnan(similarities).any():
        warnings.warn(f"NaN detected in similarities relative to position {pos_ref}. Replacing with 0.")
        similarities = torch.nan_to_num(similarities, nan=0.0)

    # Create relative positions
    relative_positions = torch.arange(num_positions) - pos_ref

    return relative_positions.numpy(), similarities.numpy()


def plot_similarity_from_position(  # noqa: C901
    plot_configurations: dict,
    plot_path: pathlib.Path | None = None,
    pos_ref: int | None = None,
    num_positions: int = 201,
    d_model: int = 128,
    seed: int = 42,
    plot: bool = True,
    use_config_key_for_label: bool = False,
    title: str | None = None,
    metric: MetricStr = "cosine",
    legend: bool = True,
    figsize: tuple[float, float] = (19, 10),
    vertical_line: bool = True,
) -> None:
    """Generates and plots the similarity relative to a reference position for multiple PE configurations.

    Args:
        plot_configurations (Dict[str, PlotConfiguration]): A dictionary where keys are unique labels
         for plot lines, and values are dictionaries containing 'type' (TSPositionalEncodingTypeStr)
         and optionally 'params' (Dict) for the PE. Example:
            {
                "frac_power_beta0.8": {"type": "fractional_power", "params": {"beta": 0.8, "kernel": "gaussian"}},
                "frac_power_beta0.5": {"type": "fractional_power", "params": {"beta": 0.5, "kernel": "gaussian"}},
                "sinusoidal": {"type": "sinusoidal"}
            }
        plot_path (pathlib.Path | None): Path to save the plot. Defaults to None (no save).
        pos_ref (int | None): Reference position. If None, defaults to center (num_positions // 2).
        num_positions (int): Number of positions for the encoding. Defaults to 201.
        d_model (int): Dimension of the model/encoding. Defaults to 128.
        seed (int): Random seed for reproducibility. Defaults to 42.
        plot (bool): Whether to display the plot using pyplot.show(). Defaults to True.
        title (str | None): Custom title for the plot. If None, a default title is generated.
        metric (MetricStr): Similarity metric ('cosine' or 'product'). Defaults to "cosine".
        use_config_key_for_label (bool): If True, uses the configuration key as the label for the plot.
            If False, generates a descriptive label based on the PE type and parameters.
        legend (bool): Whether to display the legend on the plot. Defaults to True.
        figsize (tuple): Figure size for the plot. Defaults to (19, 10).
        vertical_line (bool): Whether to draw a horizontal line at y=0 for reference. Defaults to True.
    """
    # --- Set up the random seed for reproducibility ---
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

    similarity_results = {}

    # Calculate reference position if not provided
    pos_ref = pos_ref if pos_ref is not None else num_positions // 2

    print(f"--- Generating Encodings and Calculating Similarities (Ref Pos: {pos_ref}, Metric: {metric}) ---")
    for config_key, config in plot_configurations.items():
        print(f"\nProcessing configuration: {config_key}")

        print(f"  Params: {config}")

        # Instantiate Encoder
        pos_encoder = PositionalEncodingFactory.get_positional_encoding(
            positional_encoding_arguments=config if isinstance(config, str) else config.copy(),  # type: ignore[arg-type]
            d_model=d_model,
            num_positions=num_positions,
        )

        # Get Weights
        # Shape: (1, num_positions, embedding_dim) -> (num_positions, embedding_dim)
        pe_weights = pos_encoder.encodings.detach().squeeze(0)

        # Calculate Similarity Profile
        relative_positions, similarities = calculate_similarity_from_position(
            pe_weights=pe_weights, pos_ref=pos_ref, metric=metric
        )
        if relative_positions.size > 0:  # Only store if calculation was successful
            # Store results along with the original config for labeling
            similarity_results[config_key] = {
                "positions": relative_positions,
                "similarities": similarities,
                "config": config,  # Store config for easy access later
            }

    print("\n--- Plotting Results ---")
    pyplot.figure(figsize=figsize)

    for config_key, result_data in similarity_results.items():
        relative_positions = result_data["positions"]
        similarities = result_data["similarities"]
        config = result_data["config"]
        if not isinstance(config, dict):
            pe_type = config
            params = {}
        else:
            pe_type = config["type"]
            params = {k: v for k, v in config.items() if k != "type"}

        if use_config_key_for_label:
            label = config_key
        else:
            # Create a descriptive label
            label = pe_type.replace("_", " ").strip().title()
            params_str_parts = [f"{k}={v}" for k, v in params.items()]
            if params_str_parts:
                # Optionally add type and params if key isn't descriptive enough
                label = f"{label} ({', '.join(params_str_parts)})"

        # Example of specific adjustments (keep if needed, or make configurable)
        if pe_type == "split_sinusoidal":
            similarities = similarities - 0.01
            label += " (shifted by -0.01)"

        pyplot.plot(relative_positions, similarities, label=label, alpha=0.8, linewidth=1.5)

    pyplot.xlabel("Relative Position")
    ylabel = f"{metric.capitalize()} Similarity"
    pyplot.ylabel(ylabel)
    # default_title = (
    #     f"{metric.capitalize()} Similarity relative to Position {pos_ref}\n"
    #     f"(Dim={d_model}, Num Positions={num_positions})"
    # )
    # plot_title = default_title if title is None else title
    # pyplot.title(plot_title)
    if legend:
        pyplot.legend(loc="upper right", fontsize="small")

    if vertical_line:
        pyplot.grid(True, linestyle="--", alpha=0.6)
        pyplot.axhline(0, color="black", linewidth=0.5, linestyle="--")

        if pos_ref > 0:
            pyplot.axvline(0, color="red", linestyle=":", linewidth=1.5, label=f"Ref Pos {pos_ref}")

    # Set y-axis limits for better visibility
    pyplot.ylim(0, 1.05)

    pyplot.tight_layout()

    if plot_path is not None:
        plot_path = pathlib.Path(plot_path)  # Ensure it's a Path object
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(plot_path)
        print(f"Similarity plot saved to: {plot_path.as_posix()}")

    if plot:
        pyplot.show()
    pyplot.close()


if __name__ == "__main__":
    # --- Experiment Setup ---
    num_positions_for_sim = 251
    d_model = 128
    seed = 42
    cosine_metric: MetricStr = "cosine"
    output_directory = Config.plot_dir / "positional_encodings" / "similarity"
    output_directory.mkdir(parents=True, exist_ok=True)

    # --- Plot Configurations ---
    # Example 1: different PE types
    configs_compare_types = {
        "Sinusoidal": "sinusoidal",
        "Fractional Power (β=0.8, Gauss)": {"type": "fractional_power", "beta": 0.8, "kernel": "gaussian"},
        "Fractional Power (β=1, Sinc)": {"type": "fractional_power", "beta": 1, "kernel": "sinc"},
        "Random": "random",
        # "Split Sinusoidal": "split_sinusoidal"
    }

    print("\n=== Plotting Comparison of Different PE Types ===")
    plot_similarity_from_position(
        plot_configurations=configs_compare_types,
        num_positions=num_positions_for_sim,
        d_model=d_model,
        seed=seed,
        plot_path=output_directory / f"similarity_types_d{d_model}_n{num_positions_for_sim}_{cosine_metric}.png",
        plot=True,
        metric=cosine_metric,
    )

    # Example 2: FPE with different parameters
    configs_compare_params = {
        # Sinusoidal for reference
        "Sinusoidal Ref": "sinusoidal",
        "FracPower (β=0.8, Gauss)": {"type": "fractional_power", "beta": 0.8, "kernel": "gaussian"},
        "FracPower (β=1, Sinc)": {"type": "fractional_power", "beta": 1, "kernel": "sinc"},
        "FracPower (β=2, Sinc)": {"type": "fractional_power", "beta": 2, "kernel": "sinc"},
        "FracPower (β=5, Sinc)": {"type": "fractional_power", "beta": 5, "kernel": "sinc"},
    }

    print("\n=== Plotting Comparison of Fractional Power Parameters ===")
    plot_similarity_from_position(
        plot_configurations=configs_compare_params,
        num_positions=num_positions_for_sim,
        d_model=d_model,
        seed=seed,
        plot_path=output_directory
        / f"similarity_fracpower_params_d{d_model}_n{num_positions_for_sim}_{cosine_metric}.png",
        plot=True,
        metric=cosine_metric,
        title=f"Fractional Power Similarity ({cosine_metric.capitalize()}) vs Parameters",
    )

    # Example 3: different reference positions
    configs_for_pos_ref = {
        "Sinusoidal": "sinusoidal",
        "Random": "random",
        "Fractional Power $(\\text{Gauss}, \\beta=0.8)$": {
            "type": "fractional_power",
            "beta": 0.8,
            "kernel": "gaussian",
        },
        "Fractional Power $(\\text{Sinc}, \\beta=1)$": {"type": "fractional_power", "beta": 1, "kernel": "sinc"},
        "Fractional Power $(\\text{Sinc}, \\beta=2)$": {"type": "fractional_power", "beta": 2, "kernel": "sinc"},
        "Fractional Power $(\\text{Sinc}, \\beta=5)$": {"type": "fractional_power", "beta": 5, "kernel": "sinc"},
    }
    print("\n=== Plotting Comparison for Different Reference Positions ===")
    for pos_ref_test in [0, num_positions_for_sim // 2]:
        plot_similarity_from_position(
            plot_configurations=configs_for_pos_ref,
            num_positions=num_positions_for_sim,
            pos_ref=pos_ref_test,  # Set specific reference position
            d_model=d_model,
            seed=seed,
            plot_path=output_directory
            / f"similarity_posref{pos_ref_test}_d{d_model}_n{num_positions_for_sim}_{cosine_metric}.png",
            plot=True,
            metric=cosine_metric,
            use_config_key_for_label=True,
        )

    # Example 4: dot product as metric
    print("\n=== Plotting Comparison using Dot Product Metric ===")
    plot_similarity_from_position(
        plot_configurations=configs_compare_types,  # Reuse type comparison config
        num_positions=num_positions_for_sim,
        d_model=d_model,
        seed=seed,
        plot_path=output_directory / f"similarity_dot_product_d{d_model}_n{num_positions_for_sim}.png",
        plot=True,
        metric="product",
    )

    # Example 5: Sinusoidal Cosine Similarity
    print("\n=== Plotting Absolute Sinusoidal Cosine Similarity ===")

    plot_similarity_from_position(
        plot_configurations={"Sinusoidal": "sinusoidal"},
        num_positions=num_positions_for_sim,
        pos_ref=num_positions_for_sim // 2,
        d_model=d_model,
        seed=seed,
        plot_path=output_directory / f"sinusoidal_{cosine_metric}_d{d_model}_n{num_positions_for_sim}.png",
        plot=True,
        metric=cosine_metric,
        use_config_key_for_label=True,
        legend=False,
        figsize=(14, 10),
        vertical_line=False,
    )
