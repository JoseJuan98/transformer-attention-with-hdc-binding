# -*- coding: utf-8 -*-
"""FPE-based positional encoding modules."""

# Standard imports
import math
from typing import Literal

# Third party imports
import torch
import torch.fft

FPEKernelStr = Literal["sinc", "gaussian", "triangular"]


# TODO: inherit from PoisitionalEncoding
class FPEOrigPositionalEncoding(torch.nn.Module):
    """Positional encoding using the original Fractional Power Encoding (FPE) via FFT.

    Args:
        d_model (int): The dimensionality of the embeddings (HDC dimension).
        num_positions (int, optional): The maximum sequence length. Defaults to 5000.
        beta (float): Scaling factor for similarity (controls how fast similarity decays).
        kernel (str, optional): Method for initializing the base phase vector.
            Options: 'sinc', 'gaussian', 'triangular'. Defaults to 'sinc'.
        seed (int, optional): Random seed for phase initialization. Defaults to 0.
    """

    name = "fractional_power_encoding"

    def __init__(
        self,
        d_model: int,
        num_positions: int = 5000,
        beta: float = 10.0,
        kernel: FPEKernelStr = "sinc",
        seed: int = 0,
        **kwargs,
    ):
        # Store specific args for _init_weight
        # init_args = {"beta": beta, "kernel": kernel, "seed": seed}
        # super(FPEOrigPositionalEncoding, self).__init__(d_model=d_model, num_positions=num_positions, **init_args)
        super(FPEOrigPositionalEncoding, self).__init__()

        self.d_model = d_model
        self.num_positions = num_positions
        self.beta = beta
        self.kernel = kernel
        self.seed = seed
        # Pass d_model, num_positions, and any other relevant args to _init_weight
        self.encodings = self._init_weight(
            d_model=d_model, num_positions=num_positions, beta=beta, kernel=kernel, seed=seed, **kwargs
        )

        if not isinstance(self.encodings, torch.nn.Parameter):
            raise TypeError("_init_weight must return a torch.nn.Parameter")

        if self.encodings.requires_grad:
            print(f"Warning: Positional encoding {self.__class__.__name__} is set to be learnable.")

    @staticmethod
    def _init_weight(
        d_model: int, num_positions: int, beta: float, kernel: FPEKernelStr, seed: int, **kwargs
    ) -> torch.nn.Parameter:
        """Initializes positional encodings using the FPE-FFT method."""
        torch.manual_seed(seed)

        # Initialize phase vector based on kernel type
        if kernel == "sinc":
            # Uniform random phases in [-pi, pi]
            phases = (torch.rand(d_model) * 2 * math.pi) - math.pi

        elif kernel == "gaussian":
            # Adjust scale as in original code
            phases = torch.randn(d_model)
            beta *= 3

        elif kernel == "triangular":
            # Approximate the original sampling logic
            # Create linearly spaced points and sinc^2 probabilities
            linspace = torch.linspace(-math.pi, math.pi, d_model)
            # torch.sinc(x) = sin(pi*x)/(pi*x)
            p = torch.pow(torch.sinc(linspace / math.pi), 2)
            p /= p.sum()
            # Sample indices based on probability p, then map back to linspace values
            indices = torch.multinomial(p, d_model, replacement=True)
            phases = linspace[indices]
            # Adjust scale as in original code
            beta *= 6
        else:
            raise ValueError(f"Invalid kernel type: {kernel}. Valid values are {FPEKernelStr}")

        # Prepare phase vector for Hermitian symmetry (ensures real ifft)
        D = d_model
        phases_half = phases[: ((D - 1) // 2)]
        if D % 2 == 1:
            # Odd D: [p_half, 0, -p_half_reversed]
            phases_sym = torch.cat((phases_half, torch.tensor([0.0]), -torch.flip(phases_half, dims=[0])))
        else:
            # Even D: [0, p_half, 0, -p_half_reversed]
            phases_sym = torch.cat(
                (torch.tensor([0.0]), phases_half, torch.tensor([0.0]), -torch.flip(phases_half, dims=[0]))
            )
        phases = phases_sym

        # Generate base vector using inverse FFT
        # Ensure phases tensor has correct dtype if needed (complex requires float input)
        base_fft = torch.exp(1j * phases.float())
        # Apply ifftshift before ifft
        base = torch.fft.ifft(torch.fft.ifftshift(base_fft)).real

        # Generate input positions (normalized 0 to 1)
        positions = torch.linspace(0, 1, num_positions, dtype=torch.float32)

        # Compute fractional power expansion (FPE)
        exponent = beta * positions[:, None] + 1
        # Ensure base is complex for fft, add batch dim for exponent broadcasting
        base_fft_batch = torch.fft.fft(base.cfloat()[None, :])
        fpe_complex = torch.fft.ifft(torch.pow(base_fft_batch, exponent))
        fpe = torch.real(fpe_complex)  # Shape: (num_positions, d_model)

        # Standardize the results (zero mean, unit variance per vector)
        fpe_mean = torch.mean(fpe, dim=1, keepdim=True)
        fpe_std = torch.std(fpe, dim=1, keepdim=True)
        # Add small epsilon to prevent division by zero
        fpe = (fpe - fpe_mean) / (fpe_std + 1e-8)

        # Add batch dimension for compatibility with Transformer layers
        encodings = fpe.unsqueeze(0)  # Shape: (1, num_positions, d_model)

        return torch.nn.Parameter(encodings.float(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the positional encoding for the input sequence length.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
                              Only seq_len (x.size(1)) is used.

        Returns:
            torch.Tensor: The positional encoding tensor of shape (1, seq_len, d_model).
        """
        # Return shape (1, seq_len, d_model) sliced from precomputed encodings
        return self.encodings[:, : x.size(1)]
