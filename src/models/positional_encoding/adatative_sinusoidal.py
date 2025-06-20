# -*- coding: utf-8 -*-
"""Sinusoidal Positional Encoding (SPE) from Sun Chuanhao et al. [1]

This module implements the Sinusoidal Positional Encoding (SPE) as proposed in the paper [1]. SPE introduces a learnable
transformation on top of a standard positional encoding, allowing the model to adaptively learn frequency features.

[1] Sun Chuanhao, et al. "Learning High-Frequency Functions Made Easy with Sinusoidal Positional Encoding."
arXiv, 2024, arxiv.org/abs/2407.09370.
"""

# Third party imports
import torch

# First party imports
from models.positional_encoding.base import PositionalEncoding
from models.positional_encoding.sinusoidal import SinusoidalPositionalEncoding


class AdaptiveSinusoidalPositionalEncoding(PositionalEncoding):
    r"""Implements Sinusoidal Positional Encoding (SPE) from Sun Chuanhao et al. [1].

    This method creates learnable positional encodings by applying a linear transformation followed by a sine activation
    to a set of base sinusoidal encodings. The formula is:
    .. math::
        \text{SPE}(x) = \sin(W \cdot \text{PE}_{\text{base}}(x))

    where :math:`\text{PE}_{\text{base}}(x)` is a standard, non-learnable sinusoidal positional encoding, and :math:`W`
    is a learnable weight matrix implemented as a torch.nn.Linear` layer.

    This allows the model to learn the most suitable frequency components for the task at hand.

    References:
        [1] Sun Chuanhao, et al. "Learning High-Frequency Functions Made Easy with Sinusoidal Positional Encoding."
        arXiv, 2024, arxiv.org/abs/2407.09370.

    Args:
        d_model (int): The dimensionality of the embeddings.
        num_positions (int, optional): The maximum sequence length. Defaults to 5000.
    """

    name = "adaptive_sinusoidal"

    def __init__(self, d_model: int, num_positions: int = 5000, **kwargs):
        """Initializes the SPEPositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            num_positions (int, optional): The maximum sequence length. Defaults to 5000.
        """
        # PositionalEncoding's __init__ which calls _init_weight.
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.num_positions = num_positions

        # Instantiate a base, non-learnable sinusoidal PE generator.
        self.base_pe_generator = SinusoidalPositionalEncoding(
            d_model=d_model, num_positions=num_positions, learnable=False
        )

        # Define the learnable linear transformation (W in the paper).
        # The paper suggests omitting the bias term.
        self.linear = torch.nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        # Initialize weights for the linear layer
        torch.nn.init.xavier_normal_(self.linear.weight, gain=1.0)

    @staticmethod
    def _init_weight(d_model: int, num_positions: int, **kwargs) -> torch.nn.Parameter:
        """Not used in SPEPositionalEncoding. It returns an empty parameter to fulfill the abstract method."""
        return torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates the SPE for the input tensor's shape.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model). The content of x is used to
                determine the sequence length.

        Returns:
            torch.Tensor: The Sinusoidal Positional Encoding tensor of shape (1, seq_len, d_model).
        """
        # Generate the base positional encodings, shape (1, seq_len, d_model)
        base_pe = self.base_pe_generator(x)

        # Apply the learnable transformation: Linear -> Sin
        # The output will have shape (1, seq_len, d_model)
        return torch.sin(self.linear(base_pe))
