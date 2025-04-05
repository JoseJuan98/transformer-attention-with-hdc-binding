# -*- coding: utf-8 -*-
"""Positiona encoding shortcuts."""
# Standard imports
from typing import Union

# Local imports
from .sinusoidal import SinusoidalPositionalEncoding
from .ts_circular_convolution import TimeSeriesCircularConvolutionPositionalEncoding
from .ts_elementwise_mul import TimeSeriesElementwiseMultiplicationPositionalEncoding
from .ts_sinusoidal import TimeSeriesSinusoidalPositionalEncoding

TSPositionalEncodingType = Union[
    TimeSeriesSinusoidalPositionalEncoding,
    TimeSeriesElementwiseMultiplicationPositionalEncoding,
    TimeSeriesCircularConvolutionPositionalEncoding,
]
