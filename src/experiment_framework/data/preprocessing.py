# -*- coding: utf-8 -*-
"""Time series scaling. (NOT USED IN THE CURRENT VERSION)

The TimeSeriesMeanScaler, TimeSeriesStdScaler, TimeSeriesNOPScaler, and classes in this module are adapted from the
Hugging Face Transformers library, licensed under the Apache License, Version 2.0.

Original code:
https://github.com/huggingface/transformers/blob/9e94801146ceeb3b215bbdb9492be74d7d7b7210/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1-L271

Copyright 2022 The HuggingFace Inc. team. All rights reserved.
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

The following modifications were made:
- Created TimeSeriesScalerType, TimeSeriesScalerTypeStr, and TimeSeriesScalerFactory classes.
"""
# Standard imports
from typing import Literal, Tuple, Union

# Third party imports
import torch


class TimeSeriesMeanScaler(torch.nn.Module):
    """Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data."""

    def __init__(
        self, dim: int = 1, keepdim: bool = True, minimum_scale: float = 1e-10, default_scale: float | None = None
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale
        self.default_scale = default_scale

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the TimeSeriesMeanScaler module.

        Args:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


class TimeSeriesStdScaler(torch.nn.Module):
    """Standardize features.

    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, dim: int = 1, keepdim: bool = True, minimum_scale: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the TimeSeriesStdScaler module.

        Args:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


class TimeSeriesNOPScaler(torch.nn.Module):
    """Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input."""

    def __init__(self, dim: int = 1, keepdim: bool = True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the TimeSeriesNOPScaler module.

        Args:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale


TimeSeriesScalerType = Union[TimeSeriesMeanScaler, TimeSeriesStdScaler, TimeSeriesNOPScaler]
TimeSeriesScalerTypeStr = Literal["mean", "std", "none"]


class TimeSeriesScalerFactory:
    """Factory class to create time series scalers."""

    @staticmethod
    def get_ts_scaler(scaling_method: TimeSeriesScalerTypeStr | None) -> TimeSeriesScalerType:
        """Get the scaler for time series data.

        Args:
            scaling_method (str): The scaling method to use. Options are "mean", "std", or None.

        Returns:
            dict: Dictionary containing the scaler classes.
        """
        # Scaler catalog
        scaler_catalog = {
            "mean": TimeSeriesMeanScaler,
            "std": TimeSeriesStdScaler,
            "none": TimeSeriesNOPScaler,
            None: TimeSeriesNOPScaler,
        }

        if scaling_method not in scaler_catalog.keys():
            raise ValueError(f"Invalid scaling method: {scaling_method}.  Must be 'mean', 'std', or None.")

        return scaler_catalog[scaling_method]()
