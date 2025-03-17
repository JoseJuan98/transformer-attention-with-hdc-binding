# -*- coding: utf-8 -*-
"""Time series feature embedder module.

The TimeSeriesFeatureEmbedder class in this module are adapted from the Hugging Face Transformers library, licensed
under the Apache License, Version 2.0.

Original code:
https://github.com/huggingface/transformers/blob/9e94801146ceeb3b215bbdb9492be74d7d7b7210/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1-L271

Copyright 2022 The HuggingFace Inc. team. All rights reserved.
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
# Third party imports
import torch


class TimeSeriesFeatureEmbedder(torch.nn.Module):
    """Embed a sequence of categorical features.

    Args:
        cardinalities (`list[int]`):
            List of cardinalities of the categorical features.
        embedding_dims (`list[int]`):
            List of embedding dimensions of the categorical features.
    """

    def __init__(self, cardinalities: list[int], embedding_dims: list[int]) -> None:
        super().__init__()

        self.num_features = len(cardinalities)
        self.embedders = torch.nn.ModuleList([torch.nn.Embedding(c, d) for c, d in zip(cardinalities, embedding_dims)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Time Series Feature Embedder module.

        Args:
            features (`torch.Tensor` of shape `(batch_size, sequence_length, num_features)`):
                The input tensor of categorical features.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, sum(embedding_dims))`:
                The output tensor of embedded features.
        """
        if self.num_features > 1:
            # we slice the last dimension, giving an array of length
            # self.num_features with shape (N,T) or (N)
            cat_feature_slices = torch.chunk(features, self.num_features, dim=-1)
        else:
            cat_feature_slices = [features]

        return torch.cat(
            [
                embed(cat_feature_slice.squeeze(-1))
                for embed, cat_feature_slice in zip(self.embedders, cat_feature_slices)
            ],
            dim=-1,
        )
