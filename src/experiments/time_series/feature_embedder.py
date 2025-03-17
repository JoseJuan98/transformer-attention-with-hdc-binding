# -*- coding: utf-8 -*-
# This project incorporates code from the following projects:
#
# Hugging Face Transformers
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original code:
# https://github.com/huggingface/transformers/blob/9e94801146ceeb3b215bbdb9492be74d7d7b7210/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1-L271
#
# Specifically, the following classes were adapted:
# - TimeSeriesFeatureEmbedder
"""Time series feature embedder module."""
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
