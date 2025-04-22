# -*- coding: utf-8 -*-
"""Import shortcuts for models."""
# Standard imports
from typing import Literal, Union

# First party imports
from models.architectures.time_series_classifier import EncoderOnlyTransformerTSClassifier

# Local imports
from .base_model import BaseModel

ModelType = Union[EncoderOnlyTransformerTSClassifier]
ModelTypeStr = Literal["encoder-only-transformer"]
