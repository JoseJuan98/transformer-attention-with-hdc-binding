# -*- coding: utf-8 -*-
"""Import shortcuts for models."""
# Standard imports
from typing import Literal, Union

# First party imports
from models.time_series.transformer import EncoderOnlyTransformerTSClassifier

# Local imports
from .base_model import BaseModel

ModelType = Union[EncoderOnlyTransformerTSClassifier]
ModelTypeStr = Literal["encoder-only-transformer"]
