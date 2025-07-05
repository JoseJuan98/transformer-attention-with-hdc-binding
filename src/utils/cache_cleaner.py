# -*- coding: utf-8 -*-
"""Cache Cleaner for PyTorch."""

import torch

class CacheCleaner:
    """A class to handle the cleaning of various caches in PyTorch to free up memory."""

    @staticmethod
    def clean_backend_cache() -> None:
        """Clean up the cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
