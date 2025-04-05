# -*- coding: utf-8 -*-
"""Data factory for creating the dataset configuration and data loaders based on the dataset name."""
# Standard imports
import logging
import multiprocessing
import pathlib
from typing import Dict, Optional, Tuple

# Third party imports
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# First party imports
from experiments.time_series.dataset import get_ucr_datasets
from utils.experiments.dataset_config import DatasetConfig


class DataFactory:
    """Factory class for creating the dataset configuration and data loaders based on the dataset name."""

    @staticmethod
    def get_data_loaders_and_config(
        dataset_name: str,
        logger: logging.Logger,
        extract_path: pathlib.Path,
        batch_size: int,
        seed: int = 42,
        val_split: float = 0.2,
        plot_path: Optional[pathlib.Path] = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ) -> Tuple[DatasetConfig, DataLoader, DataLoader, DataLoader]:
        """Get the dataset configuration and data loaders based on the dataset name.

        Args:
            dataset_name (str): The name of the dataset.
            logger (logging.Logger): The logger instance.
            extract_path (pathlib.Path): The path to extract the dataset.
            batch_size (int): The batch size for the data loaders.
            seed (int): The random seed for reproducibility.
            val_split (float): Percentage of training data to use for validation (0.0-1.0).
            plot_path (Optional[pathlib.Path]): The path to save the plots.
            pin_memory (bool): Whether to pin memory for faster data transfer to GPU.
            prefetch_factor (int): Number of batches to prefetch per worker.
            persistent_workers (bool): Whether to keep workers alive between epochs.

        Returns:
            DatasetConfig: The dataset configuration.
            torch.utils.data.DataLoader: The training data loader.
            torch.utils.data.DataLoader: The testing data loader.
            torch.utils.data.DataLoader: The validation data loader.
        """
        # Get the datasets
        train_dataset, test_dataset, max_len, num_classes, num_channels = get_ucr_datasets(
            dsid=dataset_name,
            extract_path=extract_path,
            plot_path=plot_path,
        )

        # Calculate validation split sizes
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size

        logger.info(f"Splitting training data: {train_size} samples for training, {val_size} samples for validation")

        # Create validation split
        train_subset, val_subset = random_split(
            dataset=train_dataset,
            lengths=[train_size, val_size],
            # Fixed seed for reproducibility
            generator=torch.Generator().manual_seed(seed),
        )

        # Create dataset configuration
        dataset_cfg = DatasetConfig(
            dataset_name=dataset_name, num_classes=num_classes, input_size=num_channels, context_length=max_len
        )

        # Determine optimal number of workers
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(cpu_count - 2, 4) if cpu_count > 4 else 1

        logger.info(f"Using {num_workers} workers for data loading")

        # Common DataLoader parameters
        dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
            "persistent_workers": persistent_workers if num_workers > 0 else False,
        }

        # Create data loaders
        train_dataloader = DataLoader(
            dataset=train_subset,
            shuffle=True,  # Shuffle training data
            **dataloader_kwargs,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            shuffle=False,  # Don't shuffle test data
            **dataloader_kwargs,
        )

        val_dataloader = DataLoader(
            dataset=val_subset,
            shuffle=False,  # Don't shuffle validation data
            **dataloader_kwargs,
        )

        # Log dataset information
        logger.info(f"Dataset '{dataset_name}' loaded successfully.")
        logger.info(f"  - Training samples: {len(train_subset)}")
        logger.info(f"  - Validation samples: {len(val_subset)}")
        logger.info(f"  - Test samples: {len(test_dataset)}")
        logger.info(f"  - Number of classes: {num_classes}")
        logger.info(f"  - Number of dimensions: {num_channels}")
        logger.info(f"  - Maximum sequence length: {max_len}")

        return dataset_cfg, train_dataloader, test_dataloader, val_dataloader

    @staticmethod
    def create_cached_dataloaders(
        datasets: Dict[str, Dataset],
        batch_size: int,
        logger: logging.Logger,
    ) -> Dict[str, DataLoader]:
        """Create cached data loaders for multiple datasets.

        This method is useful when you want to reuse the same datasets
        across multiple models or experiments.

        Args:
            datasets: Dictionary mapping split names to datasets
            batch_size: Batch size for the data loaders
            logger: Logger instance

        Returns:
            Dictionary mapping split names to data loaders
        """
        # Determine optimal number of workers
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(cpu_count - 2, 4) if cpu_count > 4 else 1

        logger.info(f"Creating cached dataloaders with {num_workers} workers")

        dataloaders = {}
        for split_name, dataset in datasets.items():
            shuffle = split_name == "train"  # Only shuffle training data

            dataloaders[split_name] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
            )

        return dataloaders
