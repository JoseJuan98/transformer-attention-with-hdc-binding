# -*- coding: utf-8 -*-
"""Data factory for creating the dataset configuration and data loaders based on the dataset name."""
# Standard imports
import logging
import multiprocessing
import pathlib
from typing import Optional, Tuple

# Third party imports
import lightning

# First party imports
from utils.experiment.dataset import UCRDataModule
from utils.experiment.dataset_config import DatasetConfig


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
    ) -> Tuple[DatasetConfig, lightning.LightningDataModule]:
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
        # Determine optimal number of workers
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(cpu_count - 2, 4) if cpu_count > 4 else 1

        logger.info(f"Using {num_workers} workers for data loading")

        data_module = UCRDataModule(
            dsid=dataset_name,
            extract_path=extract_path,
            batch_size=batch_size,
            plot_path=plot_path,
            num_workers=num_workers,
            val_split=val_split,
            logger=logger,
            seed=seed,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

        # Prepare and setup the data module
        data_module.prepare_data()
        data_module.setup("fit")

        # Create dataset configuration
        dataset_cfg = DatasetConfig(
            dataset_name=dataset_name,
            num_classes=data_module.num_classes,
            input_size=data_module.num_dimensions,
            context_length=data_module.max_len,
        )

        # Log dataset information
        logger.info(f"Dataset '{dataset_name}' loaded successfully.")
        logger.info(f"  - Training samples: {len(data_module.train_dataset)}")  # type: ignore [arg-type]
        logger.info(f"  - Validation samples: {len(data_module.val_dataset)}")  # type: ignore [arg-type]
        logger.info(f"  - Test samples: {len(data_module.test_dataset)}")  # type: ignore [arg-type]
        logger.info(f"  - Number of classes: {data_module.num_classes}")
        logger.info(f"  - Number of dimensions: {data_module.num_dimensions}")
        logger.info(f"  - Maximum sequence length: {data_module.max_len}")

        return dataset_cfg, data_module
