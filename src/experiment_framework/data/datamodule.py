# -*-- coding: utf-8 -*-
"""Lightning Data Module for UCR datasets."""
# Standard imports
import logging
import pathlib

# Third party imports
import lightning
import torch
from torch.utils.data import DataLoader, random_split

# First party imports
from experiment_framework.data.ucr_loader import get_ucr_datasets


class UCRDataModule(lightning.LightningDataModule):
    """Data module for UCR datasets.

    This class handles the downloading, preprocessing, and loading of UCR datasets for time series classification
    tasks using PyTorch Lightning. It uses the sktime library to load the datasets and standardizes the data using
    sklearn's StandardScaler. The data is then converted to PyTorch tensors and wrapped in TensorDatasets for
    training and testing.

    Args:
        dsid (str): The name of the UCR dataset.
        extract_path (pathlib.Path, str): The path to extract the dataset to.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 32.
        plot_path (str | None, optional): The path to save the plot. Defaults to None.
        num_workers (int, optional): The number of workers for the data loaders. Defaults to 0.
        val_split (float, optional): Percentage of training data to use for validation (0.0-1.0). Defaults to 0.2.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        dsid: str,
        extract_path: str | pathlib.Path,
        batch_size: int = 32,
        plot_path: pathlib.Path | None = None,
        num_workers: int = 0,
        val_split: float = 0.2,
        logger: logging.Logger | None = None,
        seed: int = 42,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        n_jobs: int = -1,
    ):
        super().__init__()
        self.dsid = dsid
        self.extract_path = pathlib.Path(extract_path) if isinstance(extract_path, str) else extract_path
        self.plot_path = pathlib.Path(plot_path) if plot_path else None
        self.val_split = val_split
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.max_len = -1
        self.num_classes = -1
        self.num_dimensions = -1
        self.logger = logger
        self.seed = seed

        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.persistent_workers = persistent_workers if num_workers > 0 else False
        self.n_jobs = n_jobs

    def setup(self, stage: str):
        """Set up the data for training and testing."""
        # Assign train/val datasets for use in dataloaders
        super(UCRDataModule, self).setup(stage=stage)

        if self.train_dataset is None and self.test_dataset is None:
            (
                train_dataset,
                self.test_dataset,
                self.max_len,
                self.num_classes,
                self.num_dimensions,
            ) = get_ucr_datasets(
                dsid=self.dsid,
                extract_path=self.extract_path,
                plot_path=self.plot_path,
                n_jobs=self.n_jobs,
            )

            # Calculate validation split sizes
            train_size = int((1 - self.val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            if self.logger:
                self.logger.info(
                    f"Splitting training data: {train_size} samples for training, {val_size} samples for validation"
                )

            # Create validation split
            train_subset, val_subset = random_split(
                dataset=train_dataset,
                lengths=[train_size, val_size],
                # Fixed seed for reproducibility
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.train_dataset = train_subset
            self.val_dataset = val_subset

    def train_dataloader(self):
        """Get the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        """Get the validation data loader."""
        # Use the test dataset as validation dataset
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        """Get the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
