# -*- coding: utf-8 -*-
"""Data factory for creating the dataset configuration and data loaders based on the dataset name."""
# Standard imports
import logging
import multiprocessing
import pathlib

# Third party imports
from torch.utils.data import DataLoader, random_split

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
        plot_path: pathlib.Path | None = None,
    ) -> tuple[DatasetConfig, DataLoader, DataLoader, DataLoader]:
        """Get the dataset configuration and data loaders based on the dataset name.

        Args:
            dataset_name (str): The name of the dataset.
            logger (logging.Logger): The logger instance.
            extract_path (pathlib.Path): The path to extract the dataset.
            batch_size (int): The batch size for the data loaders.
            plot_path (pathlib.Path | None): [Optional] The path to save the plots.

        Returns:
            DatasetConfig: The dataset configuration.
            torch.utils.data.DataLoader: The training data loader.
            torch.utils.data.DataLoader: The testing data loader.
            torch.utils.data.DataLoader: The validation data loader.
        """
        # TODO: split train between validation and train
        train_dataset, test_dataset, max_len, num_classes, num_channels = get_ucr_datasets(
            dsid=dataset_name,
            extract_path=extract_path,
            logger=logger,
            plot_path=plot_path,
        )

        train_dataset, val_dataset = random_split(
            dataset=train_dataset,
            lengths=[len(train_dataset) - len(test_dataset), len(test_dataset)],
        )

        dataset_cfg = DatasetConfig(
            dataset_name=dataset_name, num_classes=num_classes, input_size=num_channels, context_length=max_len
        )

        cpu_count = multiprocessing.cpu_count()
        num_workers = cpu_count - 2 if cpu_count > 4 else 1

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        return dataset_cfg, train_dataloader, test_dataloader, val_dataloader
