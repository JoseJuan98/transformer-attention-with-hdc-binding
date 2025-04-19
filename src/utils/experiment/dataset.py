# -*- coding: utf-8 -*-
"""Dataset module for time series classification experiments."""
# Standard imports
import logging
import pathlib

# Third party imports
import lightning
import numpy
import pandas
import torch
from joblib import Parallel, delayed
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_UCR_UEA_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


def __convert_row_univariate(series: pandas.Series, max_len: int) -> numpy.ndarray:
    """Converts a sktime univariate DataFrame to a numpy array.

    sktime returns pandas DataFrames which needs to be coverted to numpy arrays and then to PyTorch tensors. Also,
    the multivariate case needs to be handled correctly. sktime stores each dimension/variable of the time series in a
    separate column of the DataFrame, and each cell contains a pandas Series representing the time series for that
    dimension.

    Args:
        series (pandas.Series): Time series to convert.
        max_len (int): The maximum length of the time series.

    Returns:
        numpy.ndarray: The converted numpy array.
    """
    arr = numpy.zeros((max_len, 1))
    s = series.to_numpy()
    arr[: len(s), 0] = s
    return arr


def __convert_row_multivariate(row: pandas.DataFrame, num_dimensions: int, max_len: int) -> numpy.ndarray:
    """Converts a sktime multivariate DataFrame to a numpy array.

    sktime returns pandas DataFrames which needs to be coverted to numpy arrays and then to PyTorch tensors. Also,
    the multivariate case needs to be handled correctly. sktime stores each dimension/variable of the time series in a
    separate column of the DataFrame, and each cell contains a pandas Series representing the time series for that
    dimension.

    Args:
        row (pandas.DataFrame): Time series to convert.
        num_dimensions (int): The number of dimensions in the time series.
        max_len (int): The maximum length of the time series.

    Returns:
        numpy.ndarray: The converted numpy array.
    """
    arr = numpy.zeros((max_len, num_dimensions))
    for j in range(num_dimensions):
        s = row.iloc[j].to_numpy()
        arr[: len(s), j] = s
    return arr


def __convert_to_numpy(data: pandas.DataFrame, n_jobs: int = -1) -> numpy.ndarray:
    """Parallelized conversion of sktime DataFrame to numpy array."""
    num_samples = len(data)
    num_dimensions = len(data.columns)

    if num_dimensions == 1:
        max_len = max(len(data.iloc[i, 0]) for i in range(num_samples))
        results = list(
            tqdm(
                Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(__convert_row_univariate)(data.iloc[i, 0], max_len) for i in range(num_samples)
                ),
                total=num_samples,
                desc="Converting to tensor",
                unit=" samples",
            )
        )
        arr = numpy.stack(results, axis=0)
    else:
        max_len = max(len(data.iloc[i, j]) for i in range(num_samples) for j in range(num_dimensions))
        results = list(
            tqdm(
                Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(__convert_row_multivariate)(data.iloc[i, :], num_dimensions, max_len)
                    for i in range(num_samples)
                ),
                total=num_samples,
                desc="Converting to tensor",
                unit=" samples",
            )
        )
        arr = numpy.stack(results, axis=0)
    return arr


def get_ucr_datasets(
    dsid: str,
    extract_path: pathlib.Path | str,
    plot_path: pathlib.Path | None = None,
    n_jobs: int = -1,
) -> tuple[TensorDataset, TensorDataset, int, int, int]:
    """Loads and standardizes a UCR dataset using sktime.

    Datasets source: https://www.timeseriesclassification.com/dataset.php

    Args:
        dsid (str): The name of the UCR dataset.
        extract_path (`pathlib.Path`): The path to extract the dataset to.
        plot_path (`pathlib.Path`, optional): The path to save the plot. Defaults to None.
        n_jobs (int, optional): The number of jobs to use for parallel processing. Defaults to -1 (use all available cores).

    Returns:
        ~`torch.utils.data.TensorDataset`: The training dataset.
        TensorDataset: The testing dataset.
        int: The maximum sequence length.
        int: The number of classes in the dataset.
        int: The number of dimensions in the dataset.
    """
    if isinstance(extract_path, str):
        extract_path = pathlib.Path(extract_path)

    # Load data using sktime
    extract_path.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_UCR_UEA_dataset(dsid, split="train", return_X_y=True, extract_path=extract_path)
    X_test, y_test = load_UCR_UEA_dataset(dsid, split="test", return_X_y=True, extract_path=extract_path)

    X_train = __convert_to_numpy(X_train, n_jobs=n_jobs)
    X_test = __convert_to_numpy(X_test, n_jobs=n_jobs)

    # Standardize data using sklearn.preprocessing.StandardScaler
    # Each feature needs to be standarized independently. This means the data needs to be reshaped to
    # (num_cases * max_len, num_dimensions), standardize, and then reshape back.
    num_cases_train, max_len_train, num_dimensions = X_train.shape

    # max_len and num_dimensions should be the same
    num_cases_test, max_len_test, _ = X_test.shape

    if plot_path is not None and num_dimensions >= 1:
        _plot_time_series_sample(dsid=dsid, plot_path=plot_path, sample=X_train[0], num_dimensions=num_dimensions)

    # TODO: try with scale (-1, 1)
    scaler = StandardScaler(with_std=True, with_mean=True)
    X_train = X_train.reshape(num_cases_train * max_len_train, num_dimensions)
    X_train = scaler.fit_transform(X_train)
    X_train = torch.from_numpy(X_train.reshape(num_cases_train, max_len_train, num_dimensions)).float()

    X_test = X_test.reshape(num_cases_test * max_len_test, num_dimensions)
    X_test = scaler.transform(X_test)
    X_test = torch.from_numpy(X_test.reshape(num_cases_test, max_len_test, num_dimensions)).float()

    # Vectorized label mapping
    unique_labels, y_train_idx = numpy.unique(y_train, return_inverse=True)
    y_test_idx = numpy.searchsorted(unique_labels, y_test)
    y_train = torch.tensor(y_train_idx, dtype=torch.long)
    y_test = torch.tensor(y_test_idx, dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    num_classes = len(unique_labels)

    return train_dataset, test_dataset, max_len_train, num_classes, num_dimensions


def _plot_time_series_sample(
    dsid: str,
    sample: numpy.ndarray,
    num_dimensions: int,
    plot_path: pathlib.Path | None = None,
    show_plot: bool = False,
):
    pyplot.figure(figsize=(16, 9))

    # plot all dimensions in a heatmap
    data_row = sample
    if data_row.shape[0] >= data_row.shape[1]:
        data_row = data_row.T

    pyplot.imshow(data_row)
    pyplot.title(f"{dsid} Sample")

    if show_plot:
        pyplot.show()

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(plot_path)

    pyplot.close()

    # plot dimensions separately in a line plot
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(16, 9))
    for i in range(num_dimensions):
        ax.plot(data_row[:, i], label=f"Dimension {i}")
    pyplot.title(f"{dsid} Sample")
    pyplot.legend()

    if show_plot:
        pyplot.show()

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(plot_path.parent / plot_path.name.replace(".png", "_line_plot.png"))

    pyplot.close()


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
