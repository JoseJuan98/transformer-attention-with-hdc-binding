# -*- coding: utf-8 -*-
"""Dataset modules for loading and preparing the time series classification data for experiments."""
# Standard imports
import pathlib

# Third party imports
import numpy
import pandas
import torch
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_UCR_UEA_dataset
from torch.utils.data import TensorDataset
from tqdm import tqdm

# First party imports
from experiment_framework.data.visualization import _plot_time_series_sample


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

    # Distribute the number of jobs across the available cores or the number of samples if less than the number of cores
    X_train = __convert_to_numpy(X_train, n_jobs=min(n_jobs, X_train.shape[0]))
    X_test = __convert_to_numpy(X_test, n_jobs=min(n_jobs, X_test.shape[0]))

    # Standardize data using sklearn.preprocessing.StandardScaler
    # Each feature needs to be standarized independently. This means the data needs to be reshaped to
    # (num_cases * max_len, num_dimensions), standardize, and then reshape back.
    num_cases_train, max_len_train, num_dimensions = X_train.shape

    # max_len and num_dimensions should be the same
    num_cases_test, max_len_test, _ = X_test.shape

    if plot_path is not None and num_dimensions >= 1:
        _plot_time_series_sample(dsid=dsid, plot_path=plot_path, sample=X_train[0], num_dimensions=num_dimensions)

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
