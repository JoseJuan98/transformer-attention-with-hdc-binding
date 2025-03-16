# -*- coding: utf-8 -*-
"""Dataset module for time series classification experiments."""
# Standard imports
import logging
import pathlib

# Third party imports
import numpy
import pandas
import torch
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_UCR_UEA_dataset
from torch.utils.data import TensorDataset


def __convert_to_numpy(data: pandas.DataFrame) -> numpy.ndarray:
    """Converts a sktime DataFrame to a numpy array.

    sktime returns pandas DataFrames. The data needs to be coverted to numpy arrays and then to PyTorch tensors. Also,
    the multivariate case needs to be handled correctly. sktime stores each dimension/variable of the time series in a
    separate column of the DataFrame, and each cell contains a pandas Series representing the time series for that
    dimension.
    """
    num_cases = len(data)
    num_dimensions = len(data.columns)
    max_len = max(len(data.iloc[i, j]) for i in range(num_cases) for j in range(num_dimensions))

    # Initialize an array to hold the data.  Shape: (num_cases, max_len, num_dimensions)
    arr = numpy.zeros((num_cases, max_len, num_dimensions))

    for i in range(num_cases):
        for j in range(num_dimensions):
            # Pad the series with zeros to the max_len
            series = data.iloc[i, j].to_numpy()
            arr[i, : len(series), j] = series

    return arr


def get_ucr_datasets(
    dsid: str, extract_path: pathlib.Path, logger: logging.Logger | None = None
) -> tuple[TensorDataset, TensorDataset, int, int]:
    """Loads and standardizes a UCR dataset using sktime.

    Args:
        dsid (str): The name of the UCR dataset.
        extract_path (pathlib.Path): The path to extract the dataset to.

    Returns:
        tuple[TensorDataset, TensorDataset, int, int]: A tuple containing the
            training dataset, testing dataset, maximum sequence length, and
            number of classes.
    """
    # Load data using sktime
    extract_path.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_UCR_UEA_dataset(dsid, split="train", return_X_y=True, extract_path=extract_path)
    X_test, y_test = load_UCR_UEA_dataset(dsid, split="test", return_X_y=True, extract_path=extract_path)

    X_train = __convert_to_numpy(X_train)
    X_test = __convert_to_numpy(X_test)

    # Standardize data using sklearn.preprocessing.StandardScaler
    # Each feature needs to be standarized independently. This means the data needs to be reshaped to
    # (num_cases * max_len, num_dimensions), standardize, and then reshape back.
    num_cases_train, max_len, num_dimensions = X_train.shape
    # max_len and num_dimensions should be the same
    num_cases_test, _, _ = X_test.shape

    scaler = StandardScaler()
    X_train = X_train.reshape(num_cases_train * max_len, num_dimensions)
    X_train = scaler.fit_transform(X_train)
    X_train = torch.from_numpy(X_train.reshape(num_cases_train, max_len, num_dimensions)).float()

    X_test = X_test.reshape(num_cases_test * max_len, num_dimensions)
    X_test = scaler.transform(X_test)  # Use the same scaler fitted on training data
    X_test = torch.from_numpy(X_test.reshape(num_cases_test, max_len, num_dimensions)).float()

    # Convert y to numerical labels and tensors
    unique_labels = numpy.unique(numpy.concatenate((y_train, y_test)))
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    y_train = torch.tensor([label_mapping[label] for label in y_train], dtype=torch.long)
    y_test = torch.tensor([label_mapping[label] for label in y_test], dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    num_classes = len(torch.unique(y_train))

    if logger is not None:
        logger.info(f"Loaded {dsid} dataset to path: {extract_path}")
        logger.info(f"  Number of classes: {num_classes}")
        logger.info(f"  Number of training samples: {len(train_dataset)}")
        logger.info(f"  Number of testing samples: {len(test_dataset)}")
        logger.info(f"  Maximum sequence length: {max_len}")

    return train_dataset, test_dataset, max_len, num_classes
