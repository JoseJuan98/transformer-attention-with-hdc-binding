# -*- coding: utf-8 -*-
"""Dataset module for time series classification experiments."""
# Standard imports
import logging
import pathlib

# Third party imports
import numpy
import pandas
import torch
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_UCR_UEA_dataset
from torch.utils.data import TensorDataset
from tqdm import tqdm


def __convert_to_numpy(data: pandas.DataFrame) -> numpy.ndarray:
    """Converts a sktime DataFrame to a numpy array.

    sktime returns pandas DataFrames which needs to be coverted to numpy arrays and then to PyTorch tensors. Also,
    the multivariate case needs to be handled correctly. sktime stores each dimension/variable of the time series in a
    separate column of the DataFrame, and each cell contains a pandas Series representing the time series for that
    dimension.
    """
    num_samples = len(data)
    num_dimensions = len(data.columns)
    max_len = max(len(data.iloc[i, j]) for i in range(num_samples) for j in range(num_dimensions))

    # Initialize an array to hold the data.  Shape: (num_cases, max_len, num_dimensions)
    arr = numpy.zeros((num_samples, max_len, num_dimensions))

    for i in tqdm(iterable=range(num_samples), desc="Converting to numpy", unit=" samples"):
        for j in range(num_dimensions):
            # Pad the series with zeros to the max_len
            series = data.iloc[i, j].to_numpy()
            arr[i, : len(series), j] = series

    return arr


def get_ucr_datasets(
    dsid: str,
    extract_path: pathlib.Path,
    logger: logging.Logger | None = None,
    plot_path: pathlib.Path | None = None,
) -> tuple[TensorDataset, TensorDataset, int, int, int]:
    """Loads and standardizes a UCR dataset using sktime.

    Args:
        dsid (str): The name of the UCR dataset.
        extract_path (`pathlib.Path`): The path to extract the dataset to.
        logger (`logging.Logger`, optional): The logger to use. Defaults to None.
        plot_path (`pathlib.Path`, optional): The path to save the plot. Defaults to None.

    Returns:
        ~`torch.utils.data.TensorDataset`: The training dataset.
        TensorDataset: The testing dataset.
        int: The maximum sequence length.
        int: The number of classes in the dataset.
        int: The number of dimensions in the dataset.
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

    if plot_path is not None and num_dimensions >= 1:
        _plot_time_series_sample(dsid=dsid, plot_path=plot_path, sample=X_train[0], num_dimensions=num_dimensions)

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
        logger.info(f"  Number of dimensions: {num_dimensions}")

    return train_dataset, test_dataset, max_len, num_classes, num_dimensions


def _plot_time_series_sample(
    dsid: str, sample: numpy.ndarray, num_dimensions: int, plot_path: pathlib.Path | None = None
):
    pyplot.figure(figsize=(16, 9))

    # plot all dimensions in a heatmap
    data_row = sample
    if data_row.shape[0] >= data_row.shape[1]:
        data_row = data_row.T

    pyplot.imshow(data_row)
    pyplot.title(f"{dsid} Sample")
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
    pyplot.show()

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        pyplot.savefig(plot_path.parent / plot_path.name.replace(".png", "_line_plot.png"))

    pyplot.close()
