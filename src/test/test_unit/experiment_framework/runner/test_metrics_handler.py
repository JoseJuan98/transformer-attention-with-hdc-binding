# -*- coding: utf-8 -*-
"""Unit tests for the MetricsHandler class."""
# Pytest imports
import pytest

# Standard imports
import logging
import os
import pathlib
import tempfile
import unittest

# Third party imports
import numpy
import pandas

# First party imports
from experiment_framework.runner.metrics_handler import MetricsHandler


class TestGetTrainMetricsAndPlot(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_dir = self.temp_dir.name
        self.plots_dir = pathlib.Path(self.csv_dir) / "plots"
        self.plots_path = self.plots_dir / "test_plot.png"

        # Create a dummy model file for size calculation
        self.model_path = pathlib.Path(self.csv_dir) / "model.pth"
        with open(self.model_path, "wb") as f:
            f.write(os.urandom(1024 * 500))  # Create a dummy 500KB file
        self.expected_size_mb = round(self.model_path.stat().st_size / (1024**2), 4)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    def _run_test(self, df_data: dict, expected_metrics: dict):
        """Helper function to run the test."""
        metrics_path = pathlib.Path(self.csv_dir) / "metrics.csv"
        df = pandas.DataFrame(df_data)
        df.to_csv(metrics_path, index=False)

        handler = MetricsHandler(metrics_path=self.csv_dir)

        result_df = handler.get_train_metrics_and_plot(
            csv_dir=self.csv_dir,
            experiment="test_exp",
            logger=logging.getLogger("TestLogger"),
            plots_path=self.plots_path,
            show_plot=False,
        )

        # Prepare expected DataFrame
        expected_df = pandas.DataFrame([expected_metrics])
        # Ensure columns are in the same order and have same types (especially NaN handling)
        expected_df = expected_df[result_df.columns]  # Match column order
        expected_df = expected_df.astype(result_df.dtypes.to_dict())  # Match types

        pandas.testing.assert_frame_equal(result_df, expected_df)

    def test_case1_no_val_acc_column(self):
        """Test metrics.csv without a 'val_acc' column."""
        data = {
            "epoch": [0, 1, 2],
            "train_loss": [1.0, 0.8, 0.6],
            "train_acc": [0.5, 0.6, 0.7],
            # "val_acc": MISSING
            "val_loss": [0.9, 0.7, 0.5],
            "test_loss": [numpy.nan, numpy.nan, 0.55],  # Test loss often logged only at end
            "test_acc": [numpy.nan, numpy.nan, 0.75],  # Test acc often logged only at end
        }
        expected = {
            "train_loss": numpy.nan,
            "train_acc": numpy.nan,
            "val_loss": numpy.nan,
            "val_acc": numpy.nan,  # Expected default NaN
            "test_loss": 0.5500,  # Mean of non-NaN test_loss
            "test_acc": 0.7500,  # Mean of non-NaN test_acc
            "size_MB": self.expected_size_mb,
        }
        self._run_test(data, expected)

    def test_case2_val_acc_all_nan(self):
        """Test metrics.csv with 'val_acc' column containing only NaN."""
        data = {
            "epoch": [0, 1, 2],
            "train_loss": [1.0, 0.8, 0.6],
            "train_acc": [0.5, 0.6, 0.7],
            "val_acc": [numpy.nan, numpy.nan, numpy.nan],  # All NaN
            "val_loss": [0.9, 0.7, 0.5],
            "test_loss": [numpy.nan, numpy.nan, 0.55],
            "test_acc": [numpy.nan, numpy.nan, 0.75],
        }
        expected = {
            "train_loss": numpy.nan,
            "train_acc": numpy.nan,
            "val_loss": numpy.nan,
            "val_acc": numpy.nan,  # Expected default NaN
            "test_loss": 0.5500,
            "test_acc": 0.7500,
            "size_MB": self.expected_size_mb,
        }
        self._run_test(data, expected)

    def test_case3_empty_dataframe_after_drops(self):
        """Test metrics.csv that becomes empty after dropping test columns."""
        # This simulates a case where only test metrics were logged
        data = {
            "epoch": [0, 1, 2],
            "test_loss": [0.6, 0.55, 0.5],
            "test_acc": [0.7, 0.75, 0.8],
        }
        expected = {
            "train_loss": numpy.nan,
            "train_acc": numpy.nan,
            "val_loss": numpy.nan,
            "val_acc": numpy.nan,  # Expected default NaN
            "test_loss": 0.5500,  # Mean of test_loss
            "test_acc": 0.7500,  # Mean of test_acc
            "size_MB": self.expected_size_mb,
        }
        self._run_test(data, expected)

    def test_case4_normal_operation(self):
        """Test normal operation with a valid 'val_acc' column."""
        data = {
            "epoch": [0, 1, 2, 3],
            "train_loss": [1.0, 0.8, 0.6, 0.4],
            "train_acc": [0.5, 0.6, 0.7, 0.8],
            "val_acc": [0.55, 0.65, 0.78, 0.75],  # Max val_acc is 0.78 at epoch 2
            "val_loss": [0.9, 0.7, 0.5, 0.6],
            "test_loss": [numpy.nan, numpy.nan, numpy.nan, 0.55],
            "test_acc": [numpy.nan, numpy.nan, numpy.nan, 0.76],
        }
        # Expected metrics are from epoch 2 (where val_acc is max)
        expected = {
            "train_loss": 0.6000,
            "train_acc": 0.7000,
            "val_loss": 0.5000,
            "val_acc": 0.7800,
            "test_loss": 0.5500,  # Mean of non-NaN test_loss
            "test_acc": 0.7600,  # Mean of non-NaN test_acc
            "size_MB": self.expected_size_mb,
        }
        self._run_test(data, expected)

    def test_case5_metrics_file_not_found(self):
        """Test behavior when metrics.csv does not exist."""

        with pytest.raises(FileNotFoundError):
            _ = MetricsHandler.get_train_metrics_and_plot(
                csv_dir=self.csv_dir,
                experiment="test_exp_no_file",
                logger=None,
                plots_path=self.plots_path,
                show_plot=False,
            )
