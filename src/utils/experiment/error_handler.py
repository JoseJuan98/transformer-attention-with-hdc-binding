# -*- coding: utf-8 -*-
"""This module defines the ErrorHandler class for experiments."""

# Standard imports
import logging
import pathlib
import traceback
from typing import Any, Dict, Optional


class ErrorHandler:
    """Handles logging and recording of errors during experiments."""

    def __init__(self, logger: logging.Logger, error_log_path: pathlib.Path, development_mode: bool):
        """Initializes the ErrorHandler.

        Args:
            logger (logging.Logger): The logger instance to use for logging errors.
            error_log_path (pathlib.Path): The path to the file where errors should be logged.
            development_mode (bool): Flag indicating if running in development mode (re-raises exceptions).
        """
        self.logger = logger
        self.error_log_path = error_log_path
        self.development_mode = development_mode
        # Ensure the directory for the error log exists
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.errors: Dict[str, list] = {}

    def handle_error(self, dataset: str, exception: Exception, run: int, model_name: Optional[str] = None) -> None:
        """Handles, logs, and formats an error that occurred during the experiment.

        Args:
            dataset (str): The dataset name where the error occurred.
            exception (Exception): The exception that was caught.
            model_name (Optional[str]): The model name if the error is model-specific. Defaults to None.
            run (Optional[int]): The run number if the error is model-specific. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing structured information about the error.

        Raises:
            Exception: Re-raises the original exception if development_mode is True.
        """
        # Determine context and create a unique key for the error
        if model_name:
            err_banner = f"t{f'{"x" * 24}'f' {dataset} | {model_name} | Run {run} 'f'{"x" * 24}': ^100}"
            err_context = f"Error for {dataset} training {model_name} in run {run}"
            # Use a more structured key
            err_key = model_name

        else:
            err_banner = f"{f'{"x" * 24}'f' {dataset} 'f'{"x" * 24}': ^100}"
            err_context = f"Error loading or preparing {dataset}"
            # Dataset name as key for dataset-level errors
            err_key = dataset

        # Format log messages
        err_msg = f"\n\n\t{err_banner: ^100}\n\n"
        err_msg += f"{err_context}:\n\n{str(exception)}\n\n"
        tb_msg = f"Traceback:\n{traceback.format_exc()}"

        # Log messages using the provided logger
        self.logger.error(err_msg)
        self.logger.error(tb_msg)

        # Prepare structured error data to be returned
        error_dict: dict[str, Any] = {
            "dataset": dataset,
            "error": str(exception),
            "traceback": traceback.format_exc(),
        }
        if model_name:
            error_dict["model_name"] = model_name
            error_dict["run"] = run

        # Update the errors dictionary
        errors = self.errors.get(err_key, [])

        if any(e["error"] == error_dict["error"] for e in errors):
            self.logger.warning(f"Not writing duplicated error: {error_dict['error']}")

        else:
            errors.append(error_dict)
            self.errors[err_key] = errors

            # Append to the dedicated error log file
            try:
                with open(self.error_log_path, mode="a", encoding="utf-8") as f:
                    f.write(err_msg)
                    f.write(tb_msg)
                    # Separator for readability
                    f.write("\n" + "=" * 80 + "\n")

            except IOError as log_err:
                self.logger.error(f"Failed to write to error log file {self.error_log_path}: {log_err}")

        # Handle development mode: re-raise the exception to stop execution
        if self.development_mode:
            self.logger.error("[DEVELOPMENT MODE ON] Terminating run after error.")
            raise exception
