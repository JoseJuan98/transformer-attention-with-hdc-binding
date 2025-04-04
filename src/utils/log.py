# -*- coding: utf-8 -*-
"""Logger utils.

This module contains the logger utilities to be used in the project.
"""

# Standard imports
import logging
import pathlib
from typing import Dict, Literal, Optional

# First party imports
from utils.config import Config


class ExpLogger(logging.Logger):
    """A logger class with support for component-specific handlers that extends logging.Logger."""

    # Class variable to keep track of created loggers
    _instances: Dict[str, "ExpLogger"] = {}

    @classmethod
    def get_instance(
        cls,
        name: str = "main",
        level: int = logging.INFO,
        log_filename: Optional[str | pathlib.Path] = None,
        propagate: Optional[bool] = None,
        log_file_mode: Literal["w", "a"] = "w",
    ) -> "ExpLogger":
        """Get or create a logger instance.

        Args:
            name (str, optional): The logger name. Defaults to "main".
            level (int, optional): The logger level. Defaults to logging.INFO.
            log_filename (str | pathlib.Path, optional): The log file within the `utils.Config.log_dir`. Defaults to None.
            propagate (bool, optional): Whether to propagate the logs to the root logger. Defaults to None.
            log_file_mode (Literal["w", "a"], optional): The log file mode. Defaults to "w".

        Returns:
            ExpLogger: A Logger instance.
        """
        if name in cls._instances:
            return cls._instances[name]

        # Create a new logger instance
        instance = cls(name)
        instance.setup(level, log_filename, propagate, log_file_mode)
        cls._instances[name] = instance
        return instance

    def __init__(self, name: str):
        """Initialize the logger.

        Args:
            name (str): The logger name.
        """
        super().__init__(name)
        self.component_handlers: Dict[str, logging.Handler] = {}

    def setup(
        self,
        level: int = logging.INFO,
        log_filename: Optional[str | pathlib.Path] = None,
        propagate: Optional[bool] = None,
        log_file_mode: Literal["w", "a"] = "w",
    ) -> None:
        """Set up the logger with handlers.

        Args:
            level (int, optional): The logger level. Defaults to logging.INFO.
            log_filename (str | pathlib.Path, optional): The log file within the `utils.Config.log_dir`. Defaults to None.
            propagate (bool, optional): Whether to propagate the logs to the root logger. Defaults to None.
            log_file_mode (Literal["w", "a"], optional): The log file mode. Defaults to "w".
        """
        self.setLevel(level)

        if propagate is not None:
            self.propagate = propagate

        # Set up handlers if they don't exist
        if not self.handlers:
            formatter = logging.Formatter(fmt="[%(levelname)s] %(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

            # Add console handler
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.addHandler(stream_handler)

            # Add file handler if specified
            if log_filename is not None:
                log_file = Config.log_dir / log_filename

                # Create the logs dir if it doesn't exist
                log_file.parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.FileHandler(filename=log_file, mode=log_file_mode)
                file_handler.setFormatter(formatter)
                self.addHandler(file_handler)

    def add_component_handler(
        self,
        component_name: str,
        log_filename: str | pathlib.Path,
        level: int = logging.DEBUG,
        log_file_mode: Literal["w", "a"] = "w",
    ) -> None:
        """Add a component-specific file handler to the logger.

        Args:
            component_name (str): Name of the component (e.g., 'model1_training').
            level (int, optional): The log level for this component. Defaults to logging.DEBUG.
            log_file_mode (Literal["w", "a"], optional): The log file mode. Defaults to "w".
            log_filename (str | pathlib.Path, optional): The log file within the `utils.Config.log_dir`. Defaults to
                None.
        """
        if isinstance(log_filename, str):
            log_filename = pathlib.Path(log_filename)

        # Create component log directory if it doesn't exist
        log_filename.parent.mkdir(parents=True, exist_ok=True)

        # Create formatter
        formatter = logging.Formatter(fmt="[%(levelname)s] %(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Create component-specific file handler
        component_handler = logging.FileHandler(filename=log_filename, mode=log_file_mode)
        component_handler.setLevel(level)
        component_handler.setFormatter(formatter)

        # Add handler to logger
        self.addHandler(component_handler)

        # Store the handler reference
        self.component_handlers[component_name] = component_handler

    def remove_component_handler(self, component_name: str) -> None:
        """Remove a component-specific handler by name.

        Args:
            component_name (str): The name of the component handler to remove.
        """
        if component_name in self.component_handlers:
            handler = self.component_handlers[component_name]
            self.removeHandler(handler)
            handler.close()
            del self.component_handlers[component_name]

    def remove_all_component_handlers(self) -> None:
        """Remove all component-specific handlers."""
        for component_name in list(self.component_handlers.keys()):
            self.remove_component_handler(component_name)

    def msg_task(self, msg: str) -> None:
        """Print the title message for a task.

        Args:
            msg (str): The message to print.
        """
        self.info(f"\n\n{f' {msg} ':_^100}\n\n")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up component handlers."""
        self.remove_all_component_handlers()


# Register the custom ExpLogger class with the logging module
logging.setLoggerClass(ExpLogger)


# For backward compatibility
def get_logger(
    name: str = "main",
    level: int = logging.INFO,
    log_filename: Optional[str | pathlib.Path] = None,
    propagate: Optional[bool] = None,
    log_file_mode: Literal["w", "a"] = "w",
) -> ExpLogger:
    """Get a logger instance.

    Args:
        name (str, optional): The logger name. Defaults to "main".
        level (int, optional): The logger level. Defaults to logging.INFO.
        log_filename (str | pathlib.Path, optional): The log file within the `utils.Config.log_dir`. Defaults to None.
        propagate (bool, optional): Whether to propagate the logs to the root logger. Defaults to None.
        log_file_mode (Literal["w", "a"], optional): The log file mode. Defaults to "w".

    Returns:
        ExpLogger: A Logger instance.
    """
    return ExpLogger.get_instance(
        name=name,
        level=level,
        log_filename=log_filename,
        propagate=propagate,
        log_file_mode=log_file_mode,
    )


def msg_task(msg: str, logger: Optional[ExpLogger] = None) -> None:
    """Print the title message for a task.

    Args:
        msg (str): The message to print.
        logger (Logger, optional): The logger to use. Defaults to None, so it creates a new one.
    """
    if logger is None:
        logger = get_logger()

    logger.msg_task(msg)
