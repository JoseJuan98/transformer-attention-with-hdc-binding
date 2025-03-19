# -*- coding: utf-8 -*-
"""Logger utils.

This module contains the logger utilities to be used in the project.
"""

# Standard imports
import logging
import pathlib
from typing import Literal

# First party imports
from utils.config import Config


def get_logger(
    name: str = "main",
    level: int = logging.INFO,
    log_filename: str | pathlib.Path | None = None,
    propagate: bool | None = None,
    log_file_mode: Literal["w", "a"] = "w",
) -> logging.Logger:
    """Get the main logger.

    Args:
        name (str, optional): The logger name. Defaults to "main".
        level (int, optional): The logger level. Defaults to logging.INFO.
        log_filename (str, optional): The log file within the `utils.Config.log_dir`. Defaults to None.
        propagate (bool, optional): Whether to propagate the logs to the root logger. Defaults to False.
        log_file_mode (Literal["w", "a"], optional): The log file mode. Defaults to "w".
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=level)

    if propagate is not None:
        logger.propagate = propagate

    if not logger.handlers:

        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

        if log_filename is not None:
            log_file = Config.log_dir / log_filename

            # create the logs dir if it doesn't exist
            log_file.parent.mkdir(parents=True, exist_ok=True)

            logger.addHandler(
                logging.FileHandler(filename=log_file, mode=log_file_mode),
            )

    return logger


def msg_task(msg: str, logger: logging.Logger | None = None) -> None:
    """Print the title message for a task.

    Args:
        msg (str): The message to print.
        logger (logging.Logger, optional): The logger to use. Defaults to None, so it creates a new one.
    """
    if logger is None:
        logger = get_logger()

    logger.info(f"\n\n{f' {msg} ':_^100}\n\n")
