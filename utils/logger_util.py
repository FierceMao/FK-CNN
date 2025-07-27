"""
File: logger_util.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: primary logging utility for the application.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_file: str = "logs/app.log", level: int = logging.INFO) -> logging.Logger:
    """set a logger with console and file handlers.

    Args:
        name (str): logger name, usually is __name__
        log_file (str): the path to the log file
        level (int): logging level (e.g. logging.INFO)

    Returns:
        logging.Logger: configured logger instance
    """
    # make sure the log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # define format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

    # console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # file Handler with rotation
    # maxBytes=5MB, backupCount=3 means it keeps 3 old log files
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
