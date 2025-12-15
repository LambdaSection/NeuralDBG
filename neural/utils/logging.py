"""
Centralized logging configuration for Neural DSL.

This module provides a consistent logging interface across the entire framework
with proper configuration, formatters, and utilities.
"""
from __future__ import annotations

from enum import Enum
import logging
from pathlib import Path
import sys
from typing import List, Optional, Union


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def get_logger(name: str, level: Optional[Union[str, int, LogLevel]] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__ of the module)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        if isinstance(level, LogLevel):
            level = level.value
        elif isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        
        formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def setup_logging(
    level: Union[str, int, LogLevel] = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure global logging settings for the Neural DSL framework.
    
    Args:
        level: Global log level
        log_file: Optional file path to write logs to
        format_string: Custom format string for log messages
    
    Example:
        >>> setup_logging(LogLevel.DEBUG, log_file='neural.log')
    """
    if isinstance(level, LogLevel):
        level = level.value
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    handlers: List[logging.Handler] = []
    
    console_handler = logging.StreamHandler(sys.stderr)
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    console_formatter = ColoredFormatter(
        fmt=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            fmt=format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )


def set_log_level(logger: Union[str, logging.Logger], level: Union[str, int, LogLevel]) -> None:
    """
    Set log level for a specific logger.
    
    Args:
        logger: Logger instance or logger name
        level: Desired log level
    
    Example:
        >>> set_log_level('neural.parser', LogLevel.DEBUG)
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    
    if isinstance(level, LogLevel):
        level = level.value
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)


def disable_logging(logger_name: Optional[str] = None) -> None:
    """
    Disable logging for a specific logger or all loggers.
    
    Args:
        logger_name: Logger name to disable (None to disable all)
    
    Example:
        >>> disable_logging('neural.parser')
    """
    if logger_name:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)
    else:
        logging.disable(logging.CRITICAL)


def enable_logging(logger_name: Optional[str] = None, level: Union[str, int, LogLevel] = LogLevel.INFO) -> None:
    """
    Enable logging for a specific logger or all loggers.
    
    Args:
        logger_name: Logger name to enable (None to enable all)
        level: Log level to set
    
    Example:
        >>> enable_logging('neural.parser', LogLevel.DEBUG)
    """
    if logger_name:
        set_log_level(logger_name, level)
    else:
        logging.disable(logging.NOTSET)


class LogContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, logger: Union[str, logging.Logger], level: Union[str, int, LogLevel]):
        """
        Initialize log context.
        
        Args:
            logger: Logger instance or name
            level: Temporary log level
        """
        self.logger = logging.getLogger(logger) if isinstance(logger, str) else logger
        self.new_level = level
        self.old_level = self.logger.level
    
    def __enter__(self) -> logging.Logger:
        """Enter context and set new log level."""
        set_log_level(self.logger, self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore old log level."""
        self.logger.setLevel(self.old_level)


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        logger: Logger instance
        level: Log level for the messages
    
    Example:
        >>> logger = get_logger(__name__)
        >>> @log_function_call(logger)
        ... def my_function(x, y):
        ...     return x + y
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log(level, f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} returned {result}")
                return result
            except Exception as e:
                logger.exception(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        return wrapper
    return decorator
