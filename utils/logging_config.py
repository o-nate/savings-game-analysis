"""Configurations for app logging"""

import logging
import logging.config
import os
from datetime import datetime
from typing import Optional, Type
import warnings

LOG_DIR = "logs"

# Generate a unique log filename with timestamp
LOG_FILENAME = os.path.join(
    LOG_DIR, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)

LOG_FORMAT_STANDARD = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_ERROR = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

SUPPRESSED_THIRD_PARTY_LOGS = ["dotenv", "matplotlib", "pandas", "tqdm"]

# Logging configuration dictionary
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": LOG_FORMAT_STANDARD,
            "datefmt": DATE_FORMAT,
        },
        "error": {
            "format": LOG_FORMAT_ERROR,
            "datefmt": DATE_FORMAT,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "DEBUG",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": LOG_FILENAME,
            "formatter": "error",
            "level": "DEBUG",
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "matplotlib": {
            "handlers": ["console", "file"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)


def suppress_third_party_warnings(
    libraries: list[str], warning_categories: Optional[list[type]] = None
) -> None:
    """Suppress warnings and set logging level for specified third party libraries.

    Args:
        libraries: List of library names to suppress logging for
        warning_categories: Optional list of warning categories to suppress.
            Defaults to [UserWarning, RuntimeWarning] if None provided.

    Returns:
        None

    Example:
        >>> suppress_third_party_warnings(
        ...     libraries=['matplotlib', 'pandas'],
        ...     warning_categories=[UserWarning, RuntimeWarning, DeprecationWarning]
        ... )
    """
    # Set default warning categories if none provided
    if warning_categories is None:
        warning_categories = [UserWarning, RuntimeWarning, DeprecationWarning]

    # Suppress specified warning categories
    for category in warning_categories:
        warnings.filterwarnings("ignore", category=category)

    # Set logging level to CRITICAL for all specified libraries
    for library in libraries:
        logger = logging.getLogger(library)
        logger.setLevel(logging.CRITICAL)


def get_logger(name: str) -> Type[logging.Logger]:
    """Get logger for module

    Args:
        name (str): Module name, normally __name__

    Returns:
        Type[logging.Logger]: Module's logger object
    """
    return logging.getLogger(name)


# Suppress warnings on import
suppress_third_party_warnings(libraries=SUPPRESSED_THIRD_PARTY_LOGS)
