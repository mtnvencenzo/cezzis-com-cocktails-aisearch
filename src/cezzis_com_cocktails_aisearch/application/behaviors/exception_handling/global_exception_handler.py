import logging
import sys


def global_exception_handler(exc_type, exc_value, tb):
    """
    Custom global exception handler function.
    Logs the error details before calling the default hook.
    """

    logger = logging.getLogger("global_exception_handler")

    logger.exception("An unhandled exception occurred", exc_info=(exc_type, exc_value, tb))

    # Call the default exception handler to maintain normal program termination
    sys.__excepthook__(exc_type, exc_value, tb)
