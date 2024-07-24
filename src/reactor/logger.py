import logging
import os
import sys
import warnings
from contextlib import contextmanager

from . import settings

settings.set_up_logging()


# noinspection PyPep8Naming
def getLogger(name, suppress=False):
    if settings.SUPPRESS_LOGGER is not None:
        suppress = settings.SUPPRESS_LOGGER
    return logging.getLogger(name=name) if not suppress else Logger(name)


# noinspection PyPep8Naming
def getPersistentLogger(name):
    return logging.getLogger(name=name)


class Logger(object):
    # noinspection PyUnusedLocal
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass


@contextmanager
def suppress_output(warnings_: bool = True, logs_: bool = True, prints_: bool = True):
    original_stdout = sys.stdout

    if warnings_:
        # Suppress warnings
        warnings.simplefilter("ignore")

    if logs_:
        # Suppress logging
        logging.disable(logging.CRITICAL)

    if prints_:
        # Suppress print statements by redirecting stdout
        sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        if prints_:
            # Restore stdout and close the file
            sys.stdout.close()
            sys.stdout = original_stdout

        if logs_:
            # Restore logging
            logging.disable(logging.NOTSET)

        if warnings_:
            # Restore warnings
            warnings.simplefilter("default")
