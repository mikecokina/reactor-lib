import logging
import os
import sys
import warnings
from contextlib import contextmanager

from .conf.settings import settings

settings.set_up_logging()


# noinspection PyPep8Naming
def getLogger(name, suppress=False):
    if suppress:
        return Logger(name)
    if settings.SUPPRESS_LOGGER is not None:
        suppress = settings.SUPPRESS_LOGGER
    return logging.getLogger(name=name) if not suppress else Logger(name)


class CustomLogger(object):
    _instance = None

    SUPPRESS_LOGGER: bool = False
    LOGGER = getLogger('reactorlib', suppress=SUPPRESS_LOGGER)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CustomLogger, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance

    def get(self):
        return self._instance

    @classmethod
    def configure(cls, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(cls, key):
                raise ValueError("You are about to set configuration which doesn't exist")
            setattr(cls, key, value)

            if key == 'SUPPRESS_LOGGER':
                cls.LOGGER = getLogger('reactorlib', suppress=cls.SUPPRESS_LOGGER)

    @property
    def logger(self):
        return self.LOGGER

    def info(self, *args, **kwargs):
        return self.logger.info(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self.logger.error(*args, **kwargs)

    def exception(self, *args, **kwargs):
        return self.logger.exception(*args, **kwargs)

    def debug(self, *args, **kwargs):
        return self.logger.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self.logger.warning(*args, **kwargs)

    def warn(self, *args, **kwargs):
        return self.logger.warn(*args, **kwargs)


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

    def exception(self, *args, **kwargs):
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


logger = CustomLogger()
