import logging

from numpy.random import RandomState


class LoggerMixin:
    def __init__(self, *args, log_level=logging.INFO, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = self.getLogger()
        logging.basicConfig(level=logging.DEBUG)
        self.logger.setLevel(log_level)

    @classmethod
    def getLogger(cls):
        return logging.getLogger(cls.__name__)


class RandomStateMixin:
    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.random = RandomState(seed)
