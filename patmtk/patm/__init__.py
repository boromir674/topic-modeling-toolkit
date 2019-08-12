from os import path
import logging

from .dataset import TextDataset

logging.config.fileConfig(path.join(path.dirname(path.realpath(__file__)), 'logging.ini'), disable_existing_loggers=True)

logger = logging.getLogger(__name__)
