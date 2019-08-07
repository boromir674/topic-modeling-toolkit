from os import path
from .experimental_results import ExperimentalResults
import logging


logging.config.fileConfig(path.join(path.dirname(path.realpath(__file__)), 'logging.ini'), disable_existing_loggers=True)

logger = logging.getLogger(__name__)
