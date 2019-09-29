import logging
import logging.config

from .dataset import TextDataset

from .tuning import Tuner

from .modeling import TrainerFactory, Experiment
from .build_coherence import CoherenceFilesBuilder
from .pipe_handler import PipeHandler

# logging.config.fileConfig(path.join(path.dirname(path.realpath(__file__)), 'logging.ini'), disable_existing_loggers=True)
#
# logger = logging.getLogger(__name__)

from .discreetization import PoliticalSpectrumManager

political_spectrum = PoliticalSpectrumManager()

