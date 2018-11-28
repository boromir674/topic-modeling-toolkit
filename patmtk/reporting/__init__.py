import os
from patm.definitions import COLLECTIONS_DIR_PATH, RESULTS_DIR_NAME
from .model_selection import ResultsHandler

results_handler = ResultsHandler(collection_root_path=COLLECTIONS_DIR_PATH, results_dir_name=RESULTS_DIR_NAME)

from .reporter import ModelReporter
model_reporter = ModelReporter(COLLECTIONS_DIR_PATH, results_dir_name=RESULTS_DIR_NAME)

# from .graph_maker import GraphMaker
# graph_maker = GraphMaker(os.path.join(COLLECTIONS_DIR_PATH))
